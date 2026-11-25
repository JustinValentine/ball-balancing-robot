#include "vision.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>

Tracker::Tracker(const std::string& calibration, const std::string& settings, const Rig& rig) {
    cv::FileStorage c(calibration, cv::FileStorage::READ), f(settings, cv::FileStorage::READ);
    require(c.isOpened() && f.isOpened(), "Cannot open calibration or tracker settings");
    require(setting<std::string>(c,"left_guid") == rig.left && setting<std::string>(c,"right_guid") == rig.right,
            "Calibration camera IDs/order disagree with the input");
    size = {setting<int>(c,"width"), setting<int>(c,"height")};
    K[0] = setting<cv::Mat>(c,"K1"); K[1] = setting<cv::Mat>(c,"K2");
    D[0] = setting<cv::Mat>(c,"D1"); D[1] = setting<cv::Mat>(c,"D2");
    R = setting<cv::Mat>(c,"R"); T = setting<cv::Mat>(c,"T");
    P[0] = cv::Mat::eye(3,4,CV_64F);
    cv::hconcat(R,T,P[1]);
    if (!f["acrylic_thickness_mm"].empty()) {
        thickness = setting<double>(f,"acrylic_thickness_mm");
        index = setting<double>(f,"acrylic_refractive_index");
        require(std::isfinite(thickness) && thickness >= 0 && std::isfinite(index) && index > 1,
                "Invalid parallel plate optical settings");
    }
    threshold = setting<int>(f,"threshold"); bright = setting<int>(f,"bright_ball") != 0;
    roi = setting<int>(f,"roi_half_size_px");
    minRadius = setting<double>(f,"min_radius_px"); maxRadius = setting<double>(f,"max_radius_px");
    minCircularity = setting<double>(f,"min_circularity"); maxError = setting<double>(f,"max_reprojection_px");
    diameter = setting<double>(f,"ball_diameter_mm"); diameterTolerance = setting<double>(f,"diameter_tolerance_fraction");
    if (!f["balls"].empty()) count = setting<int>(f,"balls");
    require(count >= 1 && count <= 4, "balls must be between 1 and 4");
    require(threshold >= 0 && threshold <= 255 && roi > 0 && roi <= 4096 && minRadius > 0 && maxRadius > minRadius
            && minCircularity > 0 && minCircularity <= 1 && maxError > 0 && diameter >= 0 && diameterTolerance > 0,
            "Invalid detector settings");
}

std::vector<Tracker::Ball> Tracker::detect(int side, const cv::Mat& image, bool local) const {
    cv::Rect rect({}, size);
    if (local && previous[side]) {
        cv::Point p = *previous[side];
        rect &= cv::Rect(p.x-roi, p.y-roi, 2*roi+1, 2*roi+1);
    }
    cv::Mat mask;
    cv::threshold(image(rect), mask, threshold, 255, bright ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, rect.tl());
    std::vector<Ball> candidates;
    for (const auto& points : contours) {
        auto box = cv::boundingRect(points);
        if (box.x == rect.x || box.y == rect.y || box.br().x == rect.br().x || box.br().y == rect.br().y) continue;
        double area = cv::contourArea(points), radius = std::sqrt(area/CV_PI);
        if (radius < minRadius || radius > maxRadius || points.size() < 8) continue;
        double perimeter = cv::arcLength(points,true);
        double quality = 4*CV_PI*area/(perimeter*perimeter);
        if (quality < minCircularity) continue;
        std::vector<cv::Point2d> boundary, normalized;
        size_t samples = std::min<size_t>(48, points.size());
        for (size_t i=0; i<samples; ++i) boundary.emplace_back(points[i*points.size()/samples]);
        cv::undistortPoints(boundary, normalized, K[side], D[side], cv::noArray(), cv::noArray(),
                            {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 1e-7});
        // Tangent rays of a sphere lie on a small circle on the unit sphere.
        // Its plane normal points to the 3D center even for an off-axis ball.
        cv::Mat rays(static_cast<int>(samples),3,CV_64F);
        for (size_t i=0; i<samples; ++i) {
            auto p = normalized[i];
            auto v = cv::normalize(cv::Vec3d(p.x,p.y,1));
            for (int j=0;j<3;++j) rays.at<double>(i,j) = v[j];
        }
        cv::PCA fit(rays, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Vec3d axis(fit.eigenvectors.ptr<double>(2));
        if (axis[2] < 0) axis = -axis;
        if (!(axis[2] > 1e-6)) continue;
        double cosine = axis.dot(cv::Vec3d(fit.mean.ptr<double>()));
        auto moments = cv::moments(points);
        candidates.push_back({{moments.m10/moments.m00,moments.m01/moments.m00},
                              {axis[0]/axis[2],axis[1]/axis[2]},std::sqrt(std::max(0.,1-cosine*cosine)),quality,rays});
    }
    std::sort(candidates.begin(),candidates.end(),[](const Ball& a,const Ball& b){return a.quality>b.quality;});
    if (candidates.size()>8) candidates.resize(8);
    return candidates;
}

// After both parallel interfaces the outgoing line is parallel to the incident
// ray. Extrapolating it back to the normal through the camera gives this origin.
// Slab distance cancels for a plane-parallel sheet surrounded by air.
cv::Vec3d Tracker::slabOrigin(cv::Vec3d ray, cv::Vec3d normal) const {
    double cosine = ray.dot(normal);
    return normal * (thickness * (1 - cosine / std::sqrt(index*index - 1 + cosine*cosine)));
}

bool Tracker::refineSphere(const Ball& left, const Ball& right, cv::Vec3d normal,
                           cv::Vec3d& center, double radius) const {
    struct Ray { cv::Vec3d origin, direction; };
    std::array<Ray,96> rays;
    int count = 0;
    cv::Matx33d toLeft = cv::Matx33d(R).t();
    cv::Vec3d rightCenter = -(toLeft * cv::Vec3d(T));
    for (const Ball* ball : {&left,&right}) {
        for (int i=0; i<ball->directions.rows; ++i) {
            cv::Vec3d dir(ball->directions.ptr<double>(i));
            cv::Vec3d origin{};
            if (ball == &right) { dir = toLeft * dir; origin = rightCenter; }
            if (!(dir.dot(normal) > 0)) return false;
            rays[count++] = {origin + slabOrigin(dir,normal), dir};
        }
    }
    // Every outgoing silhouette ray is tangent to the same sphere. Fit radius
    // as well as position, since ball diameter and threshold bias can vary.
    for (int iter=0; iter<5; ++iter) {
        cv::Matx44d h = cv::Matx44d::zeros();
        cv::Vec4d rhs{};
        for (int i=0; i<count; ++i) {
            const Ray& ray = rays[i];
            auto offset = center-ray.origin;
            double depth = offset.dot(ray.direction);
            if (!(depth > 0)) return false;
            auto perpendicular = offset-depth*ray.direction;
            double distance = cv::norm(perpendicular);
            if (!(distance > 1e-9)) return false;
            auto grad = perpendicular / distance;
            cv::Vec4d j(grad[0],grad[1],grad[2],-1);
            h += j*j.t();
            rhs += j*(radius-distance);
        }
        cv::Vec4d step;
        if (!cv::solve(h,rhs,step,cv::DECOMP_CHOLESKY)) return false;
        center += cv::Vec3d(step[0],step[1],step[2]);
        radius += step[3];
        if (cv::norm(step) < 1e-5) break;
    }
    return radius > 0 && std::isfinite(cv::norm(center));
}

std::vector<Tracker::Pair> Tracker::pairs(const std::vector<Ball>& a, const std::vector<Ball>& b,
                                          cv::Vec3d normal) const {
    std::vector<Pair> out;
    cv::Vec3d rightNormal = cv::Matx33d(R) * normal;
    for (size_t i=0; i<a.size(); ++i) for (size_t k=0; k<b.size(); ++k) {
        const Ball &l=a[i], &r=b[k];
        cv::Vec3d oL{}, oR{};
        cv::Mat projection;
        if (thickness > 0) {
            auto dl=cv::normalize(cv::Vec3d(l.ray.x,l.ray.y,1));
            auto dr=cv::normalize(cv::Vec3d(r.ray.x,r.ray.y,1));
            if (dl.dot(normal)<=0 || dr.dot(rightNormal)<=0) continue;
            oL=slabOrigin(dl,normal); oR=slabOrigin(dr,rightNormal);
            cv::Mat adjusted = R*cv::Mat(oL)+T-cv::Mat(oR);
            cv::hconcat(R,adjusted,projection);
        }
        cv::Mat h;
        cv::triangulatePoints(P[0],thickness > 0 ? projection : P[1],std::vector<cv::Point2d>{l.ray},std::vector<cv::Point2d>{r.ray},h);
        double w = h.at<double>(3);
        if (std::abs(w) < 1e-12) continue;
        cv::Vec3d x(h.at<double>(0)/w,h.at<double>(1)/w,h.at<double>(2)/w);
        cv::Mat xr = R*cv::Mat(x+oL)+T-cv::Mat(oR);
        cv::Vec3d y(xr.ptr<double>());
        if (!(x[2]>0 && y[2]>0)) continue;
        auto residual = [&](int s, cv::Vec3d p, cv::Point2d obs) {
            return std::hypot((p[0]/p[2]-obs.x)*K[s].at<double>(0,0),
                              (p[1]/p[2]-obs.y)*K[s].at<double>(1,1));
        };
        double err = std::max(residual(0,x,l.ray),residual(1,y,r.ray));
        double d1=2*cv::norm(x)*l.sineRadius, d2=2*cv::norm(y)*r.sineRadius;
        if (err > maxError || std::abs(d1-d2) > diameterTolerance*std::max(d1,d2)) continue;
        if (diameter && std::max(std::abs(d1-diameter),std::abs(d2-diameter)) > diameterTolerance*diameter) continue;
        out.push_back({err/maxError + (2-l.quality-r.quality), (d1+d2)*0.25, x+oL, err, i, k});
    }
    return out;
}

std::optional<Measurement> Tracker::update(const cv::Mat& left, const cv::Mat& right, cv::Vec3d normal) {
    require(left.size() == size && right.size() == size && left.type()==CV_8UC1 && right.type()==CV_8UC1,
            "Images must match the calibrated MONO8 image size");
    for (bool local : {true,false}) {
        if (local && !previous[0]) continue;
        auto a = detect(0,left,local), b = detect(1,right,local);
        auto found = pairs(a,b,normal);
        if (previous[0])
            for (auto& p : found)
                p.score += (cv::norm(a[p.left].pixel-*previous[0])+cv::norm(b[p.right].pixel-*previous[1]))/(2*roi);
        auto best = std::min_element(found.begin(),found.end(),[](const Pair& x,const Pair& y){return x.score<y.score;});
        if (best == found.end()) continue;
        Measurement m{best->position,best->error};
        if (!thickness || refineSphere(a[best->left],b[best->right],normal,m.position,best->radius)) {
            previous[0]=a[best->left].pixel; previous[1]=b[best->right].pixel;
            return m;
        }
    }
    previous[0].reset(); previous[1].reset();
    return {};
}

std::vector<Measurement> Tracker::updateAll(const cv::Mat& left, const cv::Mat& right, cv::Vec3d normal) {
    if (count == 1) {
        auto m = update(left,right,normal);
        return m ? std::vector<Measurement>{*m} : std::vector<Measurement>{};
    }
    require(left.size() == size && right.size() == size && left.type()==CV_8UC1 && right.type()==CV_8UC1,
            "Images must match the calibrated MONO8 image size");
    auto a = detect(0,left,false), b = detect(1,right,false);
    auto found = pairs(a,b,normal);
    std::sort(found.begin(),found.end(),[](const Pair& x,const Pair& y){return x.score<y.score;});
    // Best matches first; each outline is used once and balls must not overlap.
    std::vector<Measurement> out;
    std::vector<bool> usedL(a.size()), usedR(b.size());
    for (const auto& p : found) {
        if (static_cast<int>(out.size()) == count) break;
        if (usedL[p.left] || usedR[p.right]) continue;
        Measurement m{p.position,p.error};
        if (thickness && !refineSphere(a[p.left],b[p.right],normal,m.position,p.radius)) continue;
        bool apart = true;
        for (const auto& o : out) apart &= cv::norm(o.position-m.position) > 2*p.radius*0.9;
        if (!apart) continue;
        usedL[p.left] = usedR[p.right] = true;
        out.push_back(m);
    }
    return out;
}
