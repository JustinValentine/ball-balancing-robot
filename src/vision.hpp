#pragma once
#include <opencv2/core.hpp>
#include <dc1394/dc1394.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T> T setting(const cv::FileStorage& file, const char* name) {
    require(!file[name].empty(), std::string("Missing setting ") + name);
    T value{};
    file[name] >> value;
    return value;
}

struct Rig { std::string left, right; };
struct Frame {
    cv::Mat pixels;
    int64_t time_us = 0;
};
using Context = std::unique_ptr<dc1394_t, decltype(&dc1394_free)>;
Context context();
void list(dc1394_t*);
class Camera {
    std::unique_ptr<dc1394camera_t, decltype(&dc1394_camera_free)> cam;
    bool capturing = false, streaming = false;
public:
    unsigned long received = 0, discarded = 0;
    Camera(dc1394_t*, const std::string& guid);
    void configure(const cv::FileStorage&);
    void start();
    int fd() const;
    bool newest(Frame&);
    ~Camera();
};

struct Measurement {
    cv::Vec3d position;
    double error;
};
class Tracker {
    struct Ball {
        cv::Point2d pixel, ray;
        double sineRadius, quality;
        cv::Mat directions;
    };
    cv::Mat K[2], D[2], P[2], R, T;
    cv::Size size;
    double thickness = 0, index = 1.49;
    cv::Vec3d slabOrigin(cv::Vec3d ray, cv::Vec3d normal) const;
    bool refineSphere(const Ball& left, const Ball& right, cv::Vec3d normal,
                      cv::Vec3d& center, double radius) const;
    std::optional<cv::Point2d> previous[2];
    int threshold, roi;
    bool bright;
    double minRadius, maxRadius, minCircularity, maxError, diameter, diameterTolerance;
    std::vector<Ball> detect(int side, const cv::Mat& image, bool local) const;
public:
    Tracker(const std::string& calibration, const std::string& settings, const Rig&);
    std::optional<Measurement> update(const cv::Mat& left, const cv::Mat& right, cv::Vec3d normal);
};
