// Ball balancing and bouncing controller for the 3-RRS plate. Header only, C++17, no dependencies.
//
// Units are SI inside the controller: metres, seconds, radians, newtons. The base frame is the
// Fusion assembly frame of platform.json (z up). Crank angles are measured from the CAD home pose
// about each leg's motor axis.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ballctl {

constexpr double kPi = 3.14159265358979323846;
constexpr double kG = 9.80665;

// ---------------------------------------------------------------- small linear algebra

using V3 = std::array<double, 3>;
using Q3 = std::array<double, 3>;  // Three crank angles.

inline V3 operator+(const V3& a, const V3& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
inline V3 operator-(const V3& a, const V3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline V3 operator*(const V3& a, double s) { return {a[0]*s, a[1]*s, a[2]*s}; }
inline V3 operator*(double s, const V3& a) { return a*s; }
inline double dot(const V3& a, const V3& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline V3 cross(const V3& a, const V3& b) {
  return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 unit(const V3& a) { return a*(1.0/norm(a)); }

struct M3 {
  double m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  V3 operator*(const V3& v) const {
    return {m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2], m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
            m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]};
  }
  M3 operator*(const M3& b) const {
    M3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) r.m[i][j] = m[i][0]*b.m[0][j]+m[i][1]*b.m[1][j]+m[i][2]*b.m[2][j];
    return r;
  }
  M3 T() const {
    M3 r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) r.m[i][j] = m[j][i];
    return r;
  }
  V3 col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
  static M3 cols(const V3& a, const V3& b, const V3& c) {
    M3 r;
    for (int i = 0; i < 3; i++) { r.m[i][0] = a[i]; r.m[i][1] = b[i]; r.m[i][2] = c[i]; }
    return r;
  }
};

// Right-handed rotation by angle about a unit axis.
inline M3 rotation(const V3& k, double angle) {
  const double c = std::cos(angle), s = std::sin(angle), v = 1-c;
  M3 r;
  r.m[0][0] = c+k[0]*k[0]*v;      r.m[0][1] = k[0]*k[1]*v-k[2]*s; r.m[0][2] = k[0]*k[2]*v+k[1]*s;
  r.m[1][0] = k[1]*k[0]*v+k[2]*s; r.m[1][1] = c+k[1]*k[1]*v;      r.m[1][2] = k[1]*k[2]*v-k[0]*s;
  r.m[2][0] = k[2]*k[0]*v-k[1]*s; r.m[2][1] = k[2]*k[1]*v+k[0]*s; r.m[2][2] = c+k[2]*k[2]*v;
  return r;
}

// Minimal rotation taking +z to the unit normal with horizontal components (nx, ny).
inline M3 tilt(double nx, double ny) {
  const double s = std::hypot(nx, ny);
  if (s < 1e-15) return M3{};
  return rotation({-ny/s, nx/s, 0}, std::asin(std::min(1.0, s)));
}

template <int N, int M>
struct Mat {
  std::array<double, N*M> d{};
  double& operator()(int i, int j) { return d[i*M+j]; }
  double operator()(int i, int j) const { return d[i*M+j]; }
  static Mat eye() {
    Mat r;
    for (int i = 0; i < std::min(N, M); i++) r(i, i) = 1;
    return r;
  }
  Mat<M, N> T() const {
    Mat<M, N> r;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++) r(j, i) = (*this)(i, j);
    return r;
  }
  Mat operator+(const Mat& b) const { Mat r; for (int i = 0; i < N*M; i++) r.d[i] = d[i]+b.d[i]; return r; }
  Mat operator-(const Mat& b) const { Mat r; for (int i = 0; i < N*M; i++) r.d[i] = d[i]-b.d[i]; return r; }
  Mat operator*(double s) const { Mat r; for (int i = 0; i < N*M; i++) r.d[i] = d[i]*s; return r; }
};

template <int N, int K, int M>
Mat<N, M> operator*(const Mat<N, K>& a, const Mat<K, M>& b) {
  Mat<N, M> r;
  for (int i = 0; i < N; i++)
    for (int k = 0; k < K; k++) {
      const double x = a(i, k);
      if (x != 0)
        for (int j = 0; j < M; j++) r(i, j) += x*b(k, j);
    }
  return r;
}

template <int N>
bool invert(Mat<N, N> a, Mat<N, N>* out) {
  Mat<N, N> r = Mat<N, N>::eye();
  for (int c = 0; c < N; c++) {
    int p = c;
    for (int i = c+1; i < N; i++)
      if (std::fabs(a(i, c)) > std::fabs(a(p, c))) p = i;
    if (std::fabs(a(p, c)) < 1e-300) return false;
    for (int j = 0; j < N; j++) { std::swap(a(c, j), a(p, j)); std::swap(r(c, j), r(p, j)); }
    const double s = 1/a(c, c);
    for (int j = 0; j < N; j++) { a(c, j) *= s; r(c, j) *= s; }
    for (int i = 0; i < N; i++)
      if (i != c && a(i, c) != 0) {
        const double f = a(i, c);
        for (int j = 0; j < N; j++) { a(i, j) -= f*a(c, j); r(i, j) -= f*r(c, j); }
      }
  }
  *out = r;
  return true;
}

// Solve f(x) = 0 for three unknowns with a numerical Jacobian.
inline bool newton3(const std::function<bool(const V3&, V3*)>& f, V3* x, double tol, int iterations,
                    double step = 1e-7) {
  for (int it = 0; it < iterations; it++) {
    V3 r;
    if (!f(*x, &r)) return false;
    if (std::max({std::fabs(r[0]), std::fabs(r[1]), std::fabs(r[2])}) < tol) return true;
    Mat<3, 3> J;
    for (int j = 0; j < 3; j++) {
      V3 y = *x;
      y[j] += step;
      V3 ry;
      if (!f(y, &ry)) return false;
      for (int i = 0; i < 3; i++) J(i, j) = (ry[i]-r[i])/step;
    }
    Mat<3, 3> Ji;
    if (!invert(J, &Ji)) return false;
    for (int i = 0; i < 3; i++) (*x)[i] -= Ji(i, 0)*r[0]+Ji(i, 1)*r[1]+Ji(i, 2)*r[2];
  }
  V3 r;
  return f(*x, &r) && std::max({std::fabs(r[0]), std::fabs(r[1]), std::fabs(r[2])}) < tol;
}

// ---------------------------------------------------------------- minimal JSON reader

struct Json {
  enum Type { Null, Number, String, Array, Object, Boolean } type = Null;
  double number = 0;
  bool boolean = false;
  std::string string;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  static Json parse(const std::string& text) {
    size_t i = 0;
    Json j = value(text, i);
    skip(text, i);
    if (i != text.size()) throw std::runtime_error("Trailing characters in JSON");
    return j;
  }
  static Json load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
  }
  bool has(const std::string& key) const { return type == Object && object.count(key); }
  const Json& operator[](const std::string& key) const {
    auto it = object.find(key);
    if (type != Object || it == object.end()) throw std::runtime_error("Missing JSON key " + key);
    return it->second;
  }
  const Json& operator[](size_t i) const {
    if (type != Array || i >= array.size()) throw std::runtime_error("JSON index out of range");
    return array[i];
  }
  double num() const {
    if (type != Number) throw std::runtime_error("Expected a JSON number");
    return number;
  }
  double get(const std::string& key, double fallback) const { return has(key) ? (*this)[key].num() : fallback; }
  const Json& section(const std::string& key) const {
    static const Json empty = [] { Json j; j.type = Object; return j; }();
    return has(key) ? (*this)[key] : empty;
  }
  V3 v3() const {
    if (type != Array || array.size() != 3) throw std::runtime_error("Expected a JSON 3-vector");
    return {array[0].num(), array[1].num(), array[2].num()};
  }

 private:
  static void skip(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
  }
  static Json value(const std::string& s, size_t& i) {
    skip(s, i);
    if (i >= s.size()) throw std::runtime_error("Unexpected end of JSON");
    Json j;
    const char c = s[i];
    if (c == '{') {
      j.type = Object;
      i++;
      skip(s, i);
      if (s[i] == '}') { i++; return j; }
      while (true) {
        skip(s, i);
        std::string key = value(s, i).string;
        skip(s, i);
        if (s[i++] != ':') throw std::runtime_error("Expected : in JSON");
        j.object[key] = value(s, i);
        skip(s, i);
        if (s[i] == ',') { i++; continue; }
        if (s[i++] != '}') throw std::runtime_error("Expected } in JSON");
        return j;
      }
    }
    if (c == '[') {
      j.type = Array;
      i++;
      skip(s, i);
      if (s[i] == ']') { i++; return j; }
      while (true) {
        j.array.push_back(value(s, i));
        skip(s, i);
        if (s[i] == ',') { i++; continue; }
        if (s[i++] != ']') throw std::runtime_error("Expected ] in JSON");
        return j;
      }
    }
    if (c == '"') {
      j.type = String;
      for (i++; i < s.size() && s[i] != '"'; i++) {
        if (s[i] == '\\' && i+1 < s.size()) i++;
        j.string += s[i];
      }
      i++;
      return j;
    }
    if (s.compare(i, 4, "true") == 0) { j.type = Boolean; j.boolean = true; i += 4; return j; }
    if (s.compare(i, 5, "false") == 0) { j.type = Boolean; i += 5; return j; }
    if (s.compare(i, 4, "null") == 0) { i += 4; return j; }
    size_t used = 0;
    j.type = Number;
    j.number = std::stod(s.substr(i, 40), &used);
    i += used;
    return j;
  }
};

// ---------------------------------------------------------------- kinematics

// Plate pose: spherical joint i sits at origin + t + R*s[i]. At home t = 0 and R = I.
struct Pose {
  M3 R;
  V3 t{0, 0, 0};
};

// Commanded plate posture: yaw about the plate normal and the horizontal components of the
// normal. These are the three directions the cranks control. The legs set x, y and height: turning
// all cranks together twists the plate, and its height falls roughly with the square of the
// twist, so height is controlled through yaw away from zero twist.
struct Posture {
  double yaw = 0, nx = 0, ny = 0;
};

class Kinematics {
 public:
  Kinematics(const Json& platform, double surface_offset)
      : surface_offset_(surface_offset) {
    const Json& legs = platform["legs"];
    V3 tips[3];
    for (int i = 0; i < 3; i++) {
      motor_[i] = legs[i]["motor_mm"].v3()*1e-3;
      axis_[i] = unit(legs[i]["axis"].v3());
      const V3 elbow = legs[i]["elbow_mm"].v3()*1e-3;
      tips[i] = legs[i]["ball_joint_mm"].v3()*1e-3;
      crank_[i] = elbow-motor_[i];
      rod_[i] = tips[i]-elbow;
    }
    origin_ = (tips[0]+tips[1]+tips[2])*(1.0/3);
    for (int i = 0; i < 3; i++) {
      socket_[i] = tips[i]-origin_;
      plane_[i] = dot(axis_[i], tips[i]-motor_[i]);
      const V3 cp = crank_[i]-axis_[i]*dot(axis_[i], crank_[i]);
      const V3 rp = rod_[i]-axis_[i]*dot(axis_[i], rod_[i]);
      crank_len_[i] = norm(cp);
      rod_len_[i] = norm(rp);
      u_[i] = cp*(1/crank_len_[i]);
      w_[i] = cross(axis_[i], u_[i]);
    }
    for (int i = 0; i < 3; i++) side_[i] = norm(tips[i]-tips[(i+1)%3]);
    // Pick the elbow branch that reproduces the CAD home pose.
    for (int i = 0; i < 3; i++) {
      branch_[i] = 1;
      double q = 0;
      if (!leg(i, tips[i], &q)) throw std::runtime_error("Home pose outside the leg workspace");
      if (std::fabs(q) > 1e-9) {
        branch_[i] = -1;
        if (!leg(i, tips[i], &q) || std::fabs(q) > 1e-9)
          throw std::runtime_error("Leg inverse kinematics does not reproduce the home pose");
      }
    }
  }

  double surface_offset() const { return surface_offset_; }
  const V3& origin() const { return origin_; }
  const V3& socket(int i) const { return socket_[i]; }

  // Plate surface centre and unit normal for a pose.
  V3 surface(const Pose& p) const { return origin_+p.t+p.R*V3{0, 0, surface_offset_}; }
  static V3 normal(const Pose& p) { return p.R.col(2); }

  // Full pose for a commanded posture. Solves x, y and height.
  bool pose(const Posture& c, Pose* out) const {
    if (std::hypot(c.nx, c.ny) > 0.5) return false;
    const M3 R = tilt(c.nx, c.ny)*rotation({0, 0, 1}, c.yaw);
    V3 x{0, 0, -0.5*c.yaw*c.yaw*0.15};  // Height grows roughly with the square of yaw.
    auto f = [&](const V3& t, V3* r) {
      for (int i = 0; i < 3; i++) (*r)[i] = dot(axis_[i], origin_+t+R*socket_[i]-motor_[i])-plane_[i];
      return true;
    };
    if (!newton3(f, &x, 1e-12, 12)) return false;
    out->R = R;
    out->t = x;
    return true;
  }

  // Height of the plate frame for a posture (metres relative to the zero-twist home).
  bool height(const Posture& c, double* z) const {
    Pose p;
    if (!pose(c, &p)) return false;
    *z = p.t[2];
    return true;
  }

  // Yaw on the same side of zero twist as yaw_guess that gives height z at this tilt.
  bool yaw_for_height(double z, double nx, double ny, double* yaw) const {
    double y = *yaw;
    for (int it = 0; it < 30; it++) {
      double z0, z1;
      if (!height({y, nx, ny}, &z0) || !height({y+1e-6, nx, ny}, &z1)) return false;
      const double slope = (z1-z0)/1e-6;
      if (std::fabs(z0-z) < 1e-10) { *yaw = y; return true; }
      if (std::fabs(slope) < 1e-4) return false;
      double step = (z-z0)/slope;
      step = std::clamp(step, -0.1, 0.1);
      if ((y+step)*y <= 0) step = -0.5*y;  // Stay on this side of zero twist.
      y += step;
    }
    return false;
  }

  // Crank angle that places leg i's spherical joint at tip.
  bool leg(int i, const V3& tip, double* q) const {
    const V3 d = tip-motor_[i];
    const V3 dp = d-axis_[i]*dot(axis_[i], d);
    const double D = norm(dp);
    const double c = (D*D+crank_len_[i]*crank_len_[i]-rod_len_[i]*rod_len_[i])/(2*D*crank_len_[i]);
    if (!(std::fabs(c) < 1)) return false;
    *q = std::atan2(dot(dp, w_[i]), dot(dp, u_[i])) + branch_[i]*std::acos(c);
    return true;
  }

  bool inverse(const Pose& p, Q3* q) const {
    for (int i = 0; i < 3; i++)
      if (!leg(i, origin_+p.t+p.R*socket_[i], &(*q)[i])) return false;
    return true;
  }

  bool inverse(const Posture& c, Q3* q) const {
    Pose p;
    return pose(c, &p) && inverse(p, q);
  }

  // Plate pose from measured crank angles. beta is the warm start for the passive elbow angles.
  bool forward(const Q3& q, Pose* out, V3* beta) const {
    V3 elbow[3];
    for (int i = 0; i < 3; i++) elbow[i] = motor_[i]+rotation(axis_[i], q[i])*crank_[i];
    auto tips = [&](const V3& b, V3* p) {
      for (int i = 0; i < 3; i++) p[i] = elbow[i]+rotation(axis_[i], q[i]+b[i])*rod_[i];
    };
    auto f = [&](const V3& b, V3* r) {
      V3 p[3];
      tips(b, p);
      for (int i = 0; i < 3; i++) (*r)[i] = norm(p[i]-p[(i+1)%3])-side_[i];
      return true;
    };
    V3 b = *beta;
    if (!newton3(f, &b, 1e-11, 20)) return false;
    V3 p[3];
    tips(b, p);
    // The CAD triangle is rigid, so the frame built on its edges is exact.
    auto frame = [](const V3* x) {
      const V3 e1 = unit(x[1]-x[0]);
      const V3 e3 = unit(cross(x[1]-x[0], x[2]-x[0]));
      return M3::cols(e1, cross(e3, e1), e3);
    };
    const M3 R = frame(p)*frame(socket_).T();
    if (R.m[2][2] < 0.5) return false;
    out->R = R;
    out->t = (p[0]+p[1]+p[2])*(1.0/3)-origin_;
    *beta = b;
    return true;
  }

  // Crank angles, elbow points and each rod's rotation about its leg axis from home, for a pose.
  bool legs_state(const Pose& p, Q3* q, V3* elbow, double* rod_angle) const {
    if (!inverse(p, q)) return false;
    for (int i = 0; i < 3; i++) {
      elbow[i] = motor_[i]+rotation(axis_[i], (*q)[i])*crank_[i];
      const V3 rod = origin_+p.t+p.R*socket_[i]-elbow[i];
      const V3 r0 = rod_[i]-axis_[i]*dot(axis_[i], rod_[i]);
      const V3 r1 = rod-axis_[i]*dot(axis_[i], rod);
      rod_angle[i] = std::atan2(dot(cross(r0, r1), axis_[i]), dot(r0, r1));
    }
    return true;
  }
  const V3& axis(int i) const { return axis_[i]; }
  const V3& leg_u(int i) const { return u_[i]; }
  const V3& leg_w(int i) const { return w_[i]; }

  // Posture (yaw, nx, ny) of a pose: R = tilt(n)*Rz(yaw).
  static Posture posture(const Pose& p) {
    const V3 n = normal(p);
    const M3 M = tilt(n[0], n[1]).T()*p.R;
    return {std::atan2(M.m[1][0], M.m[0][0]), n[0], n[1]};
  }

  // dq/d(yaw, nx, ny) by central differences.
  bool jacobian(const Posture& c, Mat<3, 3>* J) const {
    const double h = 1e-5;
    for (int j = 0; j < 3; j++) {
      Posture a = c, b = c;
      (j == 0 ? a.yaw : j == 1 ? a.nx : a.ny) += h;
      (j == 0 ? b.yaw : j == 1 ? b.nx : b.ny) -= h;
      Q3 qa, qb;
      if (!inverse(a, &qa) || !inverse(b, &qb)) return false;
      for (int i = 0; i < 3; i++) (*J)(i, j) = (qa[i]-qb[i])/(2*h);
    }
    return true;
  }

 private:
  V3 motor_[3], axis_[3], crank_[3], rod_[3], socket_[3], u_[3], w_[3], origin_{};
  double plane_[3], crank_len_[3], rod_len_[3], side_[3];
  int branch_[3];
  double surface_offset_;
};

// ---------------------------------------------------------------- trajectories

// Quintic segment between full position, velocity and acceleration states.
struct Quintic {
  double t0 = 0, T = 1;
  double c[6] = {0, 0, 0, 0, 0, 0};

  static Quintic make(double t0, double T, double p0, double v0, double a0, double p1, double v1, double a1) {
    Quintic q;
    q.t0 = t0;
    q.T = std::max(T, 1e-4);
    const double t = q.T, t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    q.c[0] = p0;
    q.c[1] = v0;
    q.c[2] = a0/2;
    const double dp = p1-p0-v0*t-a0*t2/2, dv = v1-v0-a0*t, da = a1-a0;
    q.c[3] = (10*dp-4*dv*t+da*t2/2)/t3;
    q.c[4] = (-15*dp+7*dv*t-da*t2)/t4;
    q.c[5] = (6*dp-3*dv*t+da*t2/2)/t5;
    return q;
  }
  void eval(double time, double* p, double* v, double* a) const {
    const double s = std::clamp(time-t0, 0.0, T);
    *p = c[0]+s*(c[1]+s*(c[2]+s*(c[3]+s*(c[4]+s*c[5]))));
    *v = c[1]+s*(2*c[2]+s*(3*c[3]+s*(4*c[4]+s*5*c[5])));
    *a = 2*c[2]+s*(6*c[3]+s*(12*c[4]+s*20*c[5]));
    if (time > t0+T) { *p += *v*(time-t0-T); *a = 0; }  // Coast at the final velocity.
  }
  double end() const { return t0+T; }
};

// A sequence of segments for one coordinate. Before the first segment and after the last one the
// coordinate holds (or coasts at) the boundary state.
struct Track {
  std::vector<Quintic> seg;
  double hold = 0;
  void eval(double t, double* p, double* v, double* a) const {
    if (seg.empty()) { *p = hold; *v = *a = 0; return; }
    if (t <= seg.front().t0) { seg.front().eval(seg.front().t0, p, v, a); return; }
    for (const auto& s : seg)
      if (t <= s.end()) { s.eval(t, p, v, a); return; }
    seg.back().eval(t, p, v, a);
  }
  void reset(double p) { seg.clear(); hold = p; }
  // Replace everything from time t with a segment from the current state to the target.
  void to(double t, double T, double p1, double v1, double a1) {
    double p, v, a;
    eval(t, &p, &v, &a);
    seg.assign(1, Quintic::make(t, T, p, v, a, p1, v1, a1));
  }
  void then(double T, double p1, double v1, double a1) {
    double p, v, a;
    const double t = seg.empty() ? 0 : seg.back().end();
    eval(t, &p, &v, &a);
    seg.push_back(Quintic::make(t, T, p, v, a, p1, v1, a1));
  }
  void prune(double t) {
    while (seg.size() > 1 && seg.front().end() < t) seg.erase(seg.begin());
  }
};

// ---------------------------------------------------------------- configuration

struct Config {
  // Geometry and frames.
  M3 left_from_base;            // Rotation of base vectors into the left optical frame.
  V3 left_origin{0, 0, 0};      // Left optical centre in the base frame.
  double surface_offset = 0.02855;
  // Ball.
  double radius = 0.020, rolling_gain = 0.6, restitution = 0.88, tangential = 0.6, drag = 0.126;
  // Vision.
  double exposure_to_timestamp = 0.003, noise_xy = 0.0004, noise_z = 0.0010, lost_after = 0.15,
         contact_noise = 0.1;
  // Limits.
  double crank_min = -1.45, crank_max = 0.3, tilt_max = 0.14, yaw_span = 0.2, crank_speed_max = 10,
         crank_accel_max = 800;
  // Plate model for feedforward.
  double plate_mass = 1.0, plate_com = 0.022, plate_tilt_inertia = 0.011;
  // Gravity model parameters (see GravityModel), empty until calibrated.
  std::vector<double> gravity;
  // Gravity calibration grid and timing.
  double cal_yaw_step = 8*kPi/180, cal_tilt = 7*kPi/180, cal_speed = 0.15, cal_settle = 0.4, cal_average = 0.8,
         cal_approach = 0.03;
  // Operation.
  double idle_yaw = -20*kPi/180, command_lead = 0;
  // Balancing.
  bool balance_accel_ff = false;
  double balance_q_rate = 20, balance_soft_start = 0.4;
  double balance_tau = 0.06, balance_q_pos = 400, balance_q_vel = 4, balance_q_int = 2000,
         balance_r = 40, reference_rate = 8, target_radius = 0.15;
  // Bouncing.
  double stroke_time = 0.08, decel = 1.5*kG, return_time = 0.15, hold_window = 0.015, freeze = 0.025,
         plate_speed_max = 0.8, lateral_gain = 0.7, lateral_integral = 0.15, launch_speed = 0.6,
         launch_dip = 0.012, apex_default = 0.10;

  static Config load(const std::string& path, Json* platform_out) {
    const Json j = Json::load(path);
    std::string dir = path.substr(0, path.find_last_of("/\\")+1);
    if (path.find_last_of("/\\") == std::string::npos) dir = "";
    std::string pf = j["platform"].string;
    if (!pf.empty() && pf[0] != '/') pf = dir+pf;
    *platform_out = Json::load(pf);
    Config c;
    const Json& rot = (*platform_out)["left_from_base_rotation"];
    for (int r = 0; r < 3; r++)
      for (int k = 0; k < 3; k++) c.left_from_base.m[r][k] = rot[r][k].num();
    c.left_origin = j["left_origin_mm"].v3()*1e-3;
    c.surface_offset = j["surface_offset_mm"].num()*1e-3;
    const Json& b = j.section("ball");
    c.radius = b.get("radius_mm", 20)*1e-3;
    c.rolling_gain = 1/(1+b.get("inertia_ratio", 2.0/3));
    c.restitution = b.get("restitution", c.restitution);
    c.tangential = b.get("tangential_restitution", c.tangential);
    c.drag = b.get("drag_per_m", c.drag);
    const Json& v = j.section("vision");
    c.exposure_to_timestamp = v.get("exposure_to_timestamp_us", 3000)*1e-6;
    c.noise_xy = v.get("noise_xy_mm", 0.4)*1e-3;
    c.noise_z = v.get("noise_depth_mm", 1.0)*1e-3;
    c.lost_after = v.get("lost_after_ms", 150)*1e-3;
    c.contact_noise = v.get("rolling_accel_noise_m_s2", c.contact_noise);
    const Json& l = j.section("limits");
    c.crank_min = l.get("crank_min_rad", c.crank_min);
    c.crank_max = l.get("crank_max_rad", c.crank_max);
    c.tilt_max = std::sin(l.get("tilt_max_deg", 8)*kPi/180);
    c.yaw_span = l.get("stroke_yaw_deg", 15)*kPi/180;
    c.crank_speed_max = l.get("crank_speed_max_rad_s", c.crank_speed_max);
    c.crank_accel_max = l.get("crank_accel_max_rad_s2", c.crank_accel_max);
    const Json& p = j.section("plate");
    c.plate_mass = p.get("mass_kg", c.plate_mass);
    c.plate_com = p.get("com_above_joints_mm", 22)*1e-3;
    c.plate_tilt_inertia = p.get("tilt_inertia_kg_m2", c.plate_tilt_inertia);
    c.idle_yaw = j.section("operation").get("idle_yaw_deg", -20)*kPi/180;
    if (j.has("gravity_file")) {
      std::string gf = j["gravity_file"].string;
      if (!gf.empty() && gf[0] != '/') gf = dir+gf;
      const Json g = Json::load(gf);
      for (const auto& v : g["parameters"].array) c.gravity.push_back(v.num());
    }
    const Json& gc = j.section("gravity_calibration");
    c.cal_yaw_step = gc.get("yaw_step_deg", 8)*kPi/180;
    c.cal_tilt = gc.get("tilt_deg", 7)*kPi/180;
    c.cal_speed = gc.get("speed_rad_s", c.cal_speed);
    c.cal_settle = gc.get("settle_s", c.cal_settle);
    c.cal_average = gc.get("average_s", c.cal_average);
    c.cal_approach = gc.get("approach_rad", c.cal_approach);
    c.command_lead = j.section("operation").get("command_lead_ms", 0)*1e-3;
    const Json& bl = j.section("balance");
    c.balance_tau = bl.get("tilt_time_constant_s", c.balance_tau);
    c.balance_q_rate = bl.get("q_tilt_rate", c.balance_q_rate);
    c.balance_soft_start = std::max(1e-3, bl.get("soft_start_s", c.balance_soft_start));
    if (bl.has("tilt_acceleration_feedforward")) c.balance_accel_ff = bl["tilt_acceleration_feedforward"].boolean;
    c.balance_q_pos = bl.get("q_position", c.balance_q_pos);
    c.balance_q_vel = bl.get("q_velocity", c.balance_q_vel);
    c.balance_q_int = bl.get("q_integral", c.balance_q_int);
    c.balance_r = bl.get("r_tilt", c.balance_r);
    c.reference_rate = bl.get("reference_rate_rad_s", c.reference_rate);
    c.target_radius = bl.get("target_radius_mm", 150)*1e-3;
    const Json& bo = j.section("bounce");
    c.stroke_time = bo.get("stroke_time_s", c.stroke_time);
    c.decel = bo.get("decel_g", 1.5)*kG;
    c.return_time = bo.get("return_time_s", c.return_time);
    c.plate_speed_max = bo.get("plate_speed_max_m_s", c.plate_speed_max);
    c.lateral_gain = bo.get("lateral_gain", c.lateral_gain);
    c.lateral_integral = bo.get("lateral_integral", c.lateral_integral);
    c.launch_speed = bo.get("launch_speed_m_s", c.launch_speed);
    c.launch_dip = bo.get("launch_dip_mm", 18)*1e-3;
    c.apex_default = bo.get("apex_mm", 100)*1e-3;
    return c;
  }
};

}  // namespace ballctl
