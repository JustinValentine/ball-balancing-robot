// Ball estimator, balancing, bouncing and mode supervision on top of ballctl.hpp.
#pragma once

#include "ballctl.hpp"
#include "gravity.hpp"
#include "songs.hpp"

namespace ballctl {

const V3 kGravity{0, 0, -kG};

// ---------------------------------------------------------------- plate state

// Surface centre S, unit normal n, velocity of S and angular velocity, all in the base frame.
struct PlateState {
  double t = 0;
  V3 S{0, 0, 0}, n{0, 0, 1}, v{0, 0, 0}, w{0, 0, 0}, a{0, 0, 0}, alpha{0, 0, 0};
  V3 velocity_at(const V3& p) const { return v+cross(w, p-S); }
  V3 acceleration_at(const V3& p) const { return a+cross(alpha, p-S)+cross(w, cross(w, p-S)); }
};

inline PlateState plate_state(const Kinematics& k, const Pose& p, const Pose& later, double dt, double t) {
  PlateState s;
  s.t = t;
  s.S = k.surface(p);
  s.n = Kinematics::normal(p);
  s.v = (k.surface(later)-s.S)*(1/dt);
  const M3 W = later.R*p.R.T();  // Small rotation over dt.
  s.w = V3{W.m[2][1]-W.m[1][2], W.m[0][2]-W.m[2][0], W.m[1][0]-W.m[0][1]}*(0.5/dt);
  return s;
}

class PlateHistory {
 public:
  void push(PlateState s) {
    if (!h_.empty() && s.t <= h_.back().t) return;
    if (!h_.empty()) {
      // Accelerations by differencing, low-passed over a few milliseconds.
      const PlateState& p = h_.back();
      const double dt = s.t-p.t, f = std::min(1.0, dt/0.003);
      s.a = p.a+((s.v-p.v)*(1/dt)-p.a)*f;
      s.alpha = p.alpha+((s.w-p.w)*(1/dt)-p.alpha)*f;
    }
    h_.push_back(s);
    while (h_.size() > 3000) h_.pop_front();
  }
  bool empty() const { return h_.empty(); }
  PlateState at(double t) const {
    if (h_.empty()) return PlateState{};
    if (t <= h_.front().t) return h_.front();
    if (t >= h_.back().t) {
      PlateState s = h_.back();
      const double dt = std::min(t-s.t, 0.05);
      s.S = s.S+s.v*dt;
      s.n = unit(s.n+cross(s.w, s.n)*dt);
      s.t = t;
      return s;
    }
    size_t lo = 0, hi = h_.size()-1;
    while (hi-lo > 1) {
      const size_t mid = (lo+hi)/2;
      (h_[mid].t <= t ? lo : hi) = mid;
    }
    const PlateState &a = h_[lo], &b = h_[hi];
    const double f = (t-a.t)/(b.t-a.t);
    PlateState s;
    s.t = t;
    s.S = a.S+(b.S-a.S)*f;
    s.n = unit(a.n+(b.n-a.n)*f);
    s.v = a.v+(b.v-a.v)*f;
    s.w = a.w+(b.w-a.w)*f;
    s.a = a.a+(b.a-a.a)*f;
    s.alpha = a.alpha+(b.alpha-a.alpha)*f;
    return s;
  }

 private:
  std::deque<PlateState> h_;
};

using PlateFn = std::function<PlateState(double)>;

// ---------------------------------------------------------------- ball model and estimator

// Ball velocity after striking a surface moving with velocity u and unit normal n.
inline V3 bounce(const V3& v, const V3& u, const V3& n, double en, double et) {
  const V3 r = v-u;
  const double rn = dot(r, n);
  return u+n*(-en*rn)+(r-n*rn)*et;
}

inline V3 flight_accel(const V3& v, double drag) { return kGravity-v*(drag*norm(v)); }

struct Impact {
  double t = 0;
  V3 v_in{}, v_out{}, u{}, n{};
};

struct Ball {
  double t = 0;
  V3 p{0, 0, 0}, v{0, 0, 0};
  Mat<6, 6> P;
  bool contact = false;
  // Height of the ball centre above the surface, minus the radius.
  double gap = 0;
};

class Estimator {
 public:
  explicit Estimator(const Config& c) : c_(c), restitution_(c.restitution) {
    const M3& R = c.left_from_base;
    // Camera noise is larger along the optical axis. Rotate it into the base frame.
    const double var[3] = {c.noise_xy*c.noise_xy, c.noise_xy*c.noise_xy, c.noise_z*c.noise_z};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        double s = 0;
        for (int k = 0; k < 3; k++) s += R.m[k][i]*var[k]*R.m[k][j];
        Rm_(i, j) = s;
      }
  }

  double restitution() const { return restitution_; }
  bool valid(double now) const { return valid_ && now-last_ < c_.lost_after; }
  const Ball& ball() const { return b_; }
  const std::vector<Impact>& impacts() const { return impacts_; }
  int rejected() const { return rejected_; }
  void invalidate() { valid_ = false; }
  // The plate is about to decelerate faster than gravity: let the ball fly. If it is in fact still
  // resting, the flight model meets the surface at once and returns it to contact.
  void release() { b_.contact = false; }

  // Propagate to t1, applying impacts and rolling contact along the way.
  void propagate(Ball& b, double t1, const PlateFn& plate, std::vector<Impact>* impacts,
                 bool covariance) const {
    const double rolling = c_.rolling_gain;
    while (b.t < t1-1e-9) {
      const double dt = std::min(5e-4, t1-b.t);
      const PlateState s = plate(b.t+dt);
      Mat<6, 6> F = Mat<6, 6>::eye();
      for (int i = 0; i < 3; i++) F(i, i+3) = dt;
      double qa = 0;  // Process acceleration noise (m/s^2)^2.
      if (b.contact) {
        // Rolling on a moving surface: gravity along the plate and a share of the surface's own
        // acceleration, which matters here because the plate tilts about a point off its surface.
        const V3 gt = kGravity-s.n*dot(kGravity, s.n);
        const V3 as = s.acceleration_at(b.p);
        const V3 a = gt*rolling+(as-s.n*dot(as, s.n))*(1-rolling);
        b.p = b.p+b.v*dt+a*(0.5*dt*dt);
        b.v = b.v+a*dt;
        // Stay on the surface and move with it along the normal.
        const double h = dot(s.n, b.p-s.S)-c_.radius;
        b.p = b.p-s.n*h;
        const V3 u = s.velocity_at(b.p);
        b.v = b.v-s.n*dot(s.n, b.v-u);
        b.gap = 0;
        qa = c_.contact_noise*c_.contact_noise;
      } else {
        const V3 a = flight_accel(b.v, c_.drag);
        b.p = b.p+b.v*dt+a*(0.5*dt*dt);
        b.v = b.v+flight_accel(b.v+a*(0.5*dt), c_.drag)*dt;
        qa = 0.25*0.25;
        const double h = dot(s.n, b.p-s.S)-c_.radius;
        const V3 u = s.velocity_at(b.p-s.n*c_.radius);
        b.gap = h;
        if (h < 0 && dot(s.n, b.v-u) < 0) {
          Impact im;
          im.t = b.t+dt;
          im.v_in = b.v;
          im.u = u;
          im.n = s.n;
          b.p = b.p-s.n*h;
          b.v = bounce(b.v, u, s.n, restitution_, c_.tangential);
          im.v_out = b.v;
          if (dot(s.n, b.v-u) < kStick) {
            b.contact = true;
            b.v = b.v-s.n*dot(s.n, b.v-u);
          }
          if (covariance) {
            const double sv = 0.04+0.12*norm(im.v_out-im.v_in);
            for (int i = 3; i < 6; i++) b.P(i, i) += sv*sv;
          }
          if (impacts) impacts->push_back(im);
        }
      }
      if (covariance) {
        Mat<6, 6> Q;
        for (int i = 0; i < 3; i++) {
          Q(i, i) = qa*dt*dt*dt/3;
          Q(i, i+3) = Q(i+3, i) = qa*dt*dt/2;
          Q(i+3, i+3) = qa*dt;
        }
        b.P = F*b.P*F.T()+Q;
      }
      b.t += dt;
    }
  }

  // Fuse a measurement of the ball centre (base frame) taken at exposure time t.
  bool update(double t, const V3& z, const PlateFn& plate) {
    if (!valid_ || t-last_ > c_.lost_after || t < b_.t-0.05) {
      reset(t, z, plate);
      return true;
    }
    std::vector<Impact> hits;
    propagate(b_, t, plate, &hits, true);
    // In contact the estimate is held on the surface, so judge lift-off from the camera itself:
    // two measurements clearly above the surface start a flight from the measured motion.
    {
      const PlateState s = plate(t);
      const double h = dot(s.n, z-s.S)-c_.radius;
      const double sigma = std::sqrt(dot(s.n, V3{Rm_(0, 0)*s.n[0]+Rm_(0, 1)*s.n[1]+Rm_(0, 2)*s.n[2],
                                                Rm_(1, 0)*s.n[0]+Rm_(1, 1)*s.n[1]+Rm_(1, 2)*s.n[2],
                                                Rm_(2, 0)*s.n[0]+Rm_(2, 1)*s.n[1]+Rm_(2, 2)*s.n[2]}));
      if (b_.contact && h > std::max(0.002, 4*sigma)) {
        if (++above_ >= 2) {
          const double vn = (h-above_h_)/std::max(t-above_t_, 1e-3);
          b_.contact = false;
          b_.p = b_.p+s.n*(h-dot(s.n, b_.p-s.S)+c_.radius);
          b_.v = b_.v+s.n*(vn+dot(s.n, s.velocity_at(b_.p))-dot(s.n, b_.v));
          for (int i = 0; i < 3; i++) { b_.P(i, i) += 1e-6; b_.P(i+3, i+3) += 0.3*0.3; }
          above_ = 0;
        } else {
          above_h_ = h;
          above_t_ = t;
        }
      } else {
        above_ = 0;
      }
    }
    for (const auto& h : hits) impacts_.push_back(h);
    if (impacts_.size() > 50) impacts_.erase(impacts_.begin(), impacts_.begin()+(impacts_.size()-50));
    Mat<3, 3> S;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) S(i, j) = b_.P(i, j)+Rm_(i, j);
    Mat<3, 3> Si;
    if (!invert(S, &Si)) { reset(t, z, plate); return true; }
    const V3 y = z-b_.p;
    double d2 = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) d2 += y[i]*Si(i, j)*y[j];
    if (d2 > 60) {
      if (++rejected_run_ >= 4) reset(t, z, plate);
      rejected_++;
      return false;
    }
    rejected_run_ = 0;
    Mat<6, 3> K;
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 3; j++) {
        double s = 0;
        for (int k = 0; k < 3; k++) s += b_.P(i, k)*Si(k, j);
        K(i, j) = s;
      }
    for (int i = 0; i < 3; i++) {
      b_.p[i] += K(i, 0)*y[0]+K(i, 1)*y[1]+K(i, 2)*y[2];
      b_.v[i] += K(i+3, 0)*y[0]+K(i+3, 1)*y[1]+K(i+3, 2)*y[2];
    }
    Mat<6, 6> KH;
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 3; j++) KH(i, j) = K(i, j);
    b_.P = (Mat<6, 6>::eye()-KH)*b_.P;
    last_ = t;
    classify(plate(t));
    learn(t);
    return true;
  }

 private:
  static constexpr double kStick = 0.06;  // Below this rebound speed the ball stays on the plate.

  void reset(double t, const V3& z, const PlateFn& plate) {
    b_ = Ball{};
    b_.t = t;
    b_.p = z;
    b_.P = Mat<6, 6>::eye()*1e-6;
    // A first sighting says nothing about speed; a ball near the plate rarely rolls faster than this.
    for (int i = 0; i < 3; i++) { b_.P(i, i) = Rm_(i, i); b_.P(i+3, i+3) = 0.3*0.3; }
    b_.contact = false;
    valid_ = true;
    last_ = t;
    rejected_run_ = 0;
    classify(plate(t));
  }

  void classify(const PlateState& s) {
    const double h = dot(s.n, b_.p-s.S)-c_.radius;
    const double vn = dot(s.n, b_.v-s.velocity_at(b_.p));
    b_.gap = h;
    if (b_.contact && (h > 0.004 || (h > 0.002 && vn > 0.1))) b_.contact = false;
    else if (!b_.contact && h < 0.0015 && std::fabs(vn) < 0.08) b_.contact = true;
  }

  // Refine the normal restitution from the velocity seen shortly after each strong impact.
  void learn(double t) {
    if (impacts_.empty() || b_.contact) return;
    const Impact& im = impacts_.back();
    if (learned_ == im.t || t-im.t < 0.03 || t-im.t > 0.06) return;
    learned_ = im.t;
    const double in = dot(im.n, im.v_in-im.u);
    if (in > -0.4) return;
    const V3 out = b_.v-flight_accel(b_.v, c_.drag)*(t-im.t);
    const double e = dot(im.n, out-im.u)/-in;
    restitution_ += 0.15*(std::clamp(e, 0.5, 0.98)-restitution_);
  }

  const Config& c_;
  Mat<3, 3> Rm_;
  Ball b_;
  bool valid_ = false;
  double last_ = -1e9, learned_ = -1, restitution_;
  int rejected_ = 0, rejected_run_ = 0, above_ = 0;
  double above_h_ = 0, above_t_ = 0;
  std::vector<Impact> impacts_;
};

// ---------------------------------------------------------------- balancing

// Discrete LQR for one horizontal axis. State: position error, velocity error, tilt, tilt rate,
// integral of position error. The ball (shifted coordinate) accelerates by g*rolling_gain per unit
// of normal tilt. The tilt follows the controller's command through a critically damped
// second-order filter with natural frequency 1/tau, so the tilt rate and acceleration sent to the
// servos are smooth.
struct AxisLqr {
  double K[5] = {0, 0, 0, 0, 0};

  static AxisLqr design(double dt, double gain, double tau, double qp, double qv, double qi, double r,
                        double qrate = 0) {
    const double w = 1/tau;
    Mat<5, 5> A = Mat<5, 5>::eye();
    A(0, 1) = dt;
    A(0, 2) = 0.5*dt*dt*gain;
    A(1, 2) = dt*gain;
    A(2, 3) = dt;
    A(3, 2) = -w*w*dt;
    A(3, 3) = 1-2*w*dt;
    A(4, 0) = dt;
    Mat<5, 1> B;
    B(3, 0) = w*w*dt;
    Mat<5, 5> Q;
    Q(0, 0) = qp*dt;
    Q(1, 1) = qv*dt;
    Q(3, 3) = qrate*dt;
    Q(4, 4) = qi*dt;
    const double R = r*dt;
    Mat<5, 5> P = Q;
    for (int it = 0; it < 400000; it++) {
      const Mat<1, 5> BtPA = B.T()*P*A;
      const double s = R+(B.T()*P*B)(0, 0);
      Mat<5, 5> Pn = Q+A.T()*P*A-BtPA.T()*BtPA*(1/s);
      double change = 0;
      for (int i = 0; i < 25; i++) change = std::max(change, std::fabs(Pn.d[i]-P.d[i]));
      P = Pn;
      if (change < 1e-12*std::max(1.0, P(0, 0))) break;
    }
    const Mat<1, 5> k = (B.T()*P*A)*(1/(R+(B.T()*P*B)(0, 0)));
    AxisLqr out;
    for (int i = 0; i < 5; i++) out.K[i] = k(0, i);
    return out;
  }
};

class Balancer {
 public:
  Balancer(const Config& c, double dt) : c_(c), dt_(dt) {
    lqr_ = AxisLqr::design(dt, kG*c.rolling_gain, c.balance_tau, c.balance_q_pos, c.balance_q_vel,
                           c.balance_q_int, c.balance_r, c.balance_q_rate);
  }
  const AxisLqr& lqr() const { return lqr_; }

  // Ball displacement per unit tilt from the surface moving under it: the controlled coordinate
  // is the ball position minus shift*tilt, which obeys the simple rolling model exactly.
  void set_shift(const double shift[2][2]) {
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) shift_[i][j] = shift[i][j];
  }

  void start(const V3& ball, const V3& ball_v, double nx, double ny, double nxd, double nyd) {
    const double n[2] = {nx, ny}, nd[2] = {nxd, nyd};
    for (int a = 0; a < 2; a++) {
      xr_[a] = ball[a]-shift_[a][0]*nx-shift_[a][1]*ny;
      vr_[a] = ball_v[a];
      ar_[a] = 0;
      jr_[a] = 0;
      integ_[a] = 0;
      n_[a] = n[a];   // The whole current tilt starts in the filtered feedback part.
      nd_[a] = nd[a];
    }
    path_ = false;
    elapsed_ = 0;
  }

  // Follow a path with its velocity and acceleration instead of the smoothed target.
  void path(const double* x, const double* v, const double* a) {
    for (int i = 0; i < 2; i++) { px_[i] = x[i]; pv_[i] = v[i]; pa_[i] = a[i]; }
    path_ = true;
  }
  void clear_path() { path_ = false; }

  // Advance one control period from the ball state and the measured tilt and tilt rate.
  // Returns the commanded normal tilt, its rate and acceleration. The reference's own tilt (its
  // acceleration over g*rolling_gain) goes straight to the plate; only the LQR correction, which
  // carries the camera noise, passes the second-order filter.
  void step(const V3& target, const V3& ball_p, const V3& ball_v, const double* tilt, const double* tilt_rate,
            double* n, double* ndot, double* nacc) {
    const double k = kG*c_.rolling_gain, w = c_.reference_rate, wa = 1/c_.balance_tau;
    double u[2], uff[2], jff[2];
    for (int a = 0; a < 2; a++) {
      const double p = ball_p[a]-shift_[a][0]*tilt[0]-shift_[a][1]*tilt[1];
      const double v = ball_v[a]-shift_[a][0]*tilt_rate[0]-shift_[a][1]*tilt_rate[1];
      double jerk;
      if (path_) {
        jerk = (pa_[a]-ar_[a])/dt_;
        jr_[a] += (jerk-jr_[a])*std::min(1.0, dt_/0.02);
        jerk = jr_[a];
        xr_[a] = px_[a];
        vr_[a] = pv_[a];
        ar_[a] = pa_[a];
      } else {
        // Critically damped fourth-order reference: its acceleration and jerk, and so the tilt
        // and tilt rate fed forward, start from zero, and the tilt stays within 80% of the limit.
        const double amax = 0.8*k*c_.tilt_max, w2 = w*w;
        double snap = w2*w2*(target[a]-xr_[a])-4*w2*w*vr_[a]-6*w2*ar_[a]-4*w*jr_[a];
        if ((ar_[a] >= amax && jr_[a] > 0) || (ar_[a] <= -amax && jr_[a] < 0)) { jr_[a] = 0; snap = std::min(snap, 0.0)*(ar_[a] > 0)+std::max(snap, 0.0)*(ar_[a] < 0); }
        xr_[a] += vr_[a]*dt_+0.5*ar_[a]*dt_*dt_;
        vr_[a] += ar_[a]*dt_;
        ar_[a] = std::clamp(ar_[a]+jr_[a]*dt_, -amax, amax);
        jr_[a] += snap*dt_;
        jerk = jr_[a];
      }
      uff[a] = ar_[a]/k;
      jff[a] = jerk/k;
      const double e = p-xr_[a], ed = v-vr_[a];
      u[a] = -(lqr_.K[0]*e+lqr_.K[1]*ed+lqr_.K[2]*n_[a]+lqr_.K[3]*nd_[a]+lqr_.K[4]*integ_[a]);
    }
    // Soft start: fade the correction in after (re)starting, while the ball estimate settles.
    const double fade = std::min(1.0, elapsed_/c_.balance_soft_start);
    elapsed_ += dt_;
    for (int a = 0; a < 2; a++) u[a] = n_[a]+(u[a]-n_[a])*fade*fade;
    for (int a = 0; a < 2; a++) {
      const double acc = wa*wa*(u[a]-n_[a])-2*wa*nd_[a];
      n_[a] += nd_[a]*dt_+0.5*acc*dt_*dt_;
      nd_[a] += acc*dt_;
      n[a] = uff[a]+n_[a];
      ndot[a] = jff[a]+nd_[a];
      nacc[a] = acc;
    }
    const double mag = std::hypot(n[0], n[1]);
    const bool saturated = mag > c_.tilt_max;
    if (saturated) {
      for (int a = 0; a < 2; a++) { n[a] *= c_.tilt_max/mag; ndot[a] = 0; }
    }
    for (int a = 0; a < 2; a++) {
      const double p = ball_p[a]-shift_[a][0]*tilt[0]-shift_[a][1]*tilt[1];
      if (!saturated) integ_[a] = std::clamp(integ_[a]+dt_*(p-xr_[a]), -0.05, 0.05);
    }
  }

 private:
  const Config& c_;
  double dt_;
  AxisLqr lqr_;
  double xr_[2] = {0, 0}, vr_[2] = {0, 0}, ar_[2] = {0, 0}, jr_[2] = {0, 0}, integ_[2] = {0, 0};
  double n_[2] = {0, 0}, nd_[2] = {0, 0}, elapsed_ = 0;
  double shift_[2][2] = {{0, 0}, {0, 0}};
  double px_[2] = {0, 0}, pv_[2] = {0, 0}, pa_[2] = {0, 0};
  bool path_ = false;
};

// ---------------------------------------------------------------- bouncing

// Speed that carries the ball a height h upward against gravity and quadratic drag, and the time
// to rise and fall back to the same height.
inline double launch_speed(double h, double drag) {
  if (h <= 0) return 0;
  const double vt2 = kG/drag;
  return std::sqrt(vt2*(std::exp(2*kG*h/vt2)-1));
}
inline double flight_time(double v0, double drag) {
  if (v0 <= 0) return 0;
  const double vt = std::sqrt(kG/drag);
  const double h = vt*vt/(2*kG)*std::log(1+v0*v0/(vt*vt));
  return vt/kG*std::atan(v0/vt)+vt/kG*std::acosh(std::exp(kG*h/(vt*vt)));
}

// Apex that gives a flight of duration T back to the same height.
inline double apex_for_flight(double T, double drag) {
  double lo = 0.05, hi = 8;
  for (int i = 0; i < 60; i++) {
    const double mid = 0.5*(lo+hi);
    (flight_time(mid, drag) < T ? lo : hi) = mid;
  }
  const double v = 0.5*(lo+hi), vt2 = kG/drag;
  return vt2/(2*kG)*std::log(1+v*v/vt2);
}

struct ImpactPlan {
  bool valid = false, saturated = false;
  double apex = 0;    // Apex the rebound was planned for.
  double flight = 0;  // Flight time the rebound was planned for (songs), s.

  double t = 0, yaw = 0, yaw_pre = 0, nx = 0, ny = 0, yaw_rate = 0, surface_speed = 0;
  V3 p_in{}, v_in{}, v_out{};
};

// ---------------------------------------------------------------- controller

enum Mode { kHold = 0, kBalance = 1, kBounce = 2, kJuggle = 3 };
enum Phase { kLevel = 0, kBalancing = 1, kSettling = 2, kLaunching = 3, kBouncing = 4, kCatching = 5, kCalibrating = 6,
             kJuggling = 7 };

struct Output {
  Q3 q{0, 0, 0}, qd{0, 0, 0}, tau{0, 0, 0};
  bool ok = false;           // False when the posture had no kinematic solution.
  bool limited = false;      // The speed or acceleration cap changed this command.
  int phase = kLevel;
  bool ball_valid = false, contact = false;
  V3 ball{0, 0, 0}, ball_v{0, 0, 0};
  double yaw = 0, nx = 0, ny = 0;
  double next_impact = 0, restitution = 0, apex = 0;
  V3 plate_normal{0, 0, 1};
  int note = 0, note_index = -1;  // MIDI note of a song impact that just happened, else 0.
  double note_time = 0;           // When that impact was planned to happen.
  double note_beat = 0;           // When the song's schedule wanted it.
  bool ball2_valid = false;       // Second ball, when juggling two.
  V3 ball2{0, 0, 0};
};

class Controller {
 public:
  Controller(const std::string& path, double dt)
      : c_(Config::load(path, &platform_)), k_(platform_, c_.surface_offset), est_(c_), est2_(c_), bal_(c_, dt), cal_(k_, c_),
        dt_(dt) {
    gravity_ = GravityModel::plate(c_.plate_mass, c_.plate_com);
    if (!c_.gravity.empty()) {
      if (c_.gravity.size() != GravityModel::kParams)
        throw std::runtime_error("gravity_file must hold " + std::to_string(GravityModel::kParams) + " parameters");
      for (int j = 0; j < GravityModel::kParams; j++) gravity_.theta[j] = c_.gravity[j];
    }
    apex_ = c_.apex_default;
    for (auto* t : {&yaw_, &nx_, &ny_}) t->reset(0);
    yaw_.reset(c_.idle_yaw);
    Q3 q;
    if (!k_.inverse(Posture{c_.idle_yaw, 0, 0}, &q)) throw std::runtime_error("Idle posture has no solution");
    for (int i = 0; i < 3; i++)
      if (q[i] < c_.crank_min || q[i] > c_.crank_max)
        throw std::runtime_error("Idle posture is outside the configured crank range");
    // Centre of the plate at the idle posture is the origin of targets.
    Pose p;
    k_.pose(Posture{c_.idle_yaw, 0, 0}, &p);
    centre_ = k_.surface(p);
    double z0 = 0, z1 = 0;
    if (!k_.height({c_.idle_yaw, 0, 0}, &z0) || !k_.height({c_.idle_yaw+1e-6, 0, 0}, &z1))
      throw std::runtime_error("Idle posture has no solution");
    dz_dyaw_ = (z1-z0)/1e-6;
    target_ = centre_;
    juggle_centre_ = centre_;
    Pose px, py;
    k_.pose(Posture{c_.idle_yaw, 1e-5, 0}, &px);
    k_.pose(Posture{c_.idle_yaw, 0, 1e-5}, &py);
    double shift[2][2];
    for (int i = 0; i < 2; i++) {
      shift[i][0] = (1-c_.rolling_gain)*(k_.surface(px)[i]-centre_[i])/1e-5;
      shift[i][1] = (1-c_.rolling_gain)*(k_.surface(py)[i]-centre_[i])/1e-5;
    }
    bal_.set_shift(shift);
    if (std::fabs(dz_dyaw_) < 0.03)
      throw std::runtime_error("Idle yaw is too close to zero twist to move the plate vertically");
  }

  Controller(const Controller&) = delete;
  Controller& operator=(const Controller&) = delete;
  const Config& config() const { return c_; }
  const Kinematics& kinematics() const { return k_; }
  const Json& platform() const { return platform_; }
  const AxisLqr& lqr() const { return bal_.lqr(); }
  V3 centre() const { return centre_; }

  void set_mode(int m) {
    if (m == kJuggle && request_ != kJuggle) juggle_centre_ = target_;
    if (m != kJuggle && request_ == kJuggle) {
      target_ = juggle_centre_;
      est2_.invalidate();
    }
    request_ = m;
  }
  // Target relative to the plate centre at idle, metres.
  void set_target(double x, double y) {
    bal_.clear_path();
    const double r = std::hypot(x, y), s = r > c_.target_radius ? c_.target_radius/r : 1;
    target_ = {centre_[0]+x*s, centre_[1]+y*s, 0};
  }
  void set_apex(double h) { apex_ = std::clamp(h, 0.0, 0.5); }
  // Roll the ball round a circle while balancing, or hop round it while bouncing. Radius in m
  // about the current target; zero stops. Period is one lap in seconds, lengthened if the tilt
  // limit cannot hold the ball on the circle.
  void set_circle(double radius, double period) {
    if (radius <= 0) {
      if (circle_r_ > 0) {
        bal_.clear_path();
        target_ = circle_centre_;
      }
      circle_r_ = 0;
      return;
    }
    if (circle_r_ <= 0) circle_centre_ = target_;
    circle_r_ = std::min(radius, c_.target_radius-std::hypot(circle_centre_[0]-centre_[0], circle_centre_[1]-centre_[1]));
    const double amax = 0.6*0.8*kG*c_.rolling_gain*c_.tilt_max;
    circle_period_ = std::max(period, 2*kPi*std::sqrt(circle_r_/amax));
    circle_start_ = -1;
  }

  // Bounce to a song from songs(); -1 stops. beat <= 0 keeps the song's own tempo.
  void set_song(int index, double beat) {
    song_ = index >= 0 && index < static_cast<int>(songs().size()) ? index : -1;
    song_started_ = false;
    flight_error_.clear();
    last_flight_ = 0;
    if (song_ >= 0) song_beat_ = beat > 0 ? beat : songs()[song_].beat;
  }
  int song() const { return song_; }

  // Two-ball juggling: half the distance between the two columns, m.
  void set_juggle_spacing(double half) { juggle_half_ = std::clamp(half, 0.025, 0.1); }

  // Balance along a path (relative to the plate centre): position, velocity and acceleration.
  void set_path(const double* x, const double* v, const double* a) {
    const double p[2] = {centre_[0]+x[0], centre_[1]+x[1]};
    bal_.path(p, v, a);
    target_ = {p[0], p[1], 0};
  }

  // Tracker row: capture timestamp (seconds) and the ball centre in the left optical frame (m).
  void measurement(double t_capture, const V3& left) {
    const double t = t_capture-c_.exposure_to_timestamp;
    const V3 base = c_.left_from_base.T()*left+c_.left_origin;
    if (norm(base-centre_) > 0.6) return;
    pending_.push_back({t, base});
  }

  Output step(double now, const Q3& q, const Q3& qd) {
    Output out;
    // Plate state from the measured cranks.
    Pose pose, later;
    const double h = 1e-3;
    Q3 qh;
    for (int i = 0; i < 3; i++) qh[i] = q[i]+qd[i]*h;
    V3 beta_h = beta_;
    if (k_.forward(q, &pose, &beta_) && k_.forward(qh, &later, &beta_h)) {
      history_.push(plate_state(k_, pose, later, h, now));
      out.plate_normal = Kinematics::normal(pose);
    }
    const PlateFn past = [this](double t) { return history_.at(t); };
    std::sort(pending_.begin(), pending_.end(), [](auto& a, auto& b) { return a.first < b.first; });
    // Measurements of one stereo pair share a time; with two balls each goes to its own track.
    for (size_t i = 0; i < pending_.size();) {
      size_t j = i;
      while (j < pending_.size() && pending_[j].first == pending_[i].first) j++;
      assign(pending_[i].first, i, j, past);
      i = j;
    }
    pending_.clear();
    valid2_ = est2_.valid(now) && request_ == kJuggle;
    ball2_ = est2_.ball();
    if (valid2_) est2_.propagate(ball2_, now, past, nullptr, false);

    const bool valid = est_.valid(now);
    Ball b = est_.ball();
    if (valid) est_.propagate(b, now, past, nullptr, false);

    supervise(now, valid, b);

    // Posture, its rate and acceleration at the command time.
    const double tc = now+c_.command_lead;
    double y[3], yd[3], ya[3];
    yaw_.eval(tc, &y[0], &yd[0], &ya[0]);
    if (phase_ == kBalancing || phase_ == kSettling) {
      double n[2], nd[2], na[2];
      const PlateState now_plate = history_.at(now);
      const V3 ndt = cross(now_plate.w, now_plate.n);
      const double tilt[2] = {now_plate.n[0], now_plate.n[1]}, tilt_rate[2] = {ndt[0], ndt[1]};
      if (circle_r_ > 0) circle_path(now);
      bal_.step(target_, b.p, b.v, tilt, tilt_rate, n, nd, na);
      // Balancing tilts are slow; the plate's tilt inertia torque is negligible and its
      // feedforward mostly passes camera noise to the motors.
      const double ff = c_.balance_accel_ff ? 1 : 0;
      y[1] = n[0]; yd[1] = nd[0]; ya[1] = na[0]*ff;
      y[2] = n[1]; yd[2] = nd[1]; ya[2] = na[1]*ff;
      nx_.reset(n[0]);
      ny_.reset(n[1]);
    } else {
      nx_.eval(tc, &y[1], &yd[1], &ya[1]);
      ny_.eval(tc, &y[2], &yd[2], &ya[2]);
    }
    command(y, yd, ya, q, &out);
    out.phase = phase_;
    out.ball_valid = valid;
    out.contact = b.contact;
    out.ball = b.p;
    out.ball_v = b.v;
    out.ball2_valid = valid2_;
    out.ball2 = ball2_.p;
    out.yaw = y[0];
    out.nx = y[1];
    out.ny = y[2];
    out.next_impact = plan_.valid ? plan_.t : 0;
    out.restitution = est_.restitution();
    out.apex = apex_;
    if (note_.first) {
      out.note = note_.first;
      out.note_index = note_index_;
      out.note_time = note_.second;
      out.note_beat = note_beat_;
      note_.first = 0;
    }
    return out;
  }

  // Gravity calibration: move the plate (without a ball) through test postures and fit the model.
  void start_calibration(double now, const Q3& q) {
    Pose p;
    V3 beta = beta_;
    Posture from{c_.idle_yaw, 0, 0};
    if (k_.forward(q, &p, &beta)) from = Kinematics::posture(p);
    cal_.start(now, from);
    phase_ = kCalibrating;
  }
  bool calibrating() const { return cal_.running(); }
  const GravityCalibration& calibration() const { return cal_; }

  // One control period of the calibration. tau is the torque each crank reports (N m).
  Output calibration_step(double now, const Q3& q, const Q3& qd, const Q3& tau) {
    Output out;
    double y[3], yd[3], ya[3];
    cal_.step(now, q, tau, y, yd, ya);
    command(y, yd, ya, q, &out);
    (void)qd;
    out.phase = kCalibrating;
    out.yaw = y[0];
    out.nx = y[1];
    out.ny = y[2];
    if (!cal_.running()) {
      // Hand back to the normal tracks from where the calibration stopped, then level.
      yaw_.reset(y[0]);
      nx_.reset(y[1]);
      ny_.reset(y[2]);
      level(now);
      phase_ = kLevel;
    }
    return out;
  }

  // Fit the gravity model to the calibration and use it from now on.
  bool finish_calibration(GravityModel* fitted, std::string* report) {
    if (!cal_.fit(GravityModel::plate(c_.plate_mass, c_.plate_com), fitted, report)) return false;
    gravity_ = *fitted;
    return true;
  }
  const GravityModel& gravity() const { return gravity_; }
  // Torque that holds the mechanism still at these crank angles, from the gravity model.
  bool holding_torque(const Q3& q, Q3* tau) const {
    Pose p;
    V3 beta{0, 0, 0};
    return k_.forward(q, &p, &beta) && gravity_.torque(k_, Kinematics::posture(p), tau);
  }
  void set_gravity(const GravityModel& g) { gravity_ = g; }

  // Number of control periods in which the speed or acceleration cap changed the command.
  long limited() const { return limited_; }

  // Plan the next impact for a ball in flight. Exposed for tests.
  ImpactPlan plan_impact(double now, const Ball& ball, double apex) const {
    ImpactPlan plan;
    plan.yaw = c_.idle_yaw;
    double nx, ny, ndot, nacc;
    nx_.eval(now, &nx, &ndot, &nacc);
    ny_.eval(now, &ny, &ndot, &nacc);
    plan.nx = nx;
    plan.ny = ny;
    double yaw_rate = 0;
    const double v_up = apex > 0 ? std::max(0.0, launch_speed(apex, c_.drag)+speed_bias_) : 0;
    const double T_f = std::max(flight_time(std::max(v_up, 0.3), c_.drag), 0.1);
    for (int iteration = 0; iteration < 4; iteration++) {
      Pose pose;
      if (!k_.pose(Posture{plan.yaw, plan.nx, plan.ny}, &pose)) return ImpactPlan{};
      const V3 S = k_.surface(pose), n = Kinematics::normal(pose);
      // Ballistic prediction to the planned surface.
      Ball b = ball;
      b.contact = false;
      V3 prev_p = b.p, prev_v = b.v;
      double prev_t = b.t, h_prev = dot(n, b.p-S)-c_.radius;
      bool hit = false;
      for (int s = 0; s < 3000 && !hit; s++) {
        const double dt = 5e-4;
        const V3 a = flight_accel(b.v, c_.drag);
        prev_p = b.p; prev_v = b.v; prev_t = b.t;
        b.p = b.p+b.v*dt+a*(0.5*dt*dt);
        b.v = b.v+flight_accel(b.v+a*(0.5*dt), c_.drag)*dt;
        b.t += dt;
        const double hgap = dot(n, b.p-S)-c_.radius;
        if (hgap <= 0 && dot(n, b.v) < 0) {
          const double f = h_prev/(h_prev-hgap);
          plan.t = prev_t+f*dt;
          plan.p_in = prev_p+(b.p-prev_p)*f;
          plan.v_in = prev_v+(b.v-prev_v)*f;
          hit = true;
        }
        h_prev = hgap;
      }
      if (!hit) return ImpactPlan{};
      // Surface velocity at the contact point per unit yaw rate.
      const V3 contact = plan.p_in-n*c_.radius;
      const V3 local = pose.R.T()*(contact-k_.origin()-pose.t);
      Pose p2;
      if (!k_.pose(Posture{plan.yaw+1e-5, plan.nx, plan.ny}, &p2)) return ImpactPlan{};
      const V3 D = (k_.origin()+p2.t+p2.R*local-contact)*(1/1e-5);
      // Desired rebound: the apex height, and a horizontal speed that lands near the target.
      V3 v_out;
      for (int a = 0; a < 2; a++)
        v_out[a] = c_.lateral_gain*(target_[a]-plan.p_in[a])/T_f+bias_[a];
      v_out[2] = v_up;
      const double en = est_.restitution(), et = c_.tangential;
      V3 x{plan.nx, plan.ny, yaw_rate};
      auto f = [&](const V3& u, V3* r) {
        const double s = std::hypot(u[0], u[1]);
        if (s > 0.5) return false;
        const V3 nn{u[0], u[1], std::sqrt(1-s*s)};
        *r = bounce(plan.v_in, D*u[2], nn, en, et)-v_out;
        return true;
      };
      newton3(f, &x, 1e-9, 20, 1e-6);
      const double s = std::hypot(x[0], x[1]);
      if (s > c_.tilt_max) { x[0] *= c_.tilt_max/s; x[1] *= c_.tilt_max/s; }
      const double vmax = c_.plate_speed_max/std::max(std::fabs(D[2]), 1e-6);
      yaw_rate = std::clamp(x[2], -vmax, vmax);
      plan.saturated = std::fabs(x[2]) > vmax || s > c_.tilt_max;
      plan.nx = x[0];
      plan.ny = x[1];
      // The plate meets the ball at the idle height, coming from a pre-stroke position.
      plan.yaw = c_.idle_yaw;
      plan.yaw_pre = std::clamp(c_.idle_yaw-yaw_rate*c_.stroke_time/2, c_.idle_yaw-c_.yaw_span,
                                c_.idle_yaw+c_.yaw_span);
      plan.yaw_rate = yaw_rate;
      plan.surface_speed = yaw_rate*D[2];
      const V3 nn{plan.nx, plan.ny, std::sqrt(1-plan.nx*plan.nx-plan.ny*plan.ny)};
      plan.v_out = bounce(plan.v_in, D*yaw_rate, nn, en, et);
    }
    plan.valid = true;
    plan.apex = apex;
    return plan;
  }

 private:
  // Crank command for a posture with its rate and acceleration: inverse kinematics, velocity
  // feedforward, and torque feedforward for gravity (from the gravity model) and plate inertia.
  void command(double* y, double* yd, double* ya, const Q3& q, Output* outp) {
    Output& out = *outp;
    const double tilt = std::hypot(y[1], y[2]);
    if (tilt > c_.tilt_max) { y[1] *= c_.tilt_max/tilt; y[2] *= c_.tilt_max/tilt; }
    const Posture cmd{y[0], y[1], y[2]};
    Mat<3, 3> J, Ji;
    V3 Fg;
    if (k_.inverse(cmd, &out.q) && k_.jacobian(cmd, &J) && invert(J.T(), &Ji) && gravity_.force(k_, cmd, &Fg)) {
      double z0 = 0, zp = 0, zm = 0, zx = 0, zy = 0;
      const double e = 1e-4;
      k_.height(cmd, &z0);
      k_.height({cmd.yaw+e, cmd.nx, cmd.ny}, &zp);
      k_.height({cmd.yaw-e, cmd.nx, cmd.ny}, &zm);
      k_.height({cmd.yaw, cmd.nx+e, cmd.ny}, &zx);
      k_.height({cmd.yaw, cmd.nx, cmd.ny+e}, &zy);
      const double dz = (zp-zm)/(2*e), d2z = (zp-2*z0+zm)/(e*e);
      const double zacc = dz*ya[0]+d2z*yd[0]*yd[0];
      const double mass = gravity_.theta[0] > 0 ? gravity_.theta[0] : c_.plate_mass;
      const double yaw_inertia = 2*c_.plate_tilt_inertia;
      // Gravity from the model; the plate's own acceleration from its mass and inertia.
      const double F[3] = {Fg[0]+mass*zacc*dz+yaw_inertia*ya[0],
                           Fg[1]+mass*zacc*(zx-z0)/e+c_.plate_tilt_inertia*ya[1],
                           Fg[2]+mass*zacc*(zy-z0)/e+c_.plate_tilt_inertia*ya[2]};
      const V3 offset = gravity_.offsets();
      for (int i = 0; i < 3; i++) {
        out.qd[i] = J(i, 0)*yd[0]+J(i, 1)*yd[1]+J(i, 2)*yd[2];
        out.tau[i] = Ji(i, 0)*F[0]+Ji(i, 1)*F[1]+Ji(i, 2)*F[2]+offset[i];
      }
      out.ok = true;
      for (int i = 0; i < 3; i++) {
        if (out.q[i] < c_.crank_min || out.q[i] > c_.crank_max) out.qd[i] = 0;
        out.q[i] = std::clamp(out.q[i], c_.crank_min, c_.crank_max);
      }
      last_ = out;
    } else {
      out = last_;
      out.qd = {0, 0, 0};
      out.ok = false;
    }
    limit(q, &out);
  }

  // Safety layer for every mode: the crank command may not exceed the speed and acceleration caps.
  // It follows the requested command exactly when that command is within the caps, and otherwise
  // approaches it along a braking curve, so steps and planning errors never overshoot. The caps sit
  // above what balancing and bouncing need.
  void limit(const Q3& measured, Output* out) {
    if (!commanded_) {
      prev_q_ = measured;
      prev_v_ = {0, 0, 0};
      commanded_ = true;
    }
    const double a = c_.crank_accel_max, vmax = c_.crank_speed_max;
    bool hit = false;
    for (int i = 0; i < 3; i++) {
      const double target = std::clamp(out->q[i], c_.crank_min, c_.crank_max);
      const double e = target-prev_q_[i];
      const double want = e/dt_;
      const double brake = std::sqrt(2*a*std::fabs(e));
      double v = out->qd[i]+std::clamp(want-out->qd[i], -brake, brake);
      v = std::clamp(v, prev_v_[i]-a*dt_, prev_v_[i]+a*dt_);
      v = std::clamp(v, -vmax, vmax);
      if (std::fabs(v-want) > 1e-6) hit = true;
      out->q[i] = std::fabs(v-want) > 1e-6 ? prev_q_[i]+v*dt_ : target;
      out->qd[i] = std::clamp(hit ? v : out->qd[i], -vmax, vmax);
      prev_v_[i] = (out->q[i]-prev_q_[i])/dt_;
      prev_q_[i] = out->q[i];
    }
    out->limited = hit;
    limited_ += hit;
  }

  void level(double now) {
    yaw_.to(now, 0.6, c_.idle_yaw, 0, 0);
    nx_.to(now, 0.6, 0, 0, 0);
    ny_.to(now, 0.6, 0, 0, 0);
  }

  void enter_balance(double now, const Ball& b) {
    double nx, ny, dx, dy, a;
    nx_.eval(now, &nx, &dx, &a);
    ny_.eval(now, &ny, &dy, &a);
    bal_.start(b.p, b.v, nx, ny, dx, dy);
    yaw_.to(now, 0.3, c_.idle_yaw, 0, 0);
    plan_ = ImpactPlan{};
  }

  void supervise(double now, bool valid, const Ball& b) {
    yaw_.prune(now-0.5);
    nx_.prune(now-0.5);
    ny_.prune(now-0.5);
    const int previous = phase_;
    if (!valid || request_ == kHold) {
      if (phase_ != kLevel) level(now);
      phase_ = kLevel;
      plan_ = ImpactPlan{};
      return;
    }
    const bool near_plate = b.gap < 0.03;
    // Contact during a bounce lasts milliseconds; resting means sustained contact.
    if (!b.contact) contact_since_ = -1;
    else if (contact_since_ < 0) contact_since_ = now;
    const bool resting = b.contact && now-contact_since_ > 0.15;
    if (request_ == kBalance) {
      if (phase_ == kBouncing || phase_ == kLaunching) phase_ = kCatching;
      if (phase_ == kCatching) {
        if (resting || (near_plate && std::fabs(b.v[2]) < 0.15)) phase_ = kBalancing;
        else bounce_step(now, b, 0.0);
      } else {
        phase_ = kBalancing;
      }
      if (phase_ == kBalancing && previous != kBalancing) enter_balance(now, b);
      return;
    }
    // Bounce or juggle requested. Juggling launches the first ball from the centre like a bounce;
    // a second ball dropped in by hand is then juggled with it in two columns.
    if (request_ == kJuggle) {
      const bool flying = phase_ == kBouncing || phase_ == kJuggling;
      target_ = flying ? column(0) : juggle_centre_;
      // A bounce touches the plate for milliseconds; only a resting ball ends the juggle.
      if (flying && valid2_ && !resting) {
        if (phase_ != kJuggling) {
          grid_t0_ = -1;
          strike_t_[0] = strike_t_[1] = 0;
          flight_err_[0] = flight_err_[1] = 0;
        }
        phase_ = kJuggling;
        juggle_step(now, b);
        return;
      }
      if (phase_ == kJuggling) phase_ = kBouncing;
    }
    switch (phase_) {
      case kLevel:
      case kBalancing:
      case kCatching:
        if (b.contact || near_plate) {
          phase_ = kSettling;
          if (previous != kBalancing) enter_balance(now, b);
          settle_since_ = now;
        } else {
          phase_ = kBouncing;
        }
        break;
      case kSettling: {
        const double err = std::hypot(b.p[0]-target_[0], b.p[1]-target_[1]);
        const double speed = std::hypot(b.v[0], b.v[1]);
        if (b.contact && ((err < 0.02 && speed < 0.05) || now-settle_since_ > 4)) launch(now);
        break;
      }
      case kLaunching:
        if (!released_ && now >= release_t_) {
          est_.release();
          released_ = true;
        }
        if (!b.contact && b.gap > 0.004) {
          phase_ = kBouncing;
          planned_flight_ = false;
        }
        else if (now-launch_t_ > 0.8) phase_ = kSettling, settle_since_ = now;
        break;
      case kBouncing:
        if (resting) {
          phase_ = kSettling;
          enter_balance(now, b);
          settle_since_ = now;
        } else {
          // Outer loop on height: compare each planned flight's apex with the request.
          if (!b.contact) flight_max_ = std::max(flight_max_, b.gap);
          if (planned_flight_ && b.v[2] < -0.2 && flight_max_ > 0.01) {
            speed_bias_ = std::clamp(speed_bias_+0.5*(launch_speed(flight_target_, c_.drag)-
                                                      launch_speed(flight_max_, c_.drag)), -0.4, 0.4);
            planned_flight_ = false;
          }
          bounce_step(now, b, apex_);
        }
        break;
    }
  }

  // Give each measurement of one stereo pair to a ball track. Outside juggling all go to the first
  // track (the one nearest to it when there are several); when juggling, detections are matched
  // to the two tracks by distance to their predictions and a new one starts the second track.
  void assign(double t, size_t from, size_t to, const PlateFn& past) {
    std::vector<V3> z;
    for (size_t i = from; i < to; i++) z.push_back(pending_[i].second);
    auto predicted = [&](Estimator& e, V3* p) {
      if (!e.valid(t)) return false;
      Ball b = e.ball();
      e.propagate(b, t, past, nullptr, false);
      *p = b.p;
      return true;
    };
    V3 p0, p1;
    const bool h0 = predicted(est_, &p0);
    if (request_ != kJuggle || (z.size() == 1 && !est2_.valid(t))) {
      size_t best = 0;
      if (h0)
        for (size_t i = 1; i < z.size(); i++)
          if (norm(z[i]-p0) < norm(z[best]-p0)) best = i;
      if (request_ != kJuggle || !h0 || norm(z[best]-p0) < 0.06) { est_.update(t, z[best], past); return; }
      est2_.update(t, z[best], past);  // A second ball has appeared.
      return;
    }
    const bool h1 = predicted(est2_, &p1);
    if (z.size() == 1) {
      const double d0 = h0 ? norm(z[0]-p0) : 1e9, d1 = h1 ? norm(z[0]-p1) : 1e9;
      (d0 <= d1 ? est_ : est2_).update(t, z[0], past);
      return;
    }
    // Two detections: keep the pairing with the smaller total distance.
    const double straight = (h0 ? norm(z[0]-p0) : 0)+(h1 ? norm(z[1]-p1) : 0);
    const double crossed = (h0 ? norm(z[1]-p0) : 0)+(h1 ? norm(z[0]-p1) : 0);
    const size_t a = straight <= crossed ? 0 : 1;
    est_.update(t, z[a], past);
    est2_.update(t, z[1-a], past);
  }

  V3 column(int ball) const {
    const double side = ball == 0 ? -1 : 1;
    return {juggle_centre_[0]+side*juggle_half_, juggle_centre_[1], 0};
  }

  // Strike whichever ball lands next. The balls share a grid of impact slots half a flight
  // apart; each ball owns every other slot, and each strike sends its ball to its own next slot.
  // A ball that has drifted off the grid, or bounced without a planned strike, is pulled back by
  // its next strike, so the two cannot settle into landing together.
  void juggle_step(double now, const Ball& b0) {
    if (plan_.valid && now > plan_.t+c_.hold_window) {
      const int k = plan_ball_;
      for (int a = 0; a < 2; a++)
        juggle_bias_[k][a] = std::clamp(juggle_bias_[k][a]+c_.lateral_integral*(column(k)[a]-plan_.p_in[a]), -0.3, 0.3);
      // Learn how long this ball's flights really take compared with what was asked.
      const double actual = plan_.t-strike_t_[k];
      if (strike_t_[k] > 0 && asked_[k] > 0 && actual > 0.5*asked_[k] && actual < 1.5*asked_[k])
        flight_err_[k] += 0.3*(std::clamp(actual-asked_[k], -0.08, 0.08)-flight_err_[k]);
      strike_t_[k] = plan_.t;
      asked_[k] = plan_.flight;
      lock_until_ = plan_.t+std::max(0.03, std::fabs(plan_.surface_speed)/c_.decel);
      plan_ = ImpactPlan{};
    }
    if (now < lock_until_) return;
    if (plan_.valid && plan_.t-now < c_.freeze) return;
    const Ball* balls[2] = {&b0, &ball2_};
    const double T0 = flight_time(launch_speed(apex_, c_.drag), c_.drag), half = 0.5*T0;
    ImpactPlan next[2];
    for (int k = 0; k < 2; k++) {
      if (balls[k]->contact) continue;
      target_ = column(k);
      for (int a = 0; a < 2; a++) bias_[a] = juggle_bias_[k][a];
      next[k] = plan_impact(now, *balls[k], apex_);
      if (next[k].valid && (next[k].t < now+0.03 || next[k].t > now+1.2)) next[k].valid = false;
    }
    // Keep refining the planned strike; switch only if the other ball clearly lands first and
    // there is still time to strike it.
    int A = -1;
    if (plan_.valid) {
      A = plan_ball_;
      const int o = 1-A;
      if (next[o].valid && next[o].t < plan_.t-0.02) A = o;
    } else {
      for (int k = 0; k < 2; k++)
        if (next[k].valid && (A < 0 || next[k].t < next[A].t)) A = k;
    }
    if (A < 0 || !next[A].valid) return;
    if (grid_t0_ < 0) {
      grid_t0_ = next[A].t;
      grid_owner_ = A;
    }
    // The other ball lands too soon after this strike to get one of its own, so it will rebound
    // off the plate unforced. Predict that rebound and move the grid onto it: the other ball is
    // struck normally when it next lands, and this ball is sent half a period away from it.
    const bool B_soon = next[1-A].valid && next[1-A].t-next[A].t < 0.1;
    if (B_soon) {
      const double vin = std::max(0.0, -next[1-A].v_in[2]);
      grid_t0_ = next[1-A].t+flight_time(est_.restitution()*vin, c_.drag);
      grid_owner_ = 1-A;
    }
    // This ball's own slots are grid_t0 + (2m + parity) * half; aim for the one nearest a full flight.
    const double parity = A == grid_owner_ ? 0 : 1;
    const double m = std::round((next[A].t+T0-grid_t0_-parity*half)/(2*half));
    const double slot = grid_t0_+(2*m+parity)*half;
    const double T = std::clamp(slot-next[A].t, 0.6*T0, 1.4*T0);
    const double asked = std::clamp(T-flight_err_[A], 0.5*T0, 1.5*T0);
    target_ = column(A);
    for (int a = 0; a < 2; a++) bias_[a] = juggle_bias_[A][a];
    ImpactPlan p = plan_impact(now, *balls[A], apex_for_flight(asked, c_.drag));
    // When the other ball lands within 100 ms it meets the same tilt, so strike level rather
    // than kick it sideways.
    if (B_soon) p.nx = p.ny = 0;
    if (!p.valid) return;
    p.flight = asked;
    plan_ball_ = A;
    execute(now, p);
  }

  // Plate motion for a planned impact: the stroke, the stop after it and the tilt.
  void execute(double now, const ImpactPlan& p) {
    const double left = p.t-now;
    if (left < 0.03) return;
    plan_ = p;
    if (phase_ == kJuggling) {
      // Strikes come every 200 ms or so, too often for a dip, stroke, stop and return. Swing
      // straight from wherever the plate is into the strike, so the plate moves in one smooth
      // oscillation; the stop and return after it only run if no next strike is planned.
      yaw_.to(now, left, p.yaw, p.yaw_rate, 0);
    } else if (left > c_.stroke_time+0.04) {
      yaw_.to(now, left-c_.stroke_time, p.yaw_pre, 0, 0);
      yaw_.then(c_.stroke_time, p.yaw, p.yaw_rate, 0);
    } else {
      yaw_.to(now, left, p.yaw, p.yaw_rate, 0);
    }
    // After the strike the ball is gone, so there is no need to stop hard. While juggling, ease
    // off over about a quarter of the swing; the next swing usually takes over before the end.
    const double T_d = phase_ == kJuggling ? 0.05 : std::clamp(std::fabs(p.surface_speed)/c_.decel, 0.02, 0.1);
    yaw_.then(T_d, p.yaw+p.yaw_rate*T_d/2, 0, 0);
    yaw_.then(c_.return_time, c_.idle_yaw, 0, 0);
    const double tilt_time = std::max(left-c_.hold_window, 0.03);
    nx_.to(now, tilt_time, p.nx, 0, 0);
    ny_.to(now, tilt_time, p.ny, 0, 0);
  }

  // Circle reference while balancing: the radius eases in so the path starts at rest.
  void circle_path(double now) {
    if (circle_start_ < 0) circle_start_ = now;
    const double tau = now-circle_start_, Tr = 1.5, w = 2*kPi/circle_period_;
    const double u = std::min(1.0, tau/Tr);
    const double r = circle_r_*u*u*u*(10-15*u+6*u*u);
    const double rd = tau < Tr ? circle_r_*30*u*u*(1-u)*(1-u)/Tr : 0;
    const double rdd = tau < Tr ? circle_r_*60*u*(1-u)*(1-2*u)/(Tr*Tr) : 0;
    const double th = w*tau, c = std::cos(th), s = std::sin(th);
    const double x[2] = {circle_centre_[0]+r*c, circle_centre_[1]+r*s};
    const double v[2] = {rd*c-r*w*s, rd*s+r*w*c};
    const double a[2] = {rdd*c-2*rd*w*s-r*w*w*c, rdd*s+2*rd*w*c-r*w*w*s};
    bal_.path(x, v, a);
  }

  void launch(double now) {
    phase_ = kLaunching;
    launch_t_ = now;
    const double v = c_.launch_speed;
    const double dyaw = -c_.launch_dip/dz_dyaw_;           // Down by the dip.
    const double rate = v/dz_dyaw_;                         // Upward at the launch speed.
    const double T_a = 2*c_.launch_dip/v;
    const double T_d = v/c_.decel;  // Average deceleration; the ball leaves once it exceeds g.
    double nx, ny, d, a;
    nx_.eval(now, &nx, &d, &a);
    ny_.eval(now, &ny, &d, &a);
    nx_.to(now, 0.15, 0, 0, 0);
    ny_.to(now, 0.15, 0, 0, 0);
    yaw_.to(now, 0.15, c_.idle_yaw+dyaw, 0, 0);
    yaw_.then(T_a, c_.idle_yaw, rate, 0);
    release_t_ = now+0.15+T_a;
    released_ = false;
    yaw_.then(T_d, c_.idle_yaw+rate*T_d/2, 0, 0);
    yaw_.then(c_.return_time, c_.idle_yaw, 0, 0);
    plan_ = ImpactPlan{};
  }

  void bounce_step(double now, const Ball& b, double apex) {
    if (plan_.valid && now > plan_.t+c_.hold_window) {
      // Learn a lateral bias from where the ball actually struck.
      for (int a = 0; a < 2; a++)
        bias_[a] = std::clamp(bias_[a]+c_.lateral_integral*(target_[a]-plan_.p_in[a]), -0.3, 0.3);
      lock_until_ = plan_.t+std::max(0.03, std::fabs(plan_.surface_speed)/c_.decel);
      saturated_ = plan_.saturated;
      flight_target_ = plan_.apex;
      if (song_ >= 0 && song_started_) {
        const Song& song = songs()[song_];
        // Learn how long flights of each note length really take compared with the plan.
        if (last_flight_ > 0 && last_impact_t_ > 0) {
          double& e = flight_error_[last_length_];
          e += 0.3*(std::clamp(plan_.t-last_impact_t_-last_flight_, -0.1, 0.1)-e);
        }
        last_flight_ = plan_.flight;
        last_length_ = song.notes[song_k_].second;
        // This impact played the song's current note; move the schedule on by its length.
        note_ = {song.notes[song_k_].first, plan_.t};  // 0 for a silent bounce.
        note_index_ = song_k_;
        if (std::fabs(plan_.t-song_next_t_) > 0.25) song_next_t_ = plan_.t;  // Lost the beat: restart it here.
        note_beat_ = song_next_t_;
        song_next_t_ += song.notes[song_k_].second*song_beat_;
        song_k_ = (song_k_+1)%static_cast<int>(song.notes.size());
      }
      if (circle_r_ > 0) {
        // Hop round the circle at the lap rate.
        if (last_impact_t_ > 0) hop_angle_ += 2*kPi*(plan_.t-last_impact_t_)/circle_period_;
        target_ = {circle_centre_[0]+circle_r_*std::cos(hop_angle_), circle_centre_[1]+circle_r_*std::sin(hop_angle_), 0};
      }
      last_impact_t_ = plan_.t;
      plan_ = ImpactPlan{};
      last_apex_ = flight_max_;
      flight_max_ = 0;
      planned_flight_ = apex > 0 && !saturated_ && !(song_ >= 0 && song_started_);
    }
    if (now < lock_until_) return;
    if (plan_.valid && plan_.t-now < c_.freeze) return;
    if (b.contact) return;
    if (circle_r_ > 0 && last_impact_t_ <= 0)
      target_ = {circle_centre_[0]+circle_r_*std::cos(hop_angle_), circle_centre_[1]+circle_r_*std::sin(hop_angle_), 0};
    // Pump up gradually: at most about double the last apex per strike.
    const double pump = std::min(apex, 2*std::max(last_apex_, b.gap)+0.02);
    ImpactPlan p;
    if (song_ >= 0 && apex > 0) {
      const Song& song = songs()[song_];
      const double first = apex_for_flight(song.notes[0].second*song_beat_, c_.drag);
      if (!song_started_) {
        // Bounce at the first note's height until steady, then start on the next impact.
        p = plan_impact(now, b, std::min(first, 2*std::max(last_apex_, b.gap)+0.02));
        if (p.valid && !saturated_ && std::fabs(last_apex_-first) < 0.25*first) {
          song_started_ = true;
          song_k_ = 0;
          song_next_t_ = p.t;
        }
      }
      if (song_started_) {
        // Next flight lasts until the next note's scheduled time, which also corrects drift.
        p = plan_impact(now, b, flight_target_ > 0 ? flight_target_ : first);
        const double length = song.notes[song_k_].second;
        for (int it = 0; it < 2 && p.valid; it++) {
          const double T = std::clamp(song_next_t_+length*song_beat_-p.t, 0.15, 0.65);
          const double asked = std::clamp(T-flight_error_[length], 0.12, 0.7);
          p = plan_impact(now, b, apex_for_flight(asked, c_.drag));
          p.flight = asked;
        }
      }
    } else {
      p = plan_impact(now, b, apex > 0 ? pump : 0);
    }
    if (!p.valid) return;
    execute(now, p);
  }

  Json platform_;
  Config c_;
  Kinematics k_;
  Estimator est_, est2_;
  Balancer bal_;
  GravityCalibration cal_;
  GravityModel gravity_;
  PlateHistory history_;
  std::vector<std::pair<double, V3>> pending_;
  Track yaw_, nx_, ny_;
  V3 beta_{0, 0, 0}, centre_{0, 0, 0}, target_{0, 0, 0};
  double dz_dyaw_ = 0, apex_ = 0.1, settle_since_ = 0, launch_t_ = 0, lock_until_ = 0;
  double bias_[2] = {0, 0};
  double contact_since_ = -1, flight_max_ = 0, speed_bias_ = 0, last_apex_ = 0;
  bool planned_flight_ = false, released_ = false, saturated_ = false;
  double release_t_ = 0, flight_target_ = 0;
  // Circles.
  double circle_r_ = 0, circle_period_ = 4, circle_start_ = -1, hop_angle_ = 0, last_impact_t_ = 0;
  V3 circle_centre_{0, 0, 0};
  // Songs.
  int song_ = -1, song_k_ = 0, note_index_ = -1;
  bool song_started_ = false;
  double song_beat_ = 0.25, song_next_t_ = 0, last_flight_ = 0, last_length_ = 0;
  std::map<double, double> flight_error_;  // Actual minus planned flight time, by note length.
  std::pair<int, double> note_{0, 0};
  double note_beat_ = 0;
  // Two balls.
  bool valid2_ = false;
  Ball ball2_;
  V3 juggle_centre_{0, 0, 0};
  double juggle_half_ = 0.045, juggle_bias_[2][2] = {{0, 0}, {0, 0}};
  int plan_ball_ = 0, grid_owner_ = 0;
  double grid_t0_ = -1, strike_t_[2] = {0, 0}, asked_[2] = {0, 0}, flight_err_[2] = {0, 0};
  int request_ = kHold, phase_ = kLevel;
  ImpactPlan plan_;
  Output last_;
  double dt_;
  Q3 prev_q_{0, 0, 0}, prev_v_{0, 0, 0};
  bool commanded_ = false;
  long limited_ = 0;
};

}  // namespace ballctl
