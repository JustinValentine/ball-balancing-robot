// Gravity model of the platform and its calibration from measured holding torques.
//
// The torque each crank needs to hold the mechanism still is the gradient of the potential energy
// with respect to the crank angles, plus a constant offset per motor. The potential is linear in
// the bodies' masses and first mass moments (mass times centre of mass), so the holding torque is
// Y(posture)*theta with 22 parameters:
//
//   0-3    plate: mass, then first moment along the plate's x, y, z (kg, kg m), in the plate frame
//          whose origin is the centroid of the spherical joints
//   4-9    cranks 1-3: first moment along the leg's in-plane directions u and w (kg m)
//   10-18  rods 1-3: mass at the elbow, then first moment along u and w (kg, kg m)
//   19-21  constant torque offset of each motor (N m)
//
// Moments along a leg's axis do not change its potential and are left out. The calibration moves
// the plate through test postures without a ball, approaches each one from both sides so friction
// cancels in the average, holds it, averages the torque the controllers report, and fits theta by
// least squares, pulled gently toward the prior where the data say little.
#pragma once

#include "ballctl.hpp"

namespace ballctl {

struct GravityModel {
  static constexpr int kPotential = 19, kParams = 22;
  std::array<double, kParams> theta{};

  static const char* name(int j) {
    static const char* names[kParams] = {
        "plate mass kg", "plate moment x kg m", "plate moment y kg m", "plate moment z kg m",
        "crank 1 moment u", "crank 1 moment w", "crank 2 moment u", "crank 2 moment w",
        "crank 3 moment u", "crank 3 moment w",
        "rod 1 mass kg", "rod 1 moment u", "rod 1 moment w", "rod 2 mass kg", "rod 2 moment u",
        "rod 2 moment w", "rod 3 mass kg", "rod 3 moment u", "rod 3 moment w",
        "motor 1 offset N m", "motor 2 offset N m", "motor 3 offset N m"};
    return names[j];
  }

  // A plate of known mass with its centre of mass above the joint centroid, nothing else.
  static GravityModel plate(double mass, double com_height) {
    GravityModel m;
    m.theta[0] = mass;
    m.theta[3] = mass*com_height;
    return m;
  }

  // Potential energy per unit of each potential parameter at a pose.
  static bool basis(const Kinematics& k, const Pose& p, double* U) {
    Q3 q;
    V3 elbow[3];
    double rod[3];
    if (!k.legs_state(p, &q, elbow, rod)) return false;
    U[0] = kG*p.t[2];
    for (int j = 0; j < 3; j++) U[1+j] = kG*p.R.m[2][j];
    for (int i = 0; i < 3; i++) {
      const M3 Rc = rotation(k.axis(i), q[i]), Rr = rotation(k.axis(i), rod[i]);
      U[4+2*i] = kG*(Rc*k.leg_u(i))[2];
      U[5+2*i] = kG*(Rc*k.leg_w(i))[2];
      U[10+3*i] = kG*elbow[i][2];
      U[11+3*i] = kG*(Rr*k.leg_u(i))[2];
      U[12+3*i] = kG*(Rr*k.leg_w(i))[2];
    }
    return true;
  }

  // Gradient of each basis potential with respect to the posture (yaw, nx, ny).
  static bool gradient(const Kinematics& k, const Posture& c, double G[kPotential][3]) {
    const double h = 1e-5;
    for (int a = 0; a < 3; a++) {
      Posture lo = c, hi = c;
      (a == 0 ? lo.yaw : a == 1 ? lo.nx : lo.ny) -= h;
      (a == 0 ? hi.yaw : a == 1 ? hi.nx : hi.ny) += h;
      Pose pl, ph;
      double Ul[kPotential], Uh[kPotential];
      if (!k.pose(lo, &pl) || !k.pose(hi, &ph) || !basis(k, pl, Ul) || !basis(k, ph, Uh)) return false;
      for (int j = 0; j < kPotential; j++) G[j][a] = (Uh[j]-Ul[j])/(2*h);
    }
    return true;
  }

  // Regressor: holding torque at the three cranks per unit of each parameter.
  static bool regressor(const Kinematics& k, const Posture& c, double Y[3][kParams]) {
    double G[kPotential][3];
    Mat<3, 3> J, JiT;
    if (!gradient(k, c, G) || !k.jacobian(c, &J) || !invert(J.T(), &JiT)) return false;
    for (int j = 0; j < kPotential; j++)
      for (int i = 0; i < 3; i++) Y[i][j] = JiT(i, 0)*G[j][0]+JiT(i, 1)*G[j][1]+JiT(i, 2)*G[j][2];
    for (int i = 0; i < 3; i++)
      for (int m = 0; m < 3; m++) Y[i][kPotential+m] = i == m ? 1 : 0;
    return true;
  }

  // Generalised gravity force on (yaw, nx, ny), and the motor offsets.
  bool force(const Kinematics& k, const Posture& c, V3* F) const {
    double G[kPotential][3];
    if (!gradient(k, c, G)) return false;
    *F = {0, 0, 0};
    for (int j = 0; j < kPotential; j++)
      for (int a = 0; a < 3; a++) (*F)[a] += G[j][a]*theta[j];
    return true;
  }
  V3 offsets() const { return {theta[19], theta[20], theta[21]}; }

  // Holding torque at the cranks.
  bool torque(const Kinematics& k, const Posture& c, Q3* tau) const {
    double Y[3][kParams];
    if (!regressor(k, c, Y)) return false;
    for (int i = 0; i < 3; i++) {
      (*tau)[i] = 0;
      for (int j = 0; j < kParams; j++) (*tau)[i] += Y[i][j]*theta[j];
    }
    return true;
  }
};

struct CalibrationSample {
  Q3 q{0, 0, 0}, tau{0, 0, 0};  // Means over the hold, crank angles (rad) and torques (N m).
  int direction = 0;            // +1 or -1: the side it was approached from.
  int target = 0;
};

class GravityCalibration {
 public:
  GravityCalibration(const Kinematics& k, const Config& c) : k_(k), c_(c) {}

  // Test postures: three twists around idle, each level and tilted in a ring of directions.
  std::vector<Posture> targets() const {
    std::vector<Posture> out;
    for (int y = -1; y <= 1; y++) {
      const double yaw = c_.idle_yaw+y*c_.cal_yaw_step;
      out.push_back({yaw, 0, 0});
      for (int ring = 1; ring <= 2; ring++) {
        const double s = std::sin(c_.cal_tilt*ring/2);
        for (int d = 0; d < 6; d++) {
          const double a = (d+0.5*(ring-1))*kPi/3;
          out.push_back({yaw, s*std::cos(a), s*std::sin(a)});
        }
      }
    }
    // Keep postures whose cranks, including the approach offsets, stay inside the limits.
    std::vector<Posture> ok;
    for (const auto& t : out) {
      bool fine = true;
      for (int dir : {-1, 0, 1}) {
        Q3 q;
        if (!k_.inverse(approach(t, dir), &q)) { fine = false; break; }
        for (int i = 0; i < 3; i++)
          if (q[i] < c_.crank_min+0.02 || q[i] > c_.crank_max-0.02) fine = false;
      }
      if (fine) ok.push_back(t);
    }
    return ok;
  }

  void start(double now, const Posture& from) {
    targets_ = targets();
    samples_.clear();
    index_ = 0;
    for (auto* t : {&yaw_, &nx_, &ny_}) t->reset(0);
    yaw_.reset(from.yaw);
    nx_.reset(from.nx);
    ny_.reset(from.ny);
    running_ = !targets_.empty();
    if (running_) plan(now);
  }

  bool running() const { return running_; }
  int total() const { return 2*static_cast<int>(targets_.size()); }
  int done() const { return index_; }
  const std::vector<CalibrationSample>& samples() const { return samples_; }

  // Advance with the measured cranks and torques; returns the posture command and its rate.
  void step(double now, const Q3& q, const Q3& tau, double* y, double* yd, double* ya) {
    yaw_.eval(now, &y[0], &yd[0], &ya[0]);
    nx_.eval(now, &y[1], &yd[1], &ya[1]);
    ny_.eval(now, &y[2], &yd[2], &ya[2]);
    if (!running_) return;
    if (now >= measure_from_ && now < measure_to_) {
      for (int i = 0; i < 3; i++) { sum_q_[i] += q[i]; sum_tau_[i] += tau[i]; }
      count_++;
    } else if (now >= measure_to_) {
      CalibrationSample s;
      for (int i = 0; i < 3; i++) { s.q[i] = sum_q_[i]/std::max(count_, 1); s.tau[i] = sum_tau_[i]/std::max(count_, 1); }
      s.direction = index_%2 ? -1 : 1;
      s.target = index_/2;
      if (count_ > 0) samples_.push_back(s);
      index_++;
      if (index_ >= total()) {
        running_ = false;
      } else {
        plan(now);
      }
    }
  }

  // Fit theta to the samples. Returns false with a reason if the data cannot support a fit.
  bool fit(const GravityModel& prior, GravityModel* out, std::string* report) const {
    // Average the two approaches to each posture so Coulomb friction cancels.
    struct Row { double Y[3][GravityModel::kParams]; Q3 tau; };
    std::vector<Row> rows;
    for (size_t a = 0; a+1 < samples_.size(); a++) {
      const auto &s1 = samples_[a], &s2 = samples_[a+1];
      if (s1.target != s2.target || s1.direction == s2.direction) continue;
      Q3 q;
      Row r;
      for (int i = 0; i < 3; i++) { q[i] = 0.5*(s1.q[i]+s2.q[i]); r.tau[i] = 0.5*(s1.tau[i]+s2.tau[i]); }
      Pose p;
      V3 beta{0, 0, 0};
      if (!k_.forward(q, &p, &beta) || !GravityModel::regressor(k_, Kinematics::posture(p), r.Y)) continue;
      rows.push_back(r);
      a++;
    }
    constexpr int N = GravityModel::kParams;
    if (rows.size() < 8) {
      *report = "Only " + std::to_string(rows.size()) + " usable postures; need at least 8.";
      return false;
    }
    Mat<N, N> A;
    std::array<double, N> b{};
    for (const auto& r : rows)
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < N; j++) {
          b[j] += r.Y[i][j]*r.tau[i];
          for (int m = 0; m < N; m++) A(j, m) += r.Y[i][j]*r.Y[i][m];
        }
    // Ridge toward the prior, scaled to the data, so unidentifiable combinations stay put.
    double scale = 0;
    for (int j = 0; j < N; j++) scale = std::max(scale, A(j, j));
    const double lambda = 1e-4*scale;
    for (int j = 0; j < N; j++) {
      A(j, j) += lambda;
      b[j] += lambda*prior.theta[j];
    }
    Mat<N, N> Ai;
    if (!invert(A, &Ai)) {
      *report = "The calibration data are degenerate.";
      return false;
    }
    for (int j = 0; j < N; j++) {
      out->theta[j] = 0;
      for (int m = 0; m < N; m++) out->theta[j] += Ai(j, m)*b[m];
    }
    auto rms = [&](const GravityModel& g, double* worst) {
      double s = 0;
      *worst = 0;
      for (const auto& r : rows)
        for (int i = 0; i < 3; i++) {
          double t = 0;
          for (int j = 0; j < N; j++) t += r.Y[i][j]*g.theta[j];
          s += (t-r.tau[i])*(t-r.tau[i]);
          *worst = std::max(*worst, std::fabs(t-r.tau[i]));
        }
      return std::sqrt(s/(3*rows.size()));
    };
    double worst_fit, worst_prior;
    const double fit_rms = rms(*out, &worst_fit), prior_rms = rms(prior, &worst_prior);
    // Friction: half the difference between the two approaches, per motor.
    double friction[3] = {0, 0, 0};
    int pairs = 0;
    for (size_t a = 0; a+1 < samples_.size(); a++)
      if (samples_[a].target == samples_[a+1].target && samples_[a].direction != samples_[a+1].direction) {
        for (int i = 0; i < 3; i++) friction[i] += 0.5*std::fabs(samples_[a].tau[i]-samples_[a+1].tau[i]);
        pairs++;
        a++;
      }
    char line[256];
    std::snprintf(line, sizeof(line),
                  "%zu postures. Torque residual RMS %.4f N m (worst %.4f); the prior model gave %.4f (worst %.4f).\n"
                  "Mean friction seen by approaching from both sides: %.3f %.3f %.3f N m.\n",
                  rows.size(), fit_rms, worst_fit, prior_rms, worst_prior, friction[0]/std::max(pairs, 1),
                  friction[1]/std::max(pairs, 1), friction[2]/std::max(pairs, 1));
    *report = line;
    for (int j = 0; j < N; j++) {
      std::snprintf(line, sizeof(line), "  %-22s %+.5f  (prior %+.5f)\n", GravityModel::name(j), out->theta[j],
                    prior.theta[j]);
      *report += line;
    }
    return true;
  }

 private:
  // Approach along pure twist: all three cranks then arrive from the same side, and from the
  // opposite side on the second visit, so each crank's friction cancels in the pair's average.
  Posture approach(const Posture& t, int dir) const { return {t.yaw+dir*c_.cal_approach, t.nx, t.ny}; }

  void plan(double now) {
    const Posture target = targets_[index_/2];
    const Posture from = approach(target, index_%2 ? -1 : 1);
    double y, d, a, x, z;
    yaw_.eval(now, &y, &d, &a);
    nx_.eval(now, &x, &d, &a);
    ny_.eval(now, &z, &d, &a);
    const double dist = std::max({std::fabs(from.yaw-y), std::fabs(from.nx-x)*2, std::fabs(from.ny-z)*2});
    const double T1 = std::max(0.8, 1.875*dist/c_.cal_speed), T2 = 0.6;
    yaw_.to(now, T1, from.yaw, 0, 0);
    nx_.to(now, T1, from.nx, 0, 0);
    ny_.to(now, T1, from.ny, 0, 0);
    yaw_.then(T2, target.yaw, 0, 0);
    nx_.then(T2, target.nx, 0, 0);
    ny_.then(T2, target.ny, 0, 0);
    measure_from_ = now+T1+T2+c_.cal_settle;
    measure_to_ = measure_from_+c_.cal_average;
    sum_q_ = sum_tau_ = {0, 0, 0};
    count_ = 0;
  }

  const Kinematics& k_;
  const Config& c_;
  std::vector<Posture> targets_;
  std::vector<CalibrationSample> samples_;
  Track yaw_, nx_, ny_;
  int index_ = 0, count_ = 0;
  bool running_ = false;
  double measure_from_ = 0, measure_to_ = 0;
  Q3 sum_q_{0, 0, 0}, sum_tau_{0, 0, 0};
};

}  // namespace ballctl
