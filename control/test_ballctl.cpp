// Unit checks that need no simulator: kinematics, trajectories, impact planning and LQR.
#include <cstdio>
#include <cstdlib>

#include "controller.hpp"

using namespace ballctl;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main(int argc, char** argv) {
  const std::string config = argc > 1 ? argv[1] : "control.example.json";
  Controller ctl(config, 1e-3);
  const Kinematics& k = ctl.kinematics();

  // Inverse then forward kinematics agree across the working range.
  std::srand(3);
  double worst = 0;
  for (int i = 0; i < 3000; i++) {
    auto r = [] { return std::rand()/(double)RAND_MAX-0.5; };
    const Posture c{0.1+0.5*r(), 0.25*r(), 0.25*r()};
    Pose p, f;
    Q3 q;
    V3 beta{0, 0, 0};
    CHECK(k.pose(c, &p) && k.inverse(p, &q) && k.forward(q, &f, &beta), "kinematics failed at %g %g %g", c.yaw, c.nx, c.ny);
    worst = std::max(worst, norm(f.t-p.t)+norm(Kinematics::normal(f)-Kinematics::normal(p)));
  }
  CHECK(worst < 1e-8, "round trip error %g", worst);

  // Quintic boundary conditions.
  const Quintic s = Quintic::make(1, 0.2, 0.1, -0.5, 2, 0.3, 0.7, -1);
  double p, v, a;
  s.eval(1, &p, &v, &a);
  CHECK(std::fabs(p-0.1) < 1e-12 && std::fabs(v+0.5) < 1e-12 && std::fabs(a-2) < 1e-9, "quintic start");
  s.eval(1.2, &p, &v, &a);
  CHECK(std::fabs(p-0.3) < 1e-12 && std::fabs(v-0.7) < 1e-9 && std::fabs(a+1) < 1e-6, "quintic end");

  // Drag model: launch speed and apex agree with a direct integration.
  const double drag = ctl.config().drag;
  const double v0 = launch_speed(0.15, drag);
  double z = 0, vz = v0, t = 0;
  while (vz > 0) { const double acc = -kG-drag*vz*std::fabs(vz); z += vz*1e-5+0.5*acc*1e-10; vz += acc*1e-5; t += 1e-5; }
  CHECK(std::fabs(z-0.15) < 1e-4, "apex %g for 0.15 m", z);
  double fall = 0;
  while (z > 0) { const double acc = -kG-drag*vz*std::fabs(vz); z += vz*1e-5; vz += acc*1e-5; fall += 1e-5; }
  CHECK(std::fabs(t+fall-flight_time(v0, drag)) < 1e-3, "flight time %g vs %g", t+fall, flight_time(v0, drag));

  // Impact plan: a ball falling onto the plate centre is sent back up to the apex and over the target.
  Ball b;
  b.t = 0;
  b.p = ctl.centre()+V3{0.02, -0.01, 0.12};
  b.v = {-0.05, 0.03, 0};
  ctl.set_target(0, 0);
  const ImpactPlan plan = ctl.plan_impact(0, b, 0.10);
  CHECK(plan.valid, "no impact plan");
  const double apex = std::log(1+plan.v_out[2]*plan.v_out[2]*drag/kG)/(2*drag);
  CHECK(std::fabs(apex-0.10) < 0.002, "planned apex %g", apex);
  CHECK(std::fabs(plan.surface_speed) < ctl.config().plate_speed_max+1e-9, "plate speed %g", plan.surface_speed);
  std::printf("impact plan: t %.3f s, v_in %.3f, v_out %.3f %.3f %.3f, tilt %.2f %.2f deg, plate %.3f m/s, yaw %.2f deg\n",
              plan.t, plan.v_in[2], plan.v_out[0], plan.v_out[1], plan.v_out[2], std::asin(plan.nx)*180/kPi,
              std::asin(plan.ny)*180/kPi, plan.surface_speed, plan.yaw*180/kPi);

  // LQR closed loop on the design model is stable and settles.
  const AxisLqr& L = ctl.lqr();
  double e = 0.1, ed = 0, n = 0, nd = 0, in = 0, peak = 0;
  const double g = kG*ctl.config().rolling_gain, wa = 1/ctl.config().balance_tau;
  for (int i = 0; i < 8000; i++) {
    const double u = -(L.K[0]*e+L.K[1]*ed+L.K[2]*n+L.K[3]*nd+L.K[4]*in);
    const double acc = wa*wa*(u-n)-2*wa*nd;
    e += ed*1e-3; ed += g*n*1e-3; n += nd*1e-3; nd += acc*1e-3; in += e*1e-3;
    if (i > 5000) peak = std::max(peak, std::fabs(e));
  }
  CHECK(peak < 0.001, "LQR model residual %g after 5 s", peak);
  std::printf("LQR gains %.3f %.3f %.3f %.4f %.3f\n", L.K[0], L.K[1], L.K[2], L.K[3], L.K[4]);
  std::printf(failures ? "%d failures\n" : "all checks passed\n", failures);
  return failures != 0;
}
