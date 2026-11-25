// C interface for the simulator (ctypes) and other hosts.
#include <cstring>
#include <memory>

#include "controller.hpp"

extern "C" {

struct bc_output {
  double q[3], qd[3], tau[3];
  int ok, phase, ball_valid, contact;
  double ball[3], ball_v[3];
  double yaw, nx, ny, next_impact, restitution, apex;
  double plate_normal[3];
  int limited;
  int note, note_index;
  double note_time, note_beat;
};

void* bc_create(const char* config_path, double dt, char* error, int error_size) {
  try {
    return new ballctl::Controller(config_path, dt);
  } catch (const std::exception& e) {
    if (error && error_size > 0) {
      std::strncpy(error, e.what(), error_size-1);
      error[error_size-1] = 0;
    }
    return nullptr;
  }
}

void bc_destroy(void* h) { delete static_cast<ballctl::Controller*>(h); }

void bc_set_mode(void* h, int mode) { static_cast<ballctl::Controller*>(h)->set_mode(mode); }
void bc_set_target(void* h, double x, double y) { static_cast<ballctl::Controller*>(h)->set_target(x, y); }
void bc_set_path(void* h, const double* x, const double* v, const double* a) {
  static_cast<ballctl::Controller*>(h)->set_path(x, v, a);
}
void bc_set_apex(void* h, double apex) { static_cast<ballctl::Controller*>(h)->set_apex(apex); }

void bc_measurement(void* h, double t_capture, const double* left) {
  static_cast<ballctl::Controller*>(h)->measurement(t_capture, {left[0], left[1], left[2]});
}

void bc_step(void* h, double now, const double* q, const double* qd, bc_output* out) {
  const auto o = static_cast<ballctl::Controller*>(h)->step(now, {q[0], q[1], q[2]}, {qd[0], qd[1], qd[2]});
  for (int i = 0; i < 3; i++) {
    out->q[i] = o.q[i]; out->qd[i] = o.qd[i]; out->tau[i] = o.tau[i];
    out->ball[i] = o.ball[i]; out->ball_v[i] = o.ball_v[i]; out->plate_normal[i] = o.plate_normal[i];
  }
  out->ok = o.ok; out->phase = o.phase; out->ball_valid = o.ball_valid; out->contact = o.contact;
  out->yaw = o.yaw; out->nx = o.nx; out->ny = o.ny; out->next_impact = o.next_impact;
  out->restitution = o.restitution; out->apex = o.apex; out->limited = o.limited;
  out->note = o.note; out->note_index = o.note_index; out->note_time = o.note_time; out->note_beat = o.note_beat;
}

// Circles (radius m, lap s; radius 0 stops) and songs (index from bc_song_name, -1 stops).
void bc_set_circle(void* h, double radius, double period) {
  static_cast<ballctl::Controller*>(h)->set_circle(radius, period);
}
void bc_set_song(void* h, int index, double beat) { static_cast<ballctl::Controller*>(h)->set_song(index, beat); }
int bc_song_count() { return static_cast<int>(ballctl::songs().size()); }
const char* bc_song_name(int i) { return ballctl::songs()[i].name.c_str(); }
// Notes of a song: MIDI numbers and lengths in units of its beat. Returns the note count.
int bc_song_notes(int i, int* midi, double* length, int capacity) {
  const auto& notes = ballctl::songs()[i].notes;
  for (size_t k = 0; k < notes.size() && static_cast<int>(k) < capacity; k++) {
    midi[k] = notes[k].first;
    length[k] = notes[k].second;
  }
  return static_cast<int>(notes.size());
}
double bc_song_beat(int i) { return ballctl::songs()[i].beat; }

// Plate centre at the idle posture (base frame, m) and the balance gains, for reports.
void bc_info(void* h, double* centre, double* gains) {
  auto* c = static_cast<ballctl::Controller*>(h);
  for (int i = 0; i < 3; i++) centre[i] = c->centre()[i];
  for (int i = 0; i < 5; i++) gains[i] = c->lqr().K[i];
}

static void copy_output(const ballctl::Output& o, bc_output* out) {
  for (int i = 0; i < 3; i++) {
    out->q[i] = o.q[i]; out->qd[i] = o.qd[i]; out->tau[i] = o.tau[i];
    out->ball[i] = o.ball[i]; out->ball_v[i] = o.ball_v[i]; out->plate_normal[i] = o.plate_normal[i];
  }
  out->ok = o.ok; out->phase = o.phase; out->ball_valid = o.ball_valid; out->contact = o.contact;
  out->yaw = o.yaw; out->nx = o.nx; out->ny = o.ny; out->next_impact = o.next_impact;
  out->restitution = o.restitution; out->apex = o.apex; out->limited = o.limited;
}

// Gravity calibration.
void bc_calibration_start(void* h, double now, const double* q) {
  static_cast<ballctl::Controller*>(h)->start_calibration(now, {q[0], q[1], q[2]});
}

// Returns 1 while the calibration runs and 0 once it has finished.
int bc_calibration_step(void* h, double now, const double* q, const double* qd, const double* tau, bc_output* out) {
  auto* c = static_cast<ballctl::Controller*>(h);
  copy_output(c->calibration_step(now, {q[0], q[1], q[2]}, {qd[0], qd[1], qd[2]}, {tau[0], tau[1], tau[2]}), out);
  return c->calibrating();
}

void bc_calibration_progress(void* h, int* done, int* total) {
  auto* c = static_cast<ballctl::Controller*>(h);
  *done = c->calibration().done();
  *total = c->calibration().total();
}

// Fits and installs the model. Writes the 22 parameters and a report; returns 0 on failure.
int bc_calibration_finish(void* h, double* params, char* report, int report_size) {
  ballctl::GravityModel g;
  std::string text;
  const bool ok = static_cast<ballctl::Controller*>(h)->finish_calibration(&g, &text);
  if (ok)
    for (int j = 0; j < ballctl::GravityModel::kParams; j++) params[j] = g.theta[j];
  if (report && report_size > 0) {
    std::strncpy(report, text.c_str(), report_size-1);
    report[report_size-1] = 0;
  }
  return ok;
}

void bc_gravity_get(void* h, double* params) {
  const auto& g = static_cast<ballctl::Controller*>(h)->gravity();
  for (int j = 0; j < ballctl::GravityModel::kParams; j++) params[j] = g.theta[j];
}

void bc_gravity_set(void* h, const double* params) {
  ballctl::GravityModel g;
  for (int j = 0; j < ballctl::GravityModel::kParams; j++) g.theta[j] = params[j];
  static_cast<ballctl::Controller*>(h)->set_gravity(g);
}

// Holding torque the gravity model predicts at crank angles q.
int bc_holding_torque(void* h, const double* q, double* tau) {
  ballctl::Q3 t;
  if (!static_cast<ballctl::Controller*>(h)->holding_torque({q[0], q[1], q[2]}, &t)) return 0;
  for (int i = 0; i < 3; i++) tau[i] = t[i];
  return 1;
}

// Holding torque the gravity model predicts for a posture.
int bc_gravity_torque(void* h, double yaw, double nx, double ny, double* tau) {
  auto* c = static_cast<ballctl::Controller*>(h);
  ballctl::Q3 t;
  if (!c->gravity().torque(c->kinematics(), ballctl::Posture{yaw, nx, ny}, &t)) return 0;
  for (int i = 0; i < 3; i++) tau[i] = t[i];
  return 1;
}

// Inverse kinematics for tests: crank angles for a (yaw, nx, ny) posture.
int bc_inverse(void* h, double yaw, double nx, double ny, double* q) {
  ballctl::Q3 r;
  if (!static_cast<ballctl::Controller*>(h)->kinematics().inverse(ballctl::Posture{yaw, nx, ny}, &r)) return 0;
  for (int i = 0; i < 3; i++) q[i] = r[i];
  return 1;
}

}
