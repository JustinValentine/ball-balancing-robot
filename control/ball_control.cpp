// Hardware runtime: drives the three moteus controllers from the ball tracker.
//
//   build/ball-vision track camera.yml stereo.yml tracker.yml normals.txt |
//     control/build/ball-control control.json --socketcan-iface can0 --normals normals.txt
//
// The tracker's CSV arrives on stdin. Keys are read from the terminal: h hold, b balance,
// j bounce, w/a/s/d move the target 5 mm, c centre it, +/- change the apex 10 mm, o circle on or
// off, 1-5 bounce to a song, 0 stop the song, k juggle two balls, q quit. --sound plays each song
// note with aplay. Juggling needs "balls: 2" in tracker.yml; drop the second ball in by hand.
// Any communication fault, controller fault or large tracking error stops all three motors.
//
// Gravity calibration (remove the ball first):
//   control/build/ball-control control.json --socketcan-iface can0 --gravity-cal gravity.json
// moves the plate through about 40 postures (about 4 minutes), fits the gravity model and writes
// it with the raw samples. Add "gravity_file": "gravity.json" to control.json to use it.
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "controller.hpp"
#include "moteus.h"

namespace mm = mjbots::moteus;
using namespace ballctl;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

double realtime() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec+ts.tv_nsec*1e-9;
}

struct Measurement {
  double t;
  V3 left;
};

// Reads tracker rows: left_us,right_us,valid,x_mm,y_mm,z_mm,...
class TrackerReader {
 public:
  void start() { thread_ = std::thread([this] { run(); }); }
  void join() { if (thread_.joinable()) thread_.join(); }
  std::vector<Measurement> take() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Measurement> out;
    out.swap(queue_);
    return out;
  }
  bool closed() const { return closed_; }

 private:
  void run() {
    std::string line;
    while (!g_stop && std::getline(std::cin, line)) {
      if (line.empty() || line[0] == 'l') continue;  // Header.
      std::vector<std::string> f;
      size_t start = 0;
      for (size_t i = 0; i <= line.size(); i++)
        if (i == line.size() || line[i] == ',') { f.push_back(line.substr(start, i-start)); start = i+1; }
      if (f.size() < 6 || f[2] != "1") continue;
      try {
        Measurement m{std::stod(f[0])*1e-6, {std::stod(f[3])*1e-3, std::stod(f[4])*1e-3, std::stod(f[5])*1e-3}};
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(m);
        if (queue_.size() > 100) queue_.erase(queue_.begin());
      } catch (const std::exception&) {
      }
    }
    closed_ = true;
  }
  std::thread thread_;
  std::mutex mutex_;
  std::vector<Measurement> queue_;
  std::atomic<bool> closed_{false};
};

// Plays song notes with aplay from its own thread, so the control loop never waits on audio.
class NotePlayer {
 public:
  NotePlayer() {
    char dir[] = "/tmp/ball-notes-XXXXXX";
    if (mkdtemp(dir)) dir_ = dir;
    thread_ = std::thread([this] { run(); });
  }
  ~NotePlayer() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
  void play(int midi) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(midi);
    }
    cv_.notify_one();
  }

 private:
  std::string file(int midi) {
    const std::string path = dir_+"/"+std::to_string(midi)+".wav";
    if (FILE* f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return path; }
    const int rate = 44100, n = rate*45/100;
    const double hz = note_hz(midi);
    std::vector<int16_t> pcm(n);
    for (int i = 0; i < n; i++) {
      const double t = static_cast<double>(i)/rate;
      const double v = std::sin(2*kPi*hz*t)*std::exp(-7*t)+0.25*std::sin(8*kPi*hz*t)*std::exp(-20*t);
      pcm[i] = static_cast<int16_t>(0.4*std::min(1.0, t/0.003)*v*32767);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return path;
    auto u32 = [&](uint32_t x) { std::fwrite(&x, 4, 1, f); };
    auto u16 = [&](uint16_t x) { std::fwrite(&x, 2, 1, f); };
    std::fwrite("RIFF", 1, 4, f); u32(36+2*n); std::fwrite("WAVEfmt ", 1, 8, f);
    u32(16); u16(1); u16(1); u32(rate); u32(2*rate); u16(2); u16(16);
    std::fwrite("data", 1, 4, f); u32(2*n);
    std::fwrite(pcm.data(), 2, n, f);
    std::fclose(f);
    return path;
  }
  void run() {
    while (true) {
      int midi;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return done_ || !queue_.empty(); });
        if (done_) return;
        midi = queue_.front();
        queue_.erase(queue_.begin());
      }
      if (!dir_.empty() && midi > 0)
        if (std::system(("aplay -q " + file(midi) + " >/dev/null 2>&1 &").c_str()) != 0) {}
    }
  }
  std::string dir_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<int> queue_;
  bool done_ = false;
};

class Terminal {
 public:
  Terminal() {
    fd_ = ::open("/dev/tty", O_RDONLY|O_NONBLOCK);
    if (fd_ >= 0 && tcgetattr(fd_, &saved_) == 0) {
      termios raw = saved_;
      raw.c_lflag &= ~(ICANON|ECHO);
      tcsetattr(fd_, TCSANOW, &raw);
      ok_ = true;
    }
  }
  ~Terminal() {
    if (ok_) tcsetattr(fd_, TCSANOW, &saved_);
    if (fd_ >= 0) ::close(fd_);
  }
  int key() {
    char c;
    return fd_ >= 0 && ::read(fd_, &c, 1) == 1 ? c : -1;
  }

 private:
  int fd_ = -1;
  bool ok_ = false;
  termios saved_{};
};

struct Args {
  std::string config, normals, log, gravity_cal;
  double max_torque = 1.0, watchdog = 0.05, ramp = 3.0;
  bool sound = false;
};

[[noreturn]] void usage() {
  std::cerr << "Usage: ball-control control.json [--normals FILE] [--log FILE.csv] [--max-torque NM]\n"
               "                    [--ramp S] [--gravity-cal OUT.json] [--sound]\n"
               "                    <moteus transport options, e.g. --socketcan-iface can0>\n"
               "Reads tracker CSV on stdin. Keys: h hold, b balance, j bounce, wasd target, c centre,\n"
               "+/- apex, o circle, 1-5 songs, 0 stop the song, k juggle two balls, q quit.\n";
  std::exit(2);
}

}  // namespace

int run(int argc, char** argv);

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ball-control: %s\n", e.what());
    return 1;
  }
}

int run(int argc, char** argv) {
  std::vector<std::string> all(argv, argv+argc);
  if (all.size() < 2 || all[1] == "-h" || all[1] == "--help") usage();
  const std::vector<std::string> rest = mm::Controller::ProcessTransportArgs(all);
  Args args;
  for (size_t i = 1; i < rest.size(); i++) {
    auto value = [&] { if (i+1 >= rest.size()) usage(); return rest[++i]; };
    if (rest[i] == "--normals") args.normals = value();
    else if (rest[i] == "--log") args.log = value();
    else if (rest[i] == "--max-torque") args.max_torque = std::stod(value());
    else if (rest[i] == "--ramp") args.ramp = std::stod(value());
    else if (rest[i] == "--gravity-cal") args.gravity_cal = value();
    else if (rest[i] == "--sound") args.sound = true;
    else if (args.config.empty() && rest[i][0] != '-') args.config = rest[i];
    else usage();
  }
  if (args.config.empty()) usage();

  const double dt = 1e-3;
  Controller ctl(args.config, dt);
  const Config& cfg = ctl.config();
  const Json& platform = ctl.platform();
  int ids[3];
  double zero[3], scale[3], zero_crank[3];
  for (int i = 0; i < 3; i++) {
    ids[i] = static_cast<int>(platform["moteus_ids"][i].num());
    zero[i] = platform["level_output_turns"][i].num();
    scale[i] = platform["crank_turns_per_output_turn"][i].num();
    // Crank angles (from the CAD pose) where the encoders read level_output_turns, e.g. the rest pose.
    zero_crank[i] = platform.has("zero_crank_rad") ? platform["zero_crank_rad"][i].num() : 0;
  }
  // Output turns <-> crank radians, as in plate_pose.py.
  auto crank = [&](int i, double turns) { return zero_crank[i]+2*kPi*scale[i]*(turns-zero[i]); };
  auto turns = [&](int i, double q) { return zero[i]+(q-zero_crank[i])/(2*kPi*scale[i]); };

  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  mlockall(MCL_CURRENT|MCL_FUTURE);
  sched_param sp{};
  sp.sched_priority = 80;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
    std::cerr << "Running without real-time priority (needs CAP_SYS_NICE); timing may jitter.\n";

  std::vector<std::shared_ptr<mm::Controller>> motors;
  for (int i = 0; i < 3; i++) {
    mm::Controller::Options o;
    o.id = ids[i];
    o.position_format.feedforward_torque = mm::kFloat;
    o.position_format.maximum_torque = mm::kFloat;
    o.position_format.watchdog_timeout = mm::kFloat;
    motors.push_back(std::make_shared<mm::Controller>(o));
  }
  auto transport = mm::Controller::MakeSingletonTransport({});
  std::vector<mm::CanFdFrame> frames, replies;

  Q3 q{0, 0, 0}, qd{0, 0, 0}, tau{0, 0, 0};
  int missing_run = 0;
  // One CAN cycle: send frames, read the three replies. Returns false on a fault.
  auto cycle = [&](std::string* why) {
    replies.clear();
    transport->BlockingCycle(frames.data(), frames.size(), &replies);
    bool seen[3] = {false, false, false};
    for (const auto& f : replies) {
      const auto r = mm::Query::Parse(f.data, f.size);
      for (int i = 0; i < 3; i++)
        if (f.source == ids[i]) {
          if (r.mode == mm::Mode::kFault || r.fault != 0) {
            *why = "moteus " + std::to_string(ids[i]) + " fault " + std::to_string(r.fault);
            return false;
          }
          q[i] = crank(i, r.position);
          qd[i] = 2*kPi*scale[i]*r.velocity;
          tau[i] = r.torque/scale[i];  // Output torque to crank torque.
          seen[i] = true;
        }
    }
    if (seen[0] && seen[1] && seen[2]) missing_run = 0;
    else if (++missing_run >= 3) { *why = "missing replies"; return false; }
    return true;
  };
  auto stop_all = [&] {
    frames.clear();
    for (auto& m : motors) frames.push_back(m->MakeStop());
    replies.clear();
    transport->BlockingCycle(frames.data(), frames.size(), &replies);
  };

  // Clear faults, then read where the cranks are.
  stop_all();
  frames.clear();
  for (auto& m : motors) frames.push_back(m->MakeQuery());
  std::string why;
  if (!cycle(&why) || missing_run) {
    std::cerr << "Cannot read all three controllers: " << (why.empty() ? "missing replies" : why) << "\n";
    stop_all();
    return 1;
  }
  for (int i = 0; i < 3; i++)
    if (q[i] < cfg.crank_min || q[i] > cfg.crank_max) {
      std::fprintf(stderr, "Crank %d is at %.3f rad, outside the configured range [%.3f, %.3f]. Check "
                   "level_output_turns and the crank limits.\n", i+1, q[i], cfg.crank_min, cfg.crank_max);
      stop_all();
      return 1;
    }
  Q3 idle;
  if (!ctl.kinematics().inverse(Posture{cfg.idle_yaw, 0, 0}, &idle)) return 1;
  const Q3 start = q;

  FILE* normals = args.normals.empty() ? nullptr : std::fopen(args.normals.c_str(), "a");
  FILE* log = args.log.empty() ? nullptr : std::fopen(args.log.c_str(), "w");
  if (log)
    std::fprintf(log, "t,phase,q1,q2,q3,q1_cmd,q2_cmd,q3_cmd,tau1,tau2,tau3,ball_x,ball_y,ball_z,"
                      "valid,contact,yaw,nx,ny,cycle_us\n");
  std::vector<char> log_buffer;

  const bool calibrating = !args.gravity_cal.empty();
  bool cal_started = false, cal_finished = false;
  TrackerReader tracker;
  if (!calibrating) tracker.start();
  std::unique_ptr<NotePlayer> player;
  if (args.sound) player = std::make_unique<NotePlayer>();
  bool circle = false;
  Terminal term;
  std::fprintf(stderr, "Ramping to the idle posture over %.1f s. Keys: h hold, b balance, j bounce, wasd "
               "target, c centre, +/- apex, o circle, 1-%zu songs, 0 stop the song, k juggle, q quit.\n", args.ramp, songs().size());

  timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  const double t_start = realtime();
  double tx = 0, ty = 0, apex = cfg.apex_default;
  long ticks = 0;
  int status = 0;
  while (!g_stop) {
    next.tv_nsec += 1000000;
    if (next.tv_nsec >= 1000000000) { next.tv_nsec -= 1000000000; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
    const double now = realtime();
    const double tick_start = now;

    Output out;
    const double ramp_s = (now-t_start)/args.ramp;
    if (ramp_s < 1) {
      // Smooth move from wherever the cranks were to the idle posture.
      const double s = ramp_s*ramp_s*(3-2*ramp_s), ds = 6*ramp_s*(1-ramp_s)/args.ramp;
      for (int i = 0; i < 3; i++) {
        out.q[i] = start[i]+(idle[i]-start[i])*s;
        out.qd[i] = (idle[i]-start[i])*ds;
      }
      // Fade the gravity holding torque in over the first third of the ramp, so the plate is
      // carried by feedforward, not position error, by the time the controller takes over.
      Q3 hold;
      if (ctl.holding_torque(out.q, &hold))
        for (int i = 0; i < 3; i++) out.tau[i] = hold[i]*std::min(1.0, 3*ramp_s);
      out.ok = true;
      ctl.step(now, q, qd);  // Keep the plate history and estimator current.
    } else if (calibrating && !cal_finished) {
      if (!cal_started) {
        ctl.start_calibration(now, q);
        cal_started = true;
        std::fprintf(stderr, "Gravity calibration: %d holds. Keep the plate clear and do not touch it.\n",
                     ctl.calibration().total());
      }
      out = ctl.calibration_step(now, q, qd, tau);
      if (!ctl.calibrating()) {
        cal_finished = true;
        GravityModel g;
        std::string report;
        if (!ctl.finish_calibration(&g, &report)) {
          why = "gravity calibration failed: " + report;
          g_stop = true;
        } else {
          std::fprintf(stderr, "\n%s", report.c_str());
          FILE* f = std::fopen(args.gravity_cal.c_str(), "w");
          if (!f) {
            why = "cannot write " + args.gravity_cal;
            g_stop = true;
          } else {
            std::fprintf(f, "{\n  \"parameters\": [");
            for (int j = 0; j < GravityModel::kParams; j++) std::fprintf(f, "%s%.9g", j ? ", " : "", g.theta[j]);
            std::fprintf(f, "],\n  \"names\": [");
            for (int j = 0; j < GravityModel::kParams; j++) std::fprintf(f, "%s\"%s\"", j ? ", " : "", GravityModel::name(j));
            std::fprintf(f, "],\n  \"samples\": [\n");
            const auto& samples = ctl.calibration().samples();
            for (size_t k = 0; k < samples.size(); k++) {
              const auto& smp = samples[k];
              std::fprintf(f, "    {\"target\": %d, \"direction\": %d, \"q\": [%.7f, %.7f, %.7f], \"tau\": [%.6f, %.6f, %.6f]}%s\n",
                           smp.target, smp.direction, smp.q[0], smp.q[1], smp.q[2], smp.tau[0], smp.tau[1],
                           smp.tau[2], k+1 < samples.size() ? "," : "");
            }
            std::fprintf(f, "  ]\n}\n");
            std::fclose(f);
            std::fprintf(stderr, "Wrote %s. Add \"gravity_file\": \"%s\" to %s to use it. Levelling, then press q.\n",
                         args.gravity_cal.c_str(), args.gravity_cal.c_str(), args.config.c_str());
          }
        }
      }
    } else {
      for (const auto& m : tracker.take()) ctl.measurement(m.t, m.left);
      for (int k = term.key(); k >= 0; k = term.key()) {
        switch (k) {
          case 'h': ctl.set_mode(kHold); break;
          case 'b': ctl.set_mode(kBalance); break;
          case 'k': ctl.set_mode(kJuggle); break;
          case 'j': ctl.set_mode(kBounce); break;
          case 'w': ty += 0.005; break;
          case 's': ty -= 0.005; break;
          case 'a': tx -= 0.005; break;
          case 'd': tx += 0.005; break;
          case 'c': tx = ty = 0; break;
          case '+': case '=': apex += 0.01; break;
          case '-': apex = std::max(0.02, apex-0.01); break;
          case 'q': g_stop = true; break;
          case 'o':
            circle = !circle;
            ctl.set_circle(circle ? 0.06 : 0, 4.0);
            break;
          case '0': ctl.set_song(-1, 0); break;
          default:
            if (k >= '1' && k <= '9' && k-'1' < static_cast<int>(songs().size())) {
              ctl.set_song(k-'1', 0);
              ctl.set_mode(kBounce);
              std::fprintf(stderr, "\nsong %s\n", songs()[k-'1'].name.c_str());
            }
        }
        ctl.set_target(tx, ty);
        ctl.set_apex(apex);
      }
      out = ctl.step(now, q, qd);
      if (player && out.note > 0) player->play(out.note);
    }
    for (int i = 0; i < 3; i++)
      if (std::fabs(out.q[i]-q[i]) > 0.6) {
        why = "crank " + std::to_string(i+1) + " is far from its command";
        g_stop = true;
      }
    if (g_stop) break;

    frames.clear();
    for (int i = 0; i < 3; i++) {
      mm::PositionMode::Command c;
      c.position = turns(i, out.q[i]);
      c.velocity = out.qd[i]/(2*kPi*scale[i]);
      c.feedforward_torque = out.tau[i]*scale[i];
      c.maximum_torque = args.max_torque;
      c.watchdog_timeout = args.watchdog;
      frames.push_back(motors[i]->MakePosition(c));
    }
    if (!cycle(&why)) { g_stop = true; break; }

    if (normals && ticks%5 == 0) {
      const V3 n = cfg.left_from_base*out.plate_normal;
      std::fprintf(normals, "%lld %.9f %.9f %.9f\n", static_cast<long long>(now*1e6), n[0], n[1], n[2]);
      std::fflush(normals);
    }
    if (log) {
      char row[512];
      const int len = std::snprintf(row, sizeof(row),
          "%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%d,%d,%.6f,%.6f,%.6f,%.0f\n",
          now, out.phase, q[0], q[1], q[2], out.q[0], out.q[1], out.q[2], out.tau[0], out.tau[1], out.tau[2],
          out.ball[0], out.ball[1], out.ball[2], out.ball_valid, out.contact, out.yaw, out.nx, out.ny,
          (realtime()-tick_start)*1e6);
      log_buffer.insert(log_buffer.end(), row, row+len);
    }
    if (calibrating && !cal_finished && ticks%1000 == 0)
      std::fprintf(stderr, "\rcalibration hold %d of %d   ", ctl.calibration().done(), ctl.calibration().total());
    else if (ticks%1000 == 0)
      std::fprintf(stderr, "\rphase %d ball %s contact %d target %+5.0f %+5.0f mm apex %3.0f mm e %.2f   ",
                   out.phase, out.ball_valid ? "seen" : "lost", out.contact, tx*1e3, ty*1e3, apex*1e3,
                   out.restitution);
    ticks++;
  }
  stop_all();
  if (!why.empty()) {
    std::fprintf(stderr, "\nStopped: %s\n", why.c_str());
    status = 1;
  }
  if (log) {
    std::fwrite(log_buffer.data(), 1, log_buffer.size(), log);
    std::fclose(log);
  }
  if (normals) std::fclose(normals);
  std::fprintf(stderr, "\nMotors stopped.\n");
  std::_Exit(status);  // The tracker thread may be blocked reading stdin.
}
