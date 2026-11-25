#include "vision.hpp"
#include <opencv2/imgcodecs.hpp>
#include <poll.h>
#include <array>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
namespace {
volatile std::sig_atomic_t stopping = 0;
void stop(int) { stopping = 1; }
using Clock = std::chrono::steady_clock;
constexpr char header[] = "left_us,right_us,left,right";

class Recorder {
    fs::path dir;
    std::ofstream manifest;
    int64_t interval, last = 0;
public:
    Recorder(const fs::path& path,const Rig& rig,double seconds):dir(path) {
        require(std::isfinite(seconds) && seconds>=0 && seconds<=86400, "Invalid recording interval");
        interval = std::llround(seconds*1e6);
        require(fs::create_directory(dir), "Choose a new recording directory");
        // Rig GUIDs have been parsed before construction.
        std::ofstream meta(dir/"rig.json");
        meta << "{\"left_guid\":\"" << rig.left << "\",\"right_guid\":\"" << rig.right << "\"}\n";
        meta.close();
        require(bool(meta),"Cannot save rig identity");
        manifest.exceptions(std::ios::failbit | std::ios::badbit);
        manifest.open(dir/"pairs.csv");
        manifest << header << std::endl;
    }
    void save(const Frame& l,const Frame& r) {
        if (last && l.time_us-last < interval) return;
        std::string id=std::to_string(l.time_us);
        std::string a=id+"-L.pgm", b=id+"-R.pgm";
        require(cv::imwrite((dir/a).string(),l.pixels) && cv::imwrite((dir/b).string(),r.pixels),"Cannot save stereo pair");
        manifest << l.time_us << ',' << r.time_us << ',' << a << ',' << b << std::endl;
        last=l.time_us;
    }
};

cv::Vec3d unitNormal(cv::Vec3d n) {
    double length=cv::norm(n);
    require(length>1e-9 && std::isfinite(length),"Invalid plate normal");
    return n/length;
}

// A robot can append one complete ASCII row per pose to a regular file.
// Save the read position on EOF so a partially appended row is retried.
class PlateNormals {
    struct Sample { int64_t time; cv::Vec3d normal; };
    std::ifstream input;
    cv::Vec3d fixed{0,0,1};
    std::optional<Sample> before, after;
    int maxAge=0, delay=0;
    std::optional<Sample> read() {
        auto pos=input.tellg();
        std::string row;
        std::getline(input,row);
        require(!input.bad(),"Error reading plate normal stream");
        if (input.eof()) {
            input.clear(); input.seekg(pos);
            return {};
        }
        std::istringstream fields(row);
        Sample s; std::string extra;
        require((fields>>s.time>>s.normal[0]>>s.normal[1]>>s.normal[2]) && !(fields>>extra)
                && s.time>0 && (!before || s.time>before->time),"Malformed or unordered plate normal row");
        s.normal=unitNormal(s.normal);
        return s;
    }
public:
    PlateNormals(const char* options,const char* path) {
        cv::FileStorage f(options,cv::FileStorage::READ);
        if (f["acrylic_thickness_mm"].empty() || setting<double>(f,"acrylic_thickness_mm")==0) {
            require(!path,"Plate normals supplied with acrylic correction disabled");
            return;
        }
        if (path) {
            require(fs::is_regular_file(path),"Plate normal input must be a regular, appendable file");
            input.open(path);
            require(input.is_open(),"Cannot open plate normals");
            maxAge=setting<int>(f,"normal_max_age_us");
            delay=setting<int>(f,"capture_minus_exposure_us");
            require(maxAge>=0 && delay>=0,"Invalid pose timing configuration");
        } else {
            fixed=unitNormal(setting<cv::Vec3d>(f,"fixed_plate_normal"));
            std::cerr << "Applying acrylic correction with FIXED plate normal " << fixed
                      << ". Supply a timestamped normals file for a moving platform.\n";
        }
    }
    std::optional<cv::Vec3d> at(int64_t captureTime) {
        if (!input.is_open()) return fixed;
        int64_t t=captureTime-delay;
        for (;;) {
            if (!after) after=read();
            if (!after || after->time>t) break;
            before=after; after.reset();
        }
        if (!before || t-before->time>maxAge) return {};
        if (after && after->time-t<=maxAge) {
            double weight=double(t-before->time)/(after->time-before->time);
            return unitNormal((1-weight)*before->normal+weight*after->normal);
        }
        return before->normal;
    }
};

class Output {
    Tracker tracker;
    PlateNormals normals;
    int64_t last = 0;
    cv::Vec3d prev;
public:
    Output(const char* calib,const char* options,const Rig& rig,const char* path):tracker(calib,options,rig),normals(options,path) {
        std::cout << "left_us,right_us,valid,x_mm,y_mm,z_mm,vx_mm_s,vy_mm_s,vz_mm_s,reprojection_px,processing_us\n";
    }
    void publish(const Frame& l,const Frame& r) {
        auto start=Clock::now();
        auto normal=normals.at(l.time_us);
        if (tracker.balls() > 1) {
            // One row per ball found; velocity is left to the consumer, which tells the balls apart.
            auto all=normal ? tracker.updateAll(l.pixels,r.pixels,*normal) : std::vector<Measurement>{};
            auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
            if (all.empty()) std::cout << l.time_us << ',' << r.time_us << ",0,,,,,,,," << elapsed << '\n';
            for (const auto& m : all) {
                std::cout << std::setprecision(10) << l.time_us << ',' << r.time_us << ",1,";
                for (double v:m.position.val) std::cout << v << ',';
                std::cout << ",,," << m.error << ',' << elapsed << '\n';
            }
            std::cout << std::flush;
            require(bool(std::cout),"Tracking output closed");
            return;
        }
        auto m=normal ? tracker.update(l.pixels,r.pixels,*normal) : std::nullopt;
        auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
        std::cout << std::setprecision(10) << l.time_us << ',' << r.time_us << ',' << int(bool(m)) << ',';
        if (m) {
            for (double v:m->position.val) std::cout << v << ',';
            int64_t dt=l.time_us-last;
            if (last && dt>0 && dt<=50000) {
                auto velocity=(m->position-prev)*(1e6/dt);
                for (double v:velocity.val) std::cout << v << ',';
            } else std::cout << ",,,";
            std::cout << m->error;
            last=l.time_us; prev=m->position;
        } else {
            last=0;
            std::cout << ",,,,,,";
        }
        std::cout << ',' << elapsed << std::endl;
        require(bool(std::cout),"Tracking output closed");
    }
};

int64_t integer(const std::string& s) {
    size_t end;
    auto value=std::stoll(s,&end);
    require(end==s.size() && value>0,"Invalid capture timestamp");
    return value;
}
void replay(const fs::path& dir,const char* calib,const char* settings,const char* normals) {
    cv::FileStorage identity((dir/"rig.json").string(),cv::FileStorage::READ);
    Rig rig{setting<std::string>(identity,"left_guid"),setting<std::string>(identity,"right_guid")};
    Output output(calib,settings,rig,normals);
    std::ifstream manifest(dir/"pairs.csv");
    std::string row;
    require(std::getline(manifest,row) && row==header,"Invalid pairs.csv header");
    int64_t lastL=0, lastR=0;
    while (!stopping && std::getline(manifest,row)) {
        std::istringstream line(row);
        std::array<std::string,4> item;
        for (auto& col:item) require(bool(std::getline(line,col,',')) && !col.empty(),"Invalid pairs.csv row");
        require(line.eof(),"Unexpected manifest column");
        Frame l{cv::imread((dir/item[2]).string(),cv::IMREAD_UNCHANGED),integer(item[0])};
        Frame r{cv::imread((dir/item[3]).string(),cv::IMREAD_UNCHANGED),integer(item[1])};
        require(l.time_us>lastL && r.time_us>lastR,"Non-increasing manifest timestamps");
        lastL=l.time_us; lastR=r.time_us;
        output.publish(l,r);
    }
}

void live(const std::string& cmd,int argc,char** argv) {
    cv::FileStorage f(argv[2],cv::FileStorage::READ);
    require(f.isOpened(),"Cannot read camera settings");
    Rig rig{setting<std::string>(f,"left_guid"),setting<std::string>(f,"right_guid")};
    require(rig.left!=rig.right,"Set two distinct GUIDs in camera.yml");
    int period=setting<int>(f,"trigger_period_us"), tolerance=setting<int>(f,"pair_tolerance_us");
    int offset=setting<int>(f,"right_offset_us");
    require(period>0 && tolerance>0 && tolerance*2LL<period && std::abs(static_cast<int64_t>(offset))*2<period,
            "Pair tolerance and absolute right_offset_us must each be below half a trigger period");
    auto ctx=context();
    Camera left(ctx.get(),rig.left),right(ctx.get(),rig.right);
    left.configure(f); right.configure(f);
    std::optional<Output> output;
    std::optional<Recorder> recorder;
    if (cmd=="track") output.emplace(argv[3],argv[4],rig,argc==6 ? argv[5] : nullptr);
    if (cmd=="record") recorder.emplace(argv[3],rig,argc==5 ? std::stod(argv[4]) : 1.);
    left.start(); right.start();
    std::cerr << "Both cameras armed. Enable common triggers. Ctrl-C to stop.\n";
    Camera* cams[2]={&left,&right};
    pollfd polls[2]={{left.fd(),POLLIN,0},{right.fd(),POLLIN,0}};
    require(polls[0].fd>=0 && polls[1].fd>=0,"No capture file descriptor");
    Frame frame[2];
    bool pending[2]={};
    int64_t last[2]={};
    unsigned long pairs=0, unmatched=0;
    auto report=Clock::now();
    unsigned long previousPairs=0;
    std::optional<int64_t> latestSkew;
    while (!stopping) {
        int n=poll(polls,2,200);
        if (n<0 && errno==EINTR) continue;
        require(n>=0,"Camera poll failed");
        for (int i=0;i<2;++i) {
            require(!(polls[i].revents & (POLLERR|POLLHUP|POLLNVAL)),"Camera disconnected");
            if ((polls[i].revents & POLLIN) && cams[i]->newest(frame[i])) {
                require(frame[i].time_us>last[i],"Capture clock moved backward. Restart capture.");
                last[i]=frame[i].time_us;
                unmatched+=pending[i]; pending[i]=true;
            }
        }
        if (pending[0] && pending[1]) {
            latestSkew=frame[1].time_us-frame[0].time_us;
            int64_t delta=*latestSkew-offset;
            if (std::abs(delta)<=tolerance) {
                ++pairs; pending[0]=pending[1]=false;
                if (output) output->publish(frame[0],frame[1]);
                if (recorder) recorder->save(frame[0],frame[1]);
            } else {
                pending[delta>0 ? 0 : 1]=false;
                ++unmatched;
            }
        }
        double elapsed=std::chrono::duration<double>(Clock::now()-report).count();
        if (elapsed>=1) {
            std::cerr << "pairs/s=" << (pairs-previousPairs)/elapsed << " pairs=" << pairs << " received_L/R="
                      << left.received << "/" << right.received << " corrupt_or_backlog_L/R=" << left.discarded
                      << "/" << right.discarded << " unmatched=" << unmatched << " last_compared_R_minus_L_us=";
            if (latestSkew) std::cerr << *latestSkew;
            else std::cerr << "n/a";
            std::cerr << "\n";
            report=Clock::now(); previousPairs=pairs;
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        cv::setNumThreads(1);
        std::signal(SIGINT,stop); std::signal(SIGTERM,stop);
        std::string cmd=argc>1 ? argv[1] : "";
        if (cmd=="list" && argc==2) list(context().get());
        else if (cmd=="replay" && (argc==5 || argc==6)) replay(argv[2],argv[3],argv[4],argc==6 ? argv[5] : nullptr);
        else if ((cmd=="probe" && argc==3) || (cmd=="record" && (argc==4 || argc==5)) ||
                 (cmd=="track" && (argc==5 || argc==6))) live(cmd,argc,argv);
        else {
            std::cerr << "ball-vision list | probe CAMERA.yml | record CAMERA.yml NEW_DIR [interval_seconds]\n"
                         "            track CAMERA.yml CALIB.yml TRACKER.yml [NORMALS.txt] | replay DIR CALIB.yml TRACKER.yml [NORMALS.txt]\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"; return 1;
    }
}
