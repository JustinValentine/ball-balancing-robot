#include "vision.hpp"
#include <algorithm>
#include <charconv>
#include <iomanip>
#include <iostream>

namespace {
void check(dc1394error_t e, const char* operation) {
    if (e != DC1394_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + dc1394_error_get_string(e));
}
uint64_t parseGuid(const std::string& text) {
    uint64_t value{};
    auto end = text.data() + text.size();
    auto result = std::from_chars(text.data(), end, value, 16);
    require(result.ec == std::errc{} && result.ptr == end && text.size() == 16,
            "Copy a 16-digit hex camera GUID from ball-vision list");
    return value;
}
constexpr int bufferCount = 8;
}

Context context() {
    Context ctx(dc1394_new(), dc1394_free);
    require(bool(ctx), "Cannot initialize libdc1394");
    return ctx;
}

void list(dc1394_t* ctx) {
    dc1394camera_list_t* raw{};
    check(dc1394_camera_enumerate(ctx, &raw), "enumerate");
    std::unique_ptr<dc1394camera_list_t, decltype(&dc1394_camera_free_list)> cameras(raw, dc1394_camera_free_list);
    if (!raw->num) std::cout << "No FireWire cameras found\n";
    for (unsigned i = 0; i < raw->num; ++i) {
        auto id = raw->ids[i];
        std::unique_ptr<dc1394camera_t, decltype(&dc1394_camera_free)> camera(
            dc1394_camera_new_unit(ctx, id.guid, id.unit), dc1394_camera_free);
        require(bool(camera), "Cannot open enumerated camera");
        std::cout << std::hex << std::setw(16) << std::setfill('0') << id.guid << std::dec << " " << camera->model << "\n";
        dc1394feature_info_t trigger{};
        trigger.id = DC1394_FEATURE_TRIGGER;
        if (dc1394_feature_get(camera.get(), &trigger) == DC1394_SUCCESS && trigger.available) {
            constexpr int iidcModes[] = {0, 1, 2, 3, 4, 5, 14, 15};
            std::cout << "  Trigger modes:";
            for (unsigned j = 0; j < trigger.trigger_modes.num; ++j)
                std::cout << " " << iidcModes[trigger.trigger_modes.modes[j] - DC1394_TRIGGER_MODE_MIN];
            std::cout << "  sources:";
            for (unsigned j = 0; j < trigger.trigger_sources.num; ++j) {
                auto source = trigger.trigger_sources.sources[j];
                if (source == DC1394_TRIGGER_SOURCE_SOFTWARE) std::cout << " software";
                else std::cout << " " << source - DC1394_TRIGGER_SOURCE_0;
            }
            std::cout << "  polarity configurable=" << trigger.polarity_capable << "\n";
        }
        dc1394video_modes_t modes{};
        check(dc1394_video_get_supported_modes(camera.get(), &modes), "query modes");
        for (unsigned j = 0; j < modes.num; ++j) {
            auto mode = modes.modes[j];
            if (dc1394_is_video_mode_scalable(mode)) {
                dc1394format7mode_t info{};
                check(dc1394_format7_get_mode_info(camera.get(), mode, &info), "query Format7");
                bool mono = std::find(info.color_codings.codings, info.color_codings.codings + info.color_codings.num,
                                      DC1394_COLOR_CODING_MONO8) != info.color_codings.codings + info.color_codings.num;
                std::cout << "  Format7 mode " << mode - DC1394_VIDEO_MODE_FORMAT7_0 << ": "
                          << info.max_size_x << "x" << info.max_size_y << ", MONO8=" << mono
                          << ", max packet " << info.max_packet_size << " bytes\n";
            } else if (mode == DC1394_VIDEO_MODE_640x480_MONO8) {
                dc1394framerates_t rates{};
                check(dc1394_video_get_supported_framerates(camera.get(), mode, &rates), "query rates");
                std::cout << "  Fixed VGA MONO8 IIDC rate indices:";
                for (unsigned k = 0; k < rates.num; ++k) std::cout << " " << rates.framerates[k] - DC1394_FRAMERATE_MIN;
                std::cout << "\n";
            }
        }
    }
}

Camera::Camera(dc1394_t* ctx, const std::string& guid)
    : cam(dc1394_camera_new(ctx, parseGuid(guid)), dc1394_camera_free) {
    require(bool(cam), "Cannot open camera " + guid);
}

void Camera::configure(const cv::FileStorage& f) {
    auto* c = cam.get();
    check(dc1394_video_set_operation_mode(c, DC1394_OPERATION_MODE_1394B), "enable 1394b");
    check(dc1394_video_set_iso_speed(c, DC1394_ISO_SPEED_800), "select S800");
    int format = setting<int>(f,"format7_mode");
    require(format >= -1 && format < DC1394_VIDEO_MODE_FORMAT7_NUM, "Invalid format7_mode");
    if (format >= 0) {
        auto mode = static_cast<dc1394video_mode_t>(DC1394_VIDEO_MODE_FORMAT7_0 + format);
        int packet = setting<int>(f,"packet_bytes");
        require(packet >= 0, "Invalid packet size");
        check(dc1394_video_set_mode(c, mode), "set Format7 mode (see list)");
        check(dc1394_format7_set_roi(c, mode, DC1394_COLOR_CODING_MONO8, packet ? packet : DC1394_USE_MAX_AVAIL,
                                     0, 0, 640, 480), "set VGA MONO8 ROI and packet size");
    } else {
        int rate = setting<int>(f,"fixed_rate_index");
        require(rate >= 0 && rate < DC1394_FRAMERATE_NUM, "Invalid IIDC rate index");
        check(dc1394_video_set_mode(c, DC1394_VIDEO_MODE_640x480_MONO8), "set fixed VGA MONO8");
        check(dc1394_video_set_framerate(c, static_cast<dc1394framerate_t>(DC1394_FRAMERATE_MIN + rate)), "set fixed rate");
    }
    int mode = setting<int>(f,"trigger_mode"), source = setting<int>(f,"trigger_source");
    require(mode == 0 || mode == 14, "Select edge trigger mode 0 or overlapped mode 14");
    require(source >= 0 && source <= 3, "Invalid trigger source");
    check(dc1394_external_trigger_set_mode(c, mode == 0 ? DC1394_TRIGGER_MODE_0 : DC1394_TRIGGER_MODE_14), "set trigger mode");
    check(dc1394_external_trigger_set_source(c, static_cast<dc1394trigger_source_t>(DC1394_TRIGGER_SOURCE_0 + source)), "set trigger source");
    int high = setting<int>(f,"trigger_active_high");
    require(high >= -1 && high <= 1, "Invalid trigger polarity");
    if (high >= 0) check(dc1394_external_trigger_set_polarity(c, high ? DC1394_TRIGGER_ACTIVE_HIGH : DC1394_TRIGGER_ACTIVE_LOW), "set polarity");
    double exposure = setting<double>(f,"exposure_us");
    require(std::isfinite(exposure) && exposure >= 0, "Invalid exposure");
    if (exposure > 0) {
        check(dc1394_feature_set_mode(c, DC1394_FEATURE_SHUTTER, DC1394_FEATURE_MODE_MANUAL), "manual shutter");
        check(dc1394_feature_set_absolute_control(c, DC1394_FEATURE_SHUTTER, DC1394_ON), "absolute shutter");
        check(dc1394_feature_set_absolute_value(c, DC1394_FEATURE_SHUTTER, exposure * 1e-6), "set exposure seconds");
    }
    int gain = setting<int>(f,"gain_raw");
    require(gain >= -1, "Invalid gain");
    if (gain >= 0) {
        check(dc1394_feature_set_mode(c, DC1394_FEATURE_GAIN, DC1394_FEATURE_MODE_MANUAL), "manual gain");
        dc1394bool_t absolute{};
        check(dc1394_feature_has_absolute_control(c, DC1394_FEATURE_GAIN, &absolute), "query gain units");
        if (absolute) check(dc1394_feature_set_absolute_control(c, DC1394_FEATURE_GAIN, DC1394_OFF), "select raw gain");
        check(dc1394_feature_set_value(c, DC1394_FEATURE_GAIN, gain), "set gain");
    }
    check(dc1394_external_trigger_set_power(c, DC1394_ON), "enable external trigger");
    check(dc1394_capture_setup(c, bufferCount, DC1394_CAPTURE_FLAGS_DEFAULT), "allocate capture DMA / bus bandwidth");
    capturing = true;
}
void Camera::start() {
    check(dc1394_video_set_transmission(cam.get(), DC1394_ON), "start transmission");
    streaming = true;
}
Camera::~Camera() {
    if (streaming) dc1394_video_set_transmission(cam.get(), DC1394_OFF);
    if (capturing) dc1394_capture_stop(cam.get());
}
int Camera::fd() const { return dc1394_capture_get_fileno(cam.get()); }

bool Camera::newest(Frame& out) {
    for (int n = 0; n < bufferCount; ++n) {
        dc1394video_frame_t* frame{};
        check(dc1394_capture_dequeue(cam.get(), DC1394_CAPTURE_POLICY_POLL, &frame), "dequeue");
        if (!frame) break;
        ++received;
        bool good = !frame->frames_behind && frame->timestamp && !dc1394_capture_is_frame_corrupt(cam.get(), frame)
            && frame->size[0] == 640 && frame->size[1] == 480 && frame->color_coding == DC1394_COLOR_CODING_MONO8
            && frame->stride >= 640 && frame->total_bytes >= 479ULL * frame->stride + 640;
        if (good) {
            cv::Mat(480, 640, CV_8U, frame->image, frame->stride).copyTo(out.pixels);
            out.time_us = static_cast<int64_t>(frame->timestamp);
        } else ++discarded;
        check(dc1394_capture_enqueue(cam.get(), frame), "return DMA buffer");
        if (good) return true;
    }
    return false;
}
