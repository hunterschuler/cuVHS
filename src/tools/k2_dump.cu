#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "format/video_format.h"
#include "gpu/device.h"
#include "io/raw_reader.h"
#include "pipeline/fm_demod.h"
#include "pipeline/sync_pulses.h"

struct Args {
    std::string input_path;
    std::string output_dir;
    VideoProfile profile = VideoProfile::NTSC_525_60_VHS;
    TapeSpeed tape_speed = TapeSpeed::SP;
    double sample_rate_mhz = 28.0;
    InputFormat input_format = InputFormat::U8;
    InputConditioning conditioning;
    bool format_explicit = false;
    int gpu_id = 0;
    size_t sample = 0;
    size_t samples = 0;
};

static InputFormat detect_format(const std::string& path) {
    auto pos = path.find_last_of('.');
    std::string ext = (pos == std::string::npos) ? "" : path.substr(pos + 1);
    if (ext == "u8" || ext == "raw") return InputFormat::U8;
    if (ext == "s16") return InputFormat::S16;
    if (ext == "u16") return InputFormat::U16;
    return InputFormat::U8;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [options] --sample N --samples N <input_file> <output_dir>\n"
            "Options:\n"
            "  --system <NTSC|PAL|PAL-M|MPAL>\n"
            "  --tape-speed <sp|lp|ep>\n"
            "  -f <MHz>\n"
            "  --format <u8|s16|u16>\n"
            "  --dc-correct\n"
            "  --gpu <id>\n",
            prog);
}

static bool parse_args(int argc, char** argv, Args& args) {
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            ++i;
            if (strcasecmp(argv[i], "NTSC") == 0 || strcasecmp(argv[i], "NTSC_525_60_VHS") == 0) {
                args.profile = VideoProfile::NTSC_525_60_VHS;
            } else if (strcasecmp(argv[i], "PAL") == 0 || strcasecmp(argv[i], "PAL_625_50_VHS") == 0) {
                args.profile = VideoProfile::PAL_625_50_VHS;
            } else if (strcasecmp(argv[i], "PAL-M") == 0 || strcasecmp(argv[i], "PALM") == 0 ||
                       strcasecmp(argv[i], "MPAL") == 0 || strcasecmp(argv[i], "MPAL_525_60_VHS") == 0) {
                args.profile = VideoProfile::MPAL_525_60_VHS;
            } else {
                fprintf(stderr, "Unknown system: %s\n", argv[i]);
                return false;
            }
        } else if (strcmp(argv[i], "--tape-speed") == 0 && i + 1 < argc) {
            ++i;
            if (strcasecmp(argv[i], "sp") == 0) args.tape_speed = TapeSpeed::SP;
            else if (strcasecmp(argv[i], "lp") == 0) args.tape_speed = TapeSpeed::LP;
            else if (strcasecmp(argv[i], "ep") == 0 || strcasecmp(argv[i], "slp") == 0) args.tape_speed = TapeSpeed::EP;
            else {
                fprintf(stderr, "Unknown tape speed: %s\n", argv[i]);
                return false;
            }
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            args.sample_rate_mhz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            ++i;
            args.format_explicit = true;
            if (strcmp(argv[i], "u8") == 0) args.input_format = InputFormat::U8;
            else if (strcmp(argv[i], "s16") == 0) args.input_format = InputFormat::S16;
            else if (strcmp(argv[i], "u16") == 0) args.input_format = InputFormat::U16;
            else {
                fprintf(stderr, "Unknown format: %s\n", argv[i]);
                return false;
            }
        } else if (strcmp(argv[i], "--dc-correct") == 0) {
            args.conditioning.dc_correct = true;
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            args.gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            args.sample = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            args.samples = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (argv[i][0] == '-' && std::strlen(argv[i]) > 1) {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        } else {
            if (positional == 0) args.input_path = argv[i];
            else if (positional == 1) args.output_dir = argv[i];
            else {
                fprintf(stderr, "Too many positional arguments\n");
                return false;
            }
            ++positional;
        }
    }

    if (positional != 2 || args.samples == 0) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

static bool write_f64(const std::string& path, const std::vector<double>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
    return out.good();
}

static bool write_i32(const std::string& path, const std::vector<int>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int)));
    return out.good();
}

static bool write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    if (!out) return false;
    out << text;
    return out.good();
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    if (!args.format_explicit) args.input_format = detect_format(args.input_path);

    GPUDevice gpu;
    if (!gpu.init(args.gpu_id)) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }

    VideoFormat fmt(args.profile, args.sample_rate_mhz, args.tape_speed);
    RawReader reader;
    reader.set_conditioning(args.conditioning);
    if (!reader.open(args.input_path, args.input_format)) {
        fprintf(stderr, "Failed to open input: %s\n", args.input_path.c_str());
        return 1;
    }

    std::string mkdir_cmd = "mkdir -p \"" + args.output_dir + "\"";
    if (std::system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "Failed to create output directory: %s\n", args.output_dir.c_str());
        return 1;
    }

    const size_t overlap = (size_t)FM_DEMOD_OVERLAP_SAMPLES;
    size_t lead = std::min(args.sample, overlap);
    size_t tail = overlap;
    if (args.sample + args.samples + tail > reader.total_samples()) {
        size_t end = reader.total_samples();
        tail = (args.sample + args.samples < end) ? (end - (args.sample + args.samples)) : 0;
    }

    std::vector<double> h_lead(lead);
    std::vector<double> h_raw(args.samples);
    std::vector<double> h_tail(tail);
    if (lead > 0) {
        size_t got_lead = reader.read_at(h_lead.data(), args.sample - lead, lead);
        if (got_lead != lead) {
            fprintf(stderr, "Failed to read lead context\n");
            return 1;
        }
    }
    size_t got = reader.read_at(h_raw.data(), args.sample, args.samples);
    if (got != args.samples) {
        fprintf(stderr, "Failed to read raw block at %zu (%zu samples)\n", args.sample, args.samples);
        return 1;
    }
    if (tail > 0) {
        size_t got_tail = reader.read_at(h_tail.data(), args.sample + args.samples, tail);
        if (got_tail != tail) {
            fprintf(stderr, "Failed to read tail context\n");
            return 1;
        }
    }

    FMDemodState state;
    if (!state.init(fmt, 1, static_cast<int>(args.samples))) {
        fprintf(stderr, "Failed to init FM demod state\n");
        return 1;
    }

    double* d_raw = nullptr;
    double* d_lead = nullptr;
    double* d_demod = nullptr;
    double* d_demod_05 = nullptr;
    double* d_envelope = nullptr;
    double* d_tail = nullptr;
    int* d_pulse_starts = nullptr;
    int* d_pulse_lengths = nullptr;
    int* d_pulse_count = nullptr;

    if (lead > 0) cudaMalloc(&d_lead, lead * sizeof(double));
    cudaMalloc(&d_raw, args.samples * sizeof(double));
    cudaMalloc(&d_demod, args.samples * sizeof(double));
    cudaMalloc(&d_demod_05, args.samples * sizeof(double));
    cudaMalloc(&d_envelope, args.samples * sizeof(double));
    if (tail > 0) cudaMalloc(&d_tail, tail * sizeof(double));
    cudaMalloc(&d_pulse_starts, MAX_PULSES * sizeof(int));
    cudaMalloc(&d_pulse_lengths, MAX_PULSES * sizeof(int));
    cudaMalloc(&d_pulse_count, sizeof(int));

    if (lead > 0) cudaMemcpy(d_lead, h_lead.data(), lead * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_raw, h_raw.data(), args.samples * sizeof(double), cudaMemcpyHostToDevice);
    if (tail > 0) cudaMemcpy(d_tail, h_tail.data(), tail * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_pulse_count, 0, sizeof(int));

    fm_demod(state,
             lead > 0 ? d_lead : nullptr,
             d_raw,
             tail > 0 ? d_tail : nullptr,
             d_demod,
             d_demod_05,
             nullptr,
             d_envelope,
             nullptr,
             nullptr,
             nullptr,
             nullptr,
             nullptr,
             1,
             args.samples,
             fmt);
    cudaDeviceSynchronize();

    sync_pulses(d_demod_05,
                d_pulse_starts,
                d_pulse_lengths,
                d_pulse_count,
                nullptr,
                1,
                args.samples,
                fmt);
    cudaDeviceSynchronize();

    int h_count = 0;
    cudaMemcpy(&h_count, d_pulse_count, sizeof(int), cudaMemcpyDeviceToHost);
    h_count = std::max(0, std::min(h_count, MAX_PULSES));

    std::vector<double> h_demod_05(args.samples);
    std::vector<int> h_starts(MAX_PULSES);
    std::vector<int> h_lengths(MAX_PULSES);
    cudaMemcpy(h_demod_05.data(), d_demod_05, args.samples * sizeof(double), cudaMemcpyDeviceToHost);
    if (h_count > 0) {
        cudaMemcpy(h_starts.data(), d_pulse_starts, (size_t)h_count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lengths.data(), d_pulse_lengths, (size_t)h_count * sizeof(int), cudaMemcpyDeviceToHost);
    }
    h_starts.resize((size_t)h_count);
    h_lengths.resize((size_t)h_count);

    write_f64(args.output_dir + "/gpu_demod_05.f64", h_demod_05);
    write_i32(args.output_dir + "/gpu_rawpulse_starts.i32", h_starts);
    write_i32(args.output_dir + "/gpu_rawpulse_lengths.i32", h_lengths);

    char meta[512];
    std::snprintf(meta, sizeof(meta),
                  "{\n"
                  "  \"sample\": %zu,\n"
                  "  \"samples\": %zu,\n"
                  "  \"lead\": %zu,\n"
                  "  \"tail\": %zu,\n"
                  "  \"rawpulse_count\": %d,\n"
                  "  \"pulse_threshold_hz\": %.17g,\n"
                  "  \"sample_rate\": %.17g\n"
                  "}\n",
                  args.sample,
                  args.samples,
                  lead,
                  tail,
                  h_count,
                  fmt.pulse_threshold_hz,
                  fmt.sample_rate);
    write_text(args.output_dir + "/config.json", std::string(meta));

    if (d_tail) cudaFree(d_tail);
    cudaFree(d_pulse_count);
    cudaFree(d_pulse_lengths);
    cudaFree(d_pulse_starts);
    if (d_lead) cudaFree(d_lead);
    cudaFree(d_envelope);
    cudaFree(d_demod_05);
    cudaFree(d_demod);
    cudaFree(d_raw);
    state.destroy();
    return 0;
}
