#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "format/video_format.h"
#include "gpu/device.h"
#include "io/raw_reader.h"
#include "pipeline/fm_demod.h"

static bool check_cuda(const char* where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        return false;
    }
    return true;
}

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
    size_t blocklen = 0;
    std::string filter_dir;
};

static InputFormat detect_format(const std::string& path) {
    auto pos = path.find_last_of('.');
    std::string ext = (pos == std::string::npos) ? "" : path.substr(pos + 1);
    if (ext == "u8" || ext == "raw") return InputFormat::U8;
    if (ext == "s16") return InputFormat::S16;
    if (ext == "u16") return InputFormat::U16;
    return InputFormat::U8;
}

static const char* profile_name_cli(VideoProfile profile) {
    switch (profile) {
        case VideoProfile::NTSC_525_60_VHS: return "NTSC";
        case VideoProfile::PAL_625_50_VHS: return "PAL";
        case VideoProfile::MPAL_525_60_VHS: return "PAL-M";
    }
    return "NTSC";
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [options] --sample N --blocklen N <input_file> <output_dir>\n"
            "Options:\n"
            "  --system <NTSC|PAL|PAL-M|MPAL>\n"
            "  --tape-speed <sp|lp|ep>\n"
            "  -f <MHz>\n"
            "  --format <u8|s16|u16>\n"
            "  --dc-correct\n"
            "  --filter-dir <dir>\n"
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
        } else if (strcmp(argv[i], "--filter-dir") == 0 && i + 1 < argc) {
            args.filter_dir = argv[++i];
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            args.gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            args.sample = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--blocklen") == 0 && i + 1 < argc) {
            args.blocklen = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
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

    if (positional != 2 || args.blocklen == 0) {
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

static bool write_complex_parts(const std::string& prefix, const std::vector<cufftDoubleComplex>& data) {
    std::vector<double> re(data.size()), im(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        re[i] = data[i].x;
        im[i] = data[i].y;
    }
    return write_f64(prefix + ".re.f64", re) && write_f64(prefix + ".im.f64", im);
}

static bool write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path);
    if (!out) return false;
    out << text;
    return out.good();
}

static bool read_f64(const std::string& path, std::vector<double>& out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    std::streamsize size = in.tellg();
    if (size < 0 || (size % static_cast<std::streamsize>(sizeof(double))) != 0) return false;
    out.resize(static_cast<size_t>(size / static_cast<std::streamsize>(sizeof(double))));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(out.data()), size);
    return in.good();
}

static std::vector<double> interp_real(const std::vector<double>& src, size_t dst_size) {
    std::vector<double> dst(dst_size, 0.0);
    if (src.empty() || dst_size == 0) return dst;
    if (src.size() == 1) {
        std::fill(dst.begin(), dst.end(), src[0]);
        return dst;
    }
    const double src_scale = static_cast<double>(src.size() - 1);
    const double dst_scale = static_cast<double>(dst_size - 1);
    for (size_t i = 0; i < dst_size; ++i) {
        double pos = (dst_scale > 0.0) ? (static_cast<double>(i) * src_scale / dst_scale) : 0.0;
        size_t lo = static_cast<size_t>(pos);
        size_t hi = std::min(lo + 1, src.size() - 1);
        double frac = pos - static_cast<double>(lo);
        dst[i] = src[lo] * (1.0 - frac) + src[hi] * frac;
    }
    return dst;
}

static std::vector<cufftDoubleComplex> interp_complex_parts(
    const std::vector<double>& src_re,
    const std::vector<double>& src_im,
    size_t dst_size,
    double scale)
{
    std::vector<double> dst_re = interp_real(src_re, dst_size);
    std::vector<double> dst_im = interp_real(src_im, dst_size);
    std::vector<cufftDoubleComplex> dst(dst_size);
    for (size_t i = 0; i < dst_size; ++i) {
        dst[i].x = dst_re[i] * scale;
        dst[i].y = dst_im[i] * scale;
    }
    return dst;
}

static bool override_filters_from_dir(const Args& args, FMDemodState& state) {
    if (args.filter_dir.empty()) return true;
    std::vector<double> rf_half, fv_re, fv_im, fv05_re, fv05_im;
    if (!read_f64(args.filter_dir + "/python_rfvideo_half.f64", rf_half)) return false;
    if (!read_f64(args.filter_dir + "/python_fvideo.re.f64", fv_re)) return false;
    if (!read_f64(args.filter_dir + "/python_fvideo.im.f64", fv_im)) return false;
    if (!read_f64(args.filter_dir + "/python_fvideo05.re.f64", fv05_re)) return false;
    if (!read_f64(args.filter_dir + "/python_fvideo05.im.f64", fv05_im)) return false;

    std::vector<double> rf_interp = interp_real(rf_half, static_cast<size_t>(state.freq_bins));
    for (int k = 1; k < state.freq_bins - 1; ++k) {
        rf_interp[static_cast<size_t>(k)] *= 2.0;
    }
    std::vector<double> env_rf_interp = interp_real(rf_half, static_cast<size_t>(state.env_freq_bins));
    for (int k = 1; k < state.env_freq_bins - 1; ++k) {
        env_rf_interp[static_cast<size_t>(k)] *= 2.0;
    }
    std::vector<cufftDoubleComplex> fv_interp =
        interp_complex_parts(fv_re, fv_im, static_cast<size_t>(state.freq_bins), 1.0 / static_cast<double>(state.fft_size));
    std::vector<cufftDoubleComplex> fv05_interp =
        interp_complex_parts(fv05_re, fv05_im, static_cast<size_t>(state.freq_bins), 1.0 / static_cast<double>(state.fft_size));

    cudaMemcpy(state.d_rf_filter, rf_interp.data(), state.freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_env_rf_filter, env_rf_interp.data(), state.env_freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_fvideo, fv_interp.data(), state.freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_fvideo05, fv05_interp.data(), state.freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    return true;
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
        fprintf(stderr, "Failed to create output dir: %s\n", args.output_dir.c_str());
        return 1;
    }

    std::vector<double> raw(args.blocklen);
    std::vector<double> lead(FM_DEMOD_OVERLAP_SAMPLES, 0.0);
    std::vector<double> tail(FM_DEMOD_OVERLAP_SAMPLES, 0.0);
    if (args.sample >= static_cast<size_t>(FM_DEMOD_OVERLAP_SAMPLES)) {
        reader.read_at(lead.data(), args.sample - static_cast<size_t>(FM_DEMOD_OVERLAP_SAMPLES),
                       static_cast<size_t>(FM_DEMOD_OVERLAP_SAMPLES));
    }
    reader.read_at(tail.data(), args.sample + args.blocklen, static_cast<size_t>(FM_DEMOD_OVERLAP_SAMPLES));
    size_t got = reader.read_at(raw.data(), args.sample, args.blocklen);
    if (got != args.blocklen) {
        fprintf(stderr, "Short read: wanted %zu samples, got %zu\n", args.blocklen, got);
        return 1;
    }

    double* d_raw = nullptr;
    double* d_lead = nullptr;
    double* d_tail = nullptr;
    double* d_demod = nullptr;
    double* d_demod05 = nullptr;
    double* d_demod_burst = nullptr;
    double* d_env = nullptr;
    double* d_raw_env = nullptr;
    double* d_demod_raw = nullptr;
    double* d_demod_diff = nullptr;
    double* d_demod_spikefixed = nullptr;
    cufftDoubleComplex* d_hilbert = nullptr;
    cudaMalloc(&d_lead, lead.size() * sizeof(double));
    cudaMalloc(&d_tail, tail.size() * sizeof(double));
    cudaMalloc(&d_raw, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod05, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod_burst, args.blocklen * sizeof(double));
    cudaMalloc(&d_env, args.blocklen * sizeof(double));
    cudaMalloc(&d_raw_env, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod_raw, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod_diff, args.blocklen * sizeof(double));
    cudaMalloc(&d_demod_spikefixed, args.blocklen * sizeof(double));
    cudaMalloc(&d_hilbert, args.blocklen * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_lead, lead.data(), lead.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tail, tail.data(), tail.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_raw, raw.data(), args.blocklen * sizeof(double), cudaMemcpyHostToDevice);

    FMDemodState state;
    if (!state.init(fmt, 1, static_cast<int>(args.blocklen))) {
        fprintf(stderr, "Failed to initialize FM demod state\n");
        return 1;
    }
    if (!override_filters_from_dir(args, state)) {
        fprintf(stderr, "Failed to load override filters from: %s\n", args.filter_dir.c_str());
        return 1;
    }

    fm_demod(state, d_lead, d_raw, d_tail, d_demod, d_demod05, d_demod_burst, d_env,
             d_raw_env, d_demod_raw, d_demod_diff, d_demod_spikefixed, d_hilbert,
             1, args.blocklen, fmt);
    if (!check_cuda("fm_demod launch")) return 1;
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "fm_demod sync: %s\n", cudaGetErrorString(sync_err));
        return 1;
    }

    std::vector<double> demod(args.blocklen);
    std::vector<double> demod05(args.blocklen);
    std::vector<double> demod_burst(args.blocklen);
    std::vector<double> demod_burst_cpu_candidate(args.blocklen);
    std::vector<double> env(args.blocklen);
    std::vector<double> raw_env(args.blocklen);
    std::vector<double> env_filtered(args.blocklen);
    std::vector<double> demod_raw(args.blocklen);
    std::vector<double> demod_diff(args.blocklen);
    std::vector<double> demod_spikefixed(args.blocklen);
    std::vector<cufftDoubleComplex> hilbert(args.blocklen);
    std::vector<double> rf_half(state.freq_bins);
    std::vector<double> env_rf_half(state.env_freq_bins);
    std::vector<cufftDoubleComplex> fv(state.freq_bins);
    std::vector<cufftDoubleComplex> fv05(state.freq_bins);
    if (cudaMemcpy(demod.data(), d_demod, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(demod05.data(), d_demod05, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(demod_burst.data(), d_demod_burst, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(env.data(), d_env, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(raw_env.data(), d_raw_env, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(env_filtered.data(), state.d_env_filtered, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(demod_raw.data(), d_demod_raw, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(demod_diff.data(), d_demod_diff, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(demod_spikefixed.data(), d_demod_spikefixed, args.blocklen * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(hilbert.data(), d_hilbert, args.blocklen * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(rf_half.data(), state.d_rf_filter, state.freq_bins * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(env_rf_half.data(), state.d_env_rf_filter, state.env_freq_bins * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(fv.data(), state.d_fvideo, state.freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    if (cudaMemcpy(fv05.data(), state.d_fvideo05, state.freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;

    if (!demod_burst_cpu(raw.data(), 1, args.blocklen, fmt, demod_burst_cpu_candidate.data())) {
        fprintf(stderr, "Failed to compute CPU demod_burst candidate\n");
        return 1;
    }

    for (int k = 1; k < state.freq_bins - 1; ++k) {
        rf_half[k] *= 0.5;
    }

    write_f64(args.output_dir + "/gpu_raw.f64", raw);
    write_f64(args.output_dir + "/gpu_demod.f64", demod);
    write_f64(args.output_dir + "/gpu_demod_05.f64", demod05);
    write_f64(args.output_dir + "/gpu_demod_burst.f64", demod_burst);
    write_f64(args.output_dir + "/gpu_demod_burst_cpu_candidate.f64", demod_burst_cpu_candidate);
    write_f64(args.output_dir + "/gpu_envelope.f64", env);
    write_f64(args.output_dir + "/gpu_raw_env.f64", raw_env);
    write_f64(args.output_dir + "/gpu_env_filtered.f64", env_filtered);
    write_f64(args.output_dir + "/gpu_env_rfvideo_half.f64", env_rf_half);
    write_f64(args.output_dir + "/gpu_demod_raw.f64", demod_raw);
    write_f64(args.output_dir + "/gpu_demod_diff.f64", demod_diff);
    write_f64(args.output_dir + "/gpu_demod_spikefixed.f64", demod_spikefixed);
    write_complex_parts(args.output_dir + "/gpu_hilbert", hilbert);
    write_f64(args.output_dir + "/gpu_rfvideo_half.f64", rf_half);
    write_complex_parts(args.output_dir + "/gpu_fvideo", fv);
    write_complex_parts(args.output_dir + "/gpu_fvideo05", fv05);

    char config[1024];
    std::snprintf(
        config,
        sizeof(config),
        "{\n"
        "  \"input\": \"%s\",\n"
        "  \"output_dir\": \"%s\",\n"
        "  \"sample\": %zu,\n"
        "  \"blocklen\": %zu,\n"
        "  \"system\": \"%s\",\n"
        "  \"tape_speed\": \"%s\",\n"
        "  \"sample_rate_mhz\": %.6f,\n"
        "  \"input_format\": \"%s\",\n"
        "  \"dc_correct\": %s,\n"
        "  \"fft_size\": %d,\n"
        "  \"freq_bins\": %d,\n"
        "  \"f05_offset\": %d\n"
        "}\n",
        args.input_path.c_str(),
        args.output_dir.c_str(),
        args.sample,
        args.blocklen,
        profile_name_cli(args.profile),
        tape_speed_name(args.tape_speed),
        args.sample_rate_mhz,
        input_format_name(args.input_format),
        args.conditioning.dc_correct ? "true" : "false",
        state.fft_size,
        state.freq_bins,
        state.f05_offset);
    write_text(args.output_dir + "/config.json", config);

    cudaFree(d_lead);
    cudaFree(d_tail);
    cudaFree(d_hilbert);
    cudaFree(d_demod_spikefixed);
    cudaFree(d_demod_diff);
    cudaFree(d_demod_raw);
    cudaFree(d_raw_env);
    cudaFree(d_env);
    cudaFree(d_demod_burst);
    cudaFree(d_demod05);
    cudaFree(d_demod);
    cudaFree(d_raw);
    return 0;
}
