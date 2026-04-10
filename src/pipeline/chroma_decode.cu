#include "pipeline/chroma_decode.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <complex>
#include <fstream>
#include <string>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double PI_LOCAL = 3.14159265358979323846;
static const double TAU_LOCAL = 2.0 * PI_LOCAL;

struct DigitalZpkLocal {
    std::vector<std::complex<double>> z;
    std::vector<std::complex<double>> p;
    std::complex<double> k{1.0, 0.0};
};

static int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

static const char* env_string_local(const char* name) {
    const char* value = std::getenv(name);
    return (value && value[0]) ? value : nullptr;
}

static std::vector<int> get_dump_fields_local() {
    std::vector<int> fields;
    const char* env = env_string_local("CUVHS_DUMP_FIELDS");
    if (!env) return fields;
    const char* p = env;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        char* end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        fields.push_back((int)v);
        p = end;
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
    }
    return fields;
}

static double median_inplace_local(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if ((values.size() & 1U) == 0) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.begin() + mid);
        med = 0.5 * (med + values[mid - 1]);
    }
    return med;
}

static void compute_chroma_level_adjust_local(const double* h_linelocs,
                                              const int* h_is_first_field,
                                              int fields_loaded,
                                              int lines_per_frame,
                                              int output_field_lines,
                                              double samples_per_line,
                                              std::vector<double>* out) {
    out->assign((size_t)fields_loaded * output_field_lines, 1.0);
    if (fields_loaded <= 0) return;

    for (int field = 0; field < fields_loaded; field++) {
        const double* field_linelocs = h_linelocs + (size_t)field * lines_per_frame;
        const int lineoffset = 1;
        std::vector<double> wow((size_t)output_field_lines + 1, 1.0);
        for (int seg = 0; seg <= output_field_lines; seg++) {
            int base = seg + lineoffset + 1;
            int next = base + 1;
            double w = 1.0;
            if (base >= 0 && next < lines_per_frame) {
                double line_len = field_linelocs[next] - field_linelocs[base];
                if (std::isfinite(line_len) && line_len > 0.0) {
                    w = line_len / samples_per_line;
                }
            }
            wow[(size_t)seg] = w;
        }

        std::vector<double> wow_copy = wow;
        double med = median_inplace_local(wow_copy);
        for (double& v : wow_copy) v = std::abs(v - med);
        double mad = median_inplace_local(wow_copy);
        double threshold = mad > 0.0 ? (15.0 * mad) : 0.001;

        double* dst = out->data() + (size_t)field * output_field_lines;
        for (int out_line = 0; out_line < output_field_lines; out_line++) {
            double w = wow[(size_t)out_line + 1];
            dst[out_line] = (std::abs(w - med) > threshold) ? med : w;
        }
    }
}

static void compute_chroma_scale_field_params_local(const double* h_linelocs,
                                                    const int* h_is_first_field,
                                                    int fields_loaded,
                                                    const VideoFormat& fmt,
                                                    std::vector<double>* out_coords,
                                                    std::vector<double>* out_level_adjust)
{
    const int outwidth = fmt.output_line_len;
    const int linesout = fmt.output_field_lines;
    const size_t samples_per_field = static_cast<size_t>(outwidth) * static_cast<size_t>(linesout);
    out_coords->assign(static_cast<size_t>(fields_loaded) * samples_per_field, 0.0);
    out_level_adjust->assign(static_cast<size_t>(fields_loaded) * samples_per_field, 1.0);

    for (int field = 0; field < fields_loaded; ++field) {
        const double* linelocs = h_linelocs + static_cast<size_t>(field) * fmt.lines_per_frame;
        const int lineoffset = 1;
        const int outline_offset = (lineoffset + 1) * outwidth;
        const size_t total_scaled = samples_per_field + static_cast<size_t>(outline_offset);
        const double outscale = fmt.samples_per_line / (double)outwidth;

        std::vector<double> wowfactors(total_scaled, 1.0);
        std::vector<double> coords(total_scaled, 0.0);
        for (size_t i = 0; i < total_scaled; ++i) {
            const double scaled = static_cast<double>(i) * outscale;
            if (scaled <= 0.0) {
                coords[i] = linelocs[0];
                wowfactors[i] = 1.0;
                continue;
            }
            double max_x = fmt.samples_per_line * static_cast<double>(fmt.lines_per_frame - 1);
            if (scaled >= max_x) {
                coords[i] = linelocs[fmt.lines_per_frame - 1];
                wowfactors[i] = 1.0;
                continue;
            }
            double u = scaled / fmt.samples_per_line;
            int seg = static_cast<int>(std::floor(u));
            if (seg < 0) seg = 0;
            if (seg > fmt.lines_per_frame - 2) seg = fmt.lines_per_frame - 2;
            double frac = u - static_cast<double>(seg);
            double a = linelocs[seg];
            double b = linelocs[seg + 1];
            coords[i] = a + frac * (b - a);
            wowfactors[i] = (b - a) / fmt.samples_per_line;
        }

        std::vector<double> wow_copy = wowfactors;
        const double med = median_inplace_local(wow_copy);
        for (double& v : wow_copy) v = std::abs(v - med);
        const double mad = median_inplace_local(wow_copy);
        const double threshold = mad > 0.0 ? (15.0 * mad) : 0.001;

        double* dst_coords = out_coords->data() + static_cast<size_t>(field) * samples_per_field;
        double* dst_level = out_level_adjust->data() + static_cast<size_t>(field) * samples_per_field;
        for (size_t j = 0; j < samples_per_field; ++j) {
            const size_t i = j + static_cast<size_t>(outline_offset);
            dst_coords[j] = coords[i];
            const double w = wowfactors[i];
            dst_level[j] = (std::abs(w - med) > threshold) ? med : w;
        }
    }
}

static void write_f64_file_local(const std::string& path, const double* data, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out || !data || count == 0) return;
    out.write(reinterpret_cast<const char*>(data), (std::streamsize)(count * sizeof(double)));
}

static void maybe_dump_k4_preacc_local(const char* dump_dir,
                                       size_t raw_offset,
                                       int field_idx,
                                       const double* chroma_preacc,
                                       size_t field_samples) {
    if (!dump_dir || !dump_dir[0] || !chroma_preacc || field_samples == 0) return;
    char mkdir_cmd[1024];
    std::snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", dump_dir);
    std::system(mkdir_cmd);

    char prefix[1024];
    std::snprintf(prefix, sizeof(prefix), "%s/k4_chunk_%zu_field_%d", dump_dir, raw_offset, field_idx);
    write_f64_file_local(std::string(prefix) + "_chroma_preacc.f64", chroma_preacc, field_samples);
}

static void maybe_dump_k4_stage_local(const char* dump_dir,
                                      size_t raw_offset,
                                      int field_idx,
                                      const char* suffix,
                                      const double* data,
                                      size_t field_samples) {
    if (!dump_dir || !dump_dir[0] || !suffix || !suffix[0] || !data || field_samples == 0) return;
    char mkdir_cmd[1024];
    std::snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", dump_dir);
    std::system(mkdir_cmd);
    char prefix[1024];
    std::snprintf(prefix, sizeof(prefix), "%s/k4_chunk_%zu_field_%d_%s", dump_dir, raw_offset, field_idx, suffix);
    write_f64_file_local(std::string(prefix) + ".f64", data, field_samples);
}

// ============================================================================
// FFT-friendly size (7-smooth) — same as fm_demod.cu
// ============================================================================

static bool is_7smooth(int n) {
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    while (n % 7 == 0) n /= 7;
    return n == 1;
}

static int next_fft_size(int n) {
    while (!is_7smooth(n)) n++;
    return n;
}

static DigitalZpkLocal buttap_zpk_local(int order) {
    DigitalZpkLocal out;
    for (int m = 0; m < order; ++m) {
        const double theta = PI_LOCAL * (2.0 * (double)m + 1.0 + (double)order) /
                             (2.0 * (double)order);
        out.p.emplace_back(std::polar(1.0, theta));
    }
    return out;
}

static DigitalZpkLocal lp2bp_zpk_local(const DigitalZpkLocal& in, double wo, double bw) {
    DigitalZpkLocal out;
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    out.z.reserve(in.z.size() * 2U + static_cast<std::size_t>(degree));
    out.p.reserve(in.p.size() * 2U);
    const double bw2 = bw / 2.0;
    for (const auto& z : in.z) {
        const auto temp = z * bw2;
        const auto rad = std::sqrt(temp * temp - std::complex<double>(wo * wo, 0.0));
        out.z.push_back(temp + rad);
        out.z.push_back(temp - rad);
    }
    for (const auto& p : in.p) {
        const auto temp = p * bw2;
        const auto rad = std::sqrt(temp * temp - std::complex<double>(wo * wo, 0.0));
        out.p.push_back(temp + rad);
        out.p.push_back(temp - rad);
    }
    for (int i = 0; i < degree; ++i) out.z.emplace_back(0.0, 0.0);
    out.k = in.k * std::pow(bw, degree);
    return out;
}

static DigitalZpkLocal bilinear_zpk_local(const DigitalZpkLocal& in, double fs) {
    DigitalZpkLocal out;
    out.z.reserve(in.z.size());
    out.p.reserve(in.p.size());
    const std::complex<double> fs2{2.0 * fs, 0.0};
    for (const auto& z : in.z) out.z.push_back((fs2 + z) / (fs2 - z));
    for (const auto& p : in.p) out.p.push_back((fs2 + p) / (fs2 - p));
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    for (int i = 0; i < degree; ++i) out.z.emplace_back(-1.0, 0.0);
    std::complex<double> num = in.k;
    for (const auto& z : in.z) num *= (fs2 - z);
    std::complex<double> den{1.0, 0.0};
    for (const auto& p : in.p) den *= (fs2 - p);
    out.k = num / den;
    return out;
}

static DigitalZpkLocal butter_digital_bandpass_zpk_local(int order, double low_hz, double high_hz, double fs) {
    const double warped_low = 2.0 * fs * std::tan(PI_LOCAL * low_hz / fs);
    const double warped_high = 2.0 * fs * std::tan(PI_LOCAL * high_hz / fs);
    const double bw = warped_high - warped_low;
    const double wo = std::sqrt(warped_low * warped_high);
    return bilinear_zpk_local(lp2bp_zpk_local(buttap_zpk_local(order), wo, bw), fs);
}

static std::vector<std::complex<double>> zpk_freqz_local(const DigitalZpkLocal& filt,
                                                         std::size_t wor_n,
                                                         bool whole) {
    std::vector<std::complex<double>> out(wor_n);
    const double step = whole ? (TAU_LOCAL / static_cast<double>(wor_n))
                              : ((wor_n > 1U) ? (PI_LOCAL / static_cast<double>(wor_n - 1U)) : 0.0);
    for (std::size_t i = 0; i < wor_n; ++i) {
        const double w = step * static_cast<double>(i);
        const std::complex<double> z = std::exp(std::complex<double>(0.0, w));
        std::complex<double> num = filt.k;
        for (const auto& zero : filt.z) num *= (z - zero);
        std::complex<double> den{1.0, 0.0};
        for (const auto& pole : filt.p) den *= (z - pole);
        out[i] = num / den;
    }
    return out;
}

// ============================================================================
// Butterworth BPF magnitude (bilinear transform) — same as fm_demod.cu
// ============================================================================

static double butter_bpf_mag(double omega, double omega_l, double omega_h, int order) {
    if (omega <= 0.0 || omega >= M_PI) return 0.0;
    double tl = tan(omega_l / 2.0);
    double th = tan(omega_h / 2.0);
    double t  = tan(omega / 2.0);
    double BW = th - tl;
    if (BW <= 0.0 || t == 0.0) return 0.0;
    double S = (t * t - tl * th) / (BW * t);
    return 1.0 / sqrt(1.0 + pow(S, 2.0 * order));
}

// ============================================================================
// Kernel: Apply pre-computed bandpass filter in frequency domain
// ============================================================================

__global__ void k_apply_bandpass(
    cufftDoubleComplex* __restrict__ fft_data,
    const double* __restrict__ filter_mag,  // [freq_bins] — squared magnitude
    int num_fields,
    int freq_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_fields * freq_bins;
    if (idx >= total) return;

    int bin = idx % freq_bins;
    double mag = filter_mag[bin];
    fft_data[idx].x *= mag;
    fft_data[idx].y *= mag;
}

__global__ void k_pack_fields_for_fft(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int fields_this,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int field_fft_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_per_field = output_field_lines * output_line_len;
    int total = fields_this * samples_per_field;
    if (idx >= total) return;

    int field = idx / samples_per_field;
    int in_field = idx % samples_per_field;
    int line = in_field / output_line_len;
    int col = in_field % output_line_len;

    dst[(size_t)field * field_fft_size + in_field] =
        src[(size_t)(field * output_field_lines + line) * fft_size + col];
}

__global__ void k_zero_field_tail(
    double* __restrict__ data,
    int fields_this,
    int used_len,
    int field_fft_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tail = field_fft_size - used_len;
    int total = fields_this * tail;
    if (idx >= total || tail <= 0) return;
    int field = idx / tail;
    int off = idx % tail;
    data[(size_t)field * field_fft_size + used_len + off] = 0.0;
}

__global__ void k_unpack_fields_from_fft(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int fields_this,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int field_fft_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_per_field = output_field_lines * output_line_len;
    int total = fields_this * samples_per_field;
    if (idx >= total) return;

    int field = idx / samples_per_field;
    int in_field = idx % samples_per_field;
    int line = in_field / output_line_len;
    int col = in_field % output_line_len;

    dst[(size_t)(field * output_field_lines + line) * fft_size + col] =
        src[(size_t)field * field_fft_size + in_field];
}

// ============================================================================
// Kernel: TBC resample chroma source onto the output field grid
// ============================================================================

__global__ void k_resample_chroma_source(
    const double* __restrict__ src,
    const double* __restrict__ linelocs,
    const int* __restrict__ is_first_field,
    const double* __restrict__ level_adjust,
    int apply_level_adjust,
    double source_shift,
    double* __restrict__ out,           // [chunk_lines × fft_size]
    int chunk_lines,
    int field_offset,                   // first field index in this chunk
    int output_field_lines,
    int lines_per_frame,
    int output_line_len,
    int fft_size,
    int active_line_start,
    int total_raw_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = chunk_lines * fft_size;
    if (idx >= total_samples) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;

    if (col >= output_line_len) {
        out[idx] = 0.0;
        return;
    }

    int abs_line = field_offset * output_field_lines + line_local;
    int field = abs_line / output_field_lines;
    int out_line = abs_line % output_field_lines;

    const int lineoffset = 1;
    int ll_line = out_line + lineoffset + 1;
    int ll_next = ll_line + 1;
    int ll_base = field * lines_per_frame;

    if (ll_next >= lines_per_frame) {
        out[idx] = 0.0;
        return;
    }

    double line_start = linelocs[ll_base + ll_line];
    double line_end   = linelocs[ll_base + ll_next];

    double frac = (double)col / (double)output_line_len;
    double coord = line_start + frac * (line_end - line_start) - source_shift;

    int ci = (int)coord;
    double x = coord - (double)ci;

    if (ci < 1 || ci + 2 >= total_raw_samples) {
        out[idx] = 0.0;
        return;
    }

    double p0 = src[ci - 1];
    double p1 = src[ci];
    double p2 = src[ci + 1];
    double p3 = src[ci + 2];

    double a = p2 - p0;
    double b = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    double c = 3.0 * (p1 - p2) + p3 - p0;
    double value = p1 + 0.5 * x * (a + x * (b + x * c));
    if (level_adjust && apply_level_adjust) {
        value *= level_adjust[field * output_field_lines + out_line];
    }
    out[idx] = value;
}

__global__ void k_resample_chroma_source_coords(
    const double* __restrict__ src,
    const double* __restrict__ coords,
    const double* __restrict__ level_adjust,
    int apply_level_adjust,
    double source_shift,
    double* __restrict__ out,
    int chunk_lines,
    int field_offset,
    int output_field_lines,
    int output_line_len,
    int fft_size,
    int total_raw_samples)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = chunk_lines * fft_size;
    if (idx >= total_samples) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;
    if (col >= output_line_len) {
        out[idx] = 0.0;
        return;
    }

    int abs_line = field_offset * output_field_lines + line_local;
    int out_line = abs_line % output_field_lines;
    const size_t field_sample_idx =
        static_cast<size_t>(abs_line) * static_cast<size_t>(output_line_len) + static_cast<size_t>(col);

    const double coord = coords[field_sample_idx] - source_shift;
    int ci = static_cast<int>(coord);
    double x = coord - static_cast<double>(ci);
    if (ci < 1 || ci + 2 >= total_raw_samples) {
        out[idx] = 0.0;
        return;
    }

    double p0 = src[ci - 1];
    double p1 = src[ci];
    double p2 = src[ci + 1];
    double p3 = src[ci + 2];
    double a = p2 - p0;
    double b = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    double c = 3.0 * (p1 - p2) + p3 - p0;
    double value = p1 + 0.5 * x * (a + x * (b + x * c));
    if (level_adjust && apply_level_adjust) {
        value *= level_adjust[field_sample_idx];
    }
    out[idx] = value;
}

// ============================================================================
// Kernel: Apply chroma heterodyne after resampling, matching vhs-decode's
// "upconvert_chroma" ordering more closely than the old fused path.
// ============================================================================

__global__ void k_apply_line_heterodyne(
    const double* __restrict__ src,
    double* __restrict__ out,
    int chunk_lines,
    int field_offset,
    int output_field_lines,
    int output_line_len,
    int fft_size,
    int phase_mode,
    double het_scale,
    int line_phase_bias,
    double het_phase_bias_rad,
    const int* __restrict__ field_track,
    const int* __restrict__ field_phase_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = chunk_lines * fft_size;
    if (idx >= total_samples) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;
    if (col >= output_line_len) {
        out[idx] = 0.0;
        return;
    }

    int abs_line = field_offset * output_field_lines + line_local;
    int field = abs_line / output_field_lines;
    int out_line = abs_line % output_field_lines;

    (void)field_phase_offset;
    double phase_offset = 0.0;
    if (phase_mode == 1) {
        int track = field_track[field];
        int rot = track ? 1 : 3;
        int line_phase = (((out_line + 1) * rot) + line_phase_bias) & 3;
        phase_offset = (double)line_phase * (M_PI * 0.5);
    } else if (phase_mode == 2) {
        int track = field_track[field];
        int rot = track ? 3 : 0;
        int line_phase = (((out_line + 1) * rot) + line_phase_bias) & 3;
        phase_offset = (double)line_phase * (M_PI * 0.5);
    }

    double abs_sample = (double)(out_line * output_line_len + col);
    double het = -cos(2.0 * M_PI * het_scale * abs_sample + phase_offset);
    if (het_phase_bias_rad != 0.0) {
        het = -cos(2.0 * M_PI * het_scale * abs_sample + phase_offset + het_phase_bias_rad);
    }
    out[idx] = src[idx] * het;
}

// ============================================================================
// Kernel: Frequency-domain bandpass centered at fsc
// ============================================================================

__global__ void k_chroma_bandpass(
    cufftDoubleComplex* fft_data,
    int chunk_lines,
    int freq_bins,
    int fsc_bin,
    int bandwidth_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunk_lines * freq_bins;
    if (idx >= total) return;

    int bin = idx % freq_bins;
    int lo = fsc_bin - bandwidth_bins;
    int hi = fsc_bin + bandwidth_bins;

    if (bin < lo || bin > hi) {
        fft_data[idx].x = 0.0;
        fft_data[idx].y = 0.0;
    }
}

// ============================================================================
// Kernel: Simple line comb filter, modeled after vhs-decode
// ============================================================================

__global__ void k_chroma_comb(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int lines_this,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int line_hop)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = lines_this * fft_size;
    if (idx >= total) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;

    if (col >= output_line_len) {
        dst[idx] = src[idx];
        return;
    }

    int field_local = line_local % output_field_lines;
    // Skip vblank/head-switch region and lines near boundaries.
    if (line_local < 16 || line_local >= lines_this - line_hop) {
        dst[idx] = src[idx];
        return;
    }

    if (field_local < line_hop || field_local >= output_field_lines - line_hop) {
        dst[idx] = src[idx];
        return;
    }

    int prev_idx = idx - line_hop * fft_size;
    int next_idx = idx + line_hop * fft_size;

    // Match vhs-decode's simple comb:
    // ((line * 2) - delayed - advanced) / 4
    dst[idx] = ((src[idx] * 2.0) - src[prev_idx] - src[next_idx]) * 0.25;
}

__global__ void k_ntsc_burst_deemphasis(
    double* __restrict__ data,
    int lines_this,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int burst_end)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = lines_this * fft_size;
    if (idx >= total) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;

    if (col >= output_line_len) return;

    int field_local = line_local % output_field_lines;
    if (field_local < 1 || field_local >= output_field_lines) return;

    // vhs-decode uses burstarea=(start-5, end+10) and then scales from
    // burstarea[1] + 5 onward, which is end + 15.
    if (col >= burst_end + 15) {
        data[idx] *= 2.0;
    }
}

// ============================================================================
// Kernel: ACC normalization + uint16 output
// ============================================================================

__global__ void k_chroma_acc_output(
    const double* __restrict__ chroma,
    uint16_t* __restrict__ tbc_chroma,
    int chunk_lines,
    int output_field_lines,
    int output_line_len,
    int fft_size,
    int burst_start,
    int burst_end,
    double burst_abs_ref,
    double fft_scale)
{
    int line = blockIdx.x;
    if (line >= chunk_lines) return;

    const double* line_data = chroma + line * fft_size;
    uint16_t* line_out = tbc_chroma + line * output_line_len;
    int field_local = line % output_field_lines;

    if (field_local < 16) {
        for (int col = threadIdx.x; col < output_line_len; col += blockDim.x) {
            line_out[col] = (uint16_t)32767;
        }
        return;
    }

    __shared__ double burst_sum_sq;
    __shared__ double acc_scale;

    if (threadIdx.x == 0) burst_sum_sq = 0.0;
    __syncthreads();

    int burst_len = burst_end - burst_start;
    for (int i = threadIdx.x; i < burst_len; i += blockDim.x) {
        double val = line_data[burst_start + i] * fft_scale;
        atomicAdd(&burst_sum_sq, val * val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double rms = sqrt(burst_sum_sq / (double)burst_len);
        acc_scale = (rms != 0.0) ? (burst_abs_ref / rms) : 1.0;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < output_line_len; col += blockDim.x) {
        double val = line_data[col] * fft_scale * acc_scale;
        double out_val = val + 32767.0;
        if (out_val < 0.0) out_val = 0.0;
        if (out_val > 65535.0) out_val = 65535.0;
        line_out[col] = (uint16_t)out_val;
    }
}

__global__ void k_measure_burst_phase_iq(
    const double* __restrict__ chroma,
    double* __restrict__ out_i,
    double* __restrict__ out_q,
    int fields_this,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int burst_start,
    int burst_end,
    double fft_scale,
    double fsc,
    double output_rate)
{
    int field = blockIdx.x;
    if (field >= fields_this) return;

    __shared__ double s_i[256];
    __shared__ double s_q[256];
    double acc_i = 0.0;
    double acc_q = 0.0;

    const int start_line = 16;
    const int end_line = output_field_lines - 16;
    int burst_hi = min(burst_end, output_line_len);

    for (int line = start_line + threadIdx.x; line < end_line; line += blockDim.x) {
        const double* line_data = chroma + (size_t)(field * output_field_lines + line) * fft_size;
        double I = 0.0;
        double Q = 0.0;
        for (int i = burst_start; i < burst_hi; i++) {
            double val = line_data[i] * fft_scale;
            double abs_sample = (double)line * (double)output_line_len + (double)i;
            double t = 2.0 * M_PI * fsc / output_rate * abs_sample;
            I += val * cos(t);
            Q += val * sin(t);
        }
        double mag = sqrt(I * I + Q * Q);
        if (mag > 0.0) {
            acc_i += I / mag;
            acc_q += Q / mag;
        }
    }

    s_i[threadIdx.x] = acc_i;
    s_q[threadIdx.x] = acc_q;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_i[threadIdx.x] += s_i[threadIdx.x + stride];
            s_q[threadIdx.x] += s_q[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_i[field] = s_i[0];
        out_q[field] = s_q[0];
    }
}

__global__ void k_apply_field_phase_rotation(
    cufftDoubleComplex* __restrict__ fft_data,
    const double* __restrict__ phase_adjust,
    int lines_this,
    int output_field_lines,
    int freq_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = lines_this * freq_bins;
    if (idx >= total) return;

    int line = idx / freq_bins;
    int field = line / output_field_lines;
    double theta = phase_adjust[field];
    if (theta == 0.0) return;

    double c = cos(theta);
    double s = sin(theta);
    cufftDoubleComplex v = fft_data[idx];
    fft_data[idx].x = v.x * c - v.y * s;
    fft_data[idx].y = v.x * s + v.y * c;
}

__global__ void k_apply_phase_rotation_per_field_fft(
    cufftDoubleComplex* __restrict__ fft_data,
    const double* __restrict__ phase_adjust,
    int fields_this,
    int freq_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = fields_this * freq_bins;
    if (idx >= total) return;

    int field = idx / freq_bins;
    double theta = phase_adjust[field];
    if (theta == 0.0) return;

    int bin = idx % freq_bins;
    if (bin == 0) return;

    double c = cos(theta);
    double s = sin(theta);
    cufftDoubleComplex v = fft_data[idx];
    fft_data[idx].x = v.x * c - v.y * s;
    fft_data[idx].y = v.x * s + v.y * c;
}

__global__ void k_scale_real_buffer(double* __restrict__ data, int count, double scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    data[idx] *= scale;
}


// ============================================================================
// Kernel: Per-field burst cancellation metric (GPU — replaces CPU measure)
//
// One block per field, 256 threads cooperate to measure burst cancellation.
// Adjacent-line burst sums should cancel with correct track phase (low metric).
// Wrong track → constructive addition → high metric.
// ============================================================================

__global__ void k_burst_cancellation(
    const double* __restrict__ chroma_data,   // [chunk_lines × fft_size] after IFFT
    double* __restrict__ metrics,             // [fields_in_chunk] output
    int fields_in_chunk,
    int output_field_lines,
    int fft_size,
    int burst_start,
    int burst_end,
    double fft_scale)
{
    int field = blockIdx.x;
    if (field >= fields_in_chunk) return;

    const int SKIP = 16;  // skip vblank + head switch lines
    int burst_len = burst_end - burst_start;
    if (burst_len <= 0) { if (threadIdx.x == 0) metrics[field] = 0.0; return; }

    int num_pairs = (output_field_lines - 2 * SKIP - 1) / 2;

    __shared__ double s_sum[256];
    __shared__ int s_count[256];

    double my_sum = 0.0;
    int my_count = 0;

    for (int pair = threadIdx.x; pair < num_pairs; pair += blockDim.x) {
        int line = SKIP + pair * 2;
        size_t base_a = ((size_t)field * output_field_lines + line) * fft_size;
        size_t base_b = base_a + fft_size;

        double pair_sum = 0.0;
        for (int i = burst_start; i < burst_end; i++) {
            pair_sum += fabs(chroma_data[base_a + i] * fft_scale
                          + chroma_data[base_b + i] * fft_scale);
        }
        my_sum += pair_sum / burst_len;
        my_count++;
    }

    s_sum[threadIdx.x] = my_sum;
    s_count[threadIdx.x] = my_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_count[threadIdx.x] += s_count[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        metrics[field] = (s_count[0] > 0) ? s_sum[0] / s_count[0] : 1e9;
    }
}


// ============================================================================
// Kernel: Scan burst metrics for first field exceeding threshold
//
// One thread per field. atomicMin finds the lowest bad index.
// Output: d_first_bad[0] = index of first bad field, or INT_MAX if all good.
// ============================================================================

__global__ void k_find_first_bad(
    const double* __restrict__ metrics,
    int* __restrict__ first_bad,    // single int output
    int num_fields,
    double threshold)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_fields) return;

    if (metrics[f] > threshold) {
        atomicMin(first_bad, f);
    }
}


// ============================================================================
// Host-side: Track detection + NTSC phase sequence
// ============================================================================

// NTSC 4-frame (8-field) phase rotation sequence lookup.
// Key: (is_first_field, burst_phase_quadrant, phase_delta_from_prev)
// Value: (field_phase_id, phase_offset_to_add)
// phase_delta = -1 means unknown (no previous field to compare)
struct NTSCPhaseEntry {
    int is_first;   // 1 = first field, 0 = second
    int quadrant;   // 0-3 (burst phase / 90°)
    int delta;      // 0-3 or -1 (unknown)
    int phase_id;   // field_phase_id (informational)
    int offset;     // phase rotation offset to apply (0-3)
};

static const NTSCPhaseEntry ntsc_phase_table[] = {
    // frame 1
    {1, 2,  0, 3, 0}, {0, 3,  1, 2, 1},
    // frame 2
    {1, 1,  2, 1, 1}, {0, 0,  3, 4, 0},
    // frame 3
    {1, 0,  0, 3, 2}, {0, 1,  1, 2, 3},
    // frame 4
    {1, 3,  2, 1, 3}, {0, 2,  3, 4, 2},
    // copies without phase delta (for first field in decode, no prior)
    {1, 2, -1, 3, 0}, {0, 3, -1, 2, 1},
    {1, 1, -1, 1, 1}, {0, 0, -1, 4, 0},
    {1, 0, -1, 3, 2}, {0, 1, -1, 2, 3},
    {1, 3, -1, 1, 3}, {0, 2, -1, 4, 2},
};
static const int NTSC_PHASE_TABLE_SIZE = sizeof(ntsc_phase_table) / sizeof(ntsc_phase_table[0]);

struct NTSCPhaseResult {
    int offset;    // phase rotation offset (0-3)
    int phase_id;  // field phase ID (1-4)
};

static NTSCPhaseResult lookup_ntsc_phase(int is_first, int quadrant, int delta) {
    // Try with delta first
    for (int i = 0; i < NTSC_PHASE_TABLE_SIZE; i++) {
        if (ntsc_phase_table[i].is_first == is_first &&
            ntsc_phase_table[i].quadrant == quadrant &&
            ntsc_phase_table[i].delta == delta) {
            return { ntsc_phase_table[i].offset, ntsc_phase_table[i].phase_id };
        }
    }
    // Fallback: try without delta
    for (int i = 0; i < NTSC_PHASE_TABLE_SIZE; i++) {
        if (ntsc_phase_table[i].is_first == is_first &&
            ntsc_phase_table[i].quadrant == quadrant &&
            ntsc_phase_table[i].delta == -1) {
            return { ntsc_phase_table[i].offset, ntsc_phase_table[i].phase_id };
        }
    }
    return { 0, 1 };  // fallback
}

// Measure burst cancellation metric for track detection.
// In NTSC, burst alternates 180° per line. With correct track phase,
// summing adjacent-line bursts should cancel (low metric).
// With wrong track phase, they add constructively (high metric).
static double measure_burst_cancellation(
    const double* h_chroma,  // host buffer: output_field_lines × fft_size
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int burst_start,
    int burst_end,
    double fft_scale)
{
    const int SKIP = 16;  // skip first/last 16 lines (vblank, head switch)
    int burst_len = burst_end - burst_start;
    double total = 0.0;
    int count = 0;

    for (int line = SKIP; line < output_field_lines - SKIP - 1; line += 2) {
        const double* line_a = h_chroma + (size_t)line * fft_size;
        const double* line_b = h_chroma + (size_t)(line + 1) * fft_size;

        double sum = 0.0;
        for (int i = burst_start; i < burst_end && i < output_line_len; i++) {
            double a = line_a[i] * fft_scale;
            double b = line_b[i] * fft_scale;
            sum += fabs(a + b);
        }
        total += sum / burst_len;
        count++;
    }
    return (count > 0) ? total / count : 1e9;
}

// Measure burst phase via I/Q product detection.
// Returns burst phase in degrees (0-360).
static double measure_burst_phase(
    const double* h_chroma,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int burst_start,
    int burst_end,
    double fft_scale,
    double fsc,
    double output_rate)
{
    const int SKIP = 16;
    double I_total = 0.0, Q_total = 0.0;

    for (int line = SKIP; line < output_field_lines - SKIP; line++) {
        const double* line_data = h_chroma + (size_t)line * fft_size;
        double I = 0.0, Q = 0.0;

        for (int i = burst_start; i < burst_end && i < output_line_len; i++) {
            double val = line_data[i] * fft_scale;
            double t = 2.0 * M_PI * fsc / output_rate * (double)i;
            I += val * cos(t);
            Q += val * sin(t);
        }

        double mag = sqrt(I * I + Q * Q);
        if (mag > 0.0) {
            I_total += I / mag;
            Q_total += Q / mag;
        }
    }

    double phase_rad = atan2(Q_total, I_total);
    double phase_deg = fmod(phase_rad * 180.0 / M_PI + 360.0, 360.0);
    return phase_deg;
}

struct BurstPhaseLineLocal {
    int line_number;
    double burst_phase_deg;
    double magnitude;
    double i;
    double q;
};

static bool measure_burst_phase_sequence_local(
    const double* h_chroma,
    int output_field_lines,
    int fft_size,
    int output_line_len,
    int burst_start,
    int burst_end,
    double fft_scale,
    double fsc,
    double output_rate,
    std::vector<BurstPhaseLineLocal>* phase_sequence,
    double* burst_phase_avg_deg)
{
    if (!h_chroma || !phase_sequence || !burst_phase_avg_deg) return false;
    phase_sequence->clear();
    double i_total = 0.0;
    double q_total = 0.0;
    int avg_count = 0;

    const int start_line = 0;
    const int end_line = output_field_lines;
    const int burst_check_start = 16;
    const int burst_check_end = output_field_lines - 16;
    const double coherence_threshold = 0.3;
    const int burst_hi = std::min(burst_end, output_line_len);
    if (burst_start >= burst_hi) return false;

    for (int line = start_line; line < end_line; ++line) {
        const double* line_data = h_chroma + (size_t)line * fft_size;
        double i_acc = 0.0;
        double q_acc = 0.0;
        for (int i = burst_start; i < burst_hi; ++i) {
            double val = line_data[i] * fft_scale;
            double t = 2.0 * M_PI * fsc / output_rate * (double)i;
            i_acc += val * cos(t);
            q_acc += val * sin(t);
        }
        double mag = sqrt(i_acc * i_acc + q_acc * q_acc);
        double phase_deg = 0.0;
        if (mag > 0.0) {
            phase_deg = fmod(atan2(q_acc, i_acc) * 180.0 / M_PI + 360.0, 360.0);
            if (line > burst_check_start && line < burst_check_end) {
                i_total += i_acc / mag;
                q_total += q_acc / mag;
                avg_count++;
            }
        }
        phase_sequence->push_back(BurstPhaseLineLocal{
            line,
            phase_deg,
            mag,
            i_acc,
            q_acc,
        });
    }

    if (avg_count == 0) return false;
    double coherence = hypot(i_total, q_total) / (double)avg_count;
    if (coherence < coherence_threshold) return false;
    *burst_phase_avg_deg = fmod(atan2(q_total, i_total) * 180.0 / M_PI + 360.0, 360.0);
    return true;
}

static void apply_burst_sync_to_linelocs_local(
    double* linelocs,
    int lines_per_frame,
    int output_line_len,
    const std::vector<BurstPhaseLineLocal>& phase_sequence,
    double burst_phase_avg_deg)
{
    if (!linelocs || lines_per_frame < 2) return;
    for (size_t i = 9; i < phase_sequence.size(); ++i) {
        int line = phase_sequence[i].line_number;
        if (line < 0 || line + 1 >= lines_per_frame) continue;
        double phase_delta = fmod((burst_phase_avg_deg - phase_sequence[i].burst_phase_deg) + 540.0, 360.0) - 180.0;
        double line_start = linelocs[line];
        double line_end = linelocs[line + 1];
        double line_length = line_end - line_start;
        double scale = line_length / (double)output_line_len;
        double line_adjust = (phase_delta / 360.0) * 4.0;
        linelocs[line] += line_adjust * scale;
    }
}

static bool env_flag_enabled_local(const char* name)
{
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    return !(::strcmp(v, "0") == 0 ||
             ::strcmp(v, "false") == 0 ||
             ::strcmp(v, "FALSE") == 0 ||
             ::strcmp(v, "no") == 0 ||
             ::strcmp(v, "NO") == 0);
}

static bool env_int_local(const char* name, int* out_value)
{
    const char* v = std::getenv(name);
    if (!v || !*v || !out_value) return false;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return false;
    *out_value = static_cast<int>(parsed);
    return true;
}

static bool env_double_local(const char* name, double* out_value)
{
    const char* v = std::getenv(name);
    if (!v || !*v || !out_value) return false;
    char* end = nullptr;
    double parsed = std::strtod(v, &end);
    if (end == v) return false;
    *out_value = parsed;
    return true;
}

// Process a single field through the full chroma pipeline and return
// the time-domain output in h_out (host buffer).
static void process_one_field_chroma(
    const double* d_src,
    const double* d_linelocs,
    const int* d_is_first_field,
    const double* d_level_adjust,
    bool apply_level_adjust,
    double source_shift,
    double* d_het_buf,
    cufftDoubleComplex* d_fft_buf,
    double* h_out,
    int field_idx,
    int track,
    int phase_offset,
    int output_field_lines,
    int lines_per_frame,
    int output_line_len,
    int fft_size,
    int freq_bins,
    int active_line_start,
    int phase_mode,
    int line_phase_bias,
    double het_phase_bias_rad,
    int total_raw_samples,
    double het_scale,
    int fsc_bin,
    int bandwidth_bins,
    int* d_field_track,
    int* d_field_phase_offset)
{
    int lines = output_field_lines;

    // Upload track + phase offset for this field
    cudaMemcpy(d_field_track + field_idx, &track, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_phase_offset + field_idx, &phase_offset, sizeof(int), cudaMemcpyHostToDevice);

    // Resample to output grid
    {
        int total = lines * fft_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_resample_chroma_source<<<blocks, threads>>>(
            d_src, d_linelocs, d_is_first_field, d_level_adjust, apply_level_adjust ? 1 : 0, source_shift, d_het_buf,
            lines, field_idx, output_field_lines,
            lines_per_frame, output_line_len, fft_size,
            active_line_start, total_raw_samples);
        k_apply_line_heterodyne<<<blocks, threads>>>(
            d_het_buf, d_het_buf,
            lines, field_idx, output_field_lines,
            output_line_len, fft_size,
            phase_mode, het_scale, line_phase_bias, het_phase_bias_rad,
            d_field_track, d_field_phase_offset);
    }

    // Forward FFT (need a temporary plan for 1 field)
    cufftHandle plan_r2c, plan_c2r;
    int n[] = { fft_size };
    cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, lines);
    cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, lines);

    cufftExecD2Z(plan_r2c, d_het_buf, d_fft_buf);

    // Bandpass
    {
        int total = lines * freq_bins;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_chroma_bandpass<<<blocks, threads>>>(d_fft_buf, lines, freq_bins, fsc_bin, bandwidth_bins);
    }

    // Inverse FFT
    cufftExecZ2D(plan_c2r, d_fft_buf, d_het_buf);

    // Download to host
    cudaMemcpy(h_out, d_het_buf, (size_t)lines * fft_size * sizeof(double), cudaMemcpyDeviceToHost);

    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
}

// ============================================================================
// Host entry point
// ============================================================================

void chroma_decode(const double* d_chroma_source,
                   double* d_linelocs,
                   const int* d_is_first_field,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt,
                   std::vector<int>& field_phase_ids,
                   ChromaState* state,
                   size_t raw_offset)
{
    bool uses_ntsc_family_phase = (fmt.profile != VideoProfile::PAL_625_50_VHS);
    bool do_chunk_track_retry = (fmt.profile == VideoProfile::NTSC_525_60_VHS);
    int phase_mode = 0;
    if (fmt.profile == VideoProfile::NTSC_525_60_VHS) phase_mode = 1;
    else if (fmt.profile == VideoProfile::MPAL_525_60_VHS) phase_mode = 2;
    int line_phase_bias = 0;
    env_int_local("CUVHS_LINE_PHASE_BIAS", &line_phase_bias);
    double het_phase_bias_rad = 0.0;
    env_double_local("CUVHS_HET_PHASE_BIAS_RAD", &het_phase_bias_rad);
    int output_field_lines = fmt.output_field_lines;
    int output_line_len = fmt.output_line_len;
    int fft_size = next_pow2(output_line_len);
    int freq_bins = fft_size / 2 + 1;
    int field_samples = output_field_lines * output_line_len;
    int field_fft_size = next_fft_size(field_samples);
    int field_freq_bins = field_fft_size / 2 + 1;
    const bool apply_chroma_level_adjust =
        !env_flag_enabled_local("CUVHS_DISABLE_CHROMA_LEVEL_ADJUST");

    double het_freq = fmt.fsc + fmt.chroma_under;
    double het_scale = het_freq / fmt.output_rate;
    double source_shift = 0.0;
    double final_source_shift = 4.0 * (fmt.sample_rate / 28.0e6);
    double source_shift_override = 0.0;
    if (env_double_local("CUVHS_CHROMA_SOURCE_SHIFT", &source_shift_override)) {
        final_source_shift = source_shift_override;
    } else {
        int source_shift_override_i = 0;
        if (env_int_local("CUVHS_CHROMA_SOURCE_SHIFT", &source_shift_override_i)) {
            final_source_shift = static_cast<double>(source_shift_override_i);
        }
    }

    int fsc_bin = (int)(fmt.fsc / fmt.output_rate * fft_size + 0.5);
    int bandwidth_bins = (int)(500000.0 / fmt.output_rate * fft_size + 0.5);
    if (bandwidth_bins < 10) bandwidth_bins = 10;

    int burst_start = (int)(fmt.burst_start_us * 1e-6 * fmt.output_rate + 0.5);
    int burst_end   = (int)(fmt.burst_end_us * 1e-6 * fmt.output_rate + 0.5);

    double fft_scale = 1.0 / (double)fft_size;
    double field_fft_norm = 1.0 / (double)field_fft_size;

    // ---------------------------------------------------------------
    // Final chroma filter: match vhs-decode's FChromaFinal structure
    // (Butterworth BPF, order 4, applied with filtfilt => |H|^2)
    // over the whole field signal rather than line-by-line.
    // ---------------------------------------------------------------
    double final_lower = (fmt.chroma_under / 1e6) * 0.9;
    double final_upper = (fmt.chroma_under / 1e6) * 0.75;
    double final_lo_hz = fmt.fsc - final_lower * 1e6;
    double final_hi_hz = fmt.fsc + final_upper * 1e6;
    auto final_bpf_resp = zpk_freqz_local(
        butter_digital_bandpass_zpk_local(4, final_lo_hz, final_hi_hz, fmt.output_rate),
        static_cast<std::size_t>(field_fft_size),
        true);
    auto* h_final_bpf = new double[field_freq_bins];
    for (int i = 0; i < field_freq_bins; i++) {
        h_final_bpf[i] = std::norm(final_bpf_resp[(size_t)i]) * field_fft_norm;
    }

    double* d_final_bpf_filter = nullptr;
    cudaMalloc(&d_final_bpf_filter, (size_t)field_freq_bins * sizeof(double));
    cudaMemcpy(d_final_bpf_filter, h_final_bpf,
               (size_t)field_freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_final_bpf;

    int field_stride = total_raw_samples / num_fields;
    const double* chroma_raw = d_chroma_source;

    // ---------------------------------------------------------------
    // Allocate per-field track and phase offset arrays (GPU)
    // ---------------------------------------------------------------
    int* d_field_track = nullptr;
    int* d_field_phase_offset = nullptr;
    cudaMalloc(&d_field_track, num_fields * sizeof(int));
    cudaMalloc(&d_field_phase_offset, num_fields * sizeof(int));
    double* d_level_adjust_pre = nullptr;
    double* d_level_adjust_post = nullptr;
    double* d_source_coords_post = nullptr;
    double* d_source_level_adjust_post = nullptr;
    cudaMalloc(&d_level_adjust_pre, (size_t)num_fields * output_field_lines * sizeof(double));
    cudaMalloc(&d_level_adjust_post, (size_t)num_fields * output_field_lines * sizeof(double));
    cudaMalloc(&d_source_coords_post, (size_t)num_fields * field_samples * sizeof(double));
    cudaMalloc(&d_source_level_adjust_post, (size_t)num_fields * field_samples * sizeof(double));

    {
        std::vector<double> h_linelocs((size_t)num_fields * fmt.lines_per_frame);
        std::vector<int> h_is_first(num_fields);
        cudaMemcpy(h_linelocs.data(), d_linelocs,
                   (size_t)num_fields * fmt.lines_per_frame * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_is_first.data(), d_is_first_field,
                   (size_t)num_fields * sizeof(int),
                   cudaMemcpyDeviceToHost);
        std::vector<double> h_level_adjust;
        compute_chroma_level_adjust_local(h_linelocs.data(), h_is_first.data(), num_fields,
                                          fmt.lines_per_frame, output_field_lines,
                                          fmt.samples_per_line, &h_level_adjust);
        cudaMemcpy(d_level_adjust_pre, h_level_adjust.data(),
                   (size_t)num_fields * output_field_lines * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_level_adjust_post, h_level_adjust.data(),
                   (size_t)num_fields * output_field_lines * sizeof(double),
                   cudaMemcpyHostToDevice);
        std::vector<double> h_source_coords;
        std::vector<double> h_source_level_adjust;
        compute_chroma_scale_field_params_local(h_linelocs.data(), h_is_first.data(), num_fields, fmt,
                                                &h_source_coords, &h_source_level_adjust);
        cudaMemcpy(d_source_coords_post, h_source_coords.data(),
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_source_level_adjust_post, h_source_level_adjust.data(),
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    // ---------------------------------------------------------------
    // Step 1: Track detection
    //
    // If we have state from a previous batch, use it directly.
    // Otherwise, process field 0 with both track=0 and track=1,
    // measure burst cancellation, and pick the lower metric.
    // ---------------------------------------------------------------

    // Temp buffers for single-field processing
    size_t one_field_het_bytes = (size_t)output_field_lines * fft_size * sizeof(double);
    size_t one_field_fft_bytes = (size_t)output_field_lines * freq_bins * sizeof(cufftDoubleComplex);

    double* d_det_het = nullptr;
    cufftDoubleComplex* d_det_fft = nullptr;
    cudaMalloc(&d_det_het, one_field_het_bytes);
    cudaMalloc(&d_det_fft, one_field_fft_bytes);

    auto* h_det = new double[(size_t)output_field_lines * fft_size];

    int detected_track = 0;
    double good_metric_threshold = 0.0;
    int cycle_start = 0;
    int forced_track = 0;
    bool have_forced_track = env_int_local("CUVHS_FORCE_TRACK", &forced_track);

    // NTSC 4-frame phase ID cycle (8 fields):
    static const int ntsc_phase_id_cycle[8] = { 3, 2, 1, 4, 3, 2, 1, 4 };
    static const int ntsc_phase_offset_cycle[8] = { 0, 1, 1, 0, 2, 3, 3, 2 };
    static const int mpal_phase_id_cycle[4] = { 1, 2, 3, 4 };

    if (have_forced_track) {
        detected_track = forced_track ? 1 : 0;
        good_metric_threshold = 0.0;
        cycle_start = 0;
        fprintf(stderr, "  [chroma] Forced track override: track=%d\n", detected_track);
    } else if (!uses_ntsc_family_phase) {
        field_phase_ids.clear();
    } else if (state && state->valid) {
        // ---------------------------------------------------------------
        // Continuation from previous batch — use carried state
        // ---------------------------------------------------------------
        detected_track = state->current_track;
        good_metric_threshold = state->good_metric_threshold;
        cycle_start = state->cycle_start;

        fprintf(stderr, "  [chroma] Using carried state: track=%d threshold=%.4f cycle_start=%d\n",
                detected_track, good_metric_threshold, cycle_start);
    } else {
        // ---------------------------------------------------------------
        // First batch — full auto-detection
        // ---------------------------------------------------------------
        double metric[2];
        for (int try_track = 0; try_track < 2; try_track++) {
            process_one_field_chroma(
                chroma_raw, d_linelocs, d_is_first_field, d_level_adjust_pre, apply_chroma_level_adjust, source_shift, d_det_het, d_det_fft, h_det,
                0, try_track, 0,
                output_field_lines, fmt.lines_per_frame, output_line_len,
                fft_size, freq_bins, fmt.active_line_start, phase_mode, line_phase_bias, het_phase_bias_rad, total_raw_samples,
                het_scale, fsc_bin, bandwidth_bins,
                d_field_track, d_field_phase_offset);

            metric[try_track] = measure_burst_cancellation(
                h_det, output_field_lines, fft_size, output_line_len,
                burst_start, burst_end, fft_scale);
        }

        detected_track = (metric[1] <= metric[0]) ? 1 : 0;
        good_metric_threshold = metric[detected_track] * 4.0;

        fprintf(stderr, "  [chroma] Track detect: metric[0]=%.4f metric[1]=%.4f → track=%d\n",
                metric[0], metric[1], detected_track);

        // ---------------------------------------------------------------
        // Step 2: NTSC burst phase measurement + 4-frame sequence lookup
        // ---------------------------------------------------------------

        // Re-process field 0 with correct track to get clean burst for phase measurement
        process_one_field_chroma(
            chroma_raw, d_linelocs, d_is_first_field, d_level_adjust_pre, apply_chroma_level_adjust, source_shift, d_det_het, d_det_fft, h_det,
            0, detected_track, 0,
            output_field_lines, fmt.lines_per_frame, output_line_len,
            fft_size, freq_bins, fmt.active_line_start, phase_mode, line_phase_bias, het_phase_bias_rad, total_raw_samples,
            het_scale, fsc_bin, bandwidth_bins,
            d_field_track, d_field_phase_offset);

        double burst_phase_0 = measure_burst_phase(
            h_det, output_field_lines, fft_size, output_line_len,
            burst_start, burst_end, fft_scale, fmt.fsc, fmt.output_rate);

        int quadrant_0 = ((int)(burst_phase_0 / 90.0 + 0.5)) % 4;

        // Field 0 is assumed first field; no previous burst → delta = -1
        auto phase_result_0 = lookup_ntsc_phase(1, quadrant_0, -1);
        int phase_id_0 = phase_result_0.phase_id;
        int phase_offset_0 = phase_result_0.offset;

        fprintf(stderr, "  [chroma] Field 0: burst_phase=%.1f° quadrant=%d phase_id=%d\n",
                burst_phase_0, quadrant_0, phase_id_0);

        // Find the exact first-field slot in the cycle using both phase ID and
        // phase offset, since the phase ID repeats twice in the 8-field cycle.
        cycle_start = 0;
        for (int i = 0; i < 8; i += 2) {
            if (ntsc_phase_id_cycle[i] == phase_id_0 &&
                ntsc_phase_offset_cycle[i] == phase_offset_0) {
                cycle_start = i;
                break;
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 3: Assign all tracks + phase IDs upfront
    //
    // Track alternates per field. Phase offset is 0 for all fields.
    // Track flips (tape edit points) are detected batch-parallel in
    // Step 4 via GPU burst cancellation kernel — no serial loop.
    // ---------------------------------------------------------------

    std::vector<int> h_track(num_fields, 0);
    std::vector<int> h_phase_offset(num_fields, 0);
    int current_track = detected_track;
    int phase_offset_bias = 0;
    env_int_local("CUVHS_PHASE_OFFSET_BIAS", &phase_offset_bias);

    if (uses_ntsc_family_phase) {
        field_phase_ids.resize(num_fields);
        for (int f = 0; f < num_fields; f++) {
            h_track[f] = (f & 1) ? (1 - current_track) : current_track;
            if (fmt.profile == VideoProfile::MPAL_525_60_VHS) {
                field_phase_ids[f] = mpal_phase_id_cycle[f % 4];
            } else {
                int cycle_idx = (cycle_start + f) % 8;
                field_phase_ids[f] = ntsc_phase_id_cycle[cycle_idx];
                int off = ntsc_phase_offset_cycle[cycle_idx] + phase_offset_bias;
                off %= 4;
                if (off < 0) off += 4;
                h_phase_offset[f] = off;
            }
        }
        cudaMemcpy(d_field_track, h_track.data(), num_fields * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_phase_offset, h_phase_offset.data(), num_fields * sizeof(int), cudaMemcpyHostToDevice);
    } else {
        field_phase_ids.clear();
    }

    if (fmt.profile == VideoProfile::NTSC_525_60_VHS) {
        const bool disable_burst_sync = env_flag_enabled_local("CUVHS_DISABLE_BURST_SYNC");
        std::vector<double> h_linelocs((size_t)num_fields * fmt.lines_per_frame);
        cudaMemcpy(h_linelocs.data(), d_linelocs,
                   (size_t)num_fields * fmt.lines_per_frame * sizeof(double),
                   cudaMemcpyDeviceToHost);
        std::vector<BurstPhaseLineLocal> phase_sequence;
        for (int f = 0; f < num_fields; ++f) {
            process_one_field_chroma(
                chroma_raw, d_linelocs, d_is_first_field, d_level_adjust_pre, apply_chroma_level_adjust, source_shift, d_det_het, d_det_fft, h_det,
                f, h_track[(size_t)f], h_phase_offset[(size_t)f],
                output_field_lines, fmt.lines_per_frame, output_line_len,
                fft_size, freq_bins, fmt.active_line_start, phase_mode, line_phase_bias, het_phase_bias_rad, total_raw_samples,
                het_scale, fsc_bin, bandwidth_bins,
                d_field_track, d_field_phase_offset);

            double burst_phase_avg_deg = 0.0;
            if (!measure_burst_phase_sequence_local(
                    h_det,
                    output_field_lines,
                    fft_size,
                    output_line_len,
                    burst_start,
                    burst_end,
                    fft_scale,
                    fmt.fsc,
                    fmt.output_rate,
                    &phase_sequence,
                    &burst_phase_avg_deg)) {
                continue;
            }

            if (!disable_burst_sync) {
                apply_burst_sync_to_linelocs_local(
                    h_linelocs.data() + (size_t)f * fmt.lines_per_frame,
                    fmt.lines_per_frame,
                    output_line_len,
                    phase_sequence,
                    burst_phase_avg_deg);
            }
        }
        std::vector<double> h_level_adjust_post;
        std::vector<int> h_is_first(num_fields);
        cudaMemcpy(h_is_first.data(), d_is_first_field,
                   (size_t)num_fields * sizeof(int),
                   cudaMemcpyDeviceToHost);
        compute_chroma_level_adjust_local(h_linelocs.data(), h_is_first.data(), num_fields,
                                          fmt.lines_per_frame, output_field_lines,
                                          fmt.samples_per_line, &h_level_adjust_post);
        std::vector<double> h_source_coords_post;
        std::vector<double> h_source_level_adjust_post;
        compute_chroma_scale_field_params_local(h_linelocs.data(), h_is_first.data(), num_fields, fmt,
                                                &h_source_coords_post, &h_source_level_adjust_post);
        cudaMemcpy(d_level_adjust_post, h_level_adjust_post.data(),
                   (size_t)num_fields * output_field_lines * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_source_coords_post, h_source_coords_post.data(),
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_source_level_adjust_post, h_source_level_adjust_post.data(),
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_linelocs, h_linelocs.data(),
                   (size_t)num_fields * fmt.lines_per_frame * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    // Free detection buffers (only needed for initial auto-detect above)
    cudaFree(d_det_het);
    cudaFree(d_det_fft);
    delete[] h_det;

    // ---------------------------------------------------------------
    // Step 4: Batch chroma processing with GPU-parallel flip detection
    //
    // Process all fields in chunks through het+FFT+BPF+IFFT, then
    // measure burst cancellation on GPU (one kernel, no CPU round-trips).
    // If a track flip is detected, update assignments and retry chunk.
    // Common case (no flips): one pass, one tiny metric download.
    // ---------------------------------------------------------------

    // Determine chunk size from available VRAM
    size_t bytes_per_line = (size_t)fft_size * sizeof(double)
                          + (size_t)freq_bins * sizeof(cufftDoubleComplex);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t usable = (size_t)(free_mem * 0.8);

    int max_lines = (int)(usable / bytes_per_line);
    if (max_lines < output_field_lines) max_lines = output_field_lines;

    int max_chunk_lines = 4096;
    max_chunk_lines = (max_chunk_lines / output_field_lines) * output_field_lines;
    if (max_chunk_lines < output_field_lines) max_chunk_lines = output_field_lines;
    if (max_lines > max_chunk_lines) max_lines = max_chunk_lines;

    int total_lines = num_fields * output_field_lines;
    int chunk_lines = std::min(max_lines, total_lines);
    chunk_lines = (chunk_lines / output_field_lines) * output_field_lines;
    if (chunk_lines < output_field_lines) chunk_lines = output_field_lines;

    // Allocate temp buffers with backoff. The initial estimate only accounts
    // for the het/FFT buffers themselves; cuFFT plan workspace and allocator
    // overhead can still push us over the edge.
    double* d_het_buf = nullptr;
    double* d_comb_buf = nullptr;
    cufftDoubleComplex* d_fft_buf = nullptr;
    double* d_metrics = nullptr;
    int* d_first_bad = nullptr;
    double* d_phase_i = nullptr;
    double* d_phase_q = nullptr;
    double* d_phase_adjust = nullptr;
    double* d_field_buf = nullptr;
    cufftDoubleComplex* d_field_fft = nullptr;
    cufftHandle plan_r2c = 0, plan_c2r = 0;
    cufftHandle plan_field_r2c = 0, plan_field_c2r = 0;
    int n[] = { fft_size };
    int field_n[] = { field_fft_size };
    int fields_per_chunk = 0;

    while (chunk_lines >= output_field_lines) {
        fields_per_chunk = chunk_lines / output_field_lines;
        size_t het_bytes = (size_t)chunk_lines * fft_size * sizeof(double);
        size_t fft_bytes = (size_t)chunk_lines * freq_bins * sizeof(cufftDoubleComplex);

        cudaError_t err = cudaMalloc(&d_het_buf, het_bytes);
        if (err != cudaSuccess) {
            d_het_buf = nullptr;
        } else {
            err = cudaMalloc(&d_comb_buf, het_bytes);
            if (err != cudaSuccess) {
                cudaFree(d_het_buf);
                d_het_buf = nullptr;
                d_comb_buf = nullptr;
            } else {
                err = cudaMalloc(&d_fft_buf, fft_bytes);
                if (err != cudaSuccess) {
                    cudaFree(d_comb_buf);
                    cudaFree(d_het_buf);
                    d_het_buf = nullptr;
                    d_comb_buf = nullptr;
                    d_fft_buf = nullptr;
                } else {
                    err = cudaMalloc(&d_metrics, fields_per_chunk * sizeof(double));
                    if (err != cudaSuccess) {
                        cudaFree(d_fft_buf);
                        cudaFree(d_comb_buf);
                        cudaFree(d_het_buf);
                        d_het_buf = nullptr;
                        d_comb_buf = nullptr;
                        d_fft_buf = nullptr;
                        d_metrics = nullptr;
                    } else {
                        err = cudaMalloc(&d_first_bad, sizeof(int));
                        if (err != cudaSuccess) {
                            cudaFree(d_metrics);
                            cudaFree(d_fft_buf);
                            cudaFree(d_comb_buf);
                            cudaFree(d_het_buf);
                            d_het_buf = nullptr;
                            d_comb_buf = nullptr;
                            d_fft_buf = nullptr;
                            d_metrics = nullptr;
                            d_first_bad = nullptr;
                        } else {
                            err = cudaMalloc(&d_phase_i, fields_per_chunk * sizeof(double));
                            if (err == cudaSuccess) err = cudaMalloc(&d_phase_q, fields_per_chunk * sizeof(double));
                            if (err == cudaSuccess) err = cudaMalloc(&d_phase_adjust, fields_per_chunk * sizeof(double));
                            if (err == cudaSuccess) err = cudaMalloc(&d_field_buf, (size_t)fields_per_chunk * field_fft_size * sizeof(double));
                            if (err == cudaSuccess) err = cudaMalloc(&d_field_fft, (size_t)fields_per_chunk * field_freq_bins * sizeof(cufftDoubleComplex));
                            if (err == cudaSuccess &&
                                cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins,
                                              CUFFT_D2Z, chunk_lines) == CUFFT_SUCCESS &&
                                cufftPlanMany(&plan_field_r2c, 1, field_n, NULL, 1, field_fft_size, NULL, 1, field_freq_bins,
                                              CUFFT_D2Z, fields_per_chunk) == CUFFT_SUCCESS &&
                                cufftPlanMany(&plan_field_c2r, 1, field_n, NULL, 1, field_freq_bins, NULL, 1, field_fft_size,
                                              CUFFT_Z2D, fields_per_chunk) == CUFFT_SUCCESS &&
                                cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size,
                                              CUFFT_Z2D, chunk_lines) == CUFFT_SUCCESS) {
                                break;
                            }
                            if (plan_r2c) cufftDestroy(plan_r2c);
                            if (plan_c2r) cufftDestroy(plan_c2r);
                            if (plan_field_r2c) cufftDestroy(plan_field_r2c);
                            if (plan_field_c2r) cufftDestroy(plan_field_c2r);
                            plan_r2c = 0;
                            plan_c2r = 0;
                            plan_field_r2c = 0;
                            plan_field_c2r = 0;
                            cudaFree(d_first_bad);
                            cudaFree(d_metrics);
                            cudaFree(d_fft_buf);
                            cudaFree(d_comb_buf);
                            cudaFree(d_het_buf);
                            if (d_field_fft) cudaFree(d_field_fft);
                            if (d_field_buf) cudaFree(d_field_buf);
                            if (d_phase_adjust) cudaFree(d_phase_adjust);
                            if (d_phase_q) cudaFree(d_phase_q);
                            if (d_phase_i) cudaFree(d_phase_i);
                            d_het_buf = nullptr;
                            d_comb_buf = nullptr;
                            d_fft_buf = nullptr;
                            d_metrics = nullptr;
                            d_first_bad = nullptr;
                            d_phase_i = nullptr;
                            d_phase_q = nullptr;
                            d_phase_adjust = nullptr;
                            d_field_buf = nullptr;
                            d_field_fft = nullptr;
                        }
                    }
                }
            }
        }

        int next_fields = std::max(1, fields_per_chunk / 2);
        int next_chunk_lines = next_fields * output_field_lines;
        if (next_chunk_lines == chunk_lines) break;
        chunk_lines = next_chunk_lines;
    }

    if (!d_het_buf || !d_comb_buf || !d_fft_buf || !d_metrics || !d_first_bad ||
        !d_phase_i || !d_phase_q || !d_phase_adjust || !d_field_buf || !d_field_fft ||
        !plan_r2c || !plan_c2r || !plan_field_r2c || !plan_field_c2r) {
        fprintf(stderr, "  [chroma] Failed to allocate temp buffers/plans even at one-field chunk size\n");
        if (plan_r2c) cufftDestroy(plan_r2c);
        if (plan_c2r) cufftDestroy(plan_c2r);
        if (plan_field_r2c) cufftDestroy(plan_field_r2c);
        if (plan_field_c2r) cufftDestroy(plan_field_c2r);
        if (d_first_bad) cudaFree(d_first_bad);
        if (d_metrics) cudaFree(d_metrics);
        if (d_fft_buf) cudaFree(d_fft_buf);
        if (d_comb_buf) cudaFree(d_comb_buf);
        if (d_het_buf) cudaFree(d_het_buf);
        if (d_field_fft) cudaFree(d_field_fft);
        if (d_field_buf) cudaFree(d_field_buf);
        if (d_phase_adjust) cudaFree(d_phase_adjust);
        if (d_phase_q) cudaFree(d_phase_q);
        if (d_phase_i) cudaFree(d_phase_i);
        cudaFree(d_field_track);
        cudaFree(d_field_phase_offset);
        cudaFree(d_level_adjust_pre);
        cudaFree(d_level_adjust_post);
        cudaFree(d_source_coords_post);
        cudaFree(d_source_level_adjust_post);
        cudaFree(d_final_bpf_filter);
        return;
    }

    const char* dump_k4_dir = env_string_local("CUVHS_K4_DUMP_DIR");
    static std::vector<int> dump_fields_k4 = get_dump_fields_local();
    if (dump_k4_dir && !dump_fields_k4.empty()) {
        std::vector<double> h_source_coords_dbg((size_t)num_fields * field_samples);
        std::vector<double> h_source_level_dbg((size_t)num_fields * field_samples);
        cudaMemcpy(h_source_coords_dbg.data(), d_source_coords_post,
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_source_level_dbg.data(), d_source_level_adjust_post,
                   (size_t)num_fields * field_samples * sizeof(double),
                   cudaMemcpyDeviceToHost);
        for (int df : dump_fields_k4) {
            if (df < 0 || df >= num_fields) continue;
            const double* coords_ptr = h_source_coords_dbg.data() + (size_t)df * field_samples;
            const double* level_ptr = h_source_level_dbg.data() + (size_t)df * field_samples;
            maybe_dump_k4_stage_local(
                dump_k4_dir,
                raw_offset,
                df,
                "source_coords_post",
                coords_ptr,
                (size_t)field_samples);
            maybe_dump_k4_stage_local(
                dump_k4_dir,
                raw_offset,
                df,
                "source_level_adjust_post",
                level_ptr,
                (size_t)field_samples);
        }
    }

    for (int field_start = 0; field_start < num_fields; field_start += fields_per_chunk) {
        int fields_this = std::min(fields_per_chunk, num_fields - field_start);
        int lines_this = fields_this * output_field_lines;

        if (lines_this < chunk_lines) {
            cufftDestroy(plan_r2c);
            cufftDestroy(plan_c2r);
            cufftDestroy(plan_field_r2c);
            cufftDestroy(plan_field_c2r);
            cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, lines_this);
            cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, lines_this);
            cufftPlanMany(&plan_field_r2c, 1, field_n, NULL, 1, field_fft_size, NULL, 1, field_freq_bins,
                          CUFFT_D2Z, fields_this);
            cufftPlanMany(&plan_field_c2r, 1, field_n, NULL, 1, field_freq_bins, NULL, 1, field_fft_size,
                          CUFFT_Z2D, fields_this);
        }

        // Retry loop: process chunk, check for track flips, retry if needed
        int retries = 3;
        bool chunk_ok = false;

        while (!chunk_ok && retries-- > 0) {
            if (uses_ntsc_family_phase) {
                cudaMemcpy(d_field_track + field_start, h_track.data() + field_start,
                           fields_this * sizeof(int), cudaMemcpyHostToDevice);
            }

            // Resample chroma source to final field geometry
            {
                int total = lines_this * fft_size;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                k_resample_chroma_source_coords<<<blocks, threads>>>(
                    chroma_raw,
                    d_source_coords_post,
                    d_source_level_adjust_post,
                    apply_chroma_level_adjust ? 1 : 0,
                    final_source_shift,
                    d_het_buf,
                    lines_this,
                    field_start,
                    output_field_lines,
                    output_line_len,
                    fft_size,
                    total_raw_samples);
            }

            if (dump_k4_dir && !dump_fields_k4.empty()) {
                std::vector<double> h_chroma_tbc((size_t)lines_this * fft_size);
                cudaMemcpy(h_chroma_tbc.data(), d_het_buf,
                           (size_t)lines_this * fft_size * sizeof(double),
                           cudaMemcpyDeviceToHost);
                for (int df : dump_fields_k4) {
                    int field = df;
                    if (field < field_start || field >= field_start + fields_this) continue;
                    int local_field = field - field_start;
                    std::vector<double> packed((size_t)output_field_lines * output_line_len);
                    for (int line = 0; line < output_field_lines; line++) {
                        const double* src = h_chroma_tbc.data()
                            + (size_t)(local_field * output_field_lines + line) * fft_size;
                        std::copy(src, src + output_line_len,
                                  packed.data() + (size_t)line * output_line_len);
                    }
                    maybe_dump_k4_stage_local(
                        dump_k4_dir,
                        raw_offset,
                        field,
                        "chroma_tbc",
                        packed.data(),
                        (size_t)output_field_lines * output_line_len);
                }
            }

            if (fmt.profile == VideoProfile::NTSC_525_60_VHS) {
                int total = lines_this * fft_size;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                k_ntsc_burst_deemphasis<<<blocks, threads>>>(
                    d_het_buf,
                    lines_this,
                    output_field_lines,
                    fft_size,
                    output_line_len,
                    burst_end);
            }

            if (dump_k4_dir && !dump_fields_k4.empty()) {
                std::vector<double> h_postdeemph((size_t)lines_this * fft_size);
                cudaMemcpy(h_postdeemph.data(), d_het_buf,
                           (size_t)lines_this * fft_size * sizeof(double),
                           cudaMemcpyDeviceToHost);
                for (int df : dump_fields_k4) {
                    int field = df;
                    if (field < field_start || field >= field_start + fields_this) continue;
                    int local_field = field - field_start;
                    std::vector<double> packed((size_t)output_field_lines * output_line_len);
                    for (int line = 0; line < output_field_lines; line++) {
                        const double* src = h_postdeemph.data()
                            + (size_t)(local_field * output_field_lines + line) * fft_size;
                        std::copy(src, src + output_line_len,
                                  packed.data() + (size_t)line * output_line_len);
                    }
                    maybe_dump_k4_stage_local(
                        dump_k4_dir,
                        raw_offset,
                        field,
                        "chroma_postdeemph",
                        packed.data(),
                        (size_t)output_field_lines * output_line_len);
                }
            }

            // Upconvert from demod_burst to FSC using per-line phase selection.
            {
                int total = lines_this * fft_size;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                k_apply_line_heterodyne<<<blocks, threads>>>(
                    d_het_buf, d_het_buf,
                    lines_this, field_start, output_field_lines,
                    output_line_len, fft_size,
                    phase_mode, het_scale, line_phase_bias, het_phase_bias_rad,
                    d_field_track, d_field_phase_offset);
            }

            if (dump_k4_dir && !dump_fields_k4.empty()) {
                std::vector<double> h_uphet_prephase((size_t)lines_this * fft_size);
                cudaMemcpy(h_uphet_prephase.data(), d_het_buf,
                           (size_t)lines_this * fft_size * sizeof(double),
                           cudaMemcpyDeviceToHost);
                for (int df : dump_fields_k4) {
                    int field = df;
                    if (field < field_start || field >= field_start + fields_this) continue;
                    int local_field = field - field_start;
                    std::vector<double> packed((size_t)output_field_lines * output_line_len);
                    for (int line = 0; line < output_field_lines; line++) {
                        const double* src = h_uphet_prephase.data()
                            + (size_t)(local_field * output_field_lines + line) * fft_size;
                        std::copy(src, src + output_line_len,
                                  packed.data() + (size_t)line * output_line_len);
                    }
                    maybe_dump_k4_stage_local(
                        dump_k4_dir,
                        raw_offset,
                        field,
                        "chroma_uphet_prephase",
                        packed.data(),
                        (size_t)output_field_lines * output_line_len);
                }
            }

            if (!uses_ntsc_family_phase || !do_chunk_track_retry || have_forced_track) {
                chunk_ok = true;
            } else {
                // Track detection uses the upconverted pre-phase signal, matching
                // vhs-decode's burst probe stage more closely than the old fused path.
                k_burst_cancellation<<<fields_this, 256>>>(
                    d_het_buf, d_metrics, fields_this,
                    output_field_lines, fft_size,
                    burst_start, burst_end, fft_scale);

                int sentinel = INT_MAX;
                cudaMemcpy(d_first_bad, &sentinel, sizeof(int), cudaMemcpyHostToDevice);
                {
                    int threads = 256;
                    int blocks = (fields_this + threads - 1) / threads;
                    k_find_first_bad<<<blocks, threads>>>(
                        d_metrics, d_first_bad, fields_this, good_metric_threshold);
                }
                int first_bad;
                cudaMemcpy(&first_bad, d_first_bad, sizeof(int), cudaMemcpyDeviceToHost);

                if (first_bad == INT_MAX) {
                    chunk_ok = true;
                } else {
                    int abs_f = field_start + first_bad;
                    current_track = 1 - current_track;
                    fprintf(stderr, "  [chroma] Track flip at field %d (threshold %.1f), new track=%d\n",
                            abs_f, good_metric_threshold, current_track);
                    for (int ff = abs_f; ff < num_fields; ff++) {
                        h_track[ff] = (ff & 1) ? (1 - current_track) : current_track;
                    }
                    chunk_ok = false;
                }
            }
        }

        if (fmt.profile == VideoProfile::NTSC_525_60_VHS) {
            k_measure_burst_phase_iq<<<fields_this, 256>>>(
                d_het_buf,
                d_phase_i,
                d_phase_q,
                fields_this,
                output_field_lines,
                fft_size,
                output_line_len,
                burst_start,
                burst_end,
                fft_scale,
                fmt.fsc,
                fmt.output_rate);

            std::vector<double> h_phase_i((size_t)fields_this);
            std::vector<double> h_phase_q((size_t)fields_this);
            std::vector<double> h_phase_adjust((size_t)fields_this, 0.0);
            double phase_target_deg_override = 0.0;
            const bool have_phase_target_override =
                env_double_local("CUVHS_NTSC_PHASE_TARGET_DEG", &phase_target_deg_override);
            cudaMemcpy(h_phase_i.data(), d_phase_i, (size_t)fields_this * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_phase_q.data(), d_phase_q, (size_t)fields_this * sizeof(double), cudaMemcpyDeviceToHost);
            for (int f = 0; f < fields_this; f++) {
                double phase_rad = atan2(h_phase_q[(size_t)f], h_phase_i[(size_t)f]);
                // Match vhs-decode ntsc_phase_comp default target (0 degrees).
                // The track-dependent 90/270 target belongs to heterodyne-phase
                // selection, not whole-field post upconversion phase compensation.
                double phase_target_deg = have_phase_target_override
                    ? phase_target_deg_override
                    : -25.0;
                double ntsc_phase_target_rad = phase_target_deg * (M_PI / 180.0);
                h_phase_adjust[(size_t)f] = ntsc_phase_target_rad - phase_rad;
            }
            if (field_start == 0 && fields_this > 0) {
                std::ofstream dbg("/tmp/cuvhs_phase_adjust.txt", std::ios::app);
                if (dbg) {
                    dbg << "raw_offset=" << raw_offset
                        << " field_start=" << field_start
                        << " phase_adjust0=" << h_phase_adjust[0]
                        << " I0=" << h_phase_i[0]
                        << " Q0=" << h_phase_q[0]
                        << "\n";
                }
            }
            cudaMemcpy(d_phase_adjust, h_phase_adjust.data(),
                       (size_t)fields_this * sizeof(double), cudaMemcpyHostToDevice);
        }

        if (dump_k4_dir && !dump_fields_k4.empty()) {
            std::vector<double> h_postline((size_t)lines_this * fft_size);
            cudaMemcpy(h_postline.data(), d_het_buf,
                       (size_t)lines_this * fft_size * sizeof(double),
                       cudaMemcpyDeviceToHost);
            for (int df : dump_fields_k4) {
                int field = df;
                if (field < field_start || field >= field_start + fields_this) continue;
                int local_field = field - field_start;
                std::vector<double> packed((size_t)output_field_lines * output_line_len);
                for (int line = 0; line < output_field_lines; line++) {
                    const double* src = h_postline.data()
                        + (size_t)(local_field * output_field_lines + line) * fft_size;
                    std::copy(src, src + output_line_len,
                              packed.data() + (size_t)line * output_line_len);
                }
                maybe_dump_k4_stage_local(
                    dump_k4_dir,
                    raw_offset,
                    field,
                    "chroma_postline_prephase",
                    packed.data(),
                    (size_t)output_field_lines * output_line_len);
            }
        }

        {
            int threads = 256;
            int total_samples = fields_this * field_samples;
            int blocks = (total_samples + threads - 1) / threads;
            k_pack_fields_for_fft<<<blocks, threads>>>(
                d_het_buf,
                d_field_buf,
                fields_this,
                output_field_lines,
                fft_size,
                output_line_len,
                field_fft_size);
            if (field_fft_size > field_samples) {
                int tail_total = fields_this * (field_fft_size - field_samples);
                int tail_blocks = (tail_total + threads - 1) / threads;
                k_zero_field_tail<<<tail_blocks, threads>>>(
                    d_field_buf,
                    fields_this,
                    field_samples,
                    field_fft_size);
            }
            cufftExecD2Z(plan_field_r2c, d_field_buf, d_field_fft);
            if (fmt.profile == VideoProfile::NTSC_525_60_VHS) {
                int rot_total = fields_this * field_freq_bins;
                int rot_blocks = (rot_total + threads - 1) / threads;
                k_apply_phase_rotation_per_field_fft<<<rot_blocks, threads>>>(
                    d_field_fft,
                    d_phase_adjust,
                    fields_this,
                    field_freq_bins);
            }
            cufftExecZ2D(plan_field_c2r, d_field_fft, d_field_buf);
            {
                int scale_total = fields_this * field_fft_size;
                int scale_blocks = (scale_total + threads - 1) / threads;
                k_scale_real_buffer<<<scale_blocks, threads>>>(
                    d_field_buf,
                    scale_total,
                    field_fft_norm);
            }
            int unpack_blocks = (total_samples + threads - 1) / threads;
            k_unpack_fields_from_fft<<<unpack_blocks, threads>>>(
                d_field_buf,
                d_comb_buf,
                fields_this,
                output_field_lines,
                fft_size,
                output_line_len,
                field_fft_size);
        }

        if (dump_k4_dir && !dump_fields_k4.empty()) {
            std::vector<double> h_postline((size_t)lines_this * fft_size);
            cudaMemcpy(h_postline.data(), d_comb_buf,
                       (size_t)lines_this * fft_size * sizeof(double),
                       cudaMemcpyDeviceToHost);
            for (int df : dump_fields_k4) {
                int field = df;
                if (field < field_start || field >= field_start + fields_this) continue;
                int local_field = field - field_start;
                std::vector<double> packed((size_t)output_field_lines * output_line_len);
                for (int line = 0; line < output_field_lines; line++) {
                    const double* src = h_postline.data()
                        + (size_t)(local_field * output_field_lines + line) * fft_size;
                    std::copy(src, src + output_line_len,
                              packed.data() + (size_t)line * output_line_len);
                }
                maybe_dump_k4_stage_local(
                    dump_k4_dir,
                    raw_offset,
                    field,
                    "chroma_postphase",
                    packed.data(),
                    (size_t)output_field_lines * output_line_len);
            }
        }

        {
            int threads = 256;
            int total_samples = fields_this * field_samples;
            int blocks = (total_samples + threads - 1) / threads;
            k_pack_fields_for_fft<<<blocks, threads>>>(
                d_comb_buf,
                d_field_buf,
                fields_this,
                output_field_lines,
                fft_size,
                output_line_len,
                field_fft_size);
            if (field_fft_size > field_samples) {
                int tail_total = fields_this * (field_fft_size - field_samples);
                int tail_blocks = (tail_total + threads - 1) / threads;
                k_zero_field_tail<<<tail_blocks, threads>>>(
                    d_field_buf,
                    fields_this,
                    field_samples,
                    field_fft_size);
            }
            cufftExecD2Z(plan_field_r2c, d_field_buf, d_field_fft);
            int filt_total = fields_this * field_freq_bins;
            int filt_blocks = (filt_total + threads - 1) / threads;
            k_apply_bandpass<<<filt_blocks, threads>>>(
                d_field_fft,
                d_final_bpf_filter,
                fields_this,
                field_freq_bins);
            cufftExecZ2D(plan_field_c2r, d_field_fft, d_field_buf);
            int unpack_blocks = (total_samples + threads - 1) / threads;
            k_unpack_fields_from_fft<<<unpack_blocks, threads>>>(
                d_field_buf,
                d_comb_buf,
                fields_this,
                output_field_lines,
                fft_size,
                output_line_len,
                field_fft_size);
        }

        if (dump_k4_dir && !dump_fields_k4.empty()) {
            std::vector<double> h_postphasefilter((size_t)lines_this * fft_size);
            cudaMemcpy(h_postphasefilter.data(), d_comb_buf,
                       (size_t)lines_this * fft_size * sizeof(double),
                       cudaMemcpyDeviceToHost);
            for (int df : dump_fields_k4) {
                int field = df;
                if (field < field_start || field >= field_start + fields_this) continue;
                int local_field = field - field_start;
                std::vector<double> packed((size_t)output_field_lines * output_line_len);
                for (int line = 0; line < output_field_lines; line++) {
                    const double* src = h_postphasefilter.data()
                        + (size_t)(local_field * output_field_lines + line) * fft_size;
                    std::copy(src, src + output_line_len,
                              packed.data() + (size_t)line * output_line_len);
                }
                maybe_dump_k4_stage_local(
                    dump_k4_dir,
                    raw_offset,
                    field,
                    "chroma_postphasefilter",
                    packed.data(),
                    (size_t)output_field_lines * output_line_len);
            }
        }

        {
            int total = lines_this * fft_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            int line_hop = (fmt.profile == VideoProfile::NTSC_525_60_VHS) ? 1 : 2;
            k_chroma_comb<<<blocks, threads>>>(
                d_comb_buf, d_het_buf,
                lines_this, output_field_lines,
                fft_size, output_line_len, line_hop);
        }

        if (dump_k4_dir && !dump_fields_k4.empty()) {
            std::vector<double> h_preacc((size_t)lines_this * fft_size);
            cudaMemcpy(h_preacc.data(), d_het_buf,
                       (size_t)lines_this * fft_size * sizeof(double),
                       cudaMemcpyDeviceToHost);
            for (int df : dump_fields_k4) {
                int field = df;
                if (field < field_start || field >= field_start + fields_this) continue;
                int local_field = field - field_start;
                std::vector<double> packed((size_t)output_field_lines * output_line_len);
                for (int line = 0; line < output_field_lines; line++) {
                    const double* src = h_preacc.data()
                        + (size_t)(local_field * output_field_lines + line) * fft_size;
                    std::copy(src, src + output_line_len,
                              packed.data() + (size_t)line * output_line_len);
                }
                maybe_dump_k4_preacc_local(
                    dump_k4_dir,
                    raw_offset,
                    field,
                    packed.data(),
                    (size_t)output_field_lines * output_line_len);
            }
        }

        // ACC normalization + uint16 output
        uint16_t* out_ptr = const_cast<uint16_t*>(
            d_tbc_chroma + (size_t)field_start * output_field_lines * output_line_len);

        k_chroma_acc_output<<<lines_this, 256>>>(
            d_het_buf, out_ptr,
            lines_this, output_field_lines, output_line_len, fft_size,
            burst_start, burst_end,
            fmt.burst_abs_ref, fft_scale);
    }

    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    cufftDestroy(plan_field_r2c);
    cufftDestroy(plan_field_c2r);
    cudaFree(d_het_buf);
    cudaFree(d_comb_buf);
    cudaFree(d_fft_buf);
    cudaFree(d_field_buf);
    cudaFree(d_field_fft);
    cudaFree(d_metrics);
    cudaFree(d_first_bad);
    cudaFree(d_phase_adjust);
    cudaFree(d_phase_q);
    cudaFree(d_phase_i);
    cudaFree(d_field_track);
    cudaFree(d_field_phase_offset);
    cudaFree(d_level_adjust_pre);
    cudaFree(d_level_adjust_post);
    cudaFree(d_source_coords_post);
    cudaFree(d_source_level_adjust_post);
    cudaFree(d_final_bpf_filter);
    // Save state for next batch
    if (state && uses_ntsc_family_phase) {
        state->valid = true;
        // current_track is the base track for even-indexed fields in this batch.
        // If batch had odd field count, next batch's field 0 needs the opposite parity.
        state->current_track = (num_fields & 1) ? (1 - current_track) : current_track;
        state->good_metric_threshold = good_metric_threshold;
        // Advance cycle_start by num_fields so next batch continues the sequence
        state->cycle_start = (cycle_start + num_fields) % 8;
    } else if (state) {
        state->valid = false;
    }
}
