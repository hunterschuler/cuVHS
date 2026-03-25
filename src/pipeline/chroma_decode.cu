#include "pipeline/chroma_decode.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
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

// ============================================================================
// Kernel: TBC resample raw signal + heterodyne with per-field phase control
// ============================================================================

__global__ void k_resample_raw_het(
    const double* __restrict__ raw,
    const double* __restrict__ linelocs,
    double* __restrict__ out,           // [chunk_lines × fft_size]
    int chunk_lines,
    int field_offset,                   // first field index in this chunk
    int output_field_lines,
    int lines_per_frame,
    int output_line_len,
    int fft_size,
    int active_line_start,
    int total_raw_samples,
    double het_scale,
    const int* __restrict__ field_track,        // per-field: 0 or 1
    const int* __restrict__ field_phase_offset) // per-field: 0-3 (NTSC sequence offset)
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

    int ll_line = out_line + active_line_start;
    int ll_next = ll_line + 1;
    int ll_base = field * lines_per_frame;

    if (ll_next >= lines_per_frame) {
        out[idx] = 0.0;
        return;
    }

    double line_start = linelocs[ll_base + ll_line];
    double line_end   = linelocs[ll_base + ll_next];

    double frac = (double)col / (double)output_line_len;
    double coord = line_start + frac * (line_end - line_start);

    int ci = (int)coord;
    double x = coord - (double)ci;

    if (ci < 1 || ci + 2 >= total_raw_samples) {
        out[idx] = 0.0;
        return;
    }

    double p0 = raw[ci - 1];
    double p1 = raw[ci];
    double p2 = raw[ci + 1];
    double p3 = raw[ci + 2];

    double a = p2 - p0;
    double b = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    double c = 3.0 * (p1 - p2) + p3 - p0;
    double raw_val = p1 + 0.5 * x * (a + x * (b + x * c));

    // Per-line phase: track rotation + NTSC phase offset
    // VHS NTSC: Track 0 → rot=3 (-90°/line), Track 1 → rot=1 (+90°/line)
    int track = field_track[field];
    int rot = track ? 1 : 3;
    int phase_off = field_phase_offset[field];
    int line_phase = (out_line * rot + phase_off) & 3;
    double phase_offset = (double)line_phase * (M_PI * 0.5);

    // Use absolute sample position within field for continuous phase.
    // Python generates het carrier across full field; col-only resets
    // phase each line, creating a 171° discontinuity per line.
    double abs_sample = (double)(out_line * output_line_len + col);
    double het = -cos(2.0 * M_PI * het_scale * abs_sample + phase_offset);

    out[idx] = raw_val * het;
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
// Kernel: ACC normalization + uint16 output
// ============================================================================

__global__ void k_chroma_acc_output(
    const double* __restrict__ chroma,
    uint16_t* __restrict__ tbc_chroma,
    int chunk_lines,
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
        acc_scale = (rms > 1.0) ? (burst_abs_ref / rms) : 0.0;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < output_line_len; col += blockDim.x) {
        double val = line_data[col] * fft_scale * acc_scale;
        double out_val = val + 32768.0;
        if (out_val < 0.0) out_val = 0.0;
        if (out_val > 65535.0) out_val = 65535.0;
        line_out[col] = (uint16_t)(out_val + 0.5);
    }
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

// Process a single field through the full chroma pipeline and return
// the time-domain output in h_out (host buffer).
static void process_one_field_chroma(
    const double* d_raw,
    const double* d_linelocs,
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

    // Resample + heterodyne
    {
        int total = lines * fft_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_resample_raw_het<<<blocks, threads>>>(
            d_raw, d_linelocs, d_het_buf,
            lines, field_idx, output_field_lines,
            lines_per_frame, output_line_len, fft_size,
            active_line_start, total_raw_samples, het_scale,
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

void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   double* d_scratch,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt,
                   std::vector<int>& field_phase_ids,
                   ChromaState* state)
{
    int output_field_lines = fmt.output_field_lines;
    int output_line_len = fmt.output_line_len;
    int fft_size = next_pow2(output_line_len);
    int freq_bins = fft_size / 2 + 1;

    double het_freq = fmt.fsc + fmt.chroma_under;
    double het_scale = het_freq / fmt.output_rate;

    int fsc_bin = (int)(fmt.fsc / fmt.output_rate * fft_size + 0.5);
    int bandwidth_bins = (int)(500000.0 / fmt.output_rate * fft_size + 0.5);
    if (bandwidth_bins < 10) bandwidth_bins = 10;

    int burst_start = (int)(fmt.burst_start_us * 1e-6 * fmt.output_rate + 0.5);
    int burst_end   = (int)(fmt.burst_end_us * 1e-6 * fmt.output_rate + 0.5);

    double fft_scale = 1.0 / (double)fft_size;

    // ---------------------------------------------------------------
    // Step 0: Build pre-bandpass filter for color-under band
    //
    // Python applies a 4th-order Butterworth BPF (60 kHz – 1.2 MHz)
    // at capture rate BEFORE heterodyning. Without this, the luma FM
    // carrier (3.4–4.4 MHz) gets heterodyned into the fsc band and
    // contaminates chroma output, causing cycling color artifacts.
    //
    // sosfiltfilt doubles the effective order → magnitude² response.
    // We approximate this with |H(f)|² in the frequency domain.
    //
    // The filter is built here; actual filtering happens per-field
    // in the chunk processing loop to avoid a huge temp buffer.
    // ---------------------------------------------------------------

    // Per-field stride in the raw/scratch buffers (may be padded beyond nominal spf)
    int field_stride = total_raw_samples / num_fields;

    int raw_fft_size = next_fft_size(field_stride);
    int raw_freq_bins = raw_fft_size / 2 + 1;

    double chroma_bpf_lower = 60000.0;    // 60 kHz
    double chroma_bpf_upper = 1200000.0;  // 1.2 MHz (NTSC VHS)
    int    chroma_bpf_order = 4;          // sosfiltfilt → effective 8th order

    double omega_lo = 2.0 * M_PI * chroma_bpf_lower / fmt.sample_rate;
    double omega_hi = 2.0 * M_PI * chroma_bpf_upper / fmt.sample_rate;
    double raw_fft_norm = 1.0 / (double)raw_fft_size;

    auto* h_bpf = new double[raw_freq_bins];
    for (int i = 0; i < raw_freq_bins; i++) {
        double omega = M_PI * (double)i / (double)(raw_freq_bins - 1);
        double mag = butter_bpf_mag(omega, omega_lo, omega_hi, chroma_bpf_order);
        h_bpf[i] = mag * mag * raw_fft_norm;  // |H|² × 1/N
    }

    double* d_bpf_filter = nullptr;
    cudaMalloc(&d_bpf_filter, raw_freq_bins * sizeof(double));
    cudaMemcpy(d_bpf_filter, h_bpf, raw_freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_bpf;

    // Bandpass-filter all fields: d_raw → d_scratch
    // Uses single-field temp buffers (padded + FFT), processes one field at a time
    {
        double* d_bpf_padded = nullptr;
        cufftDoubleComplex* d_bpf_fft = nullptr;
        cudaMalloc(&d_bpf_padded, (size_t)raw_fft_size * sizeof(double));
        cudaMalloc(&d_bpf_fft, (size_t)raw_freq_bins * sizeof(cufftDoubleComplex));

        cufftHandle bpf_plan_fwd = 0, bpf_plan_inv = 0;
        cufftPlan1d(&bpf_plan_fwd, raw_fft_size, CUFFT_D2Z, 1);
        cufftPlan1d(&bpf_plan_inv, raw_fft_size, CUFFT_Z2D, 1);

        for (int f = 0; f < num_fields; f++) {
            size_t field_offset = (size_t)f * field_stride;

            cudaMemset(d_bpf_padded, 0, raw_fft_size * sizeof(double));
            cudaMemcpy(d_bpf_padded, d_raw + field_offset,
                       field_stride * sizeof(double),
                       cudaMemcpyDeviceToDevice);

            cufftExecD2Z(bpf_plan_fwd, d_bpf_padded, d_bpf_fft);

            {
                int threads = 256;
                int blocks = (raw_freq_bins + threads - 1) / threads;
                k_apply_bandpass<<<blocks, threads>>>(d_bpf_fft, d_bpf_filter, 1, raw_freq_bins);
            }

            cufftExecZ2D(bpf_plan_inv, d_bpf_fft, d_bpf_padded);

            cudaMemcpy(d_scratch + field_offset, d_bpf_padded,
                       field_stride * sizeof(double),
                       cudaMemcpyDeviceToDevice);
        }

        cufftDestroy(bpf_plan_fwd);
        cufftDestroy(bpf_plan_inv);
        cudaFree(d_bpf_padded);
        cudaFree(d_bpf_fft);
    }

    cudaFree(d_bpf_filter);

    // Use the bandpass-filtered signal for all downstream chroma processing
    const double* chroma_raw = d_scratch;

    // ---------------------------------------------------------------
    // Allocate per-field track and phase offset arrays (GPU)
    // ---------------------------------------------------------------
    int* d_field_track = nullptr;
    int* d_field_phase_offset = nullptr;
    cudaMalloc(&d_field_track, num_fields * sizeof(int));
    cudaMalloc(&d_field_phase_offset, num_fields * sizeof(int));

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

    int detected_track;
    double good_metric_threshold;
    int cycle_start;

    // NTSC 4-frame phase ID cycle (8 fields):
    static const int ntsc_phase_id_cycle[8] = { 3, 2, 1, 4, 3, 2, 1, 4 };

    if (state && state->valid) {
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
                chroma_raw, d_linelocs, d_det_het, d_det_fft, h_det,
                0, try_track, 0,
                output_field_lines, fmt.lines_per_frame, output_line_len,
                fft_size, freq_bins, fmt.active_line_start, total_raw_samples,
                het_scale, fsc_bin, bandwidth_bins,
                d_field_track, d_field_phase_offset);

            metric[try_track] = measure_burst_cancellation(
                h_det, output_field_lines, fft_size, output_line_len,
                burst_start, burst_end, fft_scale);
        }

        detected_track = (metric[1] < metric[0]) ? 1 : 0;
        good_metric_threshold = metric[detected_track] * 4.0;

        fprintf(stderr, "  [chroma] Track detect: metric[0]=%.4f metric[1]=%.4f → track=%d\n",
                metric[0], metric[1], detected_track);

        // ---------------------------------------------------------------
        // Step 2: NTSC burst phase measurement + 4-frame sequence lookup
        // ---------------------------------------------------------------

        // Re-process field 0 with correct track to get clean burst for phase measurement
        process_one_field_chroma(
            chroma_raw, d_linelocs, d_det_het, d_det_fft, h_det,
            0, detected_track, 0,
            output_field_lines, fmt.lines_per_frame, output_line_len,
            fft_size, freq_bins, fmt.active_line_start, total_raw_samples,
            het_scale, fsc_bin, bandwidth_bins,
            d_field_track, d_field_phase_offset);

        double burst_phase_0 = measure_burst_phase(
            h_det, output_field_lines, fft_size, output_line_len,
            burst_start, burst_end, fft_scale, fmt.fsc, fmt.output_rate);

        int quadrant_0 = ((int)(burst_phase_0 / 90.0 + 0.5)) % 4;

        // Field 0 is assumed first field; no previous burst → delta = -1
        auto phase_result_0 = lookup_ntsc_phase(1, quadrant_0, -1);
        int phase_id_0 = phase_result_0.phase_id;

        fprintf(stderr, "  [chroma] Field 0: burst_phase=%.1f° quadrant=%d phase_id=%d\n",
                burst_phase_0, quadrant_0, phase_id_0);

        // Find where phase_id_0 appears for a first field in the cycle
        cycle_start = 0;
        for (int i = 0; i < 8; i += 2) {
            if (ntsc_phase_id_cycle[i] == phase_id_0) {
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

    std::vector<int> h_track(num_fields);
    std::vector<int> h_phase_offset(num_fields, 0);

    int current_track = detected_track;

    field_phase_ids.resize(num_fields);
    for (int f = 0; f < num_fields; f++) {
        h_track[f] = (f & 1) ? (1 - current_track) : current_track;
        field_phase_ids[f] = ntsc_phase_id_cycle[(cycle_start + f) % 8];
    }

    // Upload to GPU
    cudaMemcpy(d_field_track, h_track.data(), num_fields * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_phase_offset, h_phase_offset.data(), num_fields * sizeof(int), cudaMemcpyHostToDevice);

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

    int fields_per_chunk = chunk_lines / output_field_lines;

    // Allocate temp buffers
    double* d_het_buf = nullptr;
    cufftDoubleComplex* d_fft_buf = nullptr;
    double* d_metrics = nullptr;

    size_t het_bytes = (size_t)chunk_lines * fft_size * sizeof(double);
    size_t fft_bytes = (size_t)chunk_lines * freq_bins * sizeof(cufftDoubleComplex);

    cudaError_t err;
    err = cudaMalloc(&d_het_buf, het_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "  [chroma] Failed to alloc het buffer: %s\n", cudaGetErrorString(err));
        cudaFree(d_field_track); cudaFree(d_field_phase_offset);
        return;
    }
    err = cudaMalloc(&d_fft_buf, fft_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "  [chroma] Failed to alloc FFT buffer: %s\n", cudaGetErrorString(err));
        cudaFree(d_het_buf); cudaFree(d_field_track); cudaFree(d_field_phase_offset);
        return;
    }
    cudaMalloc(&d_metrics, fields_per_chunk * sizeof(double));
    auto* h_metrics = new double[fields_per_chunk];

    // Create cuFFT plans
    cufftHandle plan_r2c = 0, plan_c2r = 0;
    int n[] = { fft_size };

    cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, chunk_lines);
    cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, chunk_lines);

    for (int field_start = 0; field_start < num_fields; field_start += fields_per_chunk) {
        int fields_this = std::min(fields_per_chunk, num_fields - field_start);
        int lines_this = fields_this * output_field_lines;

        if (lines_this < chunk_lines) {
            cufftDestroy(plan_r2c);
            cufftDestroy(plan_c2r);
            cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, lines_this);
            cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, lines_this);
        }

        // Retry loop: process chunk, check for track flips, retry if needed
        int retries = 3;
        bool chunk_ok = false;

        while (!chunk_ok && retries-- > 0) {
            // Upload current track assignments for this chunk
            cudaMemcpy(d_field_track + field_start, h_track.data() + field_start,
                       fields_this * sizeof(int), cudaMemcpyHostToDevice);

            // Resample + heterodyne
            {
                int total = lines_this * fft_size;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                k_resample_raw_het<<<blocks, threads>>>(
                    chroma_raw, d_linelocs, d_het_buf,
                    lines_this, field_start, output_field_lines,
                    fmt.lines_per_frame, output_line_len, fft_size,
                    fmt.active_line_start, total_raw_samples, het_scale,
                    d_field_track, d_field_phase_offset);
            }

            // Forward FFT → bandpass → inverse FFT
            cufftExecD2Z(plan_r2c, d_het_buf, d_fft_buf);

            {
                int total = lines_this * freq_bins;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                k_chroma_bandpass<<<blocks, threads>>>(d_fft_buf, lines_this, freq_bins, fsc_bin, bandwidth_bins);
            }

            cufftExecZ2D(plan_c2r, d_fft_buf, d_het_buf);

            // GPU burst cancellation measurement (replaces 411 CPU round-trips)
            k_burst_cancellation<<<fields_this, 256>>>(
                d_het_buf, d_metrics, fields_this,
                output_field_lines, fft_size,
                burst_start, burst_end, fft_scale);

            // Download metrics (tiny: fields_this doubles ≈ 3 KB)
            cudaMemcpy(h_metrics, d_metrics, fields_this * sizeof(double), cudaMemcpyDeviceToHost);

            // Scan for track flips
            chunk_ok = true;
            for (int f = 0; f < fields_this; f++) {
                if (h_metrics[f] > good_metric_threshold) {
                    int abs_f = field_start + f;
                    current_track = 1 - current_track;
                    fprintf(stderr, "  [chroma] Track flip at field %d: metric %.1f (threshold %.1f), new track=%d\n",
                            abs_f, h_metrics[f], good_metric_threshold, current_track);

                    // Reassign tracks from this field onward (entire remaining batch)
                    for (int ff = abs_f; ff < num_fields; ff++) {
                        h_track[ff] = (ff & 1) ? (1 - current_track) : current_track;
                    }

                    chunk_ok = false;
                    break;  // retry this chunk with corrected tracks
                }
            }
        }

        // ACC normalization + uint16 output
        uint16_t* out_ptr = const_cast<uint16_t*>(
            d_tbc_chroma + (size_t)field_start * output_field_lines * output_line_len);

        k_chroma_acc_output<<<lines_this, 256>>>(
            d_het_buf, out_ptr,
            lines_this, output_line_len, fft_size,
            burst_start, burst_end,
            fmt.burst_abs_ref, fft_scale);
    }

    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    cudaFree(d_het_buf);
    cudaFree(d_fft_buf);
    cudaFree(d_metrics);
    cudaFree(d_field_track);
    cudaFree(d_field_phase_offset);
    delete[] h_metrics;

    // Save state for next batch
    if (state) {
        state->valid = true;
        // current_track is the base track for even-indexed fields in this batch.
        // If batch had odd field count, next batch's field 0 needs the opposite parity.
        state->current_track = (num_fields & 1) ? (1 - current_track) : current_track;
        state->good_metric_threshold = good_metric_threshold;
        // Advance cycle_start by num_fields so next batch continues the sequence
        state->cycle_start = (cycle_start + num_fields) % 8;
    }
}
