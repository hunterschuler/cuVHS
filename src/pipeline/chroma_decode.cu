#include "pipeline/chroma_decode.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Compute next power of 2 >= n
static int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

// ============================================================================
// Phase 1: TBC resample raw signal + heterodyne multiplication
// ============================================================================

__global__ void k_resample_raw_het(
    const double* __restrict__ raw,
    const double* __restrict__ linelocs,
    double* __restrict__ out,           // [chunk_lines × fft_size]
    int chunk_lines,                    // lines in this chunk
    int field_offset,                   // first field index in this chunk
    int output_field_lines,
    int lines_per_frame,
    int output_line_len,
    int fft_size,
    int active_line_start,
    int total_raw_samples,
    double het_scale,
    int phase_rotation)                 // per-line phase step: 1 or 3 (VHS NTSC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = chunk_lines * fft_size;
    if (idx >= total_samples) return;

    int line_local = idx / fft_size;
    int col = idx % fft_size;

    // Zero padding region
    if (col >= output_line_len) {
        out[idx] = 0.0;
        return;
    }

    // Decompose into field and output line (relative to chunk start)
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

    // Catmull-Rom cubic interpolation from raw signal
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

    // Heterodyne: multiply by -cos(2π × het_scale × col + line_phase)
    // VHS uses per-line phase rotation (±90° per line) to reduce crosstalk.
    // Track 0 uses rotation=3 (-90°/line), Track 1 uses rotation=1 (+90°/line).
    // phase_rotation encodes which rotation to use for even fields (0 or 1).
    int track = (field + phase_rotation) & 1;
    int rot = track ? 1 : 3;  // +90° or -90° per line
    int line_phase = (out_line * rot) & 3;
    double phase_offset = (double)line_phase * (M_PI * 0.5);
    double het = -cos(2.0 * M_PI * het_scale * (double)col + phase_offset);

    out[idx] = raw_val * het;
}

// ============================================================================
// Phase 2: Frequency-domain bandpass centered at fsc
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
// Phase 3: ACC normalization + uint16 output
// ============================================================================

__global__ void k_chroma_acc_output(
    const double* __restrict__ chroma,   // [chunk_lines × fft_size]
    uint16_t* __restrict__ tbc_chroma,   // output (offset into batch output)
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
// Host entry point — processes in VRAM-aware chunks
// ============================================================================

void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt)
{
    int output_field_lines = fmt.output_field_lines;
    int output_line_len = fmt.output_line_len;

    // Compute FFT size: next power-of-2 >= output_line_len
    int fft_size = next_pow2(output_line_len);
    int freq_bins = fft_size / 2 + 1;

    // Heterodyne frequency: fsc + chroma_under
    double het_freq = fmt.fsc + fmt.chroma_under;
    double het_scale = het_freq / fmt.output_rate;

    // FSC bin in FFT
    int fsc_bin = (int)(fmt.fsc / fmt.output_rate * fft_size + 0.5);

    // Bandwidth: ~500 kHz
    int bandwidth_bins = (int)(500000.0 / fmt.output_rate * fft_size + 0.5);
    if (bandwidth_bins < 10) bandwidth_bins = 10;

    // Burst window (samples at output rate, from VideoFormat)
    int burst_start = (int)(fmt.burst_start_us * 1e-6 * fmt.output_rate + 0.5);
    int burst_end   = (int)(fmt.burst_end_us * 1e-6 * fmt.output_rate + 0.5);

    // Determine chunk size from available VRAM
    // Each line needs: fft_size × 8 (het buf) + freq_bins × 16 (fft buf)
    size_t bytes_per_line = (size_t)fft_size * sizeof(double)
                          + (size_t)freq_bins * sizeof(cufftDoubleComplex);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Use at most 80% of free VRAM for chroma temp buffers
    size_t usable = (size_t)(free_mem * 0.8);

    int max_lines = (int)(usable / bytes_per_line);
    if (max_lines < output_field_lines) max_lines = output_field_lines;  // at least 1 field

    // Cap chunk size: cuFFT batched plans have practical limits on batch count,
    // and cuFFT workspace memory isn't accounted for in our VRAM estimate.
    // 4096 lines (~15 NTSC fields) balances throughput vs plan overhead.
    int max_chunk_lines = 4096;
    max_chunk_lines = (max_chunk_lines / output_field_lines) * output_field_lines;
    if (max_chunk_lines < output_field_lines) max_chunk_lines = output_field_lines;
    if (max_lines > max_chunk_lines) max_lines = max_chunk_lines;

    int total_lines = num_fields * output_field_lines;
    int chunk_lines = std::min(max_lines, total_lines);
    // Round down to multiple of output_field_lines for clean field boundaries
    chunk_lines = (chunk_lines / output_field_lines) * output_field_lines;
    if (chunk_lines < output_field_lines) chunk_lines = output_field_lines;

    // Allocate temp buffers for one chunk
    double* d_het_buf = nullptr;
    cufftDoubleComplex* d_fft_buf = nullptr;

    size_t het_bytes = (size_t)chunk_lines * fft_size * sizeof(double);
    size_t fft_bytes = (size_t)chunk_lines * freq_bins * sizeof(cufftDoubleComplex);

    cudaError_t err;
    err = cudaMalloc(&d_het_buf, het_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "  [chroma] Failed to alloc het buffer (%zu MB): %s\n",
                het_bytes / (1024*1024), cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_fft_buf, fft_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "  [chroma] Failed to alloc FFT buffer (%zu MB): %s\n",
                fft_bytes / (1024*1024), cudaGetErrorString(err));
        cudaFree(d_het_buf);
        return;
    }

    // Create cuFFT plans for chunk_lines
    cufftHandle plan_r2c = 0, plan_c2r = 0;
    int n[] = { fft_size };

    cufftResult cufft_err;
    cufft_err = cufftPlanMany(&plan_r2c, 1, n,
        NULL, 1, fft_size,
        NULL, 1, freq_bins,
        CUFFT_D2Z, chunk_lines);

    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "  [chroma] cuFFT R2C plan failed: %d\n", cufft_err);
        cudaFree(d_het_buf); cudaFree(d_fft_buf);
        return;
    }

    cufft_err = cufftPlanMany(&plan_c2r, 1, n,
        NULL, 1, freq_bins,
        NULL, 1, fft_size,
        CUFFT_Z2D, chunk_lines);

    if (cufft_err != CUFFT_SUCCESS) {
        fprintf(stderr, "  [chroma] cuFFT C2R plan failed: %d\n", cufft_err);
        cufftDestroy(plan_r2c);
        cudaFree(d_het_buf); cudaFree(d_fft_buf);
        return;
    }

    double fft_scale = 1.0 / (double)fft_size;
    int fields_per_chunk = chunk_lines / output_field_lines;

    // Process in chunks
    for (int field_start = 0; field_start < num_fields; field_start += fields_per_chunk) {
        int fields_this = std::min(fields_per_chunk, num_fields - field_start);
        int lines_this = fields_this * output_field_lines;

        // If last chunk is smaller, we need new cuFFT plans
        // (cuFFT batch size must match). Recreate only if needed.
        if (lines_this < chunk_lines) {
            cufftDestroy(plan_r2c);
            cufftDestroy(plan_c2r);

            cufftPlanMany(&plan_r2c, 1, n,
                NULL, 1, fft_size, NULL, 1, freq_bins,
                CUFFT_D2Z, lines_this);
            cufftPlanMany(&plan_c2r, 1, n,
                NULL, 1, freq_bins, NULL, 1, fft_size,
                CUFFT_Z2D, lines_this);
        }

        // Phase 1: TBC resample raw + heterodyne
        {
            int total_samples = lines_this * fft_size;
            int threads = 256;
            int blocks = (total_samples + threads - 1) / threads;
            k_resample_raw_het<<<blocks, threads>>>(
                d_raw, d_linelocs, d_het_buf,
                lines_this, field_start, output_field_lines,
                fmt.lines_per_frame, output_line_len, fft_size,
                fmt.active_line_start, total_raw_samples, het_scale,
                0);  // track phase guess (0 or 1) — TODO: auto-detect
        }

        // Phase 2: Forward FFT
        cufftExecD2Z(plan_r2c, d_het_buf, d_fft_buf);

        // Phase 3: Bandpass at fsc
        {
            int total = lines_this * freq_bins;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            k_chroma_bandpass<<<blocks, threads>>>(
                d_fft_buf, lines_this, freq_bins, fsc_bin, bandwidth_bins);
        }

        // Phase 4: Inverse FFT
        cufftExecZ2D(plan_c2r, d_fft_buf, d_het_buf);

        // Phase 5: ACC + output
        // Output pointer offset to the right position in d_tbc_chroma
        uint16_t* out_ptr = const_cast<uint16_t*>(
            d_tbc_chroma + (size_t)field_start * output_field_lines * output_line_len);

        k_chroma_acc_output<<<lines_this, 256>>>(
            d_het_buf, out_ptr,
            lines_this, output_line_len, fft_size,
            burst_start, burst_end,
            fmt.burst_abs_ref, fft_scale);
    }

    // Cleanup
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    cudaFree(d_het_buf);
    cudaFree(d_fft_buf);
}
