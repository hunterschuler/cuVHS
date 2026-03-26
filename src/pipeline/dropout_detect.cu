#include "pipeline/dropout_detect.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Detection parameters (matching Python vhs-decode defaults)
#define DOD_THRESHOLD_P   0.18    // 18% of field mean envelope
#define DOD_HYSTERESIS    1.25    // up_threshold = down_threshold × 1.25
#define DOD_MERGE_DIST    30      // merge dropouts closer than 30 RF samples
#define DOD_MIN_LENGTH    10      // discard dropouts shorter than 10 RF samples
#define ENV_BLOCK_SIZE    16      // local RMS block size (samples)

// ============================================================================
// Kernel 1: Compute mean envelope amplitude per field (parallel reduction)
// ============================================================================

__global__ void k_field_mean_env(
    const double* __restrict__ envelope,
    double* __restrict__ field_mean_env,
    int num_fields,
    int samples_per_field)
{
    int field = blockIdx.x;
    if (field >= num_fields) return;

    const double* field_env = envelope + (size_t)field * samples_per_field;

    // Each thread accumulates its portion
    double sum = 0.0;
    for (int i = threadIdx.x; i < samples_per_field; i += blockDim.x) {
        sum += field_env[i];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block-level reduction via shared memory
    __shared__ double warp_sums[8];  // up to 256 threads = 8 warps
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double total = 0.0;
        int nwarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < nwarps; w++)
            total += warp_sums[w];
        field_mean_env[field] = total / (double)samples_per_field;
    }
}

// ============================================================================
// Kernel 2: Detect dropouts + map to TBC positions (one thread per field)
// ============================================================================

__device__ int find_tbc_line(const double* linelocs, int als, int n_lines, double rf_pos) {
    if (rf_pos < linelocs[als] || rf_pos >= linelocs[als + n_lines])
        return -1;

    int lo = 0, hi = n_lines - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (linelocs[als + mid] <= rf_pos)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

__global__ void k_detect_and_map(
    const double* __restrict__ envelope,
    const double* __restrict__ linelocs,
    const double* __restrict__ field_mean_env,
    int* __restrict__ do_lines,
    int* __restrict__ do_starts,
    int* __restrict__ do_ends,
    int* __restrict__ do_count,
    int num_fields,
    int samples_per_field,
    int lines_per_frame,
    int output_field_lines,
    int output_line_len,
    int active_line_start)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    const double* field_ll = linelocs + (size_t)field * lines_per_frame;
    int als = active_line_start;
    int max_do = MAX_DROPOUTS_PER_FIELD;

    int* my_lines  = do_lines  + field * max_do;
    int* my_starts = do_starts + field * max_do;
    int* my_ends   = do_ends   + field * max_do;

    // Compute thresholds from mean envelope (no sqrt needed)
    double mean_env = field_mean_env[field];
    double down_thresh = mean_env * DOD_THRESHOLD_P;
    double up_thresh = down_thresh * DOD_HYSTERESIS;

    // Determine RF scan range from linelocs (active video region only)
    int field_start = field * samples_per_field;
    int scan_start = (int)field_ll[als];
    scan_start = max(scan_start, field_start);
    int scan_end = min((int)field_ll[als + output_field_lines],
                       field_start + samples_per_field);
    if (scan_end <= scan_start) {
        do_count[field] = 0;
        return;
    }

    // Phase 1: Detect dropouts in RF domain using block-based RMS + hysteresis
    // Use temporary storage in output arrays (will overwrite with TBC-mapped entries)
    int* rf_starts_buf = my_lines;
    int* rf_ends_buf = my_starts;

    int n_rf_do = 0;
    int rf_do_start = -1;
    const int MAX_RF_DO = 128;

    int num_blocks = (scan_end - scan_start + ENV_BLOCK_SIZE - 1) / ENV_BLOCK_SIZE;

    for (int blk = 0; blk < num_blocks; blk++) {
        int blk_start = scan_start + blk * ENV_BLOCK_SIZE;
        int blk_end = min(blk_start + ENV_BLOCK_SIZE, scan_end);

        // Compute block mean envelope
        double block_sum = 0.0;
        int blk_len = blk_end - blk_start;
        for (int i = blk_start; i < blk_end; i++) {
            block_sum += envelope[i];
        }
        double block_mean_env = block_sum / (double)blk_len;

        if (block_mean_env <= down_thresh) {
            // Below threshold — start or continue dropout
            if (rf_do_start < 0) {
                rf_do_start = blk_start;
            }
        } else if (block_mean_env >= up_thresh) {
            // Above threshold — end dropout if active
            if (rf_do_start >= 0) {
                int do_end = blk_start;

                // Merge with previous if close enough
                if (n_rf_do > 0 && rf_do_start - rf_ends_buf[n_rf_do - 1] <= DOD_MERGE_DIST) {
                    rf_ends_buf[n_rf_do - 1] = do_end;
                } else if (n_rf_do < MAX_RF_DO) {
                    rf_starts_buf[n_rf_do] = rf_do_start;
                    rf_ends_buf[n_rf_do] = do_end;
                    n_rf_do++;
                }
                rf_do_start = -1;
            }
        }
    }

    // Close any open dropout at end of scan
    if (rf_do_start >= 0) {
        if (n_rf_do > 0 && rf_do_start - rf_ends_buf[n_rf_do - 1] <= DOD_MERGE_DIST) {
            rf_ends_buf[n_rf_do - 1] = scan_end;
        } else if (n_rf_do < MAX_RF_DO) {
            rf_starts_buf[n_rf_do] = rf_do_start;
            rf_ends_buf[n_rf_do] = scan_end;
            n_rf_do++;
        }
    }

    // Filter out short dropouts (post-merge)
    int n_filtered = 0;
    for (int i = 0; i < n_rf_do; i++) {
        if (rf_ends_buf[i] - rf_starts_buf[i] >= DOD_MIN_LENGTH) {
            rf_starts_buf[n_filtered] = rf_starts_buf[i];
            rf_ends_buf[n_filtered] = rf_ends_buf[i];
            n_filtered++;
        }
    }
    n_rf_do = n_filtered;

    // Phase 2: Map RF dropouts to TBC line/column positions
    // Copy RF data to local memory first (we'll overwrite output arrays)
    int local_rf_starts[128];
    int local_rf_ends[128];
    for (int i = 0; i < n_rf_do; i++) {
        local_rf_starts[i] = rf_starts_buf[i];
        local_rf_ends[i] = rf_ends_buf[i];
    }

    int tbc_count = 0;
    int n_active = output_field_lines;

    for (int i = 0; i < n_rf_do && tbc_count < max_do; i++) {
        double start_rf = (double)local_rf_starts[i];
        double end_rf = (double)local_rf_ends[i];

        // Find TBC line containing start of dropout
        int start_line = find_tbc_line(field_ll, als, n_active, start_rf);
        if (start_line < 0) continue;

        // Find TBC line containing end of dropout
        int end_line = find_tbc_line(field_ll, als, n_active, end_rf - 1.0);
        if (end_line < 0) end_line = n_active - 1;

        // Create one TBC entry per affected line
        for (int l = start_line; l <= end_line && tbc_count < max_do; l++) {
            double ll_start = field_ll[als + l];
            double ll_end = field_ll[als + l + 1];
            double line_len_rf = ll_end - ll_start;
            if (line_len_rf <= 0.0) continue;

            int startx, endx;
            if (l == start_line) {
                double offset = start_rf - ll_start;
                startx = (int)(offset / line_len_rf * (double)output_line_len);
                startx = max(0, startx);
            } else {
                startx = 0;
            }

            if (l == end_line) {
                double offset = end_rf - ll_start;
                endx = (int)(offset / line_len_rf * (double)output_line_len + 0.999);
                endx = min(output_line_len, endx);
            } else {
                endx = output_line_len;
            }

            if (endx > startx) {
                my_lines[tbc_count] = l;
                my_starts[tbc_count] = startx;
                my_ends[tbc_count] = endx;
                tbc_count++;
            }
        }
    }

    do_count[field] = tbc_count;
}

// ============================================================================
// Kernel 3: Conceal dropout regions in TBC (replace from adjacent line)
// ============================================================================

__global__ void k_conceal(
    uint16_t* __restrict__ tbc_luma,
    uint16_t* __restrict__ tbc_chroma,
    const int* __restrict__ do_lines,
    const int* __restrict__ do_starts,
    const int* __restrict__ do_ends,
    const int* __restrict__ do_count,
    int num_fields,
    int output_line_len,
    int output_field_lines)
{
    // One block per field, threads iterate over dropout entries
    int field = blockIdx.x;
    if (field >= num_fields) return;

    int count = do_count[field];
    if (count == 0) return;

    size_t field_offset = (size_t)field * output_field_lines * output_line_len;
    uint16_t* luma = tbc_luma + field_offset;
    uint16_t* chroma = tbc_chroma + field_offset;

    const int* fl = do_lines  + field * MAX_DROPOUTS_PER_FIELD;
    const int* fs = do_starts + field * MAX_DROPOUTS_PER_FIELD;
    const int* fe = do_ends   + field * MAX_DROPOUTS_PER_FIELD;

    for (int d = threadIdx.x; d < count; d += blockDim.x) {
        int line = fl[d];
        int sx = fs[d];
        int ex = fe[d];

        // Source line: prefer line above, fall back to below
        int src_line = (line > 0) ? line - 1 : line + 1;
        if (src_line >= output_field_lines) continue;

        int src_off = src_line * output_line_len;
        int dst_off = line * output_line_len;

        for (int col = sx; col < ex; col++) {
            luma[dst_off + col] = luma[src_off + col];
            chroma[dst_off + col] = chroma[src_off + col];
        }
    }
}

// ============================================================================
// Host entry point
// ============================================================================

void dropout_detect(const double* d_envelope,
                    const double* d_linelocs,
                    uint16_t* d_tbc_luma,
                    uint16_t* d_tbc_chroma,
                    int* d_do_lines,
                    int* d_do_starts,
                    int* d_do_ends,
                    int* d_do_count,
                    int num_fields,
                    size_t samples_per_field,
                    const VideoFormat& fmt)
{
    // Allocate temp buffer for field mean envelope amplitude
    double* d_field_mean_env = nullptr;
    cudaMalloc(&d_field_mean_env, num_fields * sizeof(double));

    // Zero the dropout count
    cudaMemset(d_do_count, 0, num_fields * sizeof(int));

    // Kernel 1: Compute field mean envelope amplitude
    k_field_mean_env<<<num_fields, 256>>>(
        d_envelope, d_field_mean_env, num_fields, (int)samples_per_field);

    // Kernel 2: Detect dropouts and map to TBC positions
    {
        int threads = 64;
        int blocks = (num_fields + threads - 1) / threads;
        k_detect_and_map<<<blocks, threads>>>(
            d_envelope, d_linelocs, d_field_mean_env,
            d_do_lines, d_do_starts, d_do_ends, d_do_count,
            num_fields, (int)samples_per_field,
            fmt.lines_per_frame, fmt.output_field_lines,
            fmt.output_line_len, fmt.active_line_start);
    }

    // Kernel 3: Conceal dropout regions (replace from adjacent line)
    k_conceal<<<num_fields, 128>>>(
        d_tbc_luma, d_tbc_chroma,
        d_do_lines, d_do_starts, d_do_ends, d_do_count,
        num_fields, fmt.output_line_len, fmt.output_field_lines);

    cudaFree(d_field_mean_env);
}
