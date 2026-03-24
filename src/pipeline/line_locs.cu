#include "pipeline/line_locs.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Phase 1: Classify pulses by width (one thread per pulse across all fields)
// ============================================================================
//
// Pulse width ranges (matching Python lddecode/core.py get_timings()):
//   HSYNC: hsync_width ± 0.5 µs  (nominal 4.7 µs)
//   EQ:    eq_width ± 0.5 µs     (nominal 2.3 µs NTSC, 2.35 µs PAL)
//   VSYNC: vsync_width*0.5 .. vsync_width + 1 µs  (nominal 27.1/27.3 µs)
//
// Unclassified pulses get PULSE_HSYNC as fallback (matches Python behavior
// where unrecognized pulses are ignored by the state machine).

__global__ void k_classify_pulses(
    const int* pulse_lengths,       // [num_fields x MAX_PULSES]
    const int* pulse_count,         // [num_fields]
    int* pulse_types,               // [num_fields x MAX_PULSES] output
    int max_pulses,
    int num_fields,
    double hsync_min, double hsync_max,
    double eq_min, double eq_max,
    double vsync_min, double vsync_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int field = idx / max_pulses;
    int pulse = idx % max_pulses;

    if (field >= num_fields) return;
    if (pulse >= pulse_count[field]) return;

    double len = (double)pulse_lengths[field * max_pulses + pulse];

    int type;
    if (len >= vsync_min && len <= vsync_max) {
        type = PULSE_VSYNC;
    } else if (len >= hsync_min && len <= hsync_max) {
        type = PULSE_HSYNC;
    } else if (len >= eq_min && len <= eq_max) {
        // EQ1 or EQ2 — will be refined by state machine in phase 2
        type = PULSE_EQ1;
    } else {
        // Unclassified: default to HSYNC
        type = PULSE_HSYNC;
    }

    pulse_types[field * max_pulses + pulse] = type;
}

// ============================================================================
// Phase 2 + 3: Per-field processing (one thread per field)
//
// - Run simplified vblank state machine to refine EQ1/EQ2 classification
// - Find longest HSYNC run -> mean line length
// - Locate first HSYNC after VSYNC -> reference pulse
// - Assign line locations with nearest-pulse matching
// ============================================================================

__global__ void k_compute_linelocs(
    const int* pulse_starts,        // [num_fields x MAX_PULSES]
    const int* pulse_lengths,       // [num_fields x MAX_PULSES]
    const int* pulse_count,         // [num_fields]
    const int* pulse_types,         // [num_fields x MAX_PULSES]
    double* linelocs,               // [num_fields x lines_per_frame] output
    int max_pulses,
    int num_fields,
    int lines_per_frame,
    int samples_per_line,           // nominal line length in samples
    double hsync_min, double hsync_max)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    const int* starts = pulse_starts + field * max_pulses;
    const int* types  = pulse_types  + field * max_pulses;
    double* locs      = linelocs + field * lines_per_frame;
    int npc           = pulse_count[field];

    double inlinelen  = (double)samples_per_line;

    // -----------------------------------------------------------------
    // Step 1: Run vblank state machine
    //
    // Simplified version of Python run_vblank_state_machine():
    // Track state transitions: HSYNC -> EQ1 -> VSYNC -> EQ2 -> HSYNC
    // Record the index of the first HSYNC after the first EQ2 section.
    // -----------------------------------------------------------------

    int ref_pulse_idx = -1;     // first HSYNC after VSYNC/EQ2
    int state = -1;             // -1 = initial

    for (int i = 0; i < npc; i++) {
        int t = types[i];

        if (state == -1) {
            // Need a HSYNC to start
            if (t == PULSE_HSYNC) state = PULSE_HSYNC;
        } else if (state == PULSE_HSYNC) {
            if (t == PULSE_HSYNC) {
                // stay
            } else if (t == PULSE_EQ1) {
                state = PULSE_EQ1;
            } else if (t == PULSE_VSYNC) {
                state = PULSE_VSYNC;
            }
        } else if (state == PULSE_EQ1) {
            if (t == PULSE_EQ1) {
                // stay
            } else if (t == PULSE_VSYNC) {
                state = PULSE_VSYNC;
            } else if (t == PULSE_HSYNC) {
                // false alarm, back to HSYNC
                state = PULSE_HSYNC;
            }
        } else if (state == PULSE_VSYNC) {
            if (t == PULSE_VSYNC) {
                // stay
            } else if (t == PULSE_EQ1) {
                // VSYNC -> EQ2 transition
                state = PULSE_EQ2;
            } else if (t == PULSE_HSYNC) {
                // Direct VSYNC -> HSYNC (unusual, but accept it)
                ref_pulse_idx = i;
                break;
            }
        } else if (state == PULSE_EQ2) {
            if (t == PULSE_EQ1) {
                // stay in EQ2
            } else if (t == PULSE_HSYNC) {
                // Found the reference pulse
                ref_pulse_idx = i;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Step 2: Compute mean line length from longest HSYNC run
    //
    // Find the longest consecutive run of HSYNC pulses, then average
    // their spacing. This gives a robust estimate even with noisy signals.
    // Matches Python computeLineLen().
    // -----------------------------------------------------------------

    int best_run_start = -1, best_run_len = 0;
    int cur_run_start = -1, cur_run_len = 0;

    for (int i = 0; i < npc; i++) {
        if (types[i] == PULSE_HSYNC) {
            if (cur_run_start == -1) {
                cur_run_start = i;
                cur_run_len = 1;
            } else {
                cur_run_len++;
            }
        } else {
            if (cur_run_len > best_run_len) {
                best_run_start = cur_run_start;
                best_run_len = cur_run_len;
            }
            cur_run_start = -1;
            cur_run_len = 0;
        }
    }
    if (cur_run_len > best_run_len) {
        best_run_start = cur_run_start;
        best_run_len = cur_run_len;
    }

    // Average HSYNC-to-HSYNC spacing within the best run
    double meanlinelen = inlinelen;  // fallback
    if (best_run_len >= 2) {
        double sum = 0.0;
        int count = 0;
        for (int i = best_run_start + 1; i < best_run_start + best_run_len; i++) {
            double spacing = (double)(starts[i] - starts[i - 1]);
            // Only include spacings within 5% of nominal (matches Python)
            if (spacing >= inlinelen * 0.95 && spacing <= inlinelen * 1.05) {
                sum += spacing;
                count++;
            }
        }
        if (count > 0) {
            meanlinelen = sum / count;
        }
    }

    // -----------------------------------------------------------------
    // Step 3: Determine reference pulse and line number
    //
    // If state machine found a valid ref_pulse_idx, use it.
    // The reference line is hsync_start_line (first active line after vblank).
    // For NTSC first field: EQ2_end + 0.5 lines ≈ line 10.
    // Simplified: use a fixed offset based on system.
    // -----------------------------------------------------------------

    double ref_position;
    double ref_line;

    if (ref_pulse_idx >= 0) {
        ref_position = (double)starts[ref_pulse_idx];
        // The reference pulse is the first HSYNC after vblank.
        // For NTSC: this is approximately line 10 (3 EQ sections × 3 lines each + 1)
        // For PAL: approximately line 8
        // Use num_eq_pulses to compute: 3 sections × (num_eq/2) half-lines ≈
        //   num_eq * 1.5 lines, plus some offsets
        // Simplified: count from 0, treating ref as the first active video line.
        // We'll number lines starting from line 0 = last HSYNC of previous field.
        // The reference is at hsync_start_line.
        ref_line = 10.0;  // approximate; will be refined later
    } else {
        // No VSYNC found: use first HSYNC pulse as a rough reference
        // Find the first HSYNC
        ref_pulse_idx = 0;
        for (int i = 0; i < npc; i++) {
            if (types[i] == PULSE_HSYNC) {
                ref_pulse_idx = i;
                break;
            }
        }
        ref_position = (double)starts[ref_pulse_idx];
        ref_line = 0.0;
    }

    // -----------------------------------------------------------------
    // Step 4: Assign line locations
    //
    // For each output line, compute expected position from reference,
    // then find nearest HSYNC pulse within tolerance.
    // Matches Python valid_pulses_to_linelocs() from sync.pyx.
    //
    // Only match against HSYNC pulses (sorted by position, which they
    // already are since K2 scans sequentially).
    // -----------------------------------------------------------------

    // Build a compact list of HSYNC-only pulse starts for matching
    // (skip EQ/VSYNC pulses since they're at half-line intervals)
    int hsync_starts[800];  // local array — fits in registers/local mem
    int num_hsyncs = 0;
    for (int i = 0; i < npc && num_hsyncs < 800; i++) {
        if (types[i] == PULSE_HSYNC) {
            hsync_starts[num_hsyncs++] = starts[i];
        }
    }

    double max_allowed_distance = meanlinelen / 1.5;
    int cur_hsync = 0;  // monotonic cursor into hsync_starts

    for (int line = 0; line < lines_per_frame; line++) {
        // Expected position
        double expected = ref_position + meanlinelen * (line - ref_line);
        locs[line] = expected;

        if (cur_hsync >= num_hsyncs) continue;

        // Search forward for nearest HSYNC within tolerance
        double best_dist = max_allowed_distance;
        int best_pos = -1;
        int search = cur_hsync;

        while (search < num_hsyncs) {
            double dist = fabs((double)hsync_starts[search] - expected);
            if (dist <= best_dist) {
                best_dist = dist;
                best_pos = hsync_starts[search];
                cur_hsync = search;
            }
            // Check if next pulse is farther away (we've passed minimum)
            if (search + 1 < num_hsyncs) {
                double next_dist = fabs((double)hsync_starts[search + 1] - expected);
                if (next_dist > dist) break;
            } else {
                break;
            }
            search++;
        }

        if (best_pos >= 0) {
            locs[line] = (double)best_pos;
            cur_hsync++;  // consume this pulse
        }
    }
}


// ============================================================================
// Host-side entry point
// ============================================================================

void line_locs(const int* d_pulse_starts,
               const int* d_pulse_lengths,
               const int* d_pulse_count,
               double* d_linelocs,
               int num_fields,
               const VideoFormat& fmt)
{
    // Allocate temporary pulse type classification buffer
    int* d_pulse_types = nullptr;
    cudaMalloc(&d_pulse_types, (size_t)num_fields * MAX_PULSES * sizeof(int));

    // Compute classification thresholds (in samples)
    // Tolerance: ±0.5 µs for HSYNC and EQ, wider range for VSYNC
    double tol_samples = 0.5e-6 * fmt.sample_rate;

    double hsync_min = fmt.hsync_width - tol_samples;
    double hsync_max = fmt.hsync_width + tol_samples;
    double eq_min    = fmt.eq_pulse_width - tol_samples;
    double eq_max    = fmt.eq_pulse_width + tol_samples;
    double vsync_min = fmt.vsync_width * 0.5;                    // very generous lower bound
    double vsync_max = fmt.vsync_width + 1.0e-6 * fmt.sample_rate;  // +1 µs upper

    // Phase 1: classify pulses (one thread per pulse slot)
    {
        int total = num_fields * MAX_PULSES;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_classify_pulses<<<blocks, threads>>>(
            d_pulse_lengths, d_pulse_count, d_pulse_types,
            MAX_PULSES, num_fields,
            hsync_min, hsync_max,
            eq_min, eq_max,
            vsync_min, vsync_max);
    }

    // Phase 2+3: per-field state machine + mean linelen + line assignment
    {
        int threads = 256;
        int blocks = (num_fields + threads - 1) / threads;
        k_compute_linelocs<<<blocks, threads>>>(
            d_pulse_starts, d_pulse_lengths, d_pulse_count,
            d_pulse_types, d_linelocs,
            MAX_PULSES, num_fields,
            fmt.lines_per_frame, fmt.samples_per_line,
            hsync_min, hsync_max);
    }

    cudaFree(d_pulse_types);
}
