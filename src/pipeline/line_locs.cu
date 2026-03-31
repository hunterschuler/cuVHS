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
    const int* pulse_types_in,      // [num_fields x MAX_PULSES] Phase 1 nominal classification
    double* linelocs,               // [num_fields x lines_per_frame] output
    int* is_first_field,            // [num_fields] output: 1=first, 0=second, -1=unknown
    K3Debug* k3_debug,              // [num_fields] optional debug output (may be nullptr)
    int max_pulses,
    int num_fields,
    int lines_per_frame,
    int samples_per_line,           // nominal line length in samples
    double hsync_nominal,           // nominal HSYNC width in samples
    double eq_nominal,              // nominal EQ width in samples
    double vsync_nominal,           // nominal VSYNC width in samples
    double tol_samples,             // ±0.5µs in samples
    int samples_per_field,          // approximate samples in one field
    bool is_ntsc)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    const int* starts  = pulse_starts + field * max_pulses;
    const int* lengths = pulse_lengths + field * max_pulses;
    double* locs       = linelocs + field * lines_per_frame;
    int npc            = pulse_count[field];

    double inlinelen   = (double)samples_per_line;

    // -----------------------------------------------------------------
    // Step 0: Adaptive pulse classification
    //
    // Different VHS recordings have different absolute pulse widths.
    // Measure median HSYNC width per field and shift all thresholds.
    // Matches Python vhs-decode core.py get_timings().
    // -----------------------------------------------------------------

    // Generous first pass: find HSYNC-like pulses (±1.75µs / +2µs, matching Python)
    double generous_hsync_min = hsync_nominal - 3.5 * tol_samples;  // -1.75µs
    double generous_hsync_max = hsync_nominal + 4.0 * tol_samples;  // +2.0µs

    double hsync_sum = 0.0;
    int hsync_count = 0;
    for (int i = 0; i < npc; i++) {
        double len = (double)lengths[i];
        if (len >= generous_hsync_min && len <= generous_hsync_max) {
            hsync_sum += len;
            hsync_count++;
        }
    }

    double hsync_median_est = (hsync_count > 0) ? (hsync_sum / hsync_count) : hsync_nominal;
    double hsync_offset = hsync_median_est - hsync_nominal;

    double a_hsync_min = hsync_nominal + hsync_offset - tol_samples;
    double a_hsync_max = hsync_nominal + hsync_offset + tol_samples;
    double a_eq_min    = eq_nominal + hsync_offset - tol_samples;
    double a_eq_max    = eq_nominal + hsync_offset + tol_samples;
    double a_vsync_min = (vsync_nominal + hsync_offset) * 0.5;
    double a_vsync_max = vsync_nominal + hsync_offset + 2.0 * tol_samples;

    int local_types[800];
    int npc_clamped = (npc < 800) ? npc : 800;
    for (int i = 0; i < npc_clamped; i++) {
        double len = (double)lengths[i];
        if (len >= a_vsync_min && len <= a_vsync_max) {
            local_types[i] = PULSE_VSYNC;
        } else if (len >= a_hsync_min && len <= a_hsync_max) {
            local_types[i] = PULSE_HSYNC;
        } else if (len >= a_eq_min && len <= a_eq_max) {
            local_types[i] = PULSE_EQ1;
        } else {
            local_types[i] = PULSE_HSYNC;
        }
    }
    const int* types = local_types;

    // -----------------------------------------------------------------
    // Step 1: Run vblank state machine
    //
    // Simplified version of Python run_vblank_state_machine():
    // Track state transitions: HSYNC -> EQ1 -> VSYNC -> EQ2 -> HSYNC
    // Record the index of the first HSYNC after the first EQ2 section.
    // Also track last_hsync_idx and first_eq1_idx for parity detection.
    // -----------------------------------------------------------------

    int ref_pulse_idx = -1;     // first HSYNC after VSYNC/EQ2
    int last_hsync_idx = -1;    // last HSYNC before EQ1 section
    int first_eq1_idx = -1;     // first EQ1 pulse in vblank
    int last_eq2_idx = -1;      // last EQ2 pulse before ref HSYNC (for exit spacing)
    int state = -1;             // -1 = initial

    for (int i = 0; i < npc_clamped; i++) {
        int t = types[i];

        if (state == -1) {
            // Need a HSYNC to start
            if (t == PULSE_HSYNC) { state = PULSE_HSYNC; last_hsync_idx = i; }
        } else if (state == PULSE_HSYNC) {
            if (t == PULSE_HSYNC) {
                last_hsync_idx = i;
            } else if (t == PULSE_EQ1) {
                state = PULSE_EQ1;
                first_eq1_idx = i;
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
                last_hsync_idx = i;
                first_eq1_idx = -1;
            }
        } else if (state == PULSE_VSYNC) {
            if (t == PULSE_VSYNC) {
                // stay
            } else if (t == PULSE_EQ1) {
                // VSYNC -> EQ2 transition
                state = PULSE_EQ2;
                last_eq2_idx = i;
            } else if (t == PULSE_HSYNC) {
                // Direct VSYNC -> HSYNC (unusual, but accept it)
                ref_pulse_idx = i;
                break;
            }
        } else if (state == PULSE_EQ2) {
            if (t == PULSE_EQ1) {
                // stay in EQ2
                last_eq2_idx = i;
            } else if (t == PULSE_HSYNC) {
                // Found the reference pulse
                if (last_eq2_idx < 0) last_eq2_idx = i - 1;  // fallback
                ref_pulse_idx = i;
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Step 1b: Field parity detection from VSYNC pulse pattern
    //
    // NTSC first field: HSYNC→EQ1 spacing ≈ 1.0 line
    // NTSC second field: HSYNC→EQ1 spacing ≈ 0.5 lines
    // Only trust when full vblank was found (ref_pulse_idx >= 0).
    // -----------------------------------------------------------------

    int field_parity = -1;  // -1 = unknown

    // Try entry spacing first: last HSYNC before vblank → first EQ1
    if (ref_pulse_idx >= 0 && last_hsync_idx >= 0 && first_eq1_idx >= 0) {
        double entry_spacing = (double)(starts[first_eq1_idx] - starts[last_hsync_idx]);
        double entry_lines = entry_spacing / inlinelen;
        if (is_ntsc) {
            field_parity = (entry_lines > 0.75) ? 1 : 0;  // 1=first, 0=second
        } else {
            field_parity = (entry_lines > 0.75) ? 0 : 1;
        }
    }

    // Fallback: exit spacing — last EQ2 pulse → first HSYNC after vblank
    if (field_parity < 0 && ref_pulse_idx >= 0 && last_eq2_idx >= 0) {
        double exit_spacing = (double)(starts[ref_pulse_idx] - starts[last_eq2_idx]);
        double exit_lines = exit_spacing / inlinelen;
        if (is_ntsc) {
            field_parity = (exit_lines > 0.75) ? 1 : 0;
        } else {
            field_parity = (exit_lines > 0.75) ? 0 : 1;
        }
    }

    is_first_field[field] = field_parity;

    // -----------------------------------------------------------------
    // Step 2: Compute mean line length from longest HSYNC run
    //
    // Find the longest consecutive run of HSYNC pulses, then average
    // their spacing. This gives a robust estimate even with noisy signals.
    // Matches Python computeLineLen().
    // -----------------------------------------------------------------

    int best_run_start = -1, best_run_len = 0;
    int cur_run_start = -1, cur_run_len = 0;

    for (int i = 0; i < npc_clamped; i++) {
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
        // First HSYNC after VSYNC is at line 10 in NTSC
        // (lines 0-5: VSYNC, lines 6-9: EQ/transition, line 10: first active HSYNC)
        ref_line = 10.0;

        // If the state machine found a VSYNC late in the pulse list, it likely
        // locked onto the NEXT field's VSYNC (at the end of this field's chunk)
        // instead of THIS field's VSYNC (at the beginning). This happens when
        // tape damage corrupts the leading sync pulses.
        //
        // Fix: subtract one field's worth of samples to project back to where
        // this field's VSYNC should have been. Verified: this recovers the
        // correct ref_position to within ~150 samples on known-bad fields.
        if (ref_pulse_idx > npc_clamped / 2) {
            double corrected = ref_position - (double)samples_per_field;
            // Only apply if the corrected position is still positive —
            // field 0 legitimately has its VSYNC in the middle of the chunk
            // (it starts before the first VSYNC), so don't "correct" it.
            if (corrected > 0.0) {
                ref_position = corrected;
            }
        }
    } else {
        // No VSYNC found: use first HSYNC pulse as a rough reference
        // Find the first HSYNC
        ref_pulse_idx = 0;
        for (int i = 0; i < npc_clamped; i++) {
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

    // Write debug info if buffer provided
    if (k3_debug) {
        K3Debug& dbg = k3_debug[field];
        dbg.npc = npc;
        dbg.ref_pulse_idx = ref_pulse_idx;
        dbg.ref_position = ref_position;
        dbg.ref_line = ref_line;
        dbg.meanlinelen = meanlinelen;
        dbg.best_run_len = best_run_len;
        dbg.hsync_offset = hsync_offset;
        dbg.final_state = state;
        dbg.field_parity = field_parity;
        // num_hsyncs filled below
    }

    // Build a compact list of line-marker pulses for matching.
    //
    // Some troublesome captures produce an `EQ`-width pulse at the correct
    // line cadence followed immediately by a tiny bogus `H` fragment. If we
    // skip the EQ pulse and keep the tiny H fragment, the coarse linelocs turn
    // into the long-line / short-line zig-zag seen in active picture streaks.
    //
    // To stabilize this, promote `EQ1/EQ2` pulses that are immediately
    // followed by a tiny HSYNC fragment into the HSYNC candidate list, and
    // suppress the tiny fragment itself.
    int marker_starts[800];
    int marker_lengths[800];
    int marker_count = 0;
    double tiny_hsync_max = hsync_nominal * 0.75;
    double surrogate_gap_max = inlinelen * 0.25;
    for (int i = 0; i < npc_clamped && marker_count < 800; i++) {
        bool surrogate_eq = false;
        if ((types[i] == PULSE_EQ1 || types[i] == PULSE_EQ2) &&
            i + 1 < npc_clamped &&
            types[i + 1] == PULSE_HSYNC &&
            (double)lengths[i + 1] <= tiny_hsync_max &&
            (double)(starts[i + 1] - starts[i]) <= surrogate_gap_max) {
            surrogate_eq = true;
        }

        bool suppress_tiny_hsync = false;
        if (types[i] == PULSE_HSYNC &&
            (double)lengths[i] <= tiny_hsync_max &&
            i > 0 &&
            (types[i - 1] == PULSE_EQ1 || types[i - 1] == PULSE_EQ2) &&
            (double)(starts[i] - starts[i - 1]) <= surrogate_gap_max) {
            suppress_tiny_hsync = true;
        }

        if (surrogate_eq) {
            marker_starts[marker_count] = starts[i];
            marker_lengths[marker_count] = lengths[i];
            marker_count++;
        } else if (types[i] == PULSE_HSYNC && !suppress_tiny_hsync) {
            marker_starts[marker_count] = starts[i];
            marker_lengths[marker_count] = lengths[i];
            marker_count++;
        }
    }

    // Collapse near-duplicate HSYNC markers inside the same line interval.
    // Keep the pulse whose width is closest to nominal HSYNC width; that
    // favors the full-width marker and drops tiny trailing fragments.
    int hsync_starts[800];  // local array — fits in registers/local mem
    int num_hsyncs = 0;
    double duplicate_gap_max = inlinelen * 0.75;
    int cluster_start = 0;
    while (cluster_start < marker_count && num_hsyncs < 800) {
        int best_idx = cluster_start;
        double best_score = fabs((double)marker_lengths[cluster_start] - hsync_nominal);
        int cluster_end = cluster_start + 1;
        while (cluster_end < marker_count &&
               (double)(marker_starts[cluster_end] - marker_starts[cluster_start]) < duplicate_gap_max) {
            double score = fabs((double)marker_lengths[cluster_end] - hsync_nominal);
            if (score < best_score ||
                (score == best_score && marker_starts[cluster_end] < marker_starts[best_idx])) {
                best_idx = cluster_end;
                best_score = score;
            }
            cluster_end++;
        }
        hsync_starts[num_hsyncs++] = marker_starts[best_idx];
        cluster_start = cluster_end;
    }

    if (k3_debug) k3_debug[field].num_hsyncs = num_hsyncs;

    double max_allowed_distance = meanlinelen / 1.5;
    int cur_hsync = 0;  // monotonic cursor into hsync_starts
    double worst_match_dist[8];
    int worst_match_lines[8];
    for (int i = 0; i < 8; i++) {
        worst_match_dist[i] = -1.0;
        worst_match_lines[i] = -1;
    }

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

        double match_score = (best_pos >= 0) ? best_dist : max_allowed_distance;

        if (line >= 16) {
            for (int slot = 0; slot < 8; slot++) {
                if (match_score > worst_match_dist[slot]) {
                    for (int move = 7; move > slot; move--) {
                        worst_match_dist[move] = worst_match_dist[move - 1];
                        worst_match_lines[move] = worst_match_lines[move - 1];
                    }
                    worst_match_dist[slot] = match_score;
                    worst_match_lines[slot] = line;
                    break;
                }
            }
        }
    }

    if (k3_debug) {
        K3Debug& dbg = k3_debug[field];
        dbg.bad_spacing_count = 0;
        dbg.isolated_spacing_count = 0;
        dbg.large_jump_count = 0;
        dbg.min_spacing = 1.0e30;
        dbg.max_spacing = -1.0e30;
        for (int i = 0; i < 8; i++) dbg.bad_lines[i] = -1;
        for (int i = 0; i < 8; i++) {
            dbg.worst_match_dist[i] = worst_match_dist[i];
            dbg.worst_match_lines[i] = worst_match_lines[i];
        }

        int bad_line_write = 0;
        for (int line = 0; line + 1 < lines_per_frame; line++) {
            double spacing = locs[line + 1] - locs[line];
            if (spacing < dbg.min_spacing) dbg.min_spacing = spacing;
            if (spacing > dbg.max_spacing) dbg.max_spacing = spacing;

            bool bad_spacing = (line >= lines_per_frame / 20) &&
                               (spacing < inlinelen * 0.9 || spacing > inlinelen * 1.1);
            if (bad_spacing) {
                dbg.bad_spacing_count++;
                if (bad_line_write < 8) dbg.bad_lines[bad_line_write++] = line;
            }

            if (fabs(spacing - inlinelen) > 0.25 * inlinelen) {
                dbg.large_jump_count++;
            }

            if (line > lines_per_frame / 20 && line + 2 < lines_per_frame) {
                double prev_spacing = locs[line] - locs[line - 1];
                double next_spacing = locs[line + 2] - locs[line + 1];
                double neighbor_avg = 0.5 * (prev_spacing + next_spacing);
                if (fabs(spacing - neighbor_avg) > 0.15 * inlinelen) {
                    dbg.isolated_spacing_count++;
                }
            }
        }
    }
}


// ============================================================================
// Host-side entry point
// ============================================================================

void line_locs(const int* d_pulse_starts,
               const int* d_pulse_lengths,
               const int* d_pulse_count,
               int* d_pulse_types,
               double* d_linelocs,
               int* d_is_first_field,
               int num_fields,
               const VideoFormat& fmt,
               K3Debug* d_k3_debug)
{
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

    // Phase 2+3: per-field adaptive classification + state machine + line assignment
    {
        int threads = 256;
        int blocks = (num_fields + threads - 1) / threads;
        k_compute_linelocs<<<blocks, threads>>>(
            d_pulse_starts, d_pulse_lengths, d_pulse_count,
            d_pulse_types, d_linelocs, d_is_first_field,
            d_k3_debug,
            MAX_PULSES, num_fields,
            fmt.lines_per_frame, fmt.samples_per_line,
            fmt.hsync_width, fmt.eq_pulse_width, fmt.vsync_width,
            tol_samples,
            fmt.samples_per_field,
            fmt.system == VideoSystem::NTSC);
    }
}

__global__ void k_line_locs_debug_analyze(
    const double* __restrict__ linelocs,
    int* __restrict__ bad_spacing_count,
    int* __restrict__ isolated_spacing_count,
    int* __restrict__ large_jump_count,
    int num_fields,
    int lines_per_frame,
    int active_line_start,
    double nominal_spacing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_lines = num_fields * (lines_per_frame - 1);
    if (idx >= total_lines) return;

    int field = idx / (lines_per_frame - 1);
    int line = idx % (lines_per_frame - 1);
    if (line < active_line_start || line + 1 >= lines_per_frame) return;

    int base = field * lines_per_frame;
    double spacing = linelocs[base + line + 1] - linelocs[base + line];
    if (spacing < 0.9 * nominal_spacing || spacing > 1.1 * nominal_spacing) {
        atomicAdd(bad_spacing_count, 1);
    }

    double large_jump_threshold = 0.25 * nominal_spacing;
    if (fabs(spacing - nominal_spacing) > large_jump_threshold) {
        atomicAdd(large_jump_count, 1);
    }

    if (line > active_line_start && line + 2 < lines_per_frame) {
        double prev_spacing = linelocs[base + line] - linelocs[base + line - 1];
        double next_spacing = linelocs[base + line + 2] - linelocs[base + line + 1];
        double neighbor_avg = 0.5 * (prev_spacing + next_spacing);
        if (fabs(spacing - neighbor_avg) > 0.15 * nominal_spacing) {
            atomicAdd(isolated_spacing_count, 1);
        }
    }
}

void line_locs_debug_analyze(const double* d_linelocs,
                             int* d_bad_spacing_count,
                             int* d_isolated_spacing_count,
                             int* d_large_jump_count,
                             int num_fields,
                             const VideoFormat& fmt)
{
    int total = num_fields * (fmt.lines_per_frame - 1);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_line_locs_debug_analyze<<<blocks, threads>>>(
        d_linelocs,
        d_bad_spacing_count,
        d_isolated_spacing_count,
        d_large_jump_count,
        num_fields,
        fmt.lines_per_frame,
        fmt.active_line_start,
        fmt.samples_per_line);
}
