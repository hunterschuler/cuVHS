#include "pipeline/line_locs.h"
#include <cstdio>

void line_locs(const int* d_pulses,
               double* d_linelocs,
               int num_fields,
               const VideoFormat& fmt) {
    // TODO: Implement pulse classification and line location assignment
    //
    // Steps:
    // 1. Classify each pulse by width:
    //    - HSYNC: ~4.7 us (fmt.hsync_width samples)
    //    - VSYNC: ~27.1 us (fmt.vsync_width samples)
    //    - EQ pulse: ~2.3 us (fmt.eq_pulse_width samples)
    //    One thread per pulse, fully parallel.
    //
    // 2. Find field boundary: locate VSYNC → HSYNC transition
    //    Sequential scan per field (~800 pulses), one thread/warp per field.
    //
    // 3. Compute mean line length: average HSYNC-to-HSYNC spacing
    //    Parallel reduction per field.
    //
    // 4. Find reference pulse (first HSYNC after vsync)
    //    Sequential per field, trivial.
    //
    // 5. Assign line positions:
    //    expected[line] = reference_pos + line * mean_linelen
    //    For each expected position, find nearest actual HSYNC.
    //    N_fields × lines_per_frame threads.

    fprintf(stderr, "  [line_locs] STUB: %d fields\n", num_fields);
}
