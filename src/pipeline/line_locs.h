#pragma once
#include "format/video_format.h"
#include "pipeline/sync_pulses.h"   // MAX_PULSES

// Pulse type codes (matches Python lddecode/core.py)
enum PulseType : int { PULSE_HSYNC = 0, PULSE_EQ1 = 1, PULSE_VSYNC = 2, PULSE_EQ2 = 3 };

// Kernel 3: Classify Pulses + Compute Line Locations
//
// Three-phase algorithm (all GPU):
//
//   Phase 1: Classify each pulse by width (parallel, one thread per pulse)
//            -> d_pulse_types[field * MAX_PULSES + i]
//
//   Phase 2: Per-field sequential scan (one thread per field):
//            - Run vblank state machine to validate pulse sequence
//            - Find longest HSYNC run -> compute mean line length
//            - Locate first HSYNC after VSYNC -> reference pulse
//
//   Phase 3: Assign line locations (one thread per field, sequential scan):
//            For each output line, compute expected position from reference,
//            then snap to nearest real HSYNC pulse within tolerance.
//            -> d_linelocs[field * lines_per_frame]
//
// Output layout:
//   d_linelocs[field * lines_per_frame + line] = sample position of line start
//
void line_locs(const int* d_pulse_starts,
               const int* d_pulse_lengths,
               const int* d_pulse_count,
               int* d_pulse_types,
               double* d_linelocs,
               int num_fields,
               const VideoFormat& fmt);
