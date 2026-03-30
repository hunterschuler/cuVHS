#pragma once
#include "format/video_format.h"

// Maximum pulses per field. NTSC has ~263 HSYNCs + ~18 EQ/VSYNC = ~281.
// 800 is generous enough for any real signal including noise pulses.
static const int MAX_PULSES = 800;

// Kernel 2: Find Sync Pulses
//
// Scans the sync-demodulated signal (demod_05, in Hz) and detects pulses
// below the sync threshold. Matches Python findpulses_numba_raw().
//
// Output layout:
//   d_pulse_starts[field * MAX_PULSES + i]  = sample position of pulse start
//   d_pulse_lengths[field * MAX_PULSES + i] = pulse length in samples
//   d_pulse_count[field]                    = number of pulses found
//
// If d_field_offsets is non-null, uses variable field offsets; otherwise assumes
// uniform stride of samples_per_field.
void sync_pulses(const double* d_demod_05,
                 int* d_pulse_starts,
                 int* d_pulse_lengths,
                 int* d_pulse_count,
                 const size_t* d_field_offsets,  // optional: per-field offsets (or nullptr for uniform stride)
                 int num_fields,
                 size_t samples_per_field,
                 const VideoFormat& fmt);
