#pragma once
#include "format/video_format.h"

// Kernel 1: Batched FM Demodulation
//
// For each field in the batch:
//   1. FFT forward (real → complex)
//   2. Frequency-domain bandpass + deemphasis + chroma trap
//   3. FFT inverse → baseband video
//   4. Hilbert unwrap → FM demodulation
//
// Produces two outputs:
//   demod    — demodulated video (luma carrier)
//   demod_05 — demodulated sync signal (lower frequency, for pulse detection)
//
// All N fields are processed as a batched cuFFT operation.
void fm_demod(const double* d_raw,
              double* d_demod,
              double* d_demod_05,
              int num_fields,
              size_t samples_per_field,
              const VideoFormat& fmt);
