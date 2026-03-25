#pragma once
#include <cufft.h>
#include "format/video_format.h"

// Persistent state for FM demodulation.
// Created once at pipeline init, reused for every batch.
struct FMDemodState {
    // cuFFT plans (batched)
    cufftHandle plan_r2c = 0;       // real -> complex (forward)
    cufftHandle plan_c2c_inv = 0;   // complex -> complex (inverse, for analytic signal)
    cufftHandle plan_c2r = 0;       // complex -> real (inverse, for post-demod filtering)

    // FFT geometry
    int fft_size = 0;               // = samples_per_field (no padding in v1)
    int freq_bins = 0;              // fft_size / 2 + 1 (complex output size)
    int max_batch = 0;              // max fields per batch (plans are created for this)

    // Pre-computed frequency-domain filter arrays (device memory)
    double*              d_rf_filter = nullptr;   // freq_bins reals: RF bandpass mag x Hilbert weight
    cufftDoubleComplex*  d_fvideo    = nullptr;   // freq_bins complex: deemph x video LPF (pre-scaled 1/N)
    cufftDoubleComplex*  d_fvideo05  = nullptr;   // freq_bins complex: deemph x video LPF x sync LPF (pre-scaled 1/N)

    // Scratch buffers (device memory, sized for max_batch)
    cufftDoubleComplex*  d_fft_half  = nullptr;   // max_batch x freq_bins — R2C output / post-demod FFT
    cufftDoubleComplex*  d_analytic  = nullptr;   // max_batch x fft_size  — full spectrum + C2C output
    cufftDoubleComplex*  d_post_fft  = nullptr;   // max_batch x freq_bins — filtered spectrum for C2R

    int f05_offset = 32;            // sync FIR group delay (samples)

    bool init(const VideoFormat& fmt, int max_batch_size, int samples_per_field_override = 0);
    void destroy();
    ~FMDemodState() { destroy(); }

    // How many extra bytes per field this kernel needs (for pipeline VRAM budgeting)
    static size_t scratch_bytes_per_field(int samples_per_field);
};

// Initialize FM demod state. Call once at pipeline startup.
bool fm_demod_init(FMDemodState& state, const VideoFormat& fmt, int max_batch);

// Run FM demodulation on a batch of fields.
//
// Input:  d_raw[num_fields * samples_per_field]      — normalized raw RF (float64)
// Output: d_demod[num_fields * samples_per_field]    — demodulated video (float64, Hz)
//         d_demod_05[num_fields * samples_per_field] — demodulated sync  (float64, Hz)
void fm_demod(FMDemodState& state,
              const double* d_raw,
              double* d_demod,
              double* d_demod_05,
              int num_fields,
              size_t samples_per_field,
              const VideoFormat& fmt);
