#pragma once
#include <cstdint>

enum class VideoSystem { NTSC, PAL };
enum class InputFormat { U8, S16, U16 };

const char* input_format_name(InputFormat fmt);
int input_format_bytes_per_sample(InputFormat fmt);

// All the timing/frequency constants needed by the pipeline.
// Derived at construction from system + sample_rate.
struct VideoFormat {
    VideoSystem system;
    double sample_rate;          // Hz (e.g., 28e6)

    // Line and field geometry
    int    lines_per_frame;      // 525 NTSC, 625 PAL
    int    lines_per_field;      // 262/263 NTSC, 312/313 PAL
    double line_rate;            // Hz (15734.264 NTSC, 15625 PAL)
    double field_rate;           // Hz (59.94 NTSC, 50 PAL)
    double frame_rate;           // Hz (29.97 NTSC, 25 PAL)

    // Derived sample counts
    int    samples_per_line;     // at capture sample rate
    int    samples_per_field;    // approximate

    // Sync timing (in samples at capture rate)
    double hsync_width;          // ~4.7 us
    double vsync_width;          // ~27.1 us NTSC, ~27.3 us PAL
    double eq_pulse_width;       // ~2.3 us
    double bp_width;             // back porch

    // Carrier frequencies (Hz)
    double luma_carrier;         // FM carrier center
    double chroma_under;         // color-under frequency (~629 kHz NTSC, ~626 kHz PAL)
    double fsc;                  // colorburst (3.579545 MHz NTSC, 4.43361875 MHz PAL)

    // TBC output geometry (fixed at 4*fsc)
    double output_rate;          // 4 * fsc
    int    output_line_len;      // samples per output line (910 NTSC, 1135 PAL)
    int    output_field_lines;   // lines per output field

    // Burst reference for ACC normalization
    double burst_abs_ref;        // target burst amplitude (4416 NTSC SP)

    VideoFormat(VideoSystem sys, double sample_rate_mhz);
    void print_info() const;
};
