#include "format/video_format.h"
#include <cstdio>
#include <cmath>

const char* input_format_name(InputFormat fmt) {
    switch (fmt) {
        case InputFormat::U8:  return "u8";
        case InputFormat::S16: return "s16";
        case InputFormat::U16: return "u16";
    }
    return "unknown";
}

int input_format_bytes_per_sample(InputFormat fmt) {
    switch (fmt) {
        case InputFormat::U8:  return 1;
        case InputFormat::S16: return 2;
        case InputFormat::U16: return 2;
    }
    return 1;
}

VideoFormat::VideoFormat(VideoSystem sys, double sample_rate_mhz)
    : system(sys), sample_rate(sample_rate_mhz * 1e6)
{
    if (sys == VideoSystem::NTSC) {
        lines_per_frame  = 525;
        lines_per_field  = 263;    // alternates 262/263
        line_rate        = 15734.264;
        field_rate       = 59.94;
        frame_rate       = 29.97;
        fsc              = 3579545.0;
        chroma_under     = (525.0 * (30.0 / 1.001)) * 40.0;  // 629370.6 Hz (40 × fH)
        luma_carrier     = 3900000.0;   // ~3.4-4.4 MHz typical VHS NTSC
        burst_abs_ref    = 4416.0;
        burst_start_us   = 5.3;
        burst_end_us     = 7.8;
        output_line_len  = 910;
        output_field_lines = 263;
        num_eq_pulses    = 6;
        field_lines_first  = 263;
        field_lines_second = 262;

        // VHS NTSC IRE mapping (from vhsdecode format_defs/vhs.py)
        hz_ire    = 1e6 / 140.0;                    // ~7142.857 Hz/IRE
        ire0      = 4.4e6 - (hz_ire * 100.0);       // ~3,685,714 Hz at 0 IRE
        vsync_ire = -40.0;                           // sync tip at -40 IRE
    } else {
        lines_per_frame  = 625;
        lines_per_field  = 313;    // alternates 312/313
        line_rate        = 15625.0;
        field_rate       = 50.0;
        frame_rate       = 25.0;
        fsc              = 4433618.75;
        chroma_under     = ((625.0 * 25.0) * 40.0) + 1953.0;  // 626953 Hz (40 × fH + 1953)
        luma_carrier     = 3800000.0;   // ~3.4-4.4 MHz typical VHS PAL
        burst_abs_ref    = 5000.0;
        burst_start_us   = 5.6;
        burst_end_us     = 7.85;
        output_line_len  = 1135;
        output_field_lines = 313;
        num_eq_pulses    = 5;
        field_lines_first  = 312;
        field_lines_second = 313;

        // VHS PAL IRE mapping (from vhsdecode format_defs/vhs.py)
        vsync_ire = -0.3 * (100.0 / 0.7);           // ~-42.857 IRE
        hz_ire    = 1e6 / (100.0 + (-vsync_ire));    // ~7000 Hz/IRE
        ire0      = 4.8e6 - (hz_ire * 100.0);        // ~4,100,000 Hz at 0 IRE
    }

    // Derived sync levels
    sync_tip_hz        = ire0 + hz_ire * vsync_ire;
    pulse_threshold_hz = ire0 + hz_ire * (-20.0);    // -20 IRE: halfway between sync and blanking

    // TBC output scaling (ld-tools compatible uint16 range)
    // Formula: output = clip((ire_shifted * output_scale) + output_zero, 0, 65535)
    // where ire_shifted = (hz - ire0) / hz_ire - vsync_ire
    if (sys == VideoSystem::NTSC) {
        output_zero  = 1024.0;                          // 0x0400
        output_scale = (51200.0 - 1024.0) / (100.0 - vsync_ire);  // (0xC800 - 0x0400) / IRE range
        active_line_start = 10;
    } else {
        output_zero  = 256.0;                           // 0x0100
        output_scale = (54016.0 - 256.0) / (100.0 - vsync_ire);   // (0xD300 - 0x0100) / IRE range
        active_line_start = 8;  // PAL has fewer EQ pulses
    }

    output_rate = 4.0 * fsc;

    samples_per_line  = static_cast<int>(round(sample_rate / line_rate));
    samples_per_field = samples_per_line * lines_per_field;

    // Sync pulse widths in samples
    hsync_width    = 4.7e-6 * sample_rate;
    eq_pulse_width = 2.3e-6 * sample_rate;
    bp_width       = 1.5e-6 * sample_rate;

    if (sys == VideoSystem::NTSC) {
        vsync_width = 27.1e-6 * sample_rate;
    } else {
        vsync_width = 27.3e-6 * sample_rate;
    }
}

void VideoFormat::print_info() const {
    const char* sys_name = (system == VideoSystem::NTSC) ? "NTSC" : "PAL";
    fprintf(stderr, "Format: %s @ %.1f MHz capture rate\n", sys_name, sample_rate / 1e6);
    fprintf(stderr, "  Lines/frame: %d  Line rate: %.3f Hz\n", lines_per_frame, line_rate);
    fprintf(stderr, "  Samples/line: %d  Samples/field: %d\n", samples_per_line, samples_per_field);
    fprintf(stderr, "  Output: %d samples/line @ %.6f MHz (4*fsc)\n",
            output_line_len, output_rate / 1e6);
    fprintf(stderr, "  Chroma-under: %.0f Hz  Fsc: %.0f Hz\n", chroma_under, fsc);
    fprintf(stderr, "  IRE: 0=%.0f Hz  sync_tip=%.0f Hz  threshold=%.0f Hz  (%.1f Hz/IRE)\n",
            ire0, sync_tip_hz, pulse_threshold_hz, hz_ire);
}
