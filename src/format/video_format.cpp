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
        chroma_under     = 629000.0;
        luma_carrier     = 3900000.0;   // ~3.4-4.4 MHz typical VHS NTSC
        burst_abs_ref    = 4416.0;
        output_line_len  = 910;
        output_field_lines = 263;
    } else {
        lines_per_frame  = 625;
        lines_per_field  = 313;    // alternates 312/313
        line_rate        = 15625.0;
        field_rate       = 50.0;
        frame_rate       = 25.0;
        fsc              = 4433618.75;
        chroma_under     = 626000.0;
        luma_carrier     = 3800000.0;   // ~3.4-4.4 MHz typical VHS PAL
        burst_abs_ref    = 4416.0;      // TODO: confirm PAL value
        output_line_len  = 1135;
        output_field_lines = 313;
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
}
