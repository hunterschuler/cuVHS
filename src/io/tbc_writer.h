#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "format/video_format.h"

// Writes .tbc (luma), _chroma.tbc (chroma), and .tbc.json (metadata).
// TBC format: raw uint16 samples, output_line_len per line, lines_per_field per field.
struct TBCWriter {
    ~TBCWriter();

    bool open(const std::string& output_base, const VideoFormat& fmt, bool overwrite);
    void close();

    // Write one field of luma TBC data (uint16, output_line_len * output_field_lines).
    bool write_luma_field(const uint16_t* data);

    // Write one field of chroma TBC data.
    bool write_chroma_field(const uint16_t* data);

    // Record dropout for the current field.
    void add_dropout(int line, int start_x, int end_x);

    // Mark current field as first field (top) or second field (bottom).
    void set_first_field(bool is_first);

    // Set NTSC field phase ID (1-4) for the current field.
    void set_field_phase_id(int phase_id);

    // Advance to next field. Commits metadata for the current field.
    void finish_field();

    // Finalize: write .tbc.json and close files.
    bool finalize();

    int fields_written() const { return field_count; }

private:
    FILE* luma_fp = nullptr;
    FILE* chroma_fp = nullptr;
    std::string json_path;
    std::string luma_path;
    std::string chroma_path;
    VideoFormat fmt = VideoFormat(VideoSystem::NTSC, 28.0);

    int field_count = 0;

    // Per-field metadata accumulator (written to JSON at finalize)
    struct FieldMeta {
        bool is_first_field = true;
        int field_phase_id = 0;  // NTSC: 1-4, 0 = unset
        struct Dropout { int line; int start; int end; };
        std::vector<Dropout> dropouts;
    };
    std::vector<FieldMeta> field_meta;
    FieldMeta current_field;
};
