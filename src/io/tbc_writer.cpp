#include "io/tbc_writer.h"
#include <cstring>
#include <sys/stat.h>
#include <vector>

TBCWriter::~TBCWriter() {
    close();
}

bool TBCWriter::open(const std::string& output_base, const VideoFormat& format, bool overwrite) {
    fmt = format;
    luma_path = output_base + ".tbc";
    chroma_path = output_base + "_chroma.tbc";
    json_path = output_base + ".tbc.json";

    // Check for existing files
    if (!overwrite) {
        struct stat st;
        if (stat(luma_path.c_str(), &st) == 0) {
            fprintf(stderr, "Output exists: %s (use --overwrite)\n", luma_path.c_str());
            return false;
        }
    }

    luma_fp = fopen(luma_path.c_str(), "wb");
    if (!luma_fp) {
        perror(luma_path.c_str());
        return false;
    }

    chroma_fp = fopen(chroma_path.c_str(), "wb");
    if (!chroma_fp) {
        perror(chroma_path.c_str());
        fclose(luma_fp);
        luma_fp = nullptr;
        return false;
    }

    return true;
}

void TBCWriter::close() {
    if (luma_fp) { fclose(luma_fp); luma_fp = nullptr; }
    if (chroma_fp) { fclose(chroma_fp); chroma_fp = nullptr; }
}

bool TBCWriter::write_luma_field(const uint16_t* data) {
    size_t count = fmt.output_line_len * fmt.output_field_lines;
    size_t written = fwrite(data, sizeof(uint16_t), count, luma_fp);
    return written == count;
}

bool TBCWriter::write_chroma_field(const uint16_t* data) {
    size_t count = fmt.output_line_len * fmt.output_field_lines;
    size_t written = fwrite(data, sizeof(uint16_t), count, chroma_fp);
    return written == count;
}

void TBCWriter::add_dropout(int line, int start_x, int end_x) {
    current_field.dropouts.push_back({line, start_x, end_x});
}

void TBCWriter::set_first_field(bool is_first) {
    current_field.is_first_field = is_first;
}

void TBCWriter::finish_field() {
    field_meta.push_back(current_field);
    current_field = FieldMeta{};
    field_count++;
}

bool TBCWriter::finalize() {
    // Flush TBC data
    if (luma_fp) fflush(luma_fp);
    if (chroma_fp) fflush(chroma_fp);

    // Write JSON metadata (compatible with ld-tools)
    FILE* fp = fopen(json_path.c_str(), "w");
    if (!fp) {
        perror(json_path.c_str());
        return false;
    }

    const char* sys_name = (fmt.system == VideoSystem::NTSC) ? "NTSC" : "PAL";

    // Compute ld-tools compatible metadata values
    // black16bIre: uint16 value at 0 IRE (blanking level)
    // white16bIre: uint16 value at 100 IRE (peak white)
    // Formula: output = (ire_shifted * output_scale) + output_zero
    //   where ire_shifted = ire - vsync_ire (shift so sync tip = 0)
    double black16b = (0.0 - fmt.vsync_ire) * fmt.output_scale + fmt.output_zero;
    double white16b = (100.0 - fmt.vsync_ire) * fmt.output_scale + fmt.output_zero;

    // Burst and active video positions (in output samples)
    int burst_start = (int)(fmt.burst_start_us * 1e-6 * fmt.output_rate + 0.5);
    int burst_end   = (int)(fmt.burst_end_us * 1e-6 * fmt.output_rate + 0.5);

    // Active video: ~9.4 µs to ~62.5 µs for NTSC (standard values from vhs-decode)
    int active_start = (fmt.system == VideoSystem::NTSC) ? 134 : 185;
    int active_end   = (fmt.system == VideoSystem::NTSC) ? 894 : 1107;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"videoParameters\": {\n");
    fprintf(fp, "    \"system\": \"%s\",\n", sys_name);
    fprintf(fp, "    \"isSubcarrierLocked\": false,\n");
    fprintf(fp, "    \"isSourcePal\": %s,\n", (fmt.system == VideoSystem::PAL) ? "true" : "false");
    fprintf(fp, "    \"numberOfSequentialFields\": %d,\n", field_count);
    fprintf(fp, "    \"black16bIre\": %.1f,\n", black16b);
    fprintf(fp, "    \"white16bIre\": %.1f,\n", white16b);
    fprintf(fp, "    \"sampleRate\": %.0f,\n", fmt.output_rate);
    fprintf(fp, "    \"fieldWidth\": %d,\n", fmt.output_line_len);
    fprintf(fp, "    \"fieldHeight\": %d,\n", fmt.output_field_lines);
    fprintf(fp, "    \"colourBurstStart\": %d,\n", burst_start);
    fprintf(fp, "    \"colourBurstEnd\": %d,\n", burst_end);
    fprintf(fp, "    \"activeVideoStart\": %d,\n", active_start);
    fprintf(fp, "    \"activeVideoEnd\": %d,\n", active_end);
    fprintf(fp, "    \"tapeFormat\": \"VHS\",\n");
    fprintf(fp, "    \"isMapped\": false\n");
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"fields\": [\n");
    for (int i = 0; i < field_count; i++) {
        const auto& f = field_meta[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"isFirstField\": %s,\n", f.is_first_field ? "true" : "false");
        fprintf(fp, "      \"seqNo\": %d,\n", i + 1);

        fprintf(fp, "      \"dropOuts\": {\n");
        fprintf(fp, "        \"fieldLine\": [");
        for (size_t j = 0; j < f.dropouts.size(); j++) {
            if (j > 0) fprintf(fp, ", ");
            fprintf(fp, "%d", f.dropouts[j].line);
        }
        fprintf(fp, "],\n");
        fprintf(fp, "        \"startx\": [");
        for (size_t j = 0; j < f.dropouts.size(); j++) {
            if (j > 0) fprintf(fp, ", ");
            fprintf(fp, "%d", f.dropouts[j].start);
        }
        fprintf(fp, "],\n");
        fprintf(fp, "        \"endx\": [");
        for (size_t j = 0; j < f.dropouts.size(); j++) {
            if (j > 0) fprintf(fp, ", ");
            fprintf(fp, "%d", f.dropouts[j].end);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "      }\n");

        fprintf(fp, "    }%s\n", (i < field_count - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    fprintf(stderr, "Wrote %s (%d fields)\n", json_path.c_str(), field_count);
    return true;
}
