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
        if (stat(chroma_path.c_str(), &st) == 0) {
            fprintf(stderr, "Output exists: %s (use --overwrite)\n", chroma_path.c_str());
            return false;
        }
        if (stat(json_path.c_str(), &st) == 0) {
            fprintf(stderr, "Output exists: %s (use --overwrite)\n", json_path.c_str());
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

void TBCWriter::set_field_phase_id(int phase_id) {
    current_field.field_phase_id = phase_id;
}

void TBCWriter::set_file_loc(size_t file_loc) {
    current_field.file_loc = file_loc;
}

void TBCWriter::finish_field() {
    field_meta.push_back(current_field);
    current_field = FieldMeta{};
    field_count++;
}

bool TBCWriter::write_json() {
    // Write to a temp file, then atomic rename — so the JSON is never truncated on disk.
    std::string tmp_path = json_path + ".tmp";
    FILE* fp = fopen(tmp_path.c_str(), "w");
    if (!fp) {
        perror(tmp_path.c_str());
        return false;
    }

    const char* sys_name = (fmt.system == VideoSystem::NTSC) ? "NTSC" : "PAL";

    // Compute ld-tools compatible metadata values
    double black16b = (0.0 - fmt.vsync_ire) * fmt.output_scale + fmt.output_zero;
    double white16b = (100.0 - fmt.vsync_ire) * fmt.output_scale + fmt.output_zero;

    int burst_start = (int)(fmt.burst_start_us * 1e-6 * fmt.output_rate + 0.5);
    int burst_end   = (int)(fmt.burst_end_us * 1e-6 * fmt.output_rate + 0.5);

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
        fprintf(fp, "      \"fileLoc\": %zu,\n", f.file_loc);
        if (f.field_phase_id > 0) {
            fprintf(fp, "      \"fieldPhaseID\": %d,\n", f.field_phase_id);
        }

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

    // Atomic rename — on POSIX this replaces the target in one operation
    if (rename(tmp_path.c_str(), json_path.c_str()) != 0) {
        perror("rename .tbc.json");
        return false;
    }

    return true;
}

bool TBCWriter::finalize() {
    if (luma_fp) fflush(luma_fp);
    if (chroma_fp) fflush(chroma_fp);

    if (!write_json()) return false;

    fprintf(stderr, "\nWrote %s (%d fields)\n", json_path.c_str(), field_count);
    return true;
}
