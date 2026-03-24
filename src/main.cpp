#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gpu/device.h"
#include "io/raw_reader.h"
#include "io/tbc_writer.h"
#include "format/video_format.h"
#include "pipeline/pipeline.h"

struct Args {
    std::string input_path;      // "-" or empty for stdin
    std::string output_base;     // output prefix (writes .tbc, _chroma.tbc, .tbc.json)
    VideoSystem system = VideoSystem::NTSC;
    double sample_rate_mhz = 28.0;
    InputFormat input_format = InputFormat::U8;
    bool format_explicit = false; // true if user specified --format
    int gpu_id = 0;              // -1 = auto-select best GPU
    bool overwrite = false;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "cuVHS — GPU-accelerated VHS RF decoder\n"
        "\n"
        "Usage: %s [options] <input_file> <output_base>\n"
        "\n"
        "Arguments:\n"
        "  input_file       Raw RF capture (.u8, .s16, .u16, .raw), or - for stdin\n"
        "  output_base      Output prefix (creates .tbc, _chroma.tbc, .tbc.json)\n"
        "\n"
        "Options:\n"
        "  --system <NTSC|PAL>   Video system (default: NTSC)\n"
        "  -f <MHz>              Sample rate in MHz (default: 28)\n"
        "  --format <u8|s16>     Input sample format (default: auto from extension)\n"
        "  --gpu <id>            GPU device ID, -1 for auto (default: 0)\n"
        "  --overwrite           Overwrite existing output files\n"
        "  -h, --help            Show this help\n",
        prog);
}

static bool parse_args(int argc, char** argv, Args& args) {
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            i++;
            if (strcasecmp(argv[i], "NTSC") == 0) args.system = VideoSystem::NTSC;
            else if (strcasecmp(argv[i], "PAL") == 0) args.system = VideoSystem::PAL;
            else { fprintf(stderr, "Unknown system: %s\n", argv[i]); return false; }
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            args.sample_rate_mhz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            i++;
            args.format_explicit = true;
            if (strcmp(argv[i], "u8") == 0) args.input_format = InputFormat::U8;
            else if (strcmp(argv[i], "s16") == 0) args.input_format = InputFormat::S16;
            else { fprintf(stderr, "Unknown format: %s\n", argv[i]); return false; }
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            args.gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--overwrite") == 0) {
            args.overwrite = true;
        } else if (argv[i][0] == '-' && strlen(argv[i]) > 1) {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        } else {
            if (positional == 0) args.input_path = argv[i];
            else if (positional == 1) args.output_base = argv[i];
            else { fprintf(stderr, "Too many arguments\n"); return false; }
            positional++;
        }
    }
    if (positional < 2) {
        fprintf(stderr, "Error: input_file and output_base are required\n\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

static InputFormat detect_format(const std::string& path) {
    auto ext = path.substr(path.find_last_of('.') + 1);
    if (ext == "u8" || ext == "raw") return InputFormat::U8;
    if (ext == "s16") return InputFormat::S16;
    if (ext == "u16") return InputFormat::U16;
    return InputFormat::U8;  // default
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args))
        return 1;

    // Auto-detect input format from extension if not explicitly set
    if (!args.format_explicit)
        args.input_format = detect_format(args.input_path);

    // GPU setup
    GPUDevice gpu;
    if (!gpu.init(args.gpu_id)) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }
    gpu.print_info();

    // Video format parameters
    VideoFormat fmt(args.system, args.sample_rate_mhz);
    fmt.print_info();

    // Open input — stdin/pipe or file
    RawReader reader;
    bool is_stdin = (args.input_path == "-");
    if (is_stdin) {
        if (!reader.open_stdin(args.input_format)) {
            fprintf(stderr, "Failed to open stdin\n");
            return 1;
        }
        fprintf(stderr, "Input: stdin (%.1f MHz, %s, streaming)\n",
                args.sample_rate_mhz, input_format_name(args.input_format));
    } else {
        if (!reader.open(args.input_path, args.input_format)) {
            fprintf(stderr, "Failed to open input: %s\n", args.input_path.c_str());
            return 1;
        }
        if (reader.is_stream()) {
            fprintf(stderr, "Input: %s (%.1f MHz, %s, streaming FIFO)\n",
                    args.input_path.c_str(), args.sample_rate_mhz,
                    input_format_name(args.input_format));
        } else {
            fprintf(stderr, "Input: %s (%.1f MHz, %s, %.2f GB)\n",
                    args.input_path.c_str(), args.sample_rate_mhz,
                    input_format_name(args.input_format),
                    reader.size_bytes() / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // Open output
    TBCWriter writer;
    if (!writer.open(args.output_base, fmt, args.overwrite)) {
        fprintf(stderr, "Failed to open output: %s\n", args.output_base.c_str());
        return 1;
    }

    // Run pipeline
    Pipeline pipeline(gpu, fmt, reader, writer);
    if (!pipeline.run()) {
        fprintf(stderr, "Pipeline failed\n");
        return 1;
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
