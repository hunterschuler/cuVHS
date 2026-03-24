#include "pipeline/pipeline.h"
#include "pipeline/fm_demod.h"
#include "pipeline/sync_pulses.h"
#include "pipeline/line_locs.h"
#include "pipeline/hsync_refine.h"
#include "pipeline/tbc_resample.h"
#include "pipeline/chroma_decode.h"
#include "pipeline/dropout_detect.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>

Pipeline::Pipeline(const GPUDevice& gpu_, const VideoFormat& fmt_,
                   RawReader& reader_, TBCWriter& writer_)
    : gpu(gpu_), fmt(fmt_), reader(reader_), writer(writer_) {}

Pipeline::~Pipeline() {
    free_buffers();
}

size_t Pipeline::bytes_per_field() const {
    size_t raw = fmt.samples_per_field * sizeof(double);       // input (converted)
    size_t demod = fmt.samples_per_field * sizeof(double);     // demod output
    size_t demod_05 = fmt.samples_per_field * sizeof(double);  // sync demod
    size_t pulses = 1024 * 3 * sizeof(int);                    // pulse arrays (generous)
    size_t linelocs = fmt.lines_per_frame * sizeof(double);    // line locations
    size_t tbc = fmt.output_line_len * fmt.output_field_lines * sizeof(uint16_t);  // luma
    size_t chroma = tbc;                                       // chroma

    return raw + demod + demod_05 + pulses + linelocs + tbc + chroma;
}

bool Pipeline::allocate_buffers() {
    batch_size = gpu.max_batch_size(bytes_per_field(), 0.8);

    // In stream mode, use smaller batches for lower latency
    int max_batch = reader.is_stream() ? 64 : 512;

    if (batch_size < 1) batch_size = 1;
    if (batch_size > max_batch) batch_size = max_batch;

    fprintf(stderr, "Batch size: %d fields (%.1f MB per field, %.1f MB total)%s\n",
            batch_size,
            bytes_per_field() / (1024.0 * 1024.0),
            batch_size * bytes_per_field() / (1024.0 * 1024.0),
            reader.is_stream() ? " [stream mode]" : "");

    size_t n = batch_size;
    size_t spf = fmt.samples_per_field;
    size_t tbc_field_size = fmt.output_line_len * fmt.output_field_lines;

    auto alloc = [](void** ptr, size_t bytes) -> bool {
        cudaError_t err = cudaMalloc(ptr, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc(%zu bytes) failed: %s\n", bytes, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    if (!alloc(&d_raw,        n * spf * sizeof(double)))    return false;
    if (!alloc(&d_demod,      n * spf * sizeof(double)))    return false;
    if (!alloc(&d_demod_05,   n * spf * sizeof(double)))    return false;
    if (!alloc(&d_pulses,     n * 1024 * 3 * sizeof(int)))  return false;
    if (!alloc(&d_linelocs,   n * fmt.lines_per_frame * sizeof(double))) return false;
    if (!alloc(&d_tbc_luma,   n * tbc_field_size * sizeof(uint16_t)))    return false;
    if (!alloc(&d_tbc_chroma, n * tbc_field_size * sizeof(uint16_t)))    return false;

    return true;
}

void Pipeline::free_buffers() {
    auto safe_free = [](void** ptr) {
        if (*ptr) { cudaFree(*ptr); *ptr = nullptr; }
    };
    safe_free(&d_raw);
    safe_free(&d_demod);
    safe_free(&d_demod_05);
    safe_free(&d_pulses);
    safe_free(&d_linelocs);
    safe_free(&d_tbc_luma);
    safe_free(&d_tbc_chroma);
}

int Pipeline::process_batch(int num_fields) {
    size_t spf = fmt.samples_per_field;
    size_t tbc_field_size = fmt.output_line_len * fmt.output_field_lines;

    // 1. Read raw data into host buffer
    // read_next works for both file mode (sequential) and stream mode (blocking)
    auto* h_raw = new double[num_fields * spf];
    size_t samples_read = reader.read_next(h_raw, num_fields * spf);
    if (samples_read == 0) {
        delete[] h_raw;
        return 0;
    }
    int fields_loaded = static_cast<int>(samples_read / spf);
    if (fields_loaded == 0) {
        delete[] h_raw;
        return 0;
    }

    // 2. Upload to GPU
    cudaMemcpy(d_raw, h_raw, fields_loaded * spf * sizeof(double),
               cudaMemcpyHostToDevice);
    delete[] h_raw;

    // 3. FM Demodulation (Kernel 1)
    fm_demod(static_cast<double*>(d_raw),
             static_cast<double*>(d_demod),
             static_cast<double*>(d_demod_05),
             fields_loaded, spf, fmt);

    // 4. Find Sync Pulses (Kernel 2)
    sync_pulses(static_cast<double*>(d_demod_05),
                static_cast<int*>(d_pulses),
                fields_loaded, spf, fmt);

    // 5. Classify Pulses + Line Locations (Kernel 3)
    line_locs(static_cast<int*>(d_pulses),
              static_cast<double*>(d_linelocs),
              fields_loaded, fmt);

    // 6. Refine Line Locations via Hsync Correlation (Kernel 4)
    hsync_refine(static_cast<double*>(d_demod_05),
                 static_cast<double*>(d_linelocs),
                 fields_loaded, fmt);

    // 7. TBC Resample (Kernel 5)
    tbc_resample(static_cast<double*>(d_demod),
                 static_cast<double*>(d_linelocs),
                 static_cast<uint16_t*>(d_tbc_luma),
                 fields_loaded, fmt);

    // 8. Chroma Decode (Kernel 6)
    chroma_decode(static_cast<double*>(d_raw),
                  static_cast<double*>(d_linelocs),
                  static_cast<uint16_t*>(d_tbc_chroma),
                  fields_loaded, fmt);

    // 9. Dropout Detection (Kernel 7)
    dropout_detect(static_cast<double*>(d_demod),
                   static_cast<uint16_t*>(d_tbc_luma),
                   fields_loaded, spf, fmt);

    // 10. Download TBC results and write to disk
    auto* h_luma = new uint16_t[fields_loaded * tbc_field_size];
    auto* h_chroma = new uint16_t[fields_loaded * tbc_field_size];

    cudaMemcpy(h_luma, d_tbc_luma,
               fields_loaded * tbc_field_size * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_chroma, d_tbc_chroma,
               fields_loaded * tbc_field_size * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < fields_loaded; i++) {
        writer.write_luma_field(h_luma + i * tbc_field_size);
        writer.write_chroma_field(h_chroma + i * tbc_field_size);
        writer.set_first_field(i % 2 == 0);
        writer.finish_field();
    }

    delete[] h_luma;
    delete[] h_chroma;

    return fields_loaded;
}

bool Pipeline::run() {
    if (!allocate_buffers())
        return false;

    size_t total_samples = reader.total_samples();  // 0 for streams
    size_t spf = fmt.samples_per_field;
    bool streaming = reader.is_stream();

    if (streaming) {
        fprintf(stderr, "Starting decode: streaming mode, batches of %d fields\n", batch_size);
    } else {
        int total_fields_est = static_cast<int>(total_samples / spf);
        fprintf(stderr, "Starting decode: ~%d fields in batches of %d\n",
                total_fields_est, batch_size);
    }

    auto t_start = std::chrono::steady_clock::now();
    int total_fields = 0;

    while (true) {
        int fields_this_batch = batch_size;

        // In file mode, don't request more fields than remain
        if (!streaming) {
            size_t consumed = static_cast<size_t>(total_fields) * spf;
            if (consumed >= total_samples) break;
            size_t remaining_fields = (total_samples - consumed) / spf;
            if (remaining_fields == 0) break;
            fields_this_batch = std::min(batch_size, static_cast<int>(remaining_fields));
        }

        int processed = process_batch(fields_this_batch);
        if (processed == 0) break;  // EOF or error

        total_fields += processed;

        // Progress
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        double fps = (total_fields / 2.0) / elapsed;

        if (streaming) {
            fprintf(stderr, "\r  %d fields — %.1f FPS (streaming)",
                    total_fields, fps);
        } else {
            double pct = 100.0 * total_fields * spf / total_samples;
            fprintf(stderr, "\r  %d fields (%.1f%%) — %.1f FPS",
                    total_fields, pct, fps);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    fprintf(stderr, "\nDecode complete: %d fields in %.1f seconds (%.1f FPS)\n",
            total_fields, total_time, (total_fields / 2.0) / total_time);

    writer.finalize();
    return true;
}
