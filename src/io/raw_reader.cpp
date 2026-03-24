#include "io/raw_reader.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <algorithm>

RawReader::~RawReader() {
    close();
}

bool RawReader::open(const std::string& path, InputFormat format) {
    fmt = format;
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(path.c_str());
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close();
        return false;
    }
    file_size = st.st_size;
    return true;
}

void RawReader::close() {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

size_t RawReader::total_samples() const {
    return file_size / input_format_bytes_per_sample(fmt);
}

size_t RawReader::read(double* dest, size_t offset, size_t num_samples) {
    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_offset = offset * bps;
    if (byte_offset >= file_size) return 0;

    size_t available = (file_size - byte_offset) / bps;
    size_t to_read = std::min(num_samples, available);
    size_t byte_count = to_read * bps;

    // Read into a temporary buffer, then convert
    // For large reads, this could be optimized with mmap
    auto* buf = new uint8_t[byte_count];
    ssize_t n = pread(fd, buf, byte_count, byte_offset);
    if (n <= 0) {
        delete[] buf;
        return 0;
    }
    size_t samples_read = n / bps;

    switch (fmt) {
        case InputFormat::U8:
            for (size_t i = 0; i < samples_read; i++)
                dest[i] = (static_cast<double>(buf[i]) - 128.0) * 256.0;
            break;
        case InputFormat::S16: {
            auto* s = reinterpret_cast<int16_t*>(buf);
            for (size_t i = 0; i < samples_read; i++)
                dest[i] = static_cast<double>(s[i]);
            break;
        }
        case InputFormat::U16: {
            auto* u = reinterpret_cast<uint16_t*>(buf);
            for (size_t i = 0; i < samples_read; i++)
                dest[i] = static_cast<double>(u[i]) - 32768.0;
            break;
        }
    }

    delete[] buf;
    return samples_read;
}

size_t RawReader::read_raw(void* dest, size_t offset, size_t num_samples) {
    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_offset = offset * bps;
    if (byte_offset >= file_size) return 0;

    size_t available = (file_size - byte_offset) / bps;
    size_t to_read = std::min(num_samples, available);
    size_t byte_count = to_read * bps;

    ssize_t n = pread(fd, dest, byte_count, byte_offset);
    if (n <= 0) return 0;
    return n / bps;
}
