#include "gpu/device.h"
#include <cuda_runtime.h>
#include <cstdio>

bool GPUDevice::init(int id) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPUs found\n");
        return false;
    }

    // Auto-select: pick GPU with most VRAM
    if (id < 0) {
        size_t best_vram = 0;
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (prop.totalGlobalMem > best_vram) {
                best_vram = prop.totalGlobalMem;
                id = i;
            }
        }
    }

    if (id >= device_count) {
        fprintf(stderr, "GPU %d not found (have %d GPUs)\n", id, device_count);
        return false;
    }

    cudaError_t err = cudaSetDevice(id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", id, cudaGetErrorString(err));
        return false;
    }

    device_id = id;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, id);
    snprintf(name, sizeof(name), "%s", prop.name);
    vram_total = prop.totalGlobalMem;
    sm_count = prop.multiProcessorCount;
    compute_major = prop.major;
    compute_minor = prop.minor;
    max_threads_per_block = prop.maxThreadsPerBlock;
    warp_size = prop.warpSize;
    shared_mem_per_block = prop.sharedMemPerBlock;
    supports_managed_memory = (prop.managedMemory != 0);
    supports_concurrent_kernels = (prop.concurrentKernels != 0);

    // Query free VRAM
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    vram_free = free_mem;

    return true;
}

void GPUDevice::print_info() const {
    fprintf(stderr, "GPU %d: %s\n", device_id, name);
    fprintf(stderr, "  Compute: %d.%d  SMs: %d  Warp: %d\n",
            compute_major, compute_minor, sm_count, warp_size);
    fprintf(stderr, "  VRAM: %.1f GB total, %.1f GB free\n",
            vram_total / (1024.0 * 1024.0 * 1024.0),
            vram_free / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Shared mem/block: %zu KB  Max threads/block: %d\n",
            shared_mem_per_block / 1024, max_threads_per_block);
    fprintf(stderr, "  Concurrent kernels: %s  Managed memory: %s\n",
            supports_concurrent_kernels ? "yes" : "no",
            supports_managed_memory ? "yes" : "no");
}

size_t GPUDevice::query_free_vram() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

int GPUDevice::max_batch_size(size_t bytes_per_field, double vram_fraction) const {
    size_t available = static_cast<size_t>(query_free_vram() * vram_fraction);
    if (bytes_per_field == 0) return 0;
    int batch = static_cast<int>(available / bytes_per_field);
    return batch > 0 ? batch : 1;
}
