# cuVHS Logbook

## Original Design Document (from vhs-decode-faster)

### Why From Scratch

After 10 iterations of GPU optimization via monkey-patching Python functions in
vhs-decode, the best achieved was 5.7 FPS (single GPU worker) vs 8.7 FPS (6 CPU
workers). The Python field assembly pipeline has ~50 interleaved functions with
sequential state, branching, and frequent CPU-GPU data transfers. Patching
individual functions cannot overcome this.

A from-scratch C++/CUDA pipeline processes fields entirely on GPU with no Python
in the hot path. The project was named **cuVHS** (following CUDA's naming convention:
cuFFT, cuBLAS, cuDNN, etc.).

### Pipeline Overview

```
Raw samples (host) --> [bulk upload] --> GPU VRAM
  |
  +--> Kernel 1: Batched FM Demod           (cuFFT + frequency-domain filters)
  +--> Kernel 2: Find Sync Pulses           (threshold + edge detect + compaction)
  +--> Kernel 3: Classify Pulses + Linelocs (pulse classification + line assignment)
  +--> Kernel 4: Refine Line Locations      (hsync cross-correlation)
  +--> Kernel 5: TBC Resample               (cubic interpolation to 4*fsc)
  +--> Kernel 6: Chroma Decode              (color-under extract + heterodyne + ACC)
  +--> Kernel 7: Dropout Detection          (envelope threshold + concealment)
  |
  <--[bulk download]-- TBC fields (host) --> disk
```

All inter-kernel data stays in GPU VRAM. One bulk upload at start, one bulk
download at end. No Python in the loop.

### Kernel Details

**Kernel 1: Batched FM Demodulation**
- cuFFT forward (N fields x ~468k samples at 28 MHz)
- Frequency-domain bandpass, deemphasis, chroma trap
- cuFFT inverse -> baseband video
- Hilbert unwrap for FM discrimination
- Produces: demod (video) + demod_05 (sync signal)

**Kernel 2: Find Sync Pulses**
- Threshold demod_05 at sync tip level -> boolean mask
- Edge detection: diff(mask) -> rising/falling edges
- Pair edges into pulses (start, end, length)
- Stream compaction (thrust::copy_if or CUB DeviceSelect)
- Output: per-field variable-length pulse arrays, max ~800 pulses/field

**Kernel 3: Classify Pulses + Compute Line Locations (hardest)**
- Classify each pulse by width: HSYNC (~4.7us), VSYNC (~27.1us), EQ (~2.3us)
- Find field boundary: VSYNC->HSYNC transition (sequential per field, ~800 pulses)
- Compute mean line length: parallel reduction per field
- Assign line numbers: expected[line] = ref_pos + line * mean_linelen
- For each expected position, find nearest actual HSYNC
- Output: N x linelocs (525 NTSC / 625 PAL entries per field)

**Kernel 4: Refine Line Locations (Hsync Correlation)**
- Cross-correlate ~40-sample window with ideal hsync waveform
- Sub-sample peak via parabolic interpolation
- N_fields x lines_per_field x 40 = embarrassingly parallel
- Hsync template in shared memory

**Kernel 5: TBC Resample**
- Map output position to input position via linelocs
- Cubic interpolation from demod signal
- N_fields x output_line_len x output_field_lines per batch
- Memory-bound, optimize for coalesced reads

**Kernel 6: Chroma Decode (VHS Color-Under)**
- Bandpass extract chroma-under (~629 kHz NTSC, ~626 kHz PAL)
- Heterodyne up to fsc (3.58 MHz NTSC, 4.43 MHz PAL)
- Track phase detection: try both phases, pick better burst
- ACC normalization per line: burst_abs_ref / burst_rms
- Comb filtering with adjacent field (1-field dependency)
- Note: input must be in s16-equivalent range (u8 normalized during upload)

**Kernel 7: Dropout Detection**
- Compute RF envelope from demod signal
- Threshold at configurable percentage of field average
- Hysteresis for dropout start/end
- Merge nearby dropouts, filter very short ones
- Map RF positions to TBC line/column positions
- Inline concealment: replace from adjacent line

### Memory Budget

Per field @ 28 MHz NTSC:
- Raw samples: 468k x 8 bytes (float64) = 3.6 MB
- Demod (float64): 468k x 8 = 3.6 MB
- Demod_05 (float64): 468k x 8 = 3.6 MB
- Linelocs: 525 x 8 = 4 KB
- TBC luma output: 263 x 910 x 2 = 0.5 MB
- TBC chroma output: 263 x 910 x 2 = 0.5 MB
- **Total per field: ~11.6 MB**

Batch sizing is dynamic — computed at runtime from available VRAM (uses 80% of
free VRAM by default). No hardcoded GPU assumptions.

### Implementation Phases

1. **Phase 1:** FM Demodulation (K1) — foundation for everything else
2. **Phase 2:** Sync + Line Locations (K2-K3) — field boundary detection
3. **Phase 3:** Hsync Refinement + TBC Resample (K4-K5) — produces luma TBC
4. **Phase 4:** Chroma Decode (K6) — produces chroma TBC
5. **Phase 5:** Dropout Detection (K7) — quality improvement
6. **Phase 6:** Optimization — kernel fusion, CUDA streams, PCIe overlap

### Validation Strategy

Every kernel must be validated against the Python vhs-decode reference:
- Frame-by-frame PSNR/SSIM comparison (target: >60 dB PSNR)
- Pulse-by-pulse comparison for K2-K3
- Line-by-line lineloc comparison for K3-K4
- Visual inspection of test captures across NTSC/PAL, SP/LP/EP

### Open Questions

- **PAL support:** Same kernels, different constants. Parameterized at init.
- **SVHS/Betamax/other formats:** Different chroma schemes. K6 needs format-specific
  paths. K1-K5 are format-agnostic.
- **Multi-GPU:** Partition fields across GPUs. Each GPU runs full pipeline on its subset.
- **Previous-field dependency:** Chroma comb filter and some dropout compensation need
  the previous field. Process in order within each GPU, or accept 1-field latency.

---

## Project Scaffolding (2026-03-23)

### Decision: Language and Build System

Chose standalone C++17/CUDA with CMake. The project lives at
`/media/hunter/DATA/GitHub/cuVHS/`, completely independent of the Python
vhs-decode codebase.

The name **cuVHS** follows CUDA's library naming convention (cuFFT, cuBLAS, cuDNN).

### Project Structure

```
cuVHS/
├── CMakeLists.txt                    # CMake + CUDA, multi-arch (sm_60..sm_90)
├── LOGBOOK.md
└── src/
    ├── main.cpp                      # CLI entry point, arg parsing
    ├── gpu/
    │   ├── device.h                  # GPU capability detection
    │   └── device.cu                 # Auto-select best GPU, dynamic VRAM query
    ├── format/
    │   ├── video_format.h            # NTSC/PAL timing/frequency constants
    │   └── video_format.cpp          # All params derived from system + sample rate
    ├── io/
    │   ├── raw_reader.h              # Read .u8/.s16/.u16 raw RF captures
    │   ├── raw_reader.cpp            # Normalizes u8 -> s16 range during read
    │   ├── tbc_writer.h              # Write .tbc + _chroma.tbc + .tbc.json
    │   └── tbc_writer.cpp            # ld-tools compatible JSON metadata
    └── pipeline/
        ├── pipeline.h                # Orchestrator: batch sizing, data flow
        ├── pipeline.cu               # Upload -> K1..K7 -> download -> write
        ├── fm_demod.h/cu             # K1: Batched FM demodulation (cuFFT)
        ├── sync_pulses.h/cu          # K2: Sync pulse detection
        ├── line_locs.h/cu            # K3: Pulse classification + line positions
        ├── hsync_refine.h/cu         # K4: Sub-sample hsync correlation
        ├── tbc_resample.h/cu         # K5: Cubic interpolation to 4*fsc
        ├── chroma_decode.h/cu        # K6: Color-under decode + ACC
        └── dropout_detect.h/cu       # K7: Envelope-based dropout detection
```

### Design Decisions

- **Dynamic batch sizing:** `GPUDevice::max_batch_size()` queries free VRAM at
  runtime and computes how many fields fit in 80% of available memory. No hardcoded
  GPU specs. Works on any CUDA-capable GPU.
- **Multi-arch build:** CMake builds for sm_60 through sm_90 by default, covering
  Pascal through Hopper. User can override with `-DCMAKE_CUDA_ARCHITECTURES=`.
- **u8 normalization in reader:** `RawReader::read()` converts u8 input to
  s16-equivalent range `(sample - 128) * 256` at load time. This prevents the
  chroma amplitude bug discovered in vhs-decode (see vhs-decode-faster LOGBOOK.md,
  2026-03-23 entry).
- **ld-tools compatible output:** TBC files and JSON metadata follow the same schema
  used by ld-dropout-correct, ld-chroma-decoder, and tbc-video-export.
- **All kernels stubbed:** Each kernel has a header (interface + documentation) and
  a .cu file with a stub implementation that prints a message. The pipeline runs
  end-to-end with stubs, producing zeroed output.

### Initial Build and Test

**Environment:**
- Ubuntu (kernel 6.17.0-19-generic)
- GCC 13.3.0
- CUDA 12.0 (V12.0.140) via `apt install nvidia-cuda-toolkit`
- NVIDIA GeForce RTX 3090, driver 580.126.09

**Build:**
```
mkdir build && cd build && cmake .. && make -j$(nproc)
```
One compilation fix needed: missing `#include <vector>` in `tbc_writer.h`.
After that, clean build with all 14 source files compiling successfully.

**Test run** on TAPE_1 10-second clip (280 MB, 28 MHz u8 NTSC):
```
$ cuvhs -f 28 --system NTSC --overwrite ~/captures/TAPE_1/TAPE_1_10s.u8 /tmp/cuvhs_test

GPU 0: NVIDIA GeForce RTX 3090
  Compute: 8.6  SMs: 82  Warp: 32
  VRAM: 23.6 GB total, 21.5 GB free
  Shared mem/block: 48 KB  Max threads/block: 1024
  Concurrent kernels: yes  Managed memory: yes
Format: NTSC @ 28.0 MHz capture rate
  Lines/frame: 525  Line rate: 15734.264 Hz
  Samples/line: 1780  Samples/field: 468140
  Output: 910 samples/line @ 14.318180 MHz (4*fsc)
Batch size: 512 fields (11.6 MB per field, 5961.5 MB total)
Starting decode: ~598 fields in batches of 512
Decode complete: 598 fields in 5.5 seconds (54.3 FPS)
Wrote /tmp/cuvhs_test.tbc.json (598 fields)
```

Results:
- Auto-detected RTX 3090 with 21.5 GB free VRAM
- Dynamically sized batch to 512 fields (5.96 GB)
- Processed 598 fields in 2 batches
- Output: 273 MB luma TBC, 273 MB chroma TBC, 91 KB JSON metadata
- 54 FPS with stub kernels (just I/O overhead — real perf comes with kernel implementation)
- All 7 kernel stubs executed in correct order

---

## Streaming / Real-Time Decode Support (2026-03-23)

### Motivation

The VHS preservation workflow currently requires:
1. Capture raw RF to disk (can take hours for a full tape)
2. Decode raw RF to TBC (another few hours)
3. Export TBC to video (more time)

With a GPU decoder fast enough to exceed real-time, steps 1 and 2 can be
combined: pipe the capture device's output directly into cuVHS. This saves
wall-clock time and lets the operator see decoded video during capture to
catch tracking errors, tape damage, etc. immediately rather than discovering
them hours later.

### Implementation

Reworked `RawReader` to support two modes:

**File mode** (existing): Random-access reads via `pread()`. Total size known.
Used for normal decode of captured files.

**Stream mode** (new): Sequential blocking reads from stdin or a named pipe/FIFO.
No seeking, total size unknown. Used for real-time decode during capture.

Key changes:
- `open_stream(fd)` / `open_stdin()` — new constructors for stream mode
- `read_next()` — sequential read that works in both modes (replaces offset-based `read()`)
- `read_full()` — internal helper that retries partial reads from pipes (a single
  `read()` syscall on a pipe may return less than requested even when more data is
  coming; this loops until the full amount is available or EOF)
- `open()` auto-detects FIFOs via `S_ISFIFO` and switches to stream mode
- `is_stream()` / `is_seekable()` queries for the pipeline to adapt behavior

Pipeline changes:
- Stream mode uses smaller batches (max 64 fields vs 512) for lower latency
- Progress display shows field count without percentages (total unknown)
- Main loop reads sequentially until EOF instead of computing offsets

CLI: Use `-` as input_file to read from stdin. `--format` is required for stdin
since there's no file extension to auto-detect from.

### Testing

**File mode** (unchanged behavior):
```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/test
Batch size: 512 fields — 53.5 FPS
```

**Stream mode via pipe:**
```
$ cat TAPE_1_10s.u8 | cuvhs -f 28 --system NTSC --format u8 --overwrite - /tmp/test
Input: stdin (28.0 MHz, u8, streaming)
Batch size: 64 fields [stream mode]
598 fields — 47.3 FPS (streaming)
```

Both produce identical output (598 fields, 273 MB TBC files).

The intended real-time capture usage:
```
capture_device -r 28e6 | cuvhs -f 28 --system NTSC --format u8 - output
```

### Next Steps

Implement Kernel 1 (FM Demodulation) — the foundation kernel. Everything downstream
depends on the demodulated signal. This involves:
1. Setting up batched cuFFT plans
2. Precomputing frequency-domain filter coefficients
3. Implementing the Hilbert transform / FM discrimination
4. Validating against Python vhs-decode output on the 10s test clip
