# cuVHS

GPU-accelerated VHS RF signal decoder. Takes raw RF captures (`.u8` files from an RF capture device) and produces timebase-corrected (`.tbc`) output compatible with [ld-tools](https://github.com/happycube/ld-decode) for video export.

**Status:** Early development. NTSC only for now — support for additional video standards is planned.

## What it does

cuVHS replaces the decode step of [vhs-decode](https://github.com/oyvindln/vhs-decode), running the full signal processing pipeline on the GPU:

- FM demodulation
- Sync pulse detection
- Line location + hsync refinement
- TBC luma resampling
- Chroma decode (VHS color-under)
- Dropout detection

The output is a pair of `.tbc` files (luma + chroma) plus a `.tbc.json` metadata file. To get viewable video, you still need `tbc-video-export` and `ld-tools` from the vhs-decode ecosystem.

## Performance

~85 FPS on an RTX 3090.

## Requirements

- NVIDIA GPU (compute capability 6.0+)
- CUDA toolkit
- CMake 3.25+
- cuFFT (included with CUDA toolkit)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
./build/cuvhs -f <sample_rate_mhz> --overwrite <input.u8> <output_base>
```

Example with a 28 MHz capture:
```bash
./build/cuvhs -f 28 --overwrite tape.u8 tape_output
```

This produces `tape_output.tbc`, `tape_output_chroma.tbc`, and `tape_output.tbc.json`.

## Viewing output

Install [tbc-video-export](https://pypi.org/project/tbc-video-export/) and [ld-tools](https://github.com/happycube/ld-decode), then:

```bash
tbc-video-export tape_output tape_output.mkv
```

## License

TBD
