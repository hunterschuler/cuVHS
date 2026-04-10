#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_imports() -> None:
    repo = _repo_root()
    vhsdecode_root = repo / "vhs-decode"
    sys.path.insert(0, str(vhsdecode_root))
    candidates = [
        repo.parent / "VHSpp",
        repo.parent / "VHSpp" / "build",
        vhsdecode_root / "build",
        repo.parent / "vhs-decode" / "build",
    ]
    version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    found = []
    for base in candidates:
        if not base.exists():
            continue
        for path in sorted(base.rglob("vhsd_rust*.so")):
            found.append(path)
    for path in found:
        if version_tag in path.name:
            sys.path.append(str(path.parent))
            return
    if found:
        sys.path.append(str(found[0].parent))


def _write_f64(path: Path, arr) -> None:
    np.asarray(arr, dtype=np.float64).tofile(path)


def _write_complex_parts(prefix: Path, arr) -> None:
    carr = np.asarray(arr, dtype=np.complex128)
    np.asarray(carr.real, dtype=np.float64).tofile(prefix.with_suffix(".re.f64"))
    np.asarray(carr.imag, dtype=np.float64).tofile(prefix.with_suffix(".im.f64"))


def _jsonable(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _emit_config(decoder, rf, args):
    dp = rf.DecoderParams
    sp = getattr(rf, "SysParams", {})
    cfg = {
        "sample": args.sample,
        "input": args.input,
        "system": args.system,
        "tape_format": args.tape_format,
        "tape_speed": args.tape_speed,
        "inputfreq_mhz": args.inputfreq,
        "threads": args.threads,
        "blocklen": int(rf.blocklen),
        "blockcut": int(rf.blockcut),
        "blockcut_end": int(rf.blockcut_end),
        "freq_hz": float(rf.freq_hz),
        "fps": _jsonable(sp.get("FPS")),
        "frame_lines": _jsonable(sp.get("frame_lines")),
        "max_field_lines": _jsonable(sp.get("field_lines")),
        "outlinelen": _jsonable(sp.get("outlinelen")),
        "fsc_mhz": _jsonable(sp.get("fsc_mhz")),
        "hz_ire": _jsonable(dp.get("hz_ire", sp.get("hz_ire"))),
        "ire0": _jsonable(dp.get("ire0", sp.get("ire0"))),
        "vsync_ire": _jsonable(dp.get("vsync_ire", sp.get("vsync_ire"))),
        "color_under_carrier": _jsonable(dp.get("color_under_carrier")),
        "luma_carrier": _jsonable(dp.get("luma_carrier")),
        "deemph_mid": _jsonable(dp.get("deemph_mid")),
        "deemph_gain": _jsonable(dp.get("deemph_gain")),
        "deemph_q": _jsonable(dp.get("deemph_q")),
        "deemph_tau": _jsonable(dp.get("deemph_tau")),
        "video_bpf_low": _jsonable(dp.get("video_bpf_low")),
        "video_bpf_high": _jsonable(dp.get("video_bpf_high")),
        "video_bpf_order": _jsonable(dp.get("video_bpf_order")),
        "video_bpf_supergauss": _jsonable(dp.get("video_bpf_supergauss")),
        "video_lpf_extra": _jsonable(dp.get("video_lpf_extra")),
        "video_lpf_extra_order": _jsonable(dp.get("video_lpf_extra_order")),
        "video_hpf_extra": _jsonable(dp.get("video_hpf_extra")),
        "video_hpf_extra_order": _jsonable(dp.get("video_hpf_extra_order")),
        "video_lpf_freq": _jsonable(dp.get("video_lpf_freq")),
        "video_lpf_order": _jsonable(dp.get("video_lpf_order")),
        "video_lpf_supergauss": _jsonable(dp.get("video_lpf_supergauss")),
        "chroma_bpf_upper": _jsonable(dp.get("chroma_bpf_upper")),
        "chroma_bpf_lower": _jsonable(dp.get("chroma_bpf_lower", 60000.0)),
        "chroma_bpf_order": _jsonable(dp.get("chroma_bpf_order", 4)),
        "boost_bpf_low": _jsonable(dp.get("boost_bpf_low")),
        "boost_bpf_high": _jsonable(dp.get("boost_bpf_high")),
        "boost_bpf_mult": _jsonable(dp.get("boost_bpf_mult")),
        "boost_rf_linear_0": _jsonable(dp.get("boost_rf_linear_0")),
        "boost_rf_linear_20": _jsonable(dp.get("boost_rf_linear_20")),
        "boost_rf_linear_double": _jsonable(dp.get("boost_rf_linear_double")),
        "start_rf_linear": _jsonable(dp.get("start_rf_linear")),
        "video_rf_peak_freq": _jsonable(dp.get("video_rf_peak_freq")),
        "video_rf_peak_gain": _jsonable(dp.get("video_rf_peak_gain")),
        "video_rf_peak_bandwidth": _jsonable(dp.get("video_rf_peak_bandwidth")),
        "fm_audio_channel_0_freq": _jsonable(dp.get("fm_audio_channel_0_freq")),
        "fm_audio_channel_1_freq": _jsonable(dp.get("fm_audio_channel_1_freq")),
        "nonlinear_highpass_freq": _jsonable(dp.get("nonlinear_highpass_freq")),
        "nonlinear_highpass_limit_h": _jsonable(dp.get("nonlinear_highpass_limit_h")),
        "nonlinear_highpass_limit_l": _jsonable(dp.get("nonlinear_highpass_limit_l")),
        "nonlinear_scaling_1": _jsonable(dp.get("nonlinear_scaling_1")),
        "nonlinear_exp_scaling": _jsonable(dp.get("nonlinear_exp_scaling")),
        "use_sub_deemphasis": _jsonable(dp.get("use_sub_deemphasis")),
        "video_eq_present": _jsonable("video_eq" in dp and dp.get("video_eq") is not None),
        "video_eq_corner": _jsonable(dp.get("video_eq", {}).get("loband", {}).get("corner")),
        "video_eq_transition": _jsonable(dp.get("video_eq", {}).get("loband", {}).get("transition")),
        "video_eq_order_limit": _jsonable(dp.get("video_eq", {}).get("loband", {}).get("order_limit")),
        "video_eq_gain": _jsonable(dp.get("video_eq", {}).get("loband", {}).get("gain")),
        "burst_abs_ref": _jsonable(sp.get("burst_abs_ref")),
        "color_burst_us": _jsonable(sp.get("colorBurstUS")),
    }
    return cfg


def _normalize_input(raw: np.ndarray, input_path: str) -> np.ndarray:
    if input_path.endswith(".u8") or input_path.endswith(".raw") or input_path.endswith(".r8"):
        return (raw.astype(np.float64) - 128.0) * 256.0
    if input_path.endswith(".s16"):
        return raw.astype(np.float64)
    if input_path.endswith(".u16"):
        return raw.astype(np.float64) - 32768.0
    return raw.astype(np.float64)


def _build_decoder_options(args):
    import vhsdecode.formats as vf
    from vhsdecode.cmdcommons import get_extra_options, get_rf_options

    class Opts:
        pass

    opts = Opts()
    opts.chroma_trap = False
    opts.sharpness = 0
    opts.notch = None
    opts.notch_q = 10.0
    opts.debug = False
    opts.wow_level_adjust_smoothing = None
    opts.wow_interpolation_method = "linear"
    opts.dod_threshold_a = None
    opts.dod_hysteresis = 1.25
    opts.track_phase = None
    opts.high_boost = None
    opts.disable_diff_demod = False
    opts.fm_audio_notch = 0
    opts.enable_dc_offset = args.dc_correct
    opts.disable_comb = False
    opts.skip_chroma = False
    opts.nldeemp = False
    opts.subdeemp = False
    opts.y_comb = 0
    opts.cafc = False
    opts.disable_right_hsync = False
    opts.level_detect_divisor = 3
    opts.fallback_vsync = False
    opts.relaxed_line0 = False
    opts.field_order_confidence = 100
    opts.saved_levels = False
    opts.skip_hsync_refine = False
    opts.export_raw_tbc = False
    opts.tape_speed = args.tape_speed
    opts.ire0_adjust = False
    opts.detect_chroma_track_phase = False
    opts.disable_burst_hsync = False
    opts.disable_phase_correction = False
    opts.gnrc_afe = False
    opts.params_file = None
    opts.nodod = False
    opts.field_order_action = "detect"
    opts.AGC = False
    opts.noAGC = False

    rf_options = get_rf_options(opts)
    dod_threshold_p = vf.DEFAULT_THRESHOLD_P_CXADC if args.cxadc else vf.DEFAULT_THRESHOLD_P_DDD
    rf_options["dod_threshold_p"] = dod_threshold_p
    rf_options["dod_threshold_a"] = opts.dod_threshold_a
    rf_options["dod_hysteresis"] = opts.dod_hysteresis
    rf_options["track_phase"] = opts.track_phase
    rf_options["high_boost"] = opts.high_boost
    rf_options["disable_diff_demod"] = opts.disable_diff_demod
    rf_options["fm_audio_notch"] = opts.fm_audio_notch
    rf_options["disable_dc_offset"] = not opts.enable_dc_offset
    rf_options["disable_comb"] = opts.disable_comb
    rf_options["skip_chroma"] = opts.skip_chroma
    rf_options["nldeemp"] = opts.nldeemp
    rf_options["subdeemp"] = opts.subdeemp
    rf_options["y_comb"] = opts.y_comb
    rf_options["cafc"] = opts.cafc
    rf_options["disable_right_hsync"] = opts.disable_right_hsync
    rf_options["level_detect_divisor"] = opts.level_detect_divisor
    rf_options["fallback_vsync"] = opts.fallback_vsync
    rf_options["relaxed_line0"] = opts.relaxed_line0
    rf_options["field_order_confidence"] = int(max(0, min(100, opts.field_order_confidence)))
    rf_options["saved_levels"] = opts.saved_levels
    rf_options["skip_hsync_refine"] = opts.skip_hsync_refine
    rf_options["export_raw_tbc"] = opts.export_raw_tbc
    rf_options["tape_speed"] = opts.tape_speed
    rf_options["ire0_adjust"] = opts.ire0_adjust
    rf_options["detect_chroma_track_phase"] = opts.detect_chroma_track_phase
    rf_options["disable_burst_hsync"] = opts.disable_burst_hsync
    rf_options["disable_phase_correction"] = opts.disable_phase_correction
    rf_options["gnrc_afe"] = opts.gnrc_afe

    extra_options = get_extra_options(opts, True)
    extra_options["params_file"] = opts.params_file
    return rf_options, extra_options, (not opts.nodod), opts.field_order_action


def main() -> int:
    _setup_imports()

    import lddecode.utils as lddu
    from vhsdecode.process import VHSDecode

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--system", default="NTSC")
    parser.add_argument("--tape-format", default="VHS")
    parser.add_argument("--tape-speed", default="sp")
    parser.add_argument("--inputfreq", type=float, default=28.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-resample", action="store_true", default=True)
    parser.add_argument("--cxadc", action="store_true", default=True)
    parser.add_argument("--dc-correct", action="store_true", default=False)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dump_k1_vhsdecode")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    loader_input_freq = args.inputfreq if not args.no_resample else None
    sample_freq = 40.0 if not args.no_resample else args.inputfreq
    loader = lddu.make_loader(args.input, loader_input_freq)
    rf_options, extra_options, do_dod, field_order_action = _build_decoder_options(args)
    decoder = VHSDecode(
        args.input,
        None,
        loader,
        logger,
        system=args.system,
        tape_format=args.tape_format,
        doDOD=do_dod,
        threads=args.threads,
        inputfreq=sample_freq,
        rf_options=rf_options,
        extra_options=extra_options,
        field_order_action=field_order_action,
    )
    rf = decoder.rf
    raw = decoder.freader(decoder.infile, args.sample, rf.blocklen)
    # Mirror the live demodblock path, but keep intermediate arrays for parity work.
    indata_fft = np.fft.fft(raw)
    if rf._notch is not None:
        indata_fft *= rf.Filters["FVideoNotchF"]
    indata_fft *= rf.Filters["RFVideo"]

    raw_filtered = np.fft.ifft(indata_fft * rf.Filters["hilbert"]).real.astype(np.single)
    np.abs(raw_filtered, out=raw_filtered)
    raw_env = np.roll(raw_filtered, 4)
    env = rf.Filters["FEnvPost"]
    from vhsdecode.rust_utils import sosfiltfilt_rust
    env_post = sosfiltfilt_rust(env, raw_env)
    hilbert = np.fft.ifft(indata_fft * rf.Filters["hilbert"])
    from vhsdecode.demod import replace_spikes, unwrap_hilbert
    demod_raw = unwrap_hilbert(hilbert, rf.freq_hz)
    demod_diff = unwrap_hilbert(np.ediff1d(hilbert, to_begin=0), rf.freq_hz).real
    check_value = rf.options.diff_demod_check_value
    demod_spikefixed = demod_raw.copy()
    if np.max(demod_spikefixed[20:-20]) > check_value:
        demod_spikefixed = replace_spikes(demod_spikefixed, demod_diff, check_value)

    out = rf.demodblock(data=raw, cut=False)
    video = out["video"]

    _write_f64(outdir / "python_raw.f64", raw)
    _write_f64(outdir / "python_raw_normalized.f64", _normalize_input(raw, args.input))
    _write_f64(outdir / "python_demod.f64", video["demod"])
    _write_f64(outdir / "python_demod_05.f64", video["demod_05"])
    _write_f64(outdir / "python_envelope.f64", video["envelope"])
    _write_f64(outdir / "python_raw_env.f64", raw_env)
    _write_f64(outdir / "python_env_post.f64", env_post)
    _write_complex_parts(outdir / "python_hilbert", hilbert)
    _write_f64(outdir / "python_demod_raw.f64", demod_raw)
    _write_f64(outdir / "python_demod_diff.f64", demod_diff)
    _write_f64(outdir / "python_demod_spikefixed.f64", demod_spikefixed)
    filters = rf.Filters
    rfvideo = np.asarray(filters["RFVideo"])
    half = (rf.blocklen // 2) + 1
    _write_f64(outdir / "python_rfvideo_half.f64", np.abs(rfvideo[:half]))
    _write_complex_parts(outdir / "python_fvideo", filters["FVideo"])
    _write_complex_parts(outdir / "python_fvideo05", filters["FVideo05"])
    names = getattr(video.dtype, "names", None) or ()
    if "demod_burst" in names:
        _write_f64(outdir / "python_demod_burst.f64", video["demod_burst"])

    config = _emit_config(decoder, rf, args)
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
