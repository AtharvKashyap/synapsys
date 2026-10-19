"""Record raw EEG from the SynapSys ESP32 (CONFIG_SYNAPSYS_RAW_STREAM=y) into CSV trials.

Live, one labelled calibration/evaluation recording per call (needs ``pip install -e
".[record]"``)::

    python scripts/record_serial.py --port /dev/cu.usbmodem1101 --seconds 20 \
        --label YES --subject S01 --split calibration --out data/synapsys/S01/cal/yes_01.csv

The recording is sample-count based (``seconds * fs`` RAW lines), not wall-clock based,
and must be gap-free: a missing ``seq`` or a ``raw_gap`` event aborts it. Next to the
CSV + JSON sidecar it keeps ``<name>.serial.log`` (the untouched device lines, including
the device's own decisions) for later checks with ``verify_device_log.py``.

Offline, split an existing serial log (e.g. from QEMU) into one CSV per device stream::

    python scripts/record_serial.py --from-log serial.log --out-dir data/qemu_replay

Labels: a configured target (``YES``/``NO``/``HELP``) or ``REST``. Protocol notes are
in docs/calibration.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from synapsys_decoder.calibration import REST_LABEL
from synapsys_decoder.config import load_config
from synapsys_decoder.io.csv_eeg import save_csv_trial
from synapsys_decoder.io.serial_log import RAW_PREFIX, parse_serial_lines

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "test", "pilot")


def _trial_fields(args, cfg) -> dict:
    label = args.label
    if label is not None and label != REST_LABEL and label not in cfg.labels:
        raise SystemExit(f"--label must be one of {cfg.labels} or {REST_LABEL!r}")
    return {
        "true_label": label,
        "true_frequency": cfg.targets.get(label) if label else None,
        "subject_id": args.subject,
        "trial_id": args.trial_id,
        "metadata": {"split": args.split, "note": args.note, "recorded_by": "record_serial.py"},
    }


def _check_channels(stream, cfg) -> None:
    ch = stream.info.get("channels")
    if ch is not None and tuple(ch) != tuple(cfg.channels):
        raise SystemExit(f"device channels {ch} != config channels {list(cfg.channels)}")
    fs = stream.info.get("fs")
    if fs is not None and float(fs) != float(cfg.sample_rate_hz):
        raise SystemExit(f"device fs {fs} != config fs {cfg.sample_rate_hz}")


def from_log(args, cfg) -> int:
    lines = Path(args.from_log).read_text(errors="replace").splitlines()
    log = parse_serial_lines(lines)
    out = Path(args.out_dir)
    n = 0
    for i, st in enumerate(log.streams):
        if not st.seq:
            continue
        _check_channels(st, cfg)
        name = f"{i:03d}_{st.info.get('stream', 'nostream')}"
        try:
            trial = st.to_trial(
                fs=cfg.sample_rate_hz, channels=cfg.channels, **_trial_fields(args, cfg)
            )
        except ValueError as exc:
            print(f"skip {name}: {exc}")
            continue
        save_csv_trial(trial, out / f"{name}.csv", units="uV")
        print(
            f"{out / name}.csv: {trial.n_samples} samples ({trial.duration_s:.1f} s), "
            f"{len(st.decisions)} device decisions"
        )
        n += 1
    print(f"{n} stream(s) written")
    return 0 if n else 1


def live(args, cfg) -> int:
    try:
        import serial  # pyserial
    except ImportError:
        raise SystemExit('pyserial missing: pip install -e ".[record]"') from None
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    fields = _trial_fields(args, cfg)
    need = round(args.seconds * cfg.sample_rate_hz)
    lines: list[str] = []
    n_raw, last_seq = 0, None
    with serial.Serial(args.port, args.baud, timeout=1) as port:
        port.reset_input_buffer()
        t0 = time.monotonic()
        print(f"recording {args.seconds:g} s ({need} samples) from {args.port} ...")
        while n_raw < need:
            raw = port.readline()
            if not raw:
                if n_raw == 0 and time.monotonic() - t0 > args.start_timeout:
                    raise SystemExit(
                        "no RAW lines: flash with CONFIG_SYNAPSYS_RAW_STREAM=y "
                        "(idf.py menuconfig -> SynapSys -> Output) and use the native USB port"
                    )
                continue
            line = raw.decode("ascii", errors="replace").rstrip("\r\n")
            if n_raw == 0 and not line.startswith(RAW_PREFIX):
                continue  # drop the partial/old tail before the first sample
            lines.append(line)
            if '"event":"raw_gap"' in line or ('"event":"raw_info"' in line and n_raw):
                raise SystemExit(f"device reported a gap/restart mid-recording: {line}")
            if line.startswith(RAW_PREFIX):
                try:
                    seq = int(line.split()[1])
                except (IndexError, ValueError):
                    raise SystemExit(f"corrupted sample line {line!r}; re-record") from None
                if last_seq is not None and seq != last_seq + 1:
                    raise SystemExit(f"sample lost: seq {last_seq} -> {seq}; re-record")
                last_seq = seq
                n_raw += 1
                if n_raw % round(cfg.sample_rate_hz * 5) == 0:
                    print(f"  {n_raw / cfg.sample_rate_hz:5.1f} s")
    log = parse_serial_lines(lines)
    streams = [s for s in log.streams if s.seq]
    if len(streams) != 1:
        raise SystemExit(f"expected one device stream, got {len(streams)}; re-record")
    st = streams[0]
    _check_channels(st, cfg)
    trial = st.to_trial(fs=cfg.sample_rate_hz, channels=cfg.channels, **fields)
    save_csv_trial(trial, out, units="uV")
    out.with_suffix(".serial.log").write_text("\n".join(lines) + "\n")
    print(
        f"saved {out} ({trial.n_samples} samples, label {fields['true_label']}, "
        f"{len(st.decisions)} device decisions, {len(st.commands)} commands)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="serial port of the ESP32 native USB (e.g. /dev/cu.usbmodem*)")
    src.add_argument("--from-log", help="split an existing serial log into CSV trials")
    ap.add_argument("--out", help="output CSV (live mode)")
    ap.add_argument("--out-dir", help="output directory (--from-log mode)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--label", default=None, help="target label or REST")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--trial-id", default=None)
    ap.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="calibration data chooses T/M; test data is used once at the end",
    )
    ap.add_argument("--note", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "synapsys.yaml"))
    ap.add_argument("--baud", type=int, default=921600, help="ignored by native USB")
    ap.add_argument("--start-timeout", type=float, default=5.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.from_log:
        if not args.out_dir:
            ap.error("--from-log needs --out-dir")
        return from_log(args, cfg)
    if not args.out or args.label is None or args.split is None:
        ap.error("live recording needs --out, --label and --split")
    return live(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
