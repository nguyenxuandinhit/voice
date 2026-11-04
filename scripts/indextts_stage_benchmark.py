"""Benchmark the IndexTTS sidecar per stage (GPT, S2M, BigVGAN, speaker).

Spawns ``backend/engines/indextts/main.py`` directly with stage timing on,
synthesizes a fixed text set and summarizes the sidecar's content-free
``indextts_performance`` log lines. Stdlib only: run it with any Python.

    python scripts/indextts_stage_benchmark.py --texts bench.txt --voice ref.wav \
        --python <IndexTTS venv python> --lang en --runs 3 --out-dir bench_out

``--texts`` holds one segment per line; blank lines and ``#`` lines are ignored.
Use ``--env KEY=VALUE`` (repeatable) to compare one operator setting, e.g.
``--env OMNIVOICE_INDEXTTS_CUDA_KERNEL=1``. Keep everything else fixed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "backend" / "engines" / "indextts" / "main.py"
STAGES = ("speaker_s", "gpt_s", "s2m_s", "vocoder_s", "other_s")


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    stream.write(struct.pack("!I", len(body)) + body)
    stream.flush()


def _recv(stream) -> dict:
    header = stream.read(4)
    if len(header) < 4:
        raise RuntimeError("sidecar exited; rerun with --log to see its stderr")
    (n,) = struct.unpack("!I", header)
    body = b""
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise RuntimeError("short read from sidecar")
        body += chunk
    return json.loads(body.decode("utf-8"))


def _drain_stderr(stream, records: queue.Queue, log) -> None:
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line.startswith("{") and '"indextts_performance"' in line:
            try:
                records.put(json.loads(line))
                continue
            except ValueError:
                pass
        if log:
            log.write(line + "\n")
            log.flush()


def _resolve_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from engines.indextts.bootstrap import resolve_indextts_venv
        return str(resolve_indextts_venv())
    except Exception as exc:  # noqa: BLE001 - surface any resolution failure
        raise SystemExit(f"Cannot locate the IndexTTS venv ({exc}); pass --python") from exc


def _load_texts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not texts:
        raise SystemExit(f"No segments in {path}")
    return texts


def _parse_env(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.startswith("OMNIVOICE_INDEXTTS_"):
            raise SystemExit(f"--env expects OMNIVOICE_INDEXTTS_*=VALUE, got {pair!r}")
        env[key] = value
    return env


def _write_wav(path: Path, frame: dict) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(frame["sample_rate"])
        out.writeframes(base64.b64decode(frame["audio_pcm_b64"]))


def _synthesize(proc, records: queue.Queue, msg: dict, timeout_s: float) -> tuple[dict, dict, float]:
    started = time.perf_counter()
    _send(proc.stdin, msg)
    while True:
        frame = _recv(proc.stdout)
        if frame.get("op") == "audio":
            break
        if frame.get("op") == "error":
            raise RuntimeError(f"sidecar error: {frame.get('message')}")
    wall_s = time.perf_counter() - started
    try:
        record = records.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise RuntimeError("no indextts_performance line; is the sidecar up to date?") from exc
    return frame, record, wall_s


def summarize(records: list[dict]) -> dict:
    """Aggregate measured (warm) records into totals, shares and RTF."""
    infer = sum(r["infer_s"] for r in records)
    audio = sum(r["audio_s"] for r in records)
    stages = {}
    for stage in STAGES:
        values = [(r.get("stages") or {}).get(stage) for r in records]
        if any(v is None for v in values):
            stages[stage] = None
            continue
        total = sum(values)
        stages[stage] = {
            "total_s": round(total, 3),
            "mean_s": round(total / len(values), 3),
            "share": round(total / infer, 3) if infer else None,
        }
    peaks = [r["peak_vram_mb"] for r in records if r.get("peak_vram_mb") is not None]
    return {
        "segments": len(records),
        "infer_s": round(infer, 3),
        "audio_s": round(audio, 3),
        "rtf": round(infer / audio, 4) if audio else None,
        "export_s": round(sum(r["export_s"] for r in records), 3),
        "stages": stages,
        "peak_vram_mb": max(peaks) if peaks else None,
        "num_beams": records[0].get("num_beams") if records else None,
        "device": records[0].get("device") if records else None,
        "config": records[0].get("config") if records else None,
    }


def _print_summary(summary: dict, cold: dict | None) -> None:
    print(f"\nDevice {summary['device']}, beams {summary['num_beams']}, "
          f"{summary['segments']} warm segments")
    if cold:
        print(f"Cold model load: {cold['load_s']:.1f}s")
    print(f"Inference {summary['infer_s']:.2f}s for {summary['audio_s']:.2f}s audio "
          f"-> RTF {summary['rtf']}   (export {summary['export_s']:.2f}s)")
    print(f"Peak VRAM: {summary['peak_vram_mb']} MB")
    print(f"{'stage':<11}{'total s':>10}{'mean s':>10}{'share':>8}")
    for stage, value in summary["stages"].items():
        if value is None:
            print(f"{stage:<11}{'n/a (not hooked)':>28}")
        else:
            print(f"{stage:<11}{value['total_s']:>10.2f}{value['mean_s']:>10.3f}{(value['share'] or 0):>8.0%}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True, help="Reference speaker WAV")
    parser.add_argument("--lang", default="en", help="IndexTTS 2.5 language token (zh/en/ja/es/ar)")
    parser.add_argument("--python", help="IndexTTS venv interpreter (default: app's resolver)")
    parser.add_argument("--runs", type=int, default=3, help="Measured passes over the text set")
    parser.add_argument("--warmup", type=int, default=1, help="Unrecorded warm-up synthesis count")
    parser.add_argument("--env", action="append", default=[], help="OMNIVOICE_INDEXTTS_*=VALUE override")
    parser.add_argument("--out-dir", type=Path, help="Save every pass as WAV (seg_000.wav, seg_000_r1.wav, ...) for listening and quality scoring")
    parser.add_argument("--json", type=Path, help="Write summary JSON here")
    parser.add_argument("--log", type=Path, help="Write sidecar stderr here (may contain upstream text logs)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for a metrics line")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup >= 0")
    if not args.voice.is_file():
        parser.error(f"--voice not found: {args.voice}")

    texts = _load_texts(args.texts)
    overrides = _parse_env(args.env)
    env = {**os.environ, **overrides, "PYTHONUNBUFFERED": "1",
           "OMNIVOICE_INDEXTTS_STAGE_TIMING": "1"}
    python = _resolve_python(args.python)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    log = args.log.open("w", encoding="utf-8") if args.log else None
    proc = subprocess.Popen([python, str(SIDECAR)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, bufsize=0)
    records: queue.Queue = queue.Queue()
    threading.Thread(target=_drain_stderr, args=(proc.stderr, records, log), daemon=True).start()
    try:
        if _recv(proc.stdout).get("op") != "ready":
            raise RuntimeError("sidecar did not send ready")
        base = {"op": "synthesize", "ref_audio": str(args.voice.resolve()), "lang": args.lang}
        cold = None
        for i in range(args.warmup):
            _, record, _ = _synthesize(proc, records, {**base, "text": texts[i % len(texts)]}, args.timeout + 900)
            cold = cold or record
            print(f"warm-up {i + 1}/{args.warmup} done", flush=True)
        measured, walls = [], []
        for run in range(args.runs):
            for idx, text in enumerate(texts):
                frame, record, wall_s = _synthesize(proc, records, {**base, "text": text}, args.timeout)
                measured.append(record)
                walls.append(wall_s)
                if args.out_dir:
                    suffix = f"_r{run}" if run else ""
                    _write_wav(args.out_dir / f"seg_{idx:03d}{suffix}.wav", frame)
                print(f"run {run + 1}/{args.runs} segment {idx + 1}/{len(texts)} "
                      f"({len(text)} chars): infer {record['infer_s']:.2f}s, rtf {record['rtf']}", flush=True)
        _send(proc.stdin, {"op": "shutdown"})
    finally:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log:
            log.close()

    summary = summarize(measured)
    summary["wall_s"] = round(sum(walls), 3)
    summary["overrides"] = overrides
    if cold:
        summary["cold_load_s"] = cold["load_s"]
    _print_summary(summary, cold)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
