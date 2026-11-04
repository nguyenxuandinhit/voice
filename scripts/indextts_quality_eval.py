"""Score benchmark WAVs for accuracy, voice similarity and naturalness.

Reads the folders written by ``scripts/indextts_stage_benchmark.py --out-dir``
(one folder per configuration, e.g. ``results/base``) and compares every
configuration with a baseline:

* CER: Whisper transcribes each WAV; character error rate against the script
  (lower is better). Catches skipped, repeated or wrong words. For Japanese
  (``pip install pykakasi``) ``cer_reading`` also compares hiragana readings, so
  Whisper writing 7 for 七 or a homophone kanji is not counted; the gate uses it.
* Speaker similarity: cosine of WavLM speaker embeddings, output vs. the
  reference voice (higher is better).
* UTMOS: predicted naturalness on a 1-5 scale (higher is better).

These are automatic proxies. They do not judge emotion or context and do not
replace a native listener before a production decision. UTMOS is trained on
English speech, so use it to compare configurations, not as an absolute score.

Runs in a normal GPU Python with torch (not the IndexTTS venv):

    pip install faster-whisper pykakasi
    python scripts/indextts_quality_eval.py --results results --texts bench.txt \
        --voice ref.wav --lang ja --json quality.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

#: Proposed acceptance gate versus the baseline; confirm before production use.
DEFAULT_GATE = {"cer_increase": 0.02, "similarity_drop": 0.02, "utmos_drop": 0.1}
_SEGMENT = re.compile(r"^seg_(\d+)(?:_r\d+)?\.wav$")
_KANJI_DIGITS = str.maketrans("0123456789", "〇一二三四五六七八九")


def load_texts(path: Path) -> list[str]:
    """Same parsing as the benchmark so segment indices line up."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def normalize(text: str) -> str:
    """NFKC, lowercase, drop punctuation, symbols and whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZ")


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def reading(text: str, to_kana) -> str:
    """Kana reading: digits as kanji numerals first so 7年 and 七年 read alike."""
    return to_kana(unicodedata.normalize("NFKC", text).translate(_KANJI_DIGITS))


def japanese_reader():
    """Hiragana converter from pykakasi, or None when it is not installed."""
    try:
        import pykakasi
    except ImportError:
        return None
    kks = pykakasi.kakasi()
    return lambda text: "".join(item["hira"] for item in kks.convert(text))


def segment_index(name: str) -> int | None:
    match = _SEGMENT.match(name)
    return int(match.group(1)) if match else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def gate(candidate: dict, baseline: dict, limits: dict = DEFAULT_GATE) -> tuple[bool, list[str]]:
    """Pass only if no metric regresses past its limit versus the baseline."""
    reasons = []
    key = "cer_reading" if candidate.get("cer_reading") is not None         and baseline.get("cer_reading") is not None else "cer"
    if candidate[key] - baseline[key] > limits["cer_increase"]:
        reasons.append(f"{key.upper()} +{candidate[key] - baseline[key]:.3f}")
    if baseline["similarity"] - candidate["similarity"] > limits["similarity_drop"]:
        reasons.append(f"similarity -{baseline['similarity'] - candidate['similarity']:.3f}")
    if baseline["utmos"] - candidate["utmos"] > limits["utmos_drop"]:
        reasons.append(f"UTMOS -{baseline['utmos'] - candidate['utmos']:.3f}")
    return not reasons, reasons


class Scorer:
    """Loads the three models once; imports stay lazy so tests need no torch."""

    def __init__(self, whisper_model: str, device: str):
        import torch
        from faster_whisper import WhisperModel
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch, self.device = torch, device
        self.asr = WhisperModel(whisper_model, device=device,
                                compute_type="float16" if device == "cuda" else "int8")
        self.sv_features = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        self.to_kana = None
        self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    @staticmethod
    def load(path: Path):
        import librosa
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        return audio

    def transcribe(self, audio, lang: str) -> str:
        segments, _ = self.asr.transcribe(audio, language=lang, beam_size=5,
                                          condition_on_previous_text=False)
        return "".join(segment.text for segment in segments)

    def embed(self, audio):
        torch = self.torch
        inputs = self.sv_features(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.sv(**inputs).embeddings
        return torch.nn.functional.normalize(emb, dim=-1)[0]

    def utmos(self, audio) -> float:
        torch = self.torch
        with torch.no_grad():
            wave = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            return float(self.mos(wave, 16000).item())


def evaluate(scorer: Scorer, folder: Path, texts: list[str], voice_emb, lang: str) -> dict:
    to_kana = scorer.to_kana if lang == "ja" else None
    files = []
    for wav in sorted(folder.glob("seg_*.wav")):
        idx = segment_index(wav.name)
        if idx is None or idx >= len(texts):
            continue
        audio = scorer.load(wav)
        hyp = scorer.transcribe(audio, lang)
        files.append({
            "file": wav.name,
            "cer": round(cer(texts[idx], hyp), 4),
            "cer_reading": round(cer(reading(texts[idx], to_kana), reading(hyp, to_kana)), 4)
            if to_kana else None,
            "similarity": round(float(scorer.embed(audio) @ voice_emb), 4),
            "utmos": round(scorer.utmos(audio), 4),
            "transcript": hyp,
        })
    return {
        "files": files,
        "cer": mean([f["cer"] for f in files]),
        "cer_reading": mean([f["cer_reading"] for f in files if f["cer_reading"] is not None]),
        "similarity": mean([f["similarity"] for f in files]),
        "utmos": mean([f["utmos"] for f in files]),
        "worst_cer": max((f["cer"] for f in files), default=None),
    }


def summarize(results_dir: Path, scores: dict, baseline: str) -> list[dict]:
    """Join quality with the benchmark's speed JSON and apply the gate."""
    base = scores[baseline]
    base_speed = _speed(results_dir, baseline)
    rows = []
    for name, score in scores.items():
        speed = _speed(results_dir, name)
        row = {"config": name, "n": len(score["files"]), "cer": score["cer"],
               "worst_cer": score["worst_cer"], "cer_reading": score.get("cer_reading"),
               "similarity": score["similarity"],
               "utmos": score["utmos"], "rtf": speed.get("rtf")}
        if base_speed.get("infer_s") and speed.get("infer_s"):
            row["time_saved"] = round(1 - speed["infer_s"] / base_speed["infer_s"], 3)
        if name == baseline:
            row["gate"] = "baseline"
        else:
            ok, reasons = gate(score, base)
            row["gate"] = "PASS" if ok else "FAIL: " + ", ".join(reasons)
        rows.append(row)
    return rows


def _speed(results_dir: Path, name: str) -> dict:
    path = results_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Folder holding one subfolder per configuration")
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--configs", nargs="*", help="Subfolders to score (default: all)")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    names = args.configs or sorted(p.name for p in args.results.iterdir()
                                   if p.is_dir() and any(p.glob("seg_*.wav")))
    if args.baseline not in names:
        parser.error(f"baseline folder {args.baseline!r} not found in {args.results}")
    texts = load_texts(args.texts)
    scorer = Scorer(args.whisper_model, args.device)
    if args.lang == "ja":
        scorer.to_kana = japanese_reader()
        if scorer.to_kana is None:
            print("pykakasi not installed: Japanese CER compares written characters only", flush=True)
    voice_emb = scorer.embed(scorer.load(args.voice))
    scores = {}
    for name in names:
        print(f"scoring {name} ...", flush=True)
        scores[name] = evaluate(scorer, args.results / name, texts, voice_emb, args.lang)
    rows = summarize(args.results, scores, args.baseline)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps({"summary": rows, "details": scores}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
