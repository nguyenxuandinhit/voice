---
title: IndexTTS per-machine tuning and fp16 S2M default
date: 2026-11-04
summary: S2M fp16 default for IndexTTS 2.5 on CUDA plus a Settings section to tune each company machine
---

# IndexTTS per-machine tuning and fp16 S2M default

## What happened
- Colab T4 rounds 2-3 (Japanese narration, 4 lines x 3 runs): S2M fp16 cut inference 52-57% with unchanged Whisper CER, WavLM similarity and UTMOS. CFG 0 / 15 steps added 4-6 points but introduced real misreads (鍵 -> 影) or lower UTMOS. Beam 1 failed (介護 misread). The <10 GB auto split was 12% slower.
- Japanese CER was noisy (Whisper writes 7 for 七, homophone kanji). Added kana-reading CER (pykakasi) to scripts/indextts_quality_eval.py; the gate prefers it.
- faster-whisper on the current Colab image needed CUDA 12 cuBLAS/cuDNN wheels plus LD_LIBRARY_PATH.

## Changes
- backend/engines/indextts/main.py: `_default_s2m_autocast` -> fp16 for 2.5 on CUDA; `OMNIVOICE_INDEXTTS_S2M_AUTOCAST=fp32` opts out.
- backend/services/indextts_tuning.py + GET/PUT /api/settings/perf/indextts: precision, beam, split; env.* prefs; env-pinned refusal (409); stops the idle sidecar.
- Electron: features/settings/indextts-tuning.tsx on the Performance page; 11 keys in all 22 locales.
- tests/test_indextts_tuning.py, sidecar default test, api_routes.txt, mobile/contract/openapi.json (fragment generated with pinned fastapi 0.136.3 / pydantic 2.13.4).

## Verification gaps
- No full backend env locally (no torch/uvicorn): route inventory and OpenAPI export tests cannot run here; Electron renderer not type-checked or built (no node_modules/bun).
- fp16 measured only on T4 and short lines; not measured on company 8 GB GPUs, ROCm, or long paragraphs.

## Next steps
- Run on a company 8 GB GPU: speed, peak VRAM, and LOW_VRAM_SPLIT=0.
- GPT is now 67-80% of time; measure GPT precision on native-bf16 GPUs.
- scripts/performance.py low-vram/colab profiles still set beam 1, which failed the quality gate.

> Historical work record — not durable authority. Prefer docs/specs/ADRs for current decisions.
