# IndexTTS performance build

This local build adds controls for IndexTTS-2 and 2.5. One default differs
from upstream: IndexTTS 2.5 on CUDA runs its S2M stage in fp16. On a Colab T4
(Japanese narration, two rounds) that cut inference time by 52–57% with the
same Whisper CER, WavLM speaker similarity and UTMOS as fp32 (srs.md section
6.3). Everything else keeps upstream defaults, including beam 3.

## Settings → Performance → IndexTTS on this machine

Each machine saves its own choice; the values are `env.*` prefs restored on
start, and an idle IndexTTS sidecar is restarted when one changes (a busy one
picks them up the next time it loads).

| Control | Choices | Default |
|---|---|---|
| Sound rendering precision (S2M) | fp16, fp32 | fp16 |
| Beam search | 3, 1 | 3 |
| Split long sentences | Auto, Off, On | Auto (upstream splits text over 40 characters below 10 GB VRAM) |

A control is disabled when its environment variable is set outside the app.
The same settings are available as `GET`/`PUT /api/settings/perf/indextts`.

## Windows, Linux and macOS

Install the app's source dependencies as documented in [installation](install/windows.md)
and install IndexTTS through Model Catalogue. Use the same backend Python that
has those dependencies, from the repository root:

```powershell
# Preview the exact non-secret settings without loading or downloading models.
.\.venv\Scripts\python.exe scripts/performance.py --profile low-vram --dry-run
# Start the backend, then open http://127.0.0.1:3900.
.\.venv\Scripts\python.exe scripts/performance.py --profile low-vram
```

Add `--engine indextts2` to pin IndexTTS as the TTS engine for that launch,
including audiobook/narration renders, which use the active engine. The pin
overrides the Settings picker until the backend restarts without it.

On Linux/macOS use `.venv/bin/python` instead. Build the Electron web renderer
if `frontend/dist/index.html` is absent (`bun install --frozen-lockfile`, then
`bun run --cwd electron build:web`). The test-only environment created while
developing this patch is **not** a complete app installation.

| Profile | GPU jobs at once | IndexTTS beam | Text-emotion model | Other choices |
|---|---|---|---|---|
| `balanced` | 1 | upstream default (currently 3) | load on first text-emotion request | reduced precision enabled, FlashInfer disabled |
| `low-vram` | 1 | upstream default | load on first text-emotion request | FlashInfer and torch.compile disabled |
| `colab` | 1 | upstream default | load on first text-emotion request | FlashInfer disabled |

Presets override inherited worker, beam and accelerator settings so a stale
configuration cannot accidentally start several GPU jobs. They keep one TTS
engine resident. Balanced/Colab clear an inherited `TORCH_COMPILE_DISABLE`
override, restoring the backend's compilation policy; low-vram sets it to `1`.
They do not select an engine, download models, disable
watermarking, or change server authentication. Reduced precision is still
subject to upstream device capability checks, including the existing BF16 probe.

No preset selects beam 1: it was only ~6% faster in the Colab rounds and
misread words. It remains available in Settings or `OMNIVOICE_INDEXTTS_NUM_BEAMS`.
A preset cannot make a model fit every 4 GB card. If IndexTTS still spills into
shared memory, use another engine or a remote GPU; adding CPU threads does not
replace missing VRAM. Text-emotion guidance can still need extra memory when
first used. Sliders/reference emotion do not require loading the optional 2.5
text-emotion model. Legacy IndexTTS-2 keeps its original emotion loading.

## What can run in parallel?

The upstream GPU queue and serial IndexTTS sidecar are retained. A single
model has mutable reference caches; concurrent calls on it are not safe, and
several model processes duplicate weights. These presets therefore use one GPU
job and one Uvicorn process, including on Colab. Existing CPU/network preparation
can overlap. More GPU jobs only help after measuring free memory and engine
concurrency support; do not launch several Uvicorn workers on a small GPU.

## Google Colab (weak local computer)

Use `notebooks/OmniVoice_Studio_Colab.ipynb` shipped with this patch:

1. Upload the notebook into Colab, select a GPU runtime, and upload the paired
   `voicestudio-performance.patch` into `/content` using the Files sidebar.
2. Run the setup cells. The install cell checks/applies the local patch; an
   already applied patch is reused. A conflicting patch stops without resetting
   your checkout. The release bundle records the base commit to use if needed.
3. The launch cell uses `PERFORMANCE_PROFILE="colab"`. Open the UI through the
   existing Google-session port proxy. The browser runs locally; synthesis and
   storage run on Colab. The model and reference audio are on that remote VM.
4. In Model Catalogue, explicitly install/select IndexTTS and accept its model
   terms to test it; the notebook's default model download is OmniVoice, not
   IndexTTS. Create the reference voice and benchmark the desired language.
5. Download results before ending the session. Colab resources are temporary
   and GPU availability varies. An already-running backend must be stopped
   before changing the profile; re-running the launch cell alone does not
   reconfigure its process.

No public tunnel is added. Colab's session-protected proxy is for the browser;
do not paste it into the desktop Remote backend field and expect API access
without Google's session. For a persistent GPU PC/server use the supported
[remote backend](remote-gpu.md) flow instead.

## Measure before and after

Keep text, voice sample, language, model version and other workloads fixed.
Record the GPU model/free VRAM and verify IndexTTS itself reports `cuda`, not
only the parent backend. Generate once to warm up, then repeat at least twice
under each profile and listen for mistakes.

Each successful IndexTTS call emits one content-free JSON line to backend logs:

```json
{"event":"indextts_performance","device":"cuda:0","num_beams":1,"load_s":0.01,"infer_s":6.0,"export_s":0.1,"audio_s":20.0,"rtf":0.3,"allocated_vram_mb":3000.0}
```

These are illustrative numbers. RTF is inference seconds / audio seconds,
excluding model/emotion load and export. Lower is faster. `allocated_vram_mb`
is the current PyTorch allocation after generation, **not peak VRAM or total
GPU consumption**. Upstream logs additionally report `gpt_gen_time`,
`s2mel_time`, and `bigvgan_time` to locate the expensive stage.

### Per-stage timing (SRS PERF-01)

Set `OMNIVOICE_INDEXTTS_STAGE_TIMING=1` to add a `stages` object and
`peak_vram_mb` to that line: `speaker_s` (reference-voice conditioning),
`gpt_s` (autoregressive code generation), `s2m_s` (semantic-to-mel diffusion),
`vocoder_s` (BigVGAN) and `other_s` (the rest of `infer()`: text processing,
glue, CPU work). A stage shows `null` when the installed upstream build has no
matching method. Timing synchronizes CUDA at stage boundaries, so leave it off
for production runs. Peak VRAM is PyTorch's peak for that call only.

The benchmark script spawns the sidecar directly with timing on, runs a fixed
text file (one segment per line) and prints per-stage totals and shares:

```powershell
.\.venv\Scripts\python.exe scripts/indextts_stage_benchmark.py `
  --texts bench.txt --voice ref.wav --lang en --runs 3 `
  --out-dir bench_out --json baseline.json
# Same set, one change at a time:
.\.venv\Scripts\python.exe scripts/indextts_stage_benchmark.py `
  --texts bench.txt --voice ref.wav --lang en --runs 3 `
  --env OMNIVOICE_INDEXTTS_CUDA_KERNEL=1 --out-dir bench_kernel --json kernel.json
```

`--python` selects the IndexTTS venv interpreter when the app's resolver cannot
find it (set `OMNIVOICE_INDEXTTS_DIR` as for the app). `--out-dir` saves the
first pass as WAVs for A/B listening. `--log` writes sidecar stderr, which can
include upstream logs. Do not share it without checking. IndexTTS 2.5 accepts
`zh`, `en`, `ja`, `es` and `ar` only.

## Optional experiments (not automatically enabled)

For a manually launched backend, operator environment variables now support:

| Variable | Default | Meaning |
|---|---|---|
| `OMNIVOICE_INDEXTTS_NUM_BEAMS` | unset | 1–16; unset preserves upstream beam setting |
| `OMNIVOICE_INDEXTTS_LAZY_EMOTION` | `0` | `1` defers 2.5's text-emotion model until needed |
| `OMNIVOICE_INDEXTTS_CUDA_KERNEL` | `0` | `1` requests BigVGAN's custom CUDA kernel |
| `OMNIVOICE_INDEXTTS_DEEPSPEED` | `0` | `1` requests the upstream DeepSpeed path |
| `OMNIVOICE_INDEXTTS_STAGE_TIMING` | `0` | `1` logs per-stage timings and peak VRAM |
| `OMNIVOICE_INDEXTTS_TORCH_COMPILE` | `0` | `1` asks 2.5 to `torch.compile` its S2M estimator |
| `OMNIVOICE_INDEXTTS_S2M_AUTOCAST` | unset (fp16 for 2.5 on CUDA) | `fp16`/`bf16` runs S2M in reduced precision; `fp32` keeps upstream precision |
| `OMNIVOICE_INDEXTTS_S2M_STEPS` | unset (25) | 1–100 diffusion steps; fewer is faster and can lower quality |
| `OMNIVOICE_INDEXTTS_S2M_CFG` | unset (0.7) | 0–2 S2M guidance rate; 0 skips the guided half of the batch |
| `OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT` | unset | `0`/`1` forces 2.5's < 10 GB rule that splits text over 40 characters |

Upstream 2.5 hardcodes the S2M steps and guidance inside `infer()`; the
sidecar applies these overrides to the S2M call only when set. Every
performance line includes a `config` object recording the values in effect.
Score any trade-off with `scripts/indextts_quality_eval.py` (Whisper CER,
WavLM speaker similarity, UTMOS; with `pykakasi` installed, Japanese CER is
also compared by kana reading) against the default before adopting it;
these automatic scores do not replace a native listener.

Restart the backend after changing these. CUDA kernels/DeepSpeed need compatible
engine dependencies and may fail or run slower on a particular host; no extra
packages are installed by these flags. Benchmark one change at a time. The
conservative launcher deliberately disables both experimental accelerators.
