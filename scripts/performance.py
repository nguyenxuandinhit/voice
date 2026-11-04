"""Launch a single backend with explicit, conservative performance presets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROFILES = {
    'balanced': {
        'OMNIVOICE_GPU_WORKERS': '1',
        'OMNIVOICE_SINGLE_ENGINE_RESIDENT': '1',
        'OMNIVOICE_INDEXTTS_LAZY_EMOTION': '1',
        'OMNIVOICE_INDEXTTS_FP16': '1',
        'OMNIVOICE_FLASHINFER': '0',
    },
    'low-vram': {
        'OMNIVOICE_FLASHINFER': '0',
        'TORCH_COMPILE_DISABLE': '1',
    },
    'colab': {
        'OMNIVOICE_FLASHINFER': '0',
    },
}


def profile_env(profile: str, base: dict[str, str]) -> dict[str, str]:
    """Explicit launcher profile wins over inherited performance overrides.

    Keep unrelated environment (including auth) intact. A stale workers=4
    must not defeat a profile selected to fit a small GPU.
    """
    if profile not in PROFILES:
        raise ValueError(f'Unknown performance profile: {profile}')
    env = dict(base)
    env.update(PROFILES['balanced'])
    env.update(PROFILES[profile])
    # Beam 1 misread words in the quality rounds (srs.md section 6.3), so every
    # preset keeps the upstream beam and leaves it to Settings → Performance.
    env.pop('OMNIVOICE_INDEXTTS_NUM_BEAMS', None)
    if profile != 'low-vram':
        env.pop('TORCH_COMPILE_DISABLE', None)
    # Optional accelerators must be tested individually, never enabled by a preset.
    env['OMNIVOICE_INDEXTTS_CUDA_KERNEL'] = '0'
    env['OMNIVOICE_INDEXTTS_DEEPSPEED'] = '0'
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='balanced')
    parser.add_argument('--port', type=int, default=3900)
    parser.add_argument('--engine', help='Pin the TTS engine for this launch, e.g. indextts2 for narration')
    parser.add_argument('--dry-run', action='store_true', help='Print only non-secret performance settings')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be from 1 to 65535')
    env = profile_env(args.profile, os.environ)
    if args.engine:
        # Env pin wins over the Settings picker for this process only (prefs.resolve).
        env['OMNIVOICE_TTS_BACKEND'] = args.engine
    keys = set(PROFILES['balanced']) | set(PROFILES['low-vram']) | set(PROFILES['colab'])
    keys.update({'OMNIVOICE_INDEXTTS_CUDA_KERNEL', 'OMNIVOICE_INDEXTTS_DEEPSPEED', 'OMNIVOICE_TTS_BACKEND'})
    print(json.dumps({key: env[key] for key in sorted(keys) if key in env}, indent=2), flush=True)
    if args.dry_run:
        return 0
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run([
            sys.executable, '-m', 'uvicorn', 'main:app', '--app-dir', 'backend',
            '--host', '127.0.0.1', '--port', str(args.port), '--workers', '1',
        ], cwd=root, env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
