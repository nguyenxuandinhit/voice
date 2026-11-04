"""Presets limit GPU contention and never print credentials."""
import importlib.util
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('performance_launcher', ROOT / 'scripts/performance.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('profile', launcher.PROFILES)
def test_profiles_override_unsafe_inherited_workers_without_mutating_env(profile):
    base = {'OMNIVOICE_GPU_WORKERS': '8', 'OMNIVOICE_API_KEY': 'secret'}
    env = launcher.profile_env(profile, base)
    assert env['OMNIVOICE_GPU_WORKERS'] == '1'
    assert env['OMNIVOICE_SINGLE_ENGINE_RESIDENT'] == '1'
    assert env['OMNIVOICE_INDEXTTS_LAZY_EMOTION'] == '1'
    assert env['OMNIVOICE_API_KEY'] == 'secret'
    assert base['OMNIVOICE_GPU_WORKERS'] == '8'


def test_every_profile_restores_upstream_beam_default():
    for profile in launcher.PROFILES:
        env = launcher.profile_env(profile, {'OMNIVOICE_INDEXTTS_NUM_BEAMS': '1'})
        assert 'OMNIVOICE_INDEXTTS_NUM_BEAMS' not in env
    assert launcher.profile_env('low-vram', {})['TORCH_COMPILE_DISABLE'] == '1'


def test_switching_profiles_does_not_inherit_accelerator_settings():
    env = launcher.profile_env('low-vram', {'OMNIVOICE_FLASHINFER': 'graph'})
    for profile in ('balanced', 'colab'):
        selected = launcher.profile_env(profile, env)
        assert selected['OMNIVOICE_FLASHINFER'] == '0'
        assert 'TORCH_COMPILE_DISABLE' not in selected


def test_dry_run_prints_only_performance_settings(monkeypatch, capsys):
    monkeypatch.setenv('OMNIVOICE_API_KEY', 'DO-NOT-PRINT')
    assert launcher.main(['--profile', 'colab', '--dry-run']) == 0
    output = capsys.readouterr().out
    assert 'DO-NOT-PRINT' not in output
    assert json.loads(output)['OMNIVOICE_FLASHINFER'] == '0'


def test_launch_uses_same_interpreter_one_server_process_and_loopback(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)) or types.SimpleNamespace(returncode=7))
    assert launcher.main(['--profile', 'low-vram']) == 7
    argv = calls[0][0][0]
    assert argv[0] == launcher.sys.executable
    assert argv[argv.index('--workers') + 1] == '1'
    assert argv[argv.index('--host') + 1] == '127.0.0.1'
    assert calls[0][1]['env']['OMNIVOICE_GPU_WORKERS'] == '1'


def test_invalid_port_does_not_start_backend():
    with pytest.raises(SystemExit):
        launcher.main(['--port', '0'])


def test_engine_flag_pins_tts_backend_for_launch(monkeypatch, capsys):
    monkeypatch.delenv('OMNIVOICE_TTS_BACKEND', raising=False)
    assert launcher.main(['--engine', 'indextts2', '--dry-run']) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings['OMNIVOICE_TTS_BACKEND'] == 'indextts2'
    assert 'OMNIVOICE_INDEXTTS_NUM_BEAMS' not in settings  # balanced keeps beam default
    assert launcher.main(['--dry-run']) == 0
    assert 'OMNIVOICE_TTS_BACKEND' not in json.loads(capsys.readouterr().out)
