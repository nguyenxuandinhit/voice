"""Performance controls must reach the isolated engine without changing defaults."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sidecar(monkeypatch):
    for key in list(__import__('os').environ):
        if key.startswith('OMNIVOICE_INDEXTTS_'):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location(
        '_performance_sidecar', ROOT / 'backend/engines/indextts/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_torch_bf16_supported', lambda: True)
    return module


def test_opt_in_beam_reaches_both_engine_versions(sidecar, monkeypatch):
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_NUM_BEAMS', '1')
    for version in (True, False):
        kw = sidecar._build_infer_kwargs({'text': 'hello', 'num_beams': 99}, 'ref.wav', is_v25=version)
        assert kw['num_beams'] == 1  # trusted operator config, never wire input


@pytest.mark.parametrize('value', ['0', '-1', '17', '1.5', 'invalid'])
def test_invalid_beam_fails_before_inference(sidecar, monkeypatch, value):
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_NUM_BEAMS', value)
    with pytest.raises(ValueError, match='OMNIVOICE_INDEXTTS_NUM_BEAMS'):
        sidecar._build_infer_kwargs({'text': 'hello'}, 'ref.wav', is_v25=True)


def test_defaults_preserve_existing_quality_and_emotion(sidecar):
    kw = sidecar._model_init_kwargs('/model', version='2.5', reduced_precision=True)
    assert kw['use_qwen_emo'] is True
    assert kw['use_deepspeed'] is False
    assert kw['use_cuda_kernel'] is False
    assert 'num_beams' not in sidecar._build_infer_kwargs({'text': 'hi'}, 'ref.wav', is_v25=True)


def test_accelerators_and_lazy_emotion_are_operator_opt_ins(sidecar, monkeypatch):
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_CUDA_KERNEL', '1')
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_DEEPSPEED', '1')
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_LAZY_EMOTION', '1')
    kw = sidecar._model_init_kwargs('/model', version='2.5', reduced_precision=True)
    assert kw['use_qwen_emo'] is False
    assert kw['use_cuda_kernel'] is True
    assert kw['use_deepspeed'] is True
    assert 'use_qwen_emo' not in sidecar._model_init_kwargs('/model', version='2', reduced_precision=True)


def test_lazy_emotion_loads_once_and_only_for_text_guidance(sidecar, monkeypatch):
    calls = []
    upstream = types.ModuleType('indextts.infer_v2_5')
    upstream.QwenEmotion = lambda path: calls.append(path) or object()
    monkeypatch.setitem(sys.modules, 'indextts.infer_v2_5', upstream)
    monkeypatch.setattr(sidecar, '_model_version', '2.5')
    monkeypatch.setattr(sidecar, '_heartbeat', lambda *args: contextlib.nullcontext())
    model = types.SimpleNamespace(qwen_emo=None, model_dir='/model', cfg=types.SimpleNamespace(qwen_emo_path='emotion'))
    sidecar._ensure_text_emotion(model, {}, io.BytesIO())
    sidecar._ensure_text_emotion(model, {'emo_vector': [0] * 8}, io.BytesIO())
    assert calls == []
    for _ in range(2):
        sidecar._ensure_text_emotion(model, {'use_emo_text': True}, io.BytesIO())
    assert len(calls) == 1
    assert model.qwen_emo is not None


def test_failed_emotion_load_can_be_retried(sidecar, monkeypatch):
    upstream = types.ModuleType('indextts.infer_v2_5')
    def fail(path):
        raise RuntimeError('model unavailable')
    upstream.QwenEmotion = fail
    monkeypatch.setitem(sys.modules, 'indextts.infer_v2_5', upstream)
    monkeypatch.setattr(sidecar, '_model_version', '2.5')
    monkeypatch.setattr(sidecar, '_heartbeat', lambda *args: contextlib.nullcontext())
    model = types.SimpleNamespace(qwen_emo=None, model_dir='/model', cfg=types.SimpleNamespace(qwen_emo_path='emotion'))
    with pytest.raises(RuntimeError):
        sidecar._ensure_text_emotion(model, {'use_emo_text': True}, io.BytesIO())
    assert model.qwen_emo is None


def test_metrics_do_not_include_content_or_paths(sidecar, capsys):
    model = types.SimpleNamespace(device='cpu')
    sidecar._log_performance(model, load_s=2, infer_s=3, export_s=1, n_samples=48000, sr=24000)
    result = json.loads(capsys.readouterr().err.strip())
    assert result['event'] == 'indextts_performance'
    assert result['rtf'] == 1.5
    assert result['audio_s'] == 2
    assert result['device'] == 'cpu'
    assert 'text' not in result and 'ref_audio' not in result


def test_synthesis_reports_metrics_and_preserves_audio_protocol(sidecar, monkeypatch):
    calls = []
    model = types.SimpleNamespace(device='cpu', infer=lambda **kw: calls.append(kw))
    monkeypatch.setattr(sidecar, '_load_model', lambda stdout: model)
    monkeypatch.setattr(sidecar, '_ensure_text_emotion', lambda *args: None)
    monkeypatch.setattr(sidecar, '_heartbeat', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(sidecar, '_wav_to_pcm_b64', lambda path: ('audio', 24000, 48000))
    metrics = []
    monkeypatch.setattr(sidecar, '_log_performance', lambda *args, **kw: metrics.append(kw))
    frames = []
    monkeypatch.setattr(sidecar, '_send', lambda stream, frame: frames.append(frame))
    sidecar._handle_synthesize({'text': 'hello', 'ref_audio': 'ref.wav'}, io.BytesIO())
    assert len(calls) == len(metrics) == 1
    assert frames[-1]['op'] == 'audio'
    assert frames[-1]['n_samples'] == 48000
    assert metrics[0]['infer_s'] >= 0


def _fake_model(sidecar):
    calls = []

    class Gpt:
        def inference_speech(self):
            calls.append('gen')
            return self.forward()  # nested hooked call must not double-count

        def forward(self):
            calls.append('fwd')
            return 'codes'

    cfm = types.SimpleNamespace(inference=lambda: calls.append('cfm'))
    s2mel = types.SimpleNamespace(models={'cfm': cfm})
    vocoder = types.SimpleNamespace(forward=lambda: calls.append('voc'))
    return types.SimpleNamespace(gpt=Gpt(), s2mel=s2mel, bigvgan=vocoder), calls


def test_stage_timing_is_opt_in(sidecar):
    model, _ = _fake_model(sidecar)
    assert sidecar._stage_timer_for(model) is None
    assert 'inference_speech' not in vars(model.gpt)


def test_stage_timer_hooks_existing_stages_once(sidecar, monkeypatch):
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_STAGE_TIMING', '1')
    model, calls = _fake_model(sidecar)
    timer = sidecar._stage_timer_for(model)
    assert sidecar._stage_timer_for(model) is timer  # hooks installed once
    assert model.gpt.inference_speech() == 'codes'
    model.s2mel.models['cfm'].inference()
    model.bigvgan.forward()
    assert calls == ['gen', 'fwd', 'cfm', 'voc']
    report = timer.report(10.0)
    assert report['speaker_s'] is None  # not present on this model
    for stage in ('gpt_s', 's2m_s', 'vocoder_s'):
        assert 0 <= report[stage] < 1
    assert set(report) == {'speaker_s', 'gpt_s', 's2m_s', 'vocoder_s', 'other_s'}
    assert report['other_s'] > 9
    sidecar._stage_timer_for(model)
    assert timer.totals == {}  # reset per request


def test_stage_metrics_stay_content_free(sidecar, capsys):
    stages = {'speaker_s': None, 'gpt_s': 1.0, 's2m_s': 1.0, 'vocoder_s': 0.5, 'other_s': 0.5}
    sidecar._log_performance(types.SimpleNamespace(device='cpu'), load_s=0, infer_s=3,
                             export_s=0, n_samples=24000, sr=24000, stages=stages)
    result = json.loads(capsys.readouterr().err.strip())
    assert result['stages'] == stages
    assert 'peak_vram_mb' in result
    assert 'text' not in result and 'ref_audio' not in result


def test_benchmark_summary_shares_and_rtf():
    spec = importlib.util.spec_from_file_location('_stage_bench', ROOT / 'scripts/indextts_stage_benchmark.py')
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    record = {'infer_s': 4.0, 'audio_s': 8.0, 'export_s': 0.1, 'num_beams': 3, 'device': 'cuda:0',
              'peak_vram_mb': 5000.0,
              'stages': {'speaker_s': 0.0, 'gpt_s': 2.0, 's2m_s': 1.0, 'vocoder_s': 0.5, 'other_s': 0.5}}
    summary = bench.summarize([record, {**record, 'peak_vram_mb': 5200.0}])
    assert summary['rtf'] == 0.5
    assert summary['stages']['gpt_s'] == {'total_s': 4.0, 'mean_s': 2.0, 'share': 0.5}
    assert summary['peak_vram_mb'] == 5200.0
    missing = bench.summarize([{**record, 'stages': {**record['stages'], 'speaker_s': None}}])
    assert missing['stages']['speaker_s'] is None


def _fake_cfm_model(calls):
    def inference(mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        calls.append((n_timesteps, inference_cfg_rate))
        return 'mel'
    cfm = types.SimpleNamespace(inference=inference)
    return types.SimpleNamespace(s2mel=types.SimpleNamespace(models={'cfm': cfm}), low_vram=True, use_bf16=True)


def test_s2m_defaults_leave_upstream_call_untouched(sidecar):
    calls = []
    model = _fake_cfm_model(calls)
    original = model.s2mel.models['cfm'].inference
    sidecar._apply_runtime_overrides(model)
    assert model.s2mel.models['cfm'].inference is original
    assert model.low_vram is True
    config = sidecar._runtime_config(model)
    assert config['s2m_steps'] == 25 and config['s2m_cfg'] == 0.7 and config['s2m_autocast'] is None


def test_s2m_steps_and_cfg_override_positional_and_keyword_args(sidecar, monkeypatch):
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_S2M_STEPS', '12')
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_S2M_CFG', '0')
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT', '0')
    calls = []
    model = _fake_cfm_model(calls)
    sidecar._apply_runtime_overrides(model)
    # Upstream passes the step count positionally and CFG by keyword.
    assert model.s2mel.models['cfm'].inference(1, 2, 3, 4, None, 25, inference_cfg_rate=0.7) == 'mel'
    assert calls == [(12, 0.0)]
    assert model.low_vram is False
    config = sidecar._runtime_config(model)
    assert (config['s2m_steps'], config['s2m_cfg'], config['low_vram_split']) == (12, 0.0, False)


@pytest.mark.parametrize('key,value', [
    ('OMNIVOICE_INDEXTTS_S2M_STEPS', '0'), ('OMNIVOICE_INDEXTTS_S2M_STEPS', 'ten'),
    ('OMNIVOICE_INDEXTTS_S2M_CFG', '-1'), ('OMNIVOICE_INDEXTTS_S2M_CFG', 'nan'),
    ('OMNIVOICE_INDEXTTS_S2M_AUTOCAST', 'fp8'), ('OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT', 'yes'),
])
def test_invalid_s2m_settings_fail_before_inference(sidecar, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    model = types.SimpleNamespace(device='cpu', infer=lambda **kw: pytest.fail('must not infer'))
    monkeypatch.setattr(sidecar, '_load_model', lambda stdout: pytest.fail('must not load'))
    with pytest.raises(ValueError, match=key):
        sidecar._handle_synthesize({'text': 'hi', 'ref_audio': 'ref.wav'}, io.BytesIO())


def test_torch_compile_is_an_opt_in_constructor_flag(sidecar, monkeypatch):
    assert sidecar._model_init_kwargs('/m', version='2.5', reduced_precision=True)['use_torch_compile'] is False
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_TORCH_COMPILE', '1')
    assert sidecar._model_init_kwargs('/m', version='2.5', reduced_precision=True)['use_torch_compile'] is True
    assert 'use_torch_compile' not in sidecar._model_init_kwargs('/m', version='2', reduced_precision=True)


def _quality_module():
    spec = importlib.util.spec_from_file_location('_quality', ROOT / 'scripts/indextts_quality_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cer_ignores_punctuation_width_and_spacing():
    q = _quality_module()
    assert q.cer('え……どうして？七年間。', 'え どうして 七年間') == 0
    assert q.cer('ＡＢＣ', 'abc') == 0
    assert q.cer('abcd', 'abed') == 0.25
    assert q.cer('abcd', 'abcdabcd') == 1.0  # repetition is penalized
    assert q.segment_index('seg_003_r2.wav') == 3 and q.segment_index('notes.wav') is None


def test_quality_gate_flags_each_regression(tmp_path):
    q = _quality_module()
    base = {'cer': 0.05, 'similarity': 0.9, 'utmos': 3.5, 'worst_cer': 0.1, 'files': [{}]}
    assert q.gate({**base, 'cer': 0.06}, base) == (True, [])
    ok, reasons = q.gate({**base, 'cer': 0.1, 'similarity': 0.85, 'utmos': 3.3}, base)
    assert not ok and len(reasons) == 3
    (tmp_path / 'base.json').write_text(json.dumps({'infer_s': 100, 'rtf': 4.0}))
    (tmp_path / 'fast.json').write_text(json.dumps({'infer_s': 70, 'rtf': 2.8}))
    rows = q.summarize(tmp_path, {'base': base, 'fast': {**base, 'utmos': 3.0}}, 'base')
    fast = next(r for r in rows if r['config'] == 'fast')
    assert fast['time_saved'] == 0.3 and fast['gate'].startswith('FAIL')


def test_reading_cer_ignores_digit_and_homophone_spelling():
    q = _quality_module()
    kana = {'七年間': 'しちねんかん', 'お義父さん': 'おぎふさん', 'お岐阜さん': 'おぎふさん'}.get
    assert q.reading('7年間', kana) == q.reading('七年間', kana) == 'しちねんかん'
    assert q.cer(q.reading('お義父さん', kana), q.reading('お岐阜さん', kana)) == 0


def test_quality_gate_prefers_reading_cer_when_both_have_it():
    q = _quality_module()
    base = {'cer': 0.04, 'cer_reading': 0.0, 'similarity': 0.9, 'utmos': 3.5}
    assert q.gate({**base, 'cer': 0.09}, base) == (True, [])
    ok, reasons = q.gate({**base, 'cer_reading': 0.05}, base)
    assert not ok and reasons[0].startswith('CER_READING')
    assert not q.gate({**base, 'cer': 0.09}, {**base, 'cer_reading': None})[0]


def test_s2m_fp16_is_the_cuda_default_for_25_only(sidecar, monkeypatch):
    calls = []
    model = _fake_cfm_model(calls)
    model.device = 'cuda:0'
    monkeypatch.setattr(sidecar, '_model_version', '2.5')
    assert sidecar._runtime_config(model)['s2m_autocast'] == 'fp16'
    for device, version in (('cpu', '2.5'), ('cuda:0', '2')):
        monkeypatch.setattr(sidecar, '_model_version', version)
        model.device = device
        assert sidecar._runtime_config(model)['s2m_autocast'] is None
    monkeypatch.setattr(sidecar, '_model_version', '2.5')
    model.device = 'cuda:0'
    monkeypatch.setenv('OMNIVOICE_INDEXTTS_S2M_AUTOCAST', 'fp32')
    original = model.s2mel.models['cfm'].inference
    sidecar._apply_runtime_overrides(model)
    assert model.s2mel.models['cfm'].inference is original  # opt-out keeps upstream fp32
    assert sidecar._runtime_config(model)['s2m_autocast'] is None
