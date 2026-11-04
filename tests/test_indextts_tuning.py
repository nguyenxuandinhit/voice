"""Settings → Performance: per-machine IndexTTS tuning (srs.md section 6.3)."""
from __future__ import annotations

import os

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")

import pytest

@pytest.fixture
def tuning(monkeypatch):
    from core import prefs
    from services import indextts_tuning

    for var in indextts_tuning.ENV.values():
        monkeypatch.delenv(var, raising=False)
        prefs.delete("env." + var)
    monkeypatch.setattr(prefs, "_EXTERNALLY_PROVIDED", frozenset())
    stopped = []
    monkeypatch.setattr(indextts_tuning, "_stop_sidecar",
                        lambda: stopped.append(indextts_tuning.ENGINE_ID) or 1)
    yield indextts_tuning, stopped
    for var in indextts_tuning.ENV.values():
        prefs.delete("env." + var)


def test_defaults_are_the_measured_choices(tuning):
    t, _ = tuning
    state = t.state(8.0)
    assert (state["s2m_precision"], state["low_vram_split"], state["num_beams"]) == ("fp16", "auto", 3)
    assert state["upstream_splits"] is True and state["env_pinned"] == []
    assert t.state(15.0)["upstream_splits"] is False and t.state(0.0)["upstream_splits"] is False


def test_apply_persists_env_and_restarts_the_sidecar(tuning):
    from core import prefs

    t, stopped = tuning
    assert t.apply({"s2m_precision": "fp32", "low_vram_split": "off", "num_beams": 1}) is True
    assert stopped == ["indextts2"]
    assert os.environ["OMNIVOICE_INDEXTTS_S2M_AUTOCAST"] == "fp32"
    assert prefs.get("env.OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT") == "0"
    assert t.state()["num_beams"] == 1
    # Choosing the default again removes the override instead of pinning it.
    t.apply({"s2m_precision": "fp16", "low_vram_split": "auto", "num_beams": 3})
    assert "OMNIVOICE_INDEXTTS_S2M_AUTOCAST" not in os.environ
    assert prefs.get("env.OMNIVOICE_INDEXTTS_NUM_BEAMS") is None


@pytest.mark.parametrize("changes", [
    {"s2m_precision": "fp8"}, {"low_vram_split": "maybe"}, {"num_beams": 0}, {"num_beams": True},
])
def test_invalid_values_change_nothing(tuning, changes):
    t, stopped = tuning
    with pytest.raises(ValueError):
        t.apply({"low_vram_split": "off", **changes})
    assert "OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT" not in os.environ and stopped == []


def test_env_pinned_fields_are_reported_and_refused(tuning, monkeypatch):
    from core import prefs

    t, _ = tuning
    monkeypatch.setattr(prefs, "_EXTERNALLY_PROVIDED", frozenset({"OMNIVOICE_INDEXTTS_NUM_BEAMS"}))
    assert t.state()["env_pinned"] == ["num_beams"]
    with pytest.raises(PermissionError):
        t.apply({"num_beams": 1})


def test_endpoint_validates_and_maps_errors(tuning, monkeypatch):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from api.routers import settings as router
    from core import prefs

    assert router.get_indextts_tuning()["s2m_precision"] == "fp16"
    saved = router.set_indextts_tuning(router._IndexTTSTuningBody(low_vram_split="off"))
    assert saved["low_vram_split"] == "off" and saved["restarted"] is True
    for bad in ({"num_beams": 99}, {"s2m_precision": "bf16"}):
        with pytest.raises(ValidationError):
            router._IndexTTSTuningBody(**bad)
    monkeypatch.setattr(prefs, "_EXTERNALLY_PROVIDED", frozenset({"OMNIVOICE_INDEXTTS_NUM_BEAMS"}))
    with pytest.raises(HTTPException) as exc:
        router.set_indextts_tuning(router._IndexTTSTuningBody(num_beams=1))
    assert exc.value.status_code == 409
