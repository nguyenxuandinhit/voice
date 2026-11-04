"""IndexTTS sidecar tuning for one machine (Settings → Performance).

The sidecar reads these ``OMNIVOICE_INDEXTTS_*`` variables when it starts,
so a change is saved as an ``env.*`` pref (restored on the next backend
start), applied to this process's environment and takes effect when the
idle sidecar is restarted. Defaults are the measured choices in srs.md
section 6.3: fp16 S2M on CUDA and the upstream beam of 3; the < 10 GB
text-splitting rule stays upstream's unless the operator overrides it.
"""
from __future__ import annotations

import os

ENGINE_ID = "indextts2"
UPSTREAM_BEAMS = 3
#: Upstream 2.5 splits text over 40 characters on GPUs below this size.
LOW_VRAM_SPLIT_BELOW_GB = 10.0

ENV = {
    "s2m_precision": "OMNIVOICE_INDEXTTS_S2M_AUTOCAST",
    "low_vram_split": "OMNIVOICE_INDEXTTS_LOW_VRAM_SPLIT",
    "num_beams": "OMNIVOICE_INDEXTTS_NUM_BEAMS",
}
PRECISIONS = ("fp16", "fp32")
SPLITS = {"auto": None, "on": "1", "off": "0"}


def _env_value(field: str, value) -> str | None:
    """Env string for a UI value; ``None`` means unset (the default)."""
    if field == "s2m_precision":
        if value not in PRECISIONS:
            raise ValueError("s2m_precision must be fp16 or fp32")
        return None if value == "fp16" else value
    if field == "low_vram_split":
        if value not in SPLITS:
            raise ValueError("low_vram_split must be auto, on or off")
        return SPLITS[value]
    if field == "num_beams":
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
            raise ValueError("num_beams must be an integer from 1 to 16")
        return None if value == UPSTREAM_BEAMS else str(value)
    raise ValueError(f"unknown setting {field}")


def _ui_value(field: str):
    raw = os.environ.get(ENV[field])
    if field == "s2m_precision":
        return raw or "fp16"
    if field == "low_vram_split":
        return {"1": "on", "0": "off"}.get(raw or "", "auto")
    try:
        return int(raw) if raw else UPSTREAM_BEAMS
    except ValueError:
        return raw  # the sidecar rejects it with a named error


def state(vram_total_gb: float = 0.0, *, restarted: bool | None = None) -> dict:
    from core import prefs

    out = {field: _ui_value(field) for field in ENV}
    out["env_pinned"] = sorted(field for field, var in ENV.items() if prefs.is_env_shadowed(var))
    out["vram_total_gb"] = vram_total_gb
    out["upstream_splits"] = 0 < vram_total_gb < LOW_VRAM_SPLIT_BELOW_GB
    if restarted is not None:
        out["restarted"] = restarted
    return out


def _stop_sidecar() -> int:
    from services.subprocess_backend import unload_sidecar

    return unload_sidecar(ENGINE_ID)


def apply(changes: dict) -> bool:
    """Validate everything first, then persist. Returns whether a running
    sidecar was stopped now; otherwise the change applies on its next start."""
    from core import prefs

    values = {field: _env_value(field, value) for field, value in changes.items()}
    pinned = [field for field in values if prefs.is_env_shadowed(ENV[field])]
    if pinned:
        raise PermissionError(", ".join(sorted(pinned)))
    for field, value in values.items():
        var = ENV[field]
        if value is None:
            os.environ.pop(var, None)
            prefs.delete("env." + var)
        else:
            os.environ[var] = value
            prefs.set_("env." + var, value)
    return bool(values) and _stop_sidecar() > 0
