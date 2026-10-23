"""Encoder presets used by the command line interface."""

from __future__ import annotations

GIANT_MODEL = "google/siglip2-giant-opt-patch16-384"
BASE_MODEL = "google/siglip2-base-patch16-224"


def resolve_model(name: str) -> str:
    """Resolve a preset name while leaving custom model IDs and paths intact."""
    aliases = {
        "default": GIANT_MODEL,
        "giant": GIANT_MODEL,
        "base": BASE_MODEL,
    }
    return aliases.get(name, name)
