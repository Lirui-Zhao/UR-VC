"""UR-VC: episode-balanced correction of time-derived progress labels."""

from .core import CorrectionResult, correct_progress
from .models import BASE_MODEL, GIANT_MODEL, resolve_model

__all__ = [
    "BASE_MODEL",
    "CorrectionResult",
    "GIANT_MODEL",
    "correct_progress",
    "resolve_model",
]

__version__ = "0.1.0"
