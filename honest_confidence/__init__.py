"""honest-confidence — a small, deterministic honesty layer for LLM agents.

Deflate self-reported confidence toward measured accuracy (calibration), and abstain on
ungrounded / refuted claims (grounding + refuter), then measure — on a labeled benchmark —
whether that improves calibration and cuts confident wrong answers.

Extracted and generalized from a running local agent; see README for the honest framing.
"""

from .calibration import (
    calibrate_confidence,
    fit_measured_rate,
    calibration_health,
)

__all__ = [
    "calibrate_confidence",
    "fit_measured_rate",
    "calibration_health",
]

__version__ = "0.1.0"
