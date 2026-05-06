"""Device selection.

CUDA is preferred when available **and** the user opted in via
``AI_USE_GPU=1``. We don't auto-pick CUDA so that runs stay reproducible
between hosts — the operator decides explicitly.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("src_torch.device")


def select_device() -> torch.device:
    """Return the torch device to use for the entire pipeline.

    Honours ``AI_USE_GPU=1`` (or ``true``/``yes``); falls back to CPU
    when CUDA isn't available, with a single log line so the operator
    sees what happened.
    """
    requested = os.environ.get("AI_USE_GPU", "").strip().lower() in {"1", "true", "yes"}
    if requested and torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("torch device: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        if requested:
            log.warning("AI_USE_GPU=1 but CUDA not available — falling back to cpu")
        else:
            log.info("torch device: cpu (set AI_USE_GPU=1 to enable cuda)")
    return device


def use_deterministic() -> None:
    """Best-effort determinism settings.

    Only applied to the torch backend; the legacy CPU pipeline already
    uses NumPy seeds and doesn't need this.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
