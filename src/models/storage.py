"""Model persistence — save and load trained models to disk."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel

log = logging.getLogger(__name__)

_DEFAULT_DIR = os.environ.get("MODEL_DIR", "/app/data/models")

_HMAC_KEY_RAW = os.environ.get("AI_MODEL_HMAC_KEY", "")
if not _HMAC_KEY_RAW:
    raise RuntimeError("AI_MODEL_HMAC_KEY environment variable must be set")
_HMAC_KEY = _HMAC_KEY_RAW.encode()


@dataclass
class TrainedModels:
    content: ContentBasedModel
    collaborative: CollaborativeModel


def save_models(models: TrainedModels, directory: str | None = None) -> str:
    """Serialize trained models to disk with HMAC integrity signature."""
    directory = directory or _DEFAULT_DIR
    Path(directory).mkdir(parents=True, exist_ok=True)

    path = os.path.join(directory, "models.pkl")
    sig_path = path + ".sig"

    data = pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL)

    # Write model
    with open(path, "wb") as f:
        f.write(data)

    # Write HMAC signature
    signature = hmac.new(_HMAC_KEY, data, hashlib.sha256).hexdigest()
    with open(sig_path, "w") as f:
        f.write(signature)

    log.info("Models saved to %s (signed)", path)
    return path


def load_models(directory: str | None = None) -> TrainedModels | None:
    """Load trained models from disk. Verifies HMAC before deserializing."""
    directory = directory or _DEFAULT_DIR
    path = os.path.join(directory, "models.pkl")
    sig_path = path + ".sig"

    if not os.path.exists(path):
        log.warning("No saved models at %s", path)
        return None

    with open(path, "rb") as f:
        data = f.read()

    # Verify HMAC signature before deserializing — signature is MANDATORY
    if not os.path.exists(sig_path):
        log.error("No signature file at %s — refusing to load unverified model", sig_path)
        return None

    with open(sig_path, "r") as f:
        expected_sig = f.read().strip()
    actual_sig = hmac.new(_HMAC_KEY, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, actual_sig):
        log.error("Model file integrity check FAILED — possible tampering at %s", path)
        return None

    models = pickle.loads(data)

    if not isinstance(models, TrainedModels):
        log.warning("Invalid model file at %s", path)
        return None

    log.info("Models loaded from %s (verified)", path)
    return models
