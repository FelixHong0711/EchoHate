"""Hugging Face token lookup for encoder downloads.

Reads ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``, or a local ``secrets/hf_token`` if one exists.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pipeline.common.paths import ECHOHATE_ROOT

_HF_TOKEN_FILE = ECHOHATE_ROOT / "secrets" / "hf_token"
_HF_ENV_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def resolve_hf_token() -> Optional[str]:
    """Read a Hugging Face token from the environment or ``secrets/hf_token``."""
    for key in _HF_ENV_KEYS:
        value = os.getenv(key)
        if value and str(value).strip():
            return str(value).strip()
    if _HF_TOKEN_FILE.is_file():
        token = _HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def ensure_hf_token_env() -> Optional[str]:
    """Copy a resolved token into the standard Hugging Face environment variables."""
    token = resolve_hf_token()
    if not token:
        return None
    for key in _HF_ENV_KEYS:
        os.environ[key] = token
    return token


def hf_hub_auth_kwargs() -> Dict[str, Any]:
    """Keyword arguments for ``from_pretrained`` when a token is available."""
    token = ensure_hf_token_env()
    if token:
        return {"token": token}
    return {}
