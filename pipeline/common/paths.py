"""EchoHate repo root, data paths, and train/val/test split layout."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

PIPELINE_DIR = Path(__file__).resolve().parent.parent
ECHOHATE_ROOT = PIPELINE_DIR.parent


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _resolve_default_data_root() -> Path:
    env = _env("ECHOHATE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    seed_dir = _env("ECHOHATE_DATA_SEED_DIR")
    if seed_dir:
        return (ECHOHATE_ROOT / "data" / seed_dir).resolve()
    try:
        from pipeline.common.project_config import default_data_seed_dir

        return (ECHOHATE_ROOT / "data" / default_data_seed_dir()).resolve()
    except Exception:
        return (ECHOHATE_ROOT / "data" / "seed_0_model").resolve()


def default_data_root() -> Path:
    """``data/<seed_dir>/`` from ``ECHOHATE_DATA_ROOT``, ``ECHOHATE_DATA_SEED_DIR``, or config."""
    return _resolve_default_data_root()


def get_data_root() -> Path:
    """Resolve data root (re-reads env / config each call)."""
    return default_data_root()


# Snapshot at import; prefer ``get_data_root()`` inside long-running processes.
DEFAULT_DATA_ROOT = _resolve_default_data_root()
# Alias used by preprocessing callers.
DATA_ROOT = DEFAULT_DATA_ROOT


def normalize_dataset_name(name: str) -> str:
    """Return canonical dataset folder name (e.g. ``IHC``, ``DynaHate``, ``Ethos``)."""
    return (name or "").strip()


def resolve_dataset_dir(data_root: Path, name: str) -> Path:
    """Return ``data_root/<dataset>/`` for the canonical dataset name."""
    canon = normalize_dataset_name(name)
    return data_root / canon


def baseline_output_dir_name(model_name: str = "roberta-large") -> str:
    """
    Flat run folder under ``data/<seed>/<dataset>/`` for RoBERTa baseline artifacts.

    Override with env ``ECHOHATE_ROBERTA_BASELINE_DIR`` (e.g. ``roberta_large_baseline``).
    """
    override = _env("ECHOHATE_ROBERTA_BASELINE_DIR")
    if override:
        return override
    slug = (model_name or "roberta-large").strip().lower().replace("-", "_")
    if slug == "roberta_large":
        return "roberta_large_baseline"
    return f"{slug}_baseline"


def resolve_baseline_run_dir(
    data_root: Path | str,
    dataset: str,
    model_name: str = "roberta-large",
) -> Path:
    """``<data_root>/<dataset>/<baseline_output_dir_name>/`` (all metrics/checkpoints here)."""
    root = Path(data_root).expanduser().resolve()
    return resolve_dataset_dir(root, dataset) / baseline_output_dir_name(model_name)


# ---------------------------------------------------------------------------
# Split layout (former split_layout.py)
# Canonical CSVs under ``data/<dataset>/<SPLITTED_DATA_SUBDIR>/``.
#
# Default fractions are documented in ``config/echohate_project_config.json``;
# this module reads ``ECHOHATE_*`` env at import time unless
# ``ECHOHATE_SPLITTED_DATA_SUBDIR`` overrides the folder name.
#
# Splits are stratified via ``python -m pipeline.preprocessing split``.
# IHC 8/1/1: ``python -m pipeline.preprocessing ihc``.
# ---------------------------------------------------------------------------

SplitName = Literal["train", "validation", "test"]
VALIDATION_ALIASES = frozenset({"valid", "dev"})

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"
TEST_CSV = "test.csv"


def _env_frac(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got: {raw!r}") from e


TRAIN_FRAC = _env_frac("ECHOHATE_TRAIN_FRAC", 0.80)
VALIDATION_FRAC = _env_frac("ECHOHATE_VALIDATION_FRAC", 0.10)
TEST_FRAC = _env_frac("ECHOHATE_TEST_FRAC", 0.10)

if abs(TRAIN_FRAC + VALIDATION_FRAC + TEST_FRAC - 1.0) > 1e-9:
    raise ValueError("TRAIN_FRAC + VALIDATION_FRAC + TEST_FRAC must sum to 1.0")
if abs(VALIDATION_FRAC - TEST_FRAC) > 1e-9:
    raise ValueError(
        "VALIDATION_FRAC and TEST_FRAC must match (holdout is split 50/50 between them)."
    )


def normalize_split_name(name: str) -> SplitName:
    """Map CLI/env split name to train | validation | test."""
    s = (name or "").strip().lower()
    if s in ("train",):
        return "train"
    if s in ("test",):
        return "test"
    if s in ("validation",) or s in VALIDATION_ALIASES:
        return "validation"
    raise ValueError(f"Unknown split name {name!r} (use train, validation, or test)")


def split_csv_filename(split: str | SplitName) -> str:
    """CSV basename for a split."""
    key = normalize_split_name(split if isinstance(split, str) else split)
    return {  # type: ignore[misc]
        "train": TRAIN_CSV,
        "validation": VALIDATION_CSV,
        "test": TEST_CSV,
    }[key]


def resolve_split_csv(split_dir: Path | str, split: str | SplitName) -> Path:
    """Path to split CSV under ``split_dir``."""
    d = Path(split_dir)
    key = normalize_split_name(split)
    return d / split_csv_filename(key)


def splitted_data_subdir_name() -> str:
    """Tag folder with integer percents, e.g. ``splitted_data_80_10_10``."""
    a = int(round(TRAIN_FRAC * 100))
    b = int(round(VALIDATION_FRAC * 100))
    c = int(round(TEST_FRAC * 100))
    if a + b + c != 100:
        raise ValueError(f"Split percents must sum to 100, got {a}+{b}+{c}")
    return f"splitted_data_{a}_{b}_{c}"


_ECHOHATE_SPLITTED_OVERRIDE = os.environ.get("ECHOHATE_SPLITTED_DATA_SUBDIR", "").strip()
if _ECHOHATE_SPLITTED_OVERRIDE:
    SPLITTED_DATA_SUBDIR: str = _ECHOHATE_SPLITTED_OVERRIDE
else:
    SPLITTED_DATA_SUBDIR = splitted_data_subdir_name()


def split_layout_run_tag() -> str:
    """
    Short tag for RoBERTa output directory names derived from ``SPLITTED_DATA_SUBDIR``.

    Uses the first digit of each split percent bucket, e.g.:

    - ``splitted_data_60_20_20`` → ``622``
    - ``splitted_data_80_10_10`` → ``811``
    """
    prefix = "splitted_data_"
    if not SPLITTED_DATA_SUBDIR.startswith(prefix):
        raise ValueError(f"Unexpected SPLITTED_DATA_SUBDIR: {SPLITTED_DATA_SUBDIR!r}")
    rest = SPLITTED_DATA_SUBDIR[len(prefix) :]
    parts = rest.split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected three split parts in {SPLITTED_DATA_SUBDIR!r}, got {parts!r}")
    return f"{parts[0][0]}{parts[1][0]}{parts[2][0]}"
