"""
Load ``config/echohate_project_config.json`` (experiment-wide defaults).

Seeds:
  - ``random_state`` — primary for training / LLM / encoder stages (default 0).
  - ``create_split_random_state`` — stratified 811 split (default 42).

Seeds, split fractions, and training hyperparameters all live in one JSON file.
``paths`` (split layout) reads ``ECHOHATE_TRAIN_FRAC`` etc. from the environment at import time;
use ``--export-shell`` to emit env vars from JSON before running Python in the same shell.

Also provides ``set_seed`` with lazy torch/numpy imports.

Training hyperparameters live in the same JSON under ``training``.
``--export-training-shell`` prints those defaults as bash assignments.
``--export-shell`` prints seeds, split fractions, and training defaults together.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_ROOT = __import__('pipeline.common.paths', fromlist=['ECHOHATE_ROOT']).ECHOHATE_ROOT
_CONFIG_JSON = _ROOT / "config" / "echohate_project_config.json"


def project_config_path() -> Path:
    return _CONFIG_JSON


def set_seed(seed: int) -> None:
    """
    Seed Python / NumPy / PyTorch / (optional) HF, and force deterministic cuDNN.
    Torch/numpy are imported lazily so config-only callers stay light.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        from transformers import set_seed as transformers_set_seed

        transformers_set_seed(seed)
    except ImportError:
        pass


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, Any]:
    if not _CONFIG_JSON.is_file():
        return {}
    with open(_CONFIG_JSON, encoding="utf-8") as f:
        return json.load(f)


def reload_project_config_for_tests() -> None:
    load_project_config.cache_clear()


def _seeds() -> Mapping[str, Any]:
    raw = load_project_config().get("seeds") or {}
    return raw if isinstance(raw, dict) else {}


def _coalesce_seed(key: str, *, fallback: int) -> int:
    s = _seeds()
    v = s.get(key)
    if v is None:
        return int(fallback)
    return int(v)


def default_split_random_state() -> int:
    """Stratified split (``python -m preprocess split``). Default 42."""
    return _coalesce_seed("create_split_random_state", fallback=42)


def default_random_state() -> int:
    """Primary seed for --random_state and sklearn. Default 0."""
    return int(_seeds().get("random_state", 0))


def default_seed_sweep() -> tuple[int, ...]:
    """Ordered seeds for multi-seed shell loops (``seeds.seed_sweep`` in JSON)."""
    rs = default_random_state()
    raw = _seeds().get("seed_sweep")
    if raw is None:
        return (rs,)
    return tuple(int(x) for x in raw)


def reload_seed_config_for_tests() -> None:
    """Clear cached config (alias for tests that previously called ``reload_seed_config_for_tests``)."""
    reload_project_config_for_tests()


def default_data_seed_dir() -> str:
    """e.g. ``seed_42`` under ``data/`` for multi-seed layouts."""
    dl = load_project_config().get("data_layout") or {}
    if not isinstance(dl, dict):
        return f"seed_{default_random_state()}"
    ov = dl.get("default_data_seed_dir")
    if ov is None or (isinstance(ov, str) and not str(ov).strip()):
        return f"seed_{default_random_state()}"
    return str(ov).strip()


def default_dataset_name() -> str:
    dl = load_project_config().get("data_layout") or {}
    if isinstance(dl, dict) and dl.get("default_dataset"):
        return str(dl["default_dataset"])
    return "IHC"


def dataset_task_overrides() -> dict[str, str]:
    """Per-dataset prompt task (``hate`` | ``offensive``); excludes ``description`` and ``default``."""
    raw = load_project_config().get("dataset_tasks") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in raw.items():
        if key in ("description", "default") or val is None:
            continue
        task = str(val).strip().lower()
        if task in ("hate", "offensive"):
            out[str(key)] = task
    return out


def prompt_task_for_dataset(dataset: str) -> str:
    """Return ``hate`` or ``offensive`` for a dataset (SBIC → offensive; others → hate by default)."""
    ds = (dataset or "").strip()
    tasks = load_project_config().get("dataset_tasks") or {}
    default = "hate"
    if isinstance(tasks, dict):
        d = tasks.get("default")
        if d is not None and str(d).strip().lower() in ("hate", "offensive"):
            default = str(d).strip().lower()
        if ds:
            ov = tasks.get(ds)
            if ov is not None and str(ov).strip().lower() in ("hate", "offensive"):
                return str(ov).strip().lower()
    return default


def split_fractions() -> tuple[float, float, float]:
    """Train / validation / test fractions from JSON (for docs and ``--export-shell``)."""
    s = load_project_config().get("split_layout") or {}
    val = s.get("validation_frac", 0.1)
    return (
        float(s.get("train_frac", 0.8)),
        float(val),
        float(s.get("test_frac", 0.1)),
    )


def training_defaults() -> dict[str, Any]:
    """Training hyperparameters from JSON (snake_case keys as in config file), minus description."""
    raw = load_project_config().get("training")
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k != "description"}


def _training_value(key: str, default: Any, typ: type) -> Any:
    t = training_defaults()
    if key not in t:
        return default
    v = t[key]
    if typ is float:
        return float(v)
    if typ is int:
        return int(v)
    return str(v)


def export_training_shell_snippet() -> str:
    """Bash ``: \\\"${VAR:=value}\\\"`` lines so a pre-set environment variable overrides the JSON default."""
    if "training" not in load_project_config():
        return ""
    specs: list[tuple[str, str, Any, type]] = [
        ("NUM_EPOCHS", "num_epochs", 6, int),
        ("BATCH_SIZE", "batch_size", 16, int),
        ("EVAL_BATCH_SIZE", "eval_batch_size", 16, int),
        ("LEARNING_RATE", "learning_rate", "1e-5", str),
        ("WEIGHT_DECAY", "weight_decay", 0.01, float),
        ("WARMUP_RATIO", "warmup_ratio", 0.0, float),
        ("MAX_LENGTH", "max_length", 256, int),
        ("POS_WEIGHT_MODE", "pos_weight_mode", "none", str),
        ("CLASS_WEIGHT_TARGET", "class_weight_target", "pos", str),
        ("PAD_TO_MAX_LENGTH", "pad_to_max_length", 1, int),
        ("NO_EVAL_EACH_EPOCH", "no_eval_each_epoch", 0, int),
    ]
    lines: list[str] = []
    for bash_var, json_key, default, typ in specs:
        val = _training_value(json_key, default, typ)
        if typ in (int, float):
            lines.append(f': "${{{bash_var}:={val}}}"')
        else:
            esc = str(val).replace('"', '\\"')
            lines.append(f': "${{{bash_var}:={esc}}}"')
    return "\n".join(lines) + "\n"


def export_shell_snippet() -> str:
    """Bash ``: \\\"${VAR:=value}\\\"`` lines; a pre-set environment variable wins."""
    rs = int(_seeds().get("random_state", 0))
    ss = _seeds()
    sweep = ss.get("seed_sweep")
    if sweep is None:
        sweep_list: list[int] = [rs]
    else:
        sweep_list = [int(x) for x in sweep]
    train, validation, test = split_fractions()
    sub_ov = (load_project_config().get("split_layout") or {}).get("splitted_data_subdir_override")
    split_seed = default_split_random_state()
    data_seed_dir = default_data_seed_dir()
    dataset = default_dataset_name()
    lines = [
        f': "${{RANDOM_STATE:={rs}}}"',
        f': "${{SEED:={rs}}}"',
        f': "${{ECHOHATE_CANONICAL_RANDOM_STATE:={rs}}}"',
        f': "${{ECHOHATE_SPLIT_SEED:={split_seed}}}"',
        f': "${{ECHOHATE_SEEDS_COMMA:={",".join(map(str, sweep_list))}}}"',
        f': "${{AMPLEHATE_SEED_LIST:={",".join(map(str, sweep_list))}}}"',
        f': "${{ECHOHATE_TRAIN_FRAC:={train}}}"',
        f': "${{ECHOHATE_VALIDATION_FRAC:={validation}}}"',
        f': "${{ECHOHATE_VALIDATION_FRAC:={validation}}}"',
        f': "${{ECHOHATE_TEST_FRAC:={test}}}"',
        f': "${{ECHOHATE_DATA_SEED_DIR:={data_seed_dir}}}"',
        f': "${{DATASET:={dataset}}}"',
    ]
    if sub_ov:
        lines.append(f': "${{ECHOHATE_SPLITTED_DATA_SUBDIR:={sub_ov}}}"')
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="EchoHate project config helper")
    p.add_argument("--export-shell", action="store_true", help="Seeds + split env + training := defaults for eval")
    p.add_argument("--export-training-shell", action="store_true", help="Print training defaults as bash assignments.")
    p.add_argument("--print-json", action="store_true", help="Print seeds, split fractions, and training as JSON")
    args = p.parse_args()
    if args.export_shell:
        sys.stdout.write(export_shell_snippet())
        return
    if args.export_training_shell:
        sys.stdout.write(export_training_shell_snippet())
        return
    if args.print_json:
        rs = int(_seeds().get("random_state", 0))
        out = {
            "random_state": rs,
            "create_split_random_state": default_split_random_state(),
            "data_seed_dir": default_data_seed_dir(),
            "default_dataset": default_dataset_name(),
            "split_fractions": {
                "train": split_fractions()[0],
                "validation": split_fractions()[1],
                "test": split_fractions()[2],
            },
            "training": training_defaults(),
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    p.print_help()


if __name__ == "__main__":
    main()
