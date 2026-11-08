#!/usr/bin/env python3
"""
EchoHate: encoder RoBERTa training with optional mechanism auxiliary loss.

Default fine-tune hyperparameters match roberta_baseline
(batch 8, lr 1e-5, wd 0.01, warmup 0, max_length 256). Full encoder fine-tune (FFT) only.

Training uses fixed padding to max_length in DistillDataset (same effective behavior as baseline
with --pad-to-max-length).

Teacher CSV schema:
  - label / label_text (HATE|NOT_HATE or OFFENSIVE|NOT_OFFENSIVE)
  - target (verbatim span or "none")
  - protected_group
  - mechanism
  - attack_cue (verbatim predicate <=12 words or "NONE")

Loss: L_CE (+ optional contrastive mix)
  + lambda_mech * L_mech when --lambda_mech > 0

Inputs (expected under data/<dataset>/splitted_data_80_10_10/ or current split_layout.SPLITTED_DATA_SUBDIR):
  - train.csv, test.csv, validation.csv (columns: text, class)

llm file:
  - Either JSON (full_output.json list) or CSV (llm_predictions.csv / ensemble_llm_predictions.csv)
  - Must contain row_id and label (or llm_label) plus target + attack_cue fields.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from pipeline.stage2.training_dynamics import TrainingDynamicsCallback
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from tqdm import tqdm

from pipeline.common.paths import DEFAULT_DATA_ROOT, ECHOHATE_ROOT, SPLITTED_DATA_SUBDIR
from pipeline.common.project_config import default_random_state, prompt_task_for_dataset
DATA_ROOT = ECHOHATE_ROOT / "data"
from pipeline.common.mechanism_taxonomy import (
    HATE_MECHANISM_LABELS,
    NOT_HATE_MECHANISM_LABELS,
    NOT_OFFENSIVE_MECHANISM_LABELS,
    N_HATE_MECHANISM_CLASSES,
    N_OFFENSIVE_MECHANISM_CLASSES,
    OFFENSIVE_MECHANISM_LABELS,
    mechanism_labels_for_task,
    mech_to_id_for_task,
    n_mechanism_classes_for_task,
)
from pipeline.stage2.roberta_baseline import (
    ContrastiveLossCosine,
    _compute_metrics_for_trainer_best_threshold,
)



# Mechanism taxonomy for the auxiliary head (see pipeline.common.mechanism_taxonomy).
# Hate datasets use 8 labels; SBIC/offensive LLM outputs use 5.
MECHANISM_LABELS = (
    list(HATE_MECHANISM_LABELS)
    + list(NOT_HATE_MECHANISM_LABELS)
    + list(OFFENSIVE_MECHANISM_LABELS)
    + list(NOT_OFFENSIVE_MECHANISM_LABELS)
)
MECH_TO_ID = {m: i for i, m in enumerate(MECHANISM_LABELS)}
N_MECHANISM_CLASSES = len(MECHANISM_LABELS)
# Per-task auxiliary heads: hate=8, offensive (SBIC)=5.
# Extra widths accepted so existing checkpoints still load for classification-only
# eval / thr re-inference (mech aux unused for hate logits): 2, 9, 11, 13, 14.
LEGACY_N_HATE_MECHANISM_CLASSES = 8  # same as current hate default
LEGACY_N_HATE_MECHANISM_CLASSES_5WAY = 11
LEGACY_N_HATE_MECHANISM_CLASSES_9WAY = 9
LEGACY_N_MECHANISM_CLASSES = 13
LEGACY_N_MECHANISM_CLASSES_14 = 14
# 2 = binary-aux control head (gold binary on the same selected rows).
ALLOWED_N_MECHANISM_CLASSES = frozenset(
    {
        2,
        N_HATE_MECHANISM_CLASSES,
        N_OFFENSIVE_MECHANISM_CLASSES,
        N_MECHANISM_CLASSES,
        LEGACY_N_HATE_MECHANISM_CLASSES,
        LEGACY_N_HATE_MECHANISM_CLASSES_5WAY,
        LEGACY_N_HATE_MECHANISM_CLASSES_9WAY,
        LEGACY_N_MECHANISM_CLASSES,
        LEGACY_N_MECHANISM_CLASSES_14,
    }
)

def mechanism_training_metadata(task: Optional[str] = None) -> Dict[str, Any]:
    """Persist in run configs so checkpoints can be validated on load/skip."""
    if task is not None:
        labels = mechanism_labels_for_task(task)
        return {
            "prompt_task": (task or "hate").strip().lower(),
            "n_mechanism_classes": len(labels),
            "mechanism_labels": list(labels),
        }
    return {
        "n_mechanism_classes": N_MECHANISM_CLASSES,
        "mechanism_labels": list(MECHANISM_LABELS),
    }


def mechanism_head_out_features_from_state(state_dict: Dict[str, Any]) -> Optional[int]:
    for key, tensor in state_dict.items():
        if key.endswith("mechanism_head.weight"):
            return int(tensor.shape[0])
    return None


def resolve_n_mechanism_classes(
    *,
    cfg: Optional[Dict[str, Any]] = None,
    state_dict: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Prefer checkpoint head width; fall back to config metadata."""
    from_ckpt = mechanism_head_out_features_from_state(state_dict) if state_dict else None
    from_cfg: Optional[int] = None
    if cfg:
        raw = cfg.get("n_mechanism_classes")
        if raw is not None:
            from_cfg = int(raw)
        elif isinstance(cfg.get("training_config"), dict):
            raw = cfg["training_config"].get("n_mechanism_classes")
            if raw is not None:
                from_cfg = int(raw)
    if from_ckpt is not None:
        return from_ckpt
    return from_cfg


def checkpoint_mechanism_compatible(
    *,
    cfg: Optional[Dict[str, Any]] = None,
    state_dict: Optional[Dict[str, Any]] = None,
    expect_mech_head: bool = True,
) -> Tuple[bool, str]:
    """Return (ok, message). Used by checkpoint checks and eval loaders."""
    if not expect_mech_head:
        return True, "mechanism head disabled (lambda_mech=0); checkpoint compatible"

    saved_n: Optional[int] = None
    from_ckpt = mechanism_head_out_features_from_state(state_dict) if state_dict else None
    from_cfg: Optional[int] = None
    if cfg:
        raw = cfg.get("n_mechanism_classes")
        if raw is not None:
            from_cfg = int(raw)
        elif isinstance(cfg.get("training_config"), dict):
            raw = cfg["training_config"].get("n_mechanism_classes")
            if raw is not None:
                from_cfg = int(raw)
    # Prefer checkpoint head width (ground truth); config may stale-write union size 13.
    saved_n = from_ckpt if from_ckpt is not None else from_cfg

    if saved_n is None:
        return False, (
            "cannot verify mechanism-head size; retrain or sync mech run config / encoder_config.json required"
        )
    if saved_n not in ALLOWED_N_MECHANISM_CLASSES:
        return False, (
            f"mechanism-head size {saved_n} not in "
            f"{sorted(ALLOWED_N_MECHANISM_CLASSES)} (hate={N_HATE_MECHANISM_CLASSES}, "
            f"offensive={N_OFFENSIVE_MECHANISM_CLASSES}, union={N_MECHANISM_CLASSES})"
        )
    if saved_n == N_MECHANISM_CLASSES:
        return True, (
            f"mechanism-head ok (union {saved_n} classes; "
            f"prefer hate={N_HATE_MECHANISM_CLASSES} / offensive={N_OFFENSIVE_MECHANISM_CLASSES} for new runs)"
        )
    return True, f"mechanism-head ok ({saved_n} classes)"

# 1:1 translations (preserve char indices) for matching llm cues/targets to encoder text
_CUE_MATCH_CHAR_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00a0": " ",  # NBSP
    }
)


def _normalize_cue_matching_surface(s: str) -> str:
    """
    Length-preserving cleanup for cue/target vs encoder text: NFC when it does
    not change length (avoids index skew), then 1:1 smart-quote/dash/NBSP mapping.
    """
    if not s:
        return s
    t = unicodedata.normalize("NFC", s)
    if len(t) != len(s):
        t = s
    return t.translate(_CUE_MATCH_CHAR_TRANSLATE)


def _collapse_whitespace_with_index_map(s: str) -> Tuple[str, List[int]]:
    """
    Collapse consecutive whitespace to a single space; lower case for comparison only elsewhere.
    Returns (collapsed, collapsed_idx -> original char index mapping).
    """
    out: List[str] = []
    idx_map: List[int] = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _span_from_collapsed_match(
    collapsed: str, index_map: List[int], pos: int, match_len: int
) -> Optional[Tuple[int, int]]:
    if pos < 0 or match_len <= 0 or pos + match_len > len(collapsed):
        return None
    if not index_map or pos + match_len - 1 >= len(index_map):
        return None
    orig_start = index_map[pos]
    orig_end = index_map[pos + match_len - 1] + 1
    return (orig_start, orig_end)


def _parse_llm_label(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, str):
        s = val.strip().upper()
        if s in ("HATE", "OFFENSIVE", "1", "TRUE"):
            return 1
        if s in ("NOT_HATE", "NOT_OFFENSIVE", "0", "FALSE"):
            return 0
        return 0
    return int(val)


def _compute_ce_class_weights(labels: np.ndarray, mode: str) -> Optional[torch.Tensor]:
    """Match baseline CE weighting exactly."""
    m = (mode or "freq").strip().lower()
    if m == "none":
        return None
    n = float(len(labels))
    n0 = max(float((labels == 0).sum()), 1.0)
    n1 = max(float((labels == 1).sum()), 1.0)
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    if m == "sqrt":
        w0 = float(np.sqrt(w0))
        w1 = float(np.sqrt(w1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def _normalize_for_span(s: str, collapse_repeated: bool = False) -> Tuple[str, List[Tuple[int, int]]]:
    """Lowercase; keep alnum; collapse non-alnum runs to spaces. Return (norm, segments->orig indices)."""
    if not s:
        return "", []
    s_lower = s.lower()
    norm_chars: List[str] = []
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < len(s_lower):
        if s_lower[i].isalnum():
            norm_chars.append(s_lower[i])
            segments.append((i, i + 1))
            i += 1
        else:
            run_start = i
            while i < len(s_lower) and not s_lower[i].isalnum():
                i += 1
            if norm_chars and norm_chars[-1] != " ":
                norm_chars.append(" ")
                segments.append((run_start, i))
            elif segments and segments[-1][1] == run_start:
                segments[-1] = (segments[-1][0], i)
            continue
    if not collapse_repeated:
        return "".join(norm_chars), segments
    out_chars: List[str] = []
    out_segments: List[Tuple[int, int]] = []
    j = 0
    while j < len(norm_chars):
        ch = norm_chars[j]
        start_seg = segments[j][0]
        j += 1
        while j < len(norm_chars) and norm_chars[j] == ch and ch != " ":
            j += 1
        end_seg = segments[j - 1][1]
        out_chars.append(ch)
        out_segments.append((start_seg, end_seg))
    return "".join(out_chars), out_segments


_NORM_MATCH_RATIO = 2.0
_NORM_MATCH_RATIO_SHORT_CUE = 1.5
_NORM_MATCH_SHORT_CUE_LEN = 4
_NORM_MATCH_ABSOLUTE_SLACK = 20
_NORM_MATCH_MIN_ALNUM_FRAC = 0.6


def _get_cue_span_in_text(text: str, cue: Optional[str]) -> Optional[Tuple[int, int]]:
    """Return (start, end) char offsets if cue appears (exact or normalized with guardrails)."""
    if not cue or not isinstance(cue, str):
        return None
    c = cue.strip()
    if not c or c.upper() in ("NONE", "N/A"):
        return None
    if c.lower() == "none":
        return None
    # Surface normalization on **full sentence** and cue (case, unicode, punctuation)
    base_text = _normalize_cue_matching_surface(text)
    base_cue = _normalize_cue_matching_surface(c)
    # 1) Case-insensitive match after whitespace collapse (tabs/newlines → single space)
    t_coll, t_map = _collapse_whitespace_with_index_map(base_text)
    c_coll, _ = _collapse_whitespace_with_index_map(base_cue)
    if c_coll.strip():
        cc = c_coll.strip()
        pos = t_coll.lower().find(cc.lower())
        if pos >= 0:
            span = _span_from_collapsed_match(t_coll, t_map, pos, len(cc))
            if span is not None:
                return span
    # 2) Case-insensitive on normalized surface (spacing as in string)
    text_lower = base_text.lower()
    c_lower = base_cue.lower()
    pos = text_lower.find(c_lower)
    if pos >= 0:
        return (pos, pos + len(base_cue))
    # 3) Alnum / gap-insensitive path (same as before, on normalized surface)
    text_norm, text_segments = _normalize_for_span(base_text, collapse_repeated=True)
    cue_norm, _ = _normalize_for_span(base_cue, collapse_repeated=True)
    cue_norm = cue_norm.strip().replace(" ", "")
    if not cue_norm:
        return None
    text_norm_nospace = text_norm.replace(" ", "")
    idx = text_norm_nospace.find(cue_norm)
    if idx < 0:
        return None
    norm_nospace_to_norm = [i for i, ch in enumerate(text_norm) if ch != " "]
    if idx + len(cue_norm) > len(norm_nospace_to_norm):
        return None
    start_norm = norm_nospace_to_norm[idx]
    end_norm = norm_nospace_to_norm[idx + len(cue_norm) - 1] + 1
    if not text_segments or end_norm > len(text_segments):
        return None
    orig_start = text_segments[start_norm][0]
    orig_end = text_segments[end_norm - 1][1]
    orig_len = orig_end - orig_start
    ratio = _NORM_MATCH_RATIO_SHORT_CUE if len(c) <= _NORM_MATCH_SHORT_CUE_LEN else _NORM_MATCH_RATIO
    max_span = min(int(ratio * len(c)), len(c) + _NORM_MATCH_ABSOLUTE_SLACK)
    if orig_len > max_span:
        return None
    span_slice = base_text[orig_start:orig_end]
    alnum_count = sum(1 for ch in span_slice if ch.isalnum())
    if len(span_slice) > 0 and alnum_count / len(span_slice) < _NORM_MATCH_MIN_ALNUM_FRAC:
        return None
    return (orig_start, orig_end)


def load_llm_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load llm outputs into row_id -> record. Supports .json (list) and .csv."""
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, dict) and "records" in records:
            records = records["records"]
        if not isinstance(records, list):
            raise ValueError(f"Expected list in {path}, got {type(records)}")
        out = {}
        for r in records:
            if isinstance(r, dict) and "row_id" in r:
                out[str(r["row_id"])] = r
        return out
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "row_id" not in df.columns:
            raise ValueError(f"CSV missing row_id: {path}")
        out = {}
        for _, row in df.iterrows():
            rid = str(row["row_id"])
            out[rid] = row.to_dict()
        return out
    raise ValueError(f"Unsupported llm file type: {path}")


def _build_validation_plus_aug_dataframe(data_root: Path, dataset_name: str) -> pd.DataFrame:
    """Union validation rows with optional aug-only rows from augmented_data/augmented_validation.csv."""
    base = data_root / dataset_name
    validation_path = base / SPLITTED_DATA_SUBDIR / "validation.csv"
    aug_path = base / "augmented_data" / "augmented_validation.csv"
    validation_df = pd.read_csv(validation_path)
    if "row_id" not in validation_df.columns:
        validation_df["row_id"] = [f"validation_{i}" for i in range(len(validation_df))]
    validation_df["source"] = "validation"
    if not aug_path.exists():
        return validation_df
    aug_df = pd.read_csv(aug_path)
    if "source" in aug_df.columns:
        aug_df = aug_df[aug_df["source"].astype(str).str.lower().eq("aug")].copy()
    else:
        aug_df = aug_df.copy()
        aug_df["source"] = "aug"
    all_cols = list(dict.fromkeys(list(validation_df.columns) + list(aug_df.columns)))
    for c in all_cols:
        if c not in validation_df.columns:
            validation_df[c] = None
        if c not in aug_df.columns:
            aug_df[c] = None
    return pd.concat([validation_df[all_cols], aug_df[all_cols]], ignore_index=True)


def _looks_like_llm_only_csv(df: pd.DataFrame) -> bool:
    c = set(df.columns.astype(str))
    if "text" in c and "class" in c:
        return False
    return bool({"row_id", "llm_label"}.issubset(c) and ("target" in c or "attack_cue" in c))


def load_splits_and_llm(
    data_root: Path,
    dataset_name: str,
    llm_path: Optional[Path],
    validation_file: Optional[Path] = None,
    use_llm_as_gt_for_aug: bool = False,
    mech_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Load train/test/validation; merge llm cues/labels for CE (+ optional mechanism aux)."""
    base = data_root / dataset_name / SPLITTED_DATA_SUBDIR
    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")
    validation_read_path = validation_file if validation_file is not None else (base / "validation.csv")
    validation_df = pd.read_csv(validation_read_path)

    llm_lookup_patch: Dict[str, Dict[str, Any]] = {}
    if ("text" not in validation_df.columns or "class" not in validation_df.columns) and _looks_like_llm_only_csv(
        validation_df
    ):
        print(
            f"WARNING: --validation_file looks like a llm-only CSV (columns={list(validation_df.columns)}). "
            f"This usually means --validation_file and --llm_file were swapped. Rebuilding validation rows from "
            f"{SPLITTED_DATA_SUBDIR}/validation.csv + augmented_data/ (if present). "
            f"llm rows loaded from that path: {validation_read_path}"
        )
        llm_lookup_patch = load_llm_lookup(validation_read_path)
        validation_df = _build_validation_plus_aug_dataframe(data_root, dataset_name)

    for df in (train_df, test_df, validation_df):
        if "text" not in df.columns or "class" not in df.columns:
            raise ValueError(f"Expected 'text' and 'class'. Got: {list(df.columns)}")
        df["binary_label"] = df["class"].astype(int)

    train_df["row_id"] = [f"train_{i}" for i in range(len(train_df))]
    test_df["row_id"] = [f"test_{i}" for i in range(len(test_df))]
    if "row_id" not in validation_df.columns:
        validation_df["row_id"] = [f"validation_{i}" for i in range(len(validation_df))]

    llm_lookup = load_llm_lookup(llm_path) if llm_path else {}
    if llm_lookup_patch:
        # Misassigned validation path contained llm rows; primary --llm_file wins on row_id clash.
        llm_lookup = {**llm_lookup_patch, **llm_lookup}
        print(
            f"Merged llm rows from misassigned validation path (patch={len(llm_lookup_patch)} keys, "
            f"total lookup={len(llm_lookup)})"
        )
    if llm_path:
        print(f"llm file: {llm_path} (records={len(llm_lookup)})")
    if validation_file:
        print(f"Validation file: {validation_read_path} (rows={len(validation_df)})")

    validation_df["original_label"] = validation_df["binary_label"]

    def get_llm_label(rid: str) -> Optional[int]:
        if rid not in llm_lookup:
            return None
        r = llm_lookup[rid]
        # full_output.json uses llm_label; CSV uses label
        v = r.get("llm_label", r.get("label", r.get("label_text")))
        if v is None:
            return None
        return _parse_llm_label(v)

    def get_field(rid: str, key: str) -> Any:
        if rid not in llm_lookup:
            return None
        return llm_lookup[rid].get(key)

    validation_df["llm_label"] = validation_df["row_id"].apply(get_llm_label)
    # Preserve existing augmented annotations, but backfill missing validation values from llm lookup.
    def _fill_from_llm(df: pd.DataFrame, col: str, llm_key: str) -> None:
        if col not in df.columns:
            df[col] = None
        miss = df[col].isna() | df[col].astype(str).str.strip().eq("")
        if miss.any():
            df.loc[miss, col] = df.loc[miss, "row_id"].apply(lambda rid: get_field(rid, llm_key))

    _fill_from_llm(validation_df, "llm_target", "target")
    _fill_from_llm(validation_df, "llm_attack_cue", "attack_cue")
    _fill_from_llm(validation_df, "llm_mechanism", "mechanism")

    # Source column is optional; used when validation_file includes augmented rows
    if "source" not in validation_df.columns:
        validation_df["source"] = "validation"

    validation_df["train_label"] = validation_df["original_label"]
    if use_llm_as_gt_for_aug:
        is_aug = validation_df["source"].astype(str).str.lower().eq("aug")
        # Prefer second-pass llm label if present; otherwise llm_label from llm file
        if "t2_label" in validation_df.columns:
            validation_df.loc[is_aug, "train_label"] = validation_df.loc[is_aug, "t2_label"].apply(_parse_llm_label)
        else:
            validation_df.loc[is_aug, "train_label"] = validation_df.loc[is_aug, "llm_label"].apply(lambda x: _parse_llm_label(x) if x is not None else 0)
        # Agreement gating uses original_label; for augmented rows, treat llm-confirmed label as GT
        validation_df.loc[is_aug, "original_label"] = validation_df.loc[is_aug, "train_label"]
    validation_df["agreed"] = (
        (validation_df["llm_label"] == validation_df["original_label"]) & validation_df["llm_label"].notna()
    )

    mech_id_map = mech_to_id if mech_to_id is not None else MECH_TO_ID

    # Mechanism aux: require llm agreement and a valid mechanism label for this task taxonomy.
    def has_mechanism_row(row) -> bool:
        if not bool(row.get("agreed")):
            return False
        mech = row.get("llm_mechanism")
        if mech is None or not isinstance(mech, str):
            return False
        mech = mech.strip()
        if not mech or mech.lower() == "none" or mech.upper() == "N/A":
            return False
        return mech in mech_id_map
    validation_df["has_mechanism"] = validation_df.apply(has_mechanism_row, axis=1)

    train_df["source"] = "train"
    train_df["train_label"] = train_df["binary_label"]
    train_df["llm_label"] = None
    train_df["llm_target"] = None
    train_df["llm_attack_cue"] = None
    train_df["llm_mechanism"] = None
    train_df["agreed"] = False
    train_df["has_mechanism"] = False

    # Phi selected-hard (and similar) llm CSVs use **train_** row_ids; merge so mechanism
    # auxiliary loss applies on those rows when the llm agrees with gold labels.
    if llm_lookup:
        n_llm_hits = 0
        for pos, rid in enumerate(train_df["row_id"].astype(str).tolist()):
            if rid not in llm_lookup:
                continue
            n_llm_hits += 1
            rec = llm_lookup[rid]
            idx = train_df.index[pos]
            tl_raw = rec.get("llm_label", rec.get("label", rec.get("label_text")))
            if tl_raw is not None and str(tl_raw).strip() != "":
                train_df.at[idx, "llm_label"] = int(_parse_llm_label(tl_raw))
            train_df.at[idx, "llm_target"] = rec.get("target")
            train_df.at[idx, "llm_attack_cue"] = rec.get("attack_cue")
            train_df.at[idx, "llm_mechanism"] = rec.get("mechanism")
        train_df["agreed"] = train_df["llm_label"].notna() & (
            train_df["llm_label"] == train_df["train_label"]
        )
        train_df["has_mechanism"] = train_df.apply(has_mechanism_row, axis=1)
        print(
            f"  train llm merge: lookup_hits={n_llm_hits}  agreed={int(train_df['agreed'].sum())}  "
            f"has_mechanism={int(train_df['has_mechanism'].sum())}"
        )

    cols = [
        "row_id", "text", "class", "train_label", "source", "binary_label",
        "llm_label", "llm_target", "llm_attack_cue", "llm_mechanism",
        "agreed", "has_mechanism",
    ]
    combined = train_df[cols].copy()
    print(f"Train rows for CE: {len(combined)} (train.csv only; validation held out for tuning)")
    rd = validation_df
    n_hate = int((rd["original_label"] == 1).sum())
    n_agree = int(rd["agreed"].sum())
    n_mech = int(rd["has_mechanism"].sum())
    n_miss_llm = int(rd["llm_label"].isna().sum())
    print(
        f"  [distill diagnostics] HATE rows={n_hate}  agreed={n_agree}  "
        f"has_mechanism={n_mech}  missing_llm_label={n_miss_llm}"
    )
    return combined, test_df, train_df, validation_df, llm_lookup, validation_df


class MechEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        use_mechanism_head: bool = False,
        n_mechanism_classes: Optional[int] = None,
    ):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        # Expose config/num_labels for HF Trainer compatibility.
        self.config = self.backbone.config
        self.num_labels = 2
        hidden = int(self.backbone.config.hidden_size)
        self.use_mechanism_head = use_mechanism_head
        n_mech = int(n_mechanism_classes) if n_mechanism_classes is not None else N_MECHANISM_CLASSES
        self.mechanism_head = nn.Linear(hidden, n_mech) if use_mechanism_head else None
        # Exact same baseline CE classifier head.
        self.classifier = nn.Linear(hidden, 2)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        token_type_ids=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            out = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        else:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        last_hidden = out.last_hidden_state
        pooled = last_hidden[:, 0, :]
        logits = self.classifier(pooled)
        outd = {"logits": logits, "pooled_cls": pooled}
        if self.mechanism_head is not None:
            outd["mech_logits"] = self.mechanism_head(pooled)
        return outd


class DistillTrainer(Trainer):
    """HF Trainer path aligned with baseline, with mechanism auxiliary added in compute_loss."""

    def __init__(
        self,
        *args,
        ce_class_weight: Optional[torch.Tensor] = None,
        ce_contrastive_lambda: Optional[float] = None,
        contrastive_margin: float = 0.5,
        lambda_mech: float = 0.0,
        lambda_mech_ramp_ratio: float = 0.1,
        mech_debug_every_steps: int = 200,
        enforce_mech_per_batch: bool = True,
        mech_indices: Optional[List[int]] = None,
        train_batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ce_class_weight = ce_class_weight
        self.ce_contrastive_lambda: Optional[float] = None
        self._contrastive_loss: Optional[nn.Module] = None
        if ce_contrastive_lambda is not None:
            lam = float(ce_contrastive_lambda)
            if not (0.0 < lam <= 1.0):
                raise ValueError(
                    f"ce_contrastive_lambda must be in (0, 1], got {lam}. "
                    "Omit the flag (None) for CE-only; λ=0 is pure contrastive, not 'off'."
                )
            self.ce_contrastive_lambda = lam
            self._contrastive_loss = ContrastiveLossCosine(margin=float(contrastive_margin))
        self.lambda_mech = float(lambda_mech)
        self.lambda_mech_ramp_ratio = float(lambda_mech_ramp_ratio)
        self.mech_debug_every_steps = max(1, int(mech_debug_every_steps))
        self.enforce_mech_per_batch = bool(enforce_mech_per_batch)
        self.mech_indices = list(mech_indices or [])
        self.train_batch_size = max(1, int(train_batch_size))

    def get_train_dataloader(self):
        if not self.enforce_mech_per_batch or not self.mech_indices:
            return super().get_train_dataloader()
        n = len(self.train_dataset)
        mech_set = set(self.mech_indices)
        non_mech = [i for i in range(n) if i not in mech_set]
        if not non_mech:
            return super().get_train_dataloader()
        sampler = StratifiedAuxBatchSampler(self.mech_indices, non_mech, self.train_batch_size)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        mech_label = inputs.pop("mech_label", None)
        has_mech = inputs.pop("has_mech", None)
        mech_weight = inputs.pop("mech_weight", None)

        want_mech = self.lambda_mech > 0.0
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        logits = outputs["logits"]
        pooled_cls = outputs.get("pooled_cls")
        mech_logits = outputs.get("mech_logits")

        w = self.ce_class_weight
        if w is not None:
            w = w.to(logits.device, dtype=logits.dtype)
            ce_loss = F.cross_entropy(logits, labels, weight=w)
        else:
            ce_loss = F.cross_entropy(logits, labels)
        ce_mix_loss = ce_loss
        ctr_loss_val = 0.0
        if self._contrastive_loss is not None and self.ce_contrastive_lambda is not None:
            lam = float(self.ce_contrastive_lambda)
            ctr = self._contrastive_loss(logits.float(), labels.view(-1))
            ctr_loss_val = float(ctr.detach().item())
            ce_mix_loss = lam * ce_loss + (1.0 - lam) * ctr
        loss = ce_mix_loss

        def _ramp(lam: float, ratio: float) -> float:
            gs = int(getattr(self.state, "global_step", 0) or 0)
            ms = int(getattr(self.state, "max_steps", 0) or 0)
            ramp_steps = int(ratio * ms) if ms > 0 else 0
            if gs < ramp_steps:
                return 0.0
            denom = max(1, ms - ramp_steps) if ms > 0 else 1
            return float(lam) * (gs - ramp_steps) / denom

        mech_loss_val = 0.0
        mask_sum_val = 0.0
        lambda_m = 0.0
        if want_mech and mech_logits is not None and mech_label is not None and has_mech is not None:
            lambda_m = _ramp(self.lambda_mech, self.lambda_mech_ramp_ratio)
            if lambda_m > 0:
                per_sample_mech = F.cross_entropy(
                    mech_logits,
                    mech_label.to(logits.device).long(),
                    reduction="none",
                    ignore_index=-100,
                )
                mask_m = has_mech.to(logits.device).float()
                if mech_weight is not None:
                    mask_m = mask_m * mech_weight.to(logits.device).float()
                mask_sum_val = float(mask_m.sum().item())
                if mask_sum_val > 0:
                    mech_loss = (per_sample_mech * mask_m).sum() / (mask_m.sum() + 1e-8)
                    mech_loss_val = float(mech_loss.detach().item())
                    loss = loss + lambda_m * mech_loss

        gs_now = int(getattr(self.state, "global_step", 0) or 0)
        if gs_now % self.mech_debug_every_steps == 0:
            ce_v = float(ce_loss.detach().item())
            ce_mix_v = float(ce_mix_loss.detach().item())
            mech_to_ce = (float(lambda_m) * mech_loss_val) / (ce_v + 1e-8)
            print(
                "[mech-debug] "
                f"step={gs_now} "
                f"ce_loss={ce_v:.6f} "
                f"ce_mix_loss={ce_mix_v:.6f} "
                f"ctr_loss={ctr_loss_val:.6f} "
                f"mech_loss={mech_loss_val:.6f} "
                f"lambda_m={float(lambda_m):.6f} "
                f"mask_m.sum()={mask_sum_val:.2f} "
                f"mech_to_ce={mech_to_ce:.6f}"
            )

        return (loss, outputs) if return_outputs else loss


def apply_mech_aux_mode(
    *,
    mode: str,
    labels: Sequence[int],
    mech_labels: List[int],
    has_mech_list: List[int],
    n_mechanism_classes: int,
    random_state: int,
) -> Tuple[List[int], int, Dict[str, Any]]:
    """Rewrite aux targets for W2 controls. Returns (mech_labels, n_classes, meta)."""
    mode = str(mode or "mechanism").strip().lower()
    meta: Dict[str, Any] = {"mech_aux_mode": mode, "n_aux_rows": int(sum(has_mech_list))}
    if mode == "mechanism":
        return mech_labels, int(n_mechanism_classes), meta

    if mode == "binary":
        # Same selected mask; aux head predicts gold binary (matched emphasis, no taxonomy).
        out = []
        for y, has in zip(labels, has_mech_list):
            if int(has) == 1:
                out.append(int(y))
            else:
                out.append(-100)
        meta["n_mechanism_classes_effective"] = 2
        return out, 2, meta

    if mode == "shuffle_within_class":
        # Preserve per-class mechanism multiset; destroy within-class pairing.
        rng = np.random.default_rng(int(random_state) + 17_029)
        by_class: Dict[int, List[int]] = {0: [], 1: []}
        for y, m, has in zip(labels, mech_labels, has_mech_list):
            if int(has) == 1 and int(m) >= 0:
                by_class[int(y)].append(int(m))
        for c in (0, 1):
            rng.shuffle(by_class[c])
        meta["n_shuffled_class0"] = len(by_class[0])
        meta["n_shuffled_class1"] = len(by_class[1])
        out = []
        ptr = {0: 0, 1: 0}
        for y, m, has in zip(labels, mech_labels, has_mech_list):
            if int(has) == 1 and int(m) >= 0:
                c = int(y)
                out.append(int(by_class[c][ptr[c]]))
                ptr[c] += 1
            else:
                out.append(-100)
        # Sanity: frequencies preserved
        from collections import Counter

        meta["mech_hist_before"] = dict(Counter(m for m, h in zip(mech_labels, has_mech_list) if int(h) == 1 and int(m) >= 0))
        meta["mech_hist_after"] = dict(Counter(m for m, h in zip(out, has_mech_list) if int(h) == 1 and int(m) >= 0))
        meta["n_mechanism_classes_effective"] = int(n_mechanism_classes)
        return out, int(n_mechanism_classes), meta

    raise ValueError(f"unknown --mech_aux_mode={mode!r}")


def _build_distill_precomputed_rows(df: pd.DataFrame, tokenizer: Any, max_length: int) -> List[Dict[str, Any]]:
    """Encode each row once (distill-style, no per-step tokenize)."""
    df = df.reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for i in tqdm(range(len(df)), desc="Pre-tokenize train", leave=False):
        row = df.iloc[i]
        text = str(row["text"]) if isinstance(row["text"], str) else ""
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        mech = row.get("llm_mechanism")
        mech_id = -100
        has_mech = 0
        if isinstance(mech, str):
            mech = mech.strip()
            if mech in MECH_TO_ID and bool(row.get("has_mechanism")):
                mech_id = MECH_TO_ID[mech]
                has_mech = 1
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": int(row["train_label"]),
                "mech_label": mech_id,
                "has_mech": has_mech,
            }
        )
    return rows


class DistillPreTokenizedDataset(Dataset):
    """Rows from _build_distill_precomputed_rows (no tokenizer in __getitem__)."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class DistillCollator:
    """Pad input_ids / attention_mask to batch max length (baseline-style)."""

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f["label"] for f in features]
        mech_label = [f["mech_label"] for f in features]
        has_mech = [f["has_mech"] for f in features]
        input_ids = [f["input_ids"][: self.max_length] for f in features]
        attention_masks = [f["attention_mask"][: self.max_length] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.float32),
            "mech_label": torch.tensor(mech_label, dtype=torch.long),
            "has_mech": torch.tensor(has_mech, dtype=torch.float32),
        }


class DistillDataset(Dataset):
    """distill-style dataset: tokenize in __getitem__, fixed max_length padding."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        mech_labels: List[int],
        has_mech_list: List[int],
        mech_weight_list: Optional[List[float]],
        tokenizer: Any,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.mech_labels = mech_labels
        self.has_mech_list = has_mech_list
        self.mech_weight_list = mech_weight_list if mech_weight_list is not None else [1.0] * len(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": int(self.labels[idx]),
            "mech_label": int(self.mech_labels[idx]),
            "has_mech": int(self.has_mech_list[idx]),
            "mech_weight": float(self.mech_weight_list[idx]),
        }


def _collate_distill(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "mech_label": torch.tensor([b["mech_label"] for b in batch], dtype=torch.long),
        "has_mech": torch.tensor([b["has_mech"] for b in batch], dtype=torch.float32),
        "mech_weight": torch.tensor([b.get("mech_weight", 1.0) for b in batch], dtype=torch.float32),
    }


def _predict_probs_distill_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_distill,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=(device == "cuda"),
    )
    probs_all: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(np.atleast_1d(probs).tolist())
    return np.array(probs_all)


def _predict_probs_with_collator(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_distill,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    probs_all: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs_all.extend(np.atleast_1d(probs).tolist())
    return np.array(probs_all)


class StratifiedAuxBatchSampler:
    """Ensure at least 1 aux sample per batch when aux pool is non-empty."""

    def __init__(self, aux_idx: List[int], non_aux_idx: List[int], batch_size: int):
        self.aux_idx = list(aux_idx)
        self.non_aux_idx = list(non_aux_idx)
        self.batch_size = batch_size
        self.n_samples = len(aux_idx) + len(non_aux_idx)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        aux = self.aux_idx[:]
        non = self.non_aux_idx[:]
        random.shuffle(aux)
        random.shuffle(non)
        ai, ni = 0, 0
        for _ in range(len(self)):
            batch = []
            if aux:
                if ai >= len(aux):
                    random.shuffle(aux)
                    ai = 0
                batch.append(aux[ai])
                ai += 1
            while len(batch) < self.batch_size:
                if not non:
                    break
                if ni >= len(non):
                    random.shuffle(non)
                    ni = 0
                batch.append(non[ni])
                ni += 1
            yield batch


def _metrics_from_preds(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "auc": float(auc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _sweep_thresholds_macro_f1(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, Any], List[Dict[str, float]]]:
    thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    best_t = 0.5
    best_metrics: Optional[Dict[str, Any]] = None
    curve: List[Dict[str, float]] = []
    best_f1 = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        m = _metrics_from_preds(labels, preds, probs)
        f1m = float(m["f1_macro"])
        curve.append({"threshold": float(t), "f1_macro": f1m})
        if f1m > best_f1:
            best_f1 = f1m
            best_t = float(t)
            best_metrics = m
    return best_t, (best_metrics or _metrics_from_preds(labels, (probs >= 0.5).astype(int), probs)), curve


def _compute_metrics_for_trainer(eval_pred) -> Dict[str, float]:
    """Metrics callback for HF Trainer epoch evaluation/checkpoint selection."""
    pred_arr = eval_pred.predictions
    if isinstance(pred_arr, (tuple, list)) and len(pred_arr) > 0:
        pred_arr = pred_arr[0]
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    labels = np.asarray(eval_pred.label_ids, dtype=np.float64).ravel().astype(int)
    if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
        row_max = np.max(pred_arr, axis=-1, keepdims=True)
        expv = np.exp(pred_arr - row_max)
        probs = (expv[:, 1] / np.sum(expv, axis=-1)).astype(np.float64)
        preds = np.argmax(pred_arr, axis=-1).astype(int)
    else:
        logits = np.squeeze(pred_arr, axis=-1) if pred_arr.ndim > 1 else pred_arr
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
    m = _metrics_from_preds(labels, preds, probs)
    return {
        "accuracy": float(m["accuracy"]),
        "f1_macro": float(m["f1_macro"]),
        "precision_macro": float(m["precision_macro"]),
        "recall_macro": float(m["recall_macro"]),
        "auc_macro": float(m["auc"]),
    }


def _predict_probs_from_rows(
    model: nn.Module,
    rows: List[Dict[str, Any]],
    collator: DistillCollator,
    device: str,
    batch_size: int,
) -> np.ndarray:
    probs_all: List[float] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = collator(rows[i : i + batch_size])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(np.atleast_1d(probs).tolist())
    return np.array(probs_all)


def set_seed(seed: int) -> None:
    from pipeline.common.project_config import set_seed as _set_seed

    _set_seed(seed)


def verify_training_device(device: str, model: nn.Module) -> None:
    """Log device info; fail if GPU expected but model not on CUDA; tiny GPU matmul on CUDA."""
    p = next(model.parameters())
    print("=" * 60)
    print("  Device check")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  Selected training device: {device}")
    print(f"  Model parameters on: {p.device}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device is 'cuda' but torch.cuda.is_available() is False.")
        if p.device.type != "cuda":
            raise RuntimeError(f"Model is not on GPU (got {p.device}).")
        x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
        y = x @ x
        torch.cuda.synchronize()
        del x, y
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print("  GPU compute sanity check: OK")
    else:
        print("  WARNING: Training on CPU — expect much slower steps than on GPU.")
    print("=" * 60)


def train_one(
    combined_train: pd.DataFrame,
    eval_df: pd.DataFrame,
    args: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    pipeline_started_at_utc = datetime.now(timezone.utc).isoformat()
    pipeline_t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # distill-style training input pipeline: tokenize in dataset __getitem__ with fixed max_length.
    texts = combined_train["text"].fillna("").astype(str).tolist()
    labels = combined_train["train_label"].astype(int).tolist()
    labels_np = np.asarray(labels, dtype=np.int64)
    ce_class_weight = _compute_ce_class_weights(labels_np, args.pos_weight_mode)
    weight_target = (args.class_weight_target or "pos").strip().lower()
    print(
        f"pos_weight_mode={args.pos_weight_mode} class_weight_target={args.class_weight_target} "
        f"ce_class_weight={None if ce_class_weight is None else ce_class_weight.tolist()}"
    )
    task_mech_to_id: Dict[str, int] = getattr(args, "task_mech_to_id", None) or MECH_TO_ID
    n_mech_classes = int(getattr(args, "n_mechanism_classes", 0) or len(task_mech_to_id))
    prompt_task = str(getattr(args, "prompt_task", "hate"))
    print(f"prompt_task={prompt_task} n_mechanism_classes={n_mech_classes}")

    mech_labels: List[int] = []
    has_mech_list: List[int] = []
    mech_weight_list: List[float] = []
    for _, row in combined_train.iterrows():
        mech = row.get("llm_mechanism")
        if isinstance(mech, str):
            mech = mech.strip()
        if bool(row.get("has_mechanism")) and isinstance(mech, str) and mech in task_mech_to_id:
            mech_labels.append(task_mech_to_id[mech])
            has_mech_list.append(1)
            mech_weight_list.append(float(row.get("mech_weight", 1.0)))
        else:
            mech_labels.append(-100)
            has_mech_list.append(0)
            mech_weight_list.append(0.0)

    if float(getattr(args, "lambda_mech", 0.0) or 0.0) > 0.0 and sum(has_mech_list) == 0:
        raise SystemExit(
            "lambda_mech > 0 but no training row has a valid mechanism label. "
            "Pass --accepted_selected_k_csv (columns: row_id, mechanism) or --llm_file. "
            "row_id values are train_0, train_1, ... in train.csv order, and mechanism "
            "must be one of the closed labels for this dataset's task "
            "(hate: 8 labels; SBIC/offensive: 5 labels). "
            "See README.md and examples/mechanism_labels.csv."
        )

    mech_aux_mode = str(getattr(args, "mech_aux_mode", "mechanism") or "mechanism")
    mech_labels, n_mech_classes, mech_aux_meta = apply_mech_aux_mode(
        mode=mech_aux_mode,
        labels=labels,
        mech_labels=mech_labels,
        has_mech_list=has_mech_list,
        n_mechanism_classes=n_mech_classes,
        random_state=int(args.random_state),
    )
    print(
        f"[mech_aux_mode] mode={mech_aux_mode} n_aux_classes={n_mech_classes} "
        f"meta={ {k: mech_aux_meta[k] for k in mech_aux_meta if k != 'mech_hist_before' and k != 'mech_hist_after'} }"
    )
    if mech_aux_mode == "shuffle_within_class":
        print(
            f"[mech_aux_mode] hist_before={mech_aux_meta.get('mech_hist_before')} "
            f"hist_after={mech_aux_meta.get('mech_hist_after')}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = DistillDataset(
        texts=texts,
        labels=labels,
        mech_labels=mech_labels,
        has_mech_list=has_mech_list,
        mech_weight_list=mech_weight_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    eval_dataset = DistillDataset(
        texts=eval_df["text"].fillna("").astype(str).tolist(),
        labels=eval_df["binary_label"].astype(int).tolist(),
        mech_labels=[-100] * len(eval_df),
        has_mech_list=[0] * len(eval_df),
        mech_weight_list=[0.0] * len(eval_df),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    model = MechEncoder(
        args.model_name,
        use_mechanism_head=(args.lambda_mech > 0),
        n_mechanism_classes=n_mech_classes if args.lambda_mech > 0 else None,
    )
    model = model.to(device)
    verify_training_device(device, model)
    train_batch = max(1, int(args.batch_size))
    eval_batch = max(1, int(args.eval_batch_size))
    warmup_steps = int(((len(dataset) + train_batch - 1) // train_batch) * args.num_epochs * args.warmup_ratio)
    ta_kw: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(args.num_epochs),
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        # Julian / DGX: keep at most one HF checkpoint (best by eval_f1_macro).
        save_total_limit=1,
        seed=int(args.random_state),
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        fp16=bool(torch.cuda.is_available()),
        bf16=False,
        # Keep non-CUDA runs on CPU. Trainer would otherwise pick MPS on Mac and
        # leave weights on a device the post-training forward does not use.
        use_cpu=not bool(torch.cuda.is_available()),
    )
    training_args = TrainingArguments(**{k: v for k, v in ta_kw.items() if v is not None})
    dynamics_callback: Optional[TrainingDynamicsCallback] = None
    if bool(getattr(args, "track_dynamics", False)):
        dynamics_callback = TrainingDynamicsCallback(
            train_dataset=dataset,
            train_df=combined_train,
            collate_fn=_collate_distill,
            eval_batch_size=eval_batch,
            device=device,
            output_dir=output_dir,
            label_col="train_label",
        )
    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics_for_trainer_best_threshold,
        data_collator=_collate_distill,
        ce_class_weight=ce_class_weight,
        ce_contrastive_lambda=args.ce_contrastive_lambda,
        contrastive_margin=args.contrastive_margin,
        lambda_mech=float(args.lambda_mech),
        lambda_mech_ramp_ratio=float(args.lambda_mech_ramp_ratio),
        mech_debug_every_steps=int(args.mech_debug_every_steps),
        enforce_mech_per_batch=bool(args.enforce_mech_per_batch),
        mech_indices=[i for i, v in enumerate(has_mech_list) if int(v) == 1],
        train_batch_size=train_batch,
        callbacks=[dynamics_callback] if dynamics_callback is not None else None,
    )
    train_t0 = time.perf_counter()
    setup_wall_sec = float(train_t0 - pipeline_t0)
    train_result = trainer.train()
    train_wall_sec = float(time.perf_counter() - train_t0)
    hf_train_metrics: Dict[str, Any] = {}
    if train_result is not None and getattr(train_result, "metrics", None):
        for _k, _v in train_result.metrics.items():
            if isinstance(_v, bool) or _v is None or isinstance(_v, str):
                hf_train_metrics[_k] = _v
            elif hasattr(_v, "item"):
                hf_train_metrics[_k] = _v.item()
            elif isinstance(_v, (float, int)):
                hf_train_metrics[_k] = _v
            else:
                hf_train_metrics[_k] = str(_v)
    print(
        f"\n⏱ Timing (wall clock): setup={setup_wall_sec:.1f}s  Trainer.train={train_wall_sec:.1f}s"
        + (
            f"  HF train_runtime={hf_train_metrics.get('train_runtime', 'n/a')}"
            if hf_train_metrics
            else ""
        )
    )

    dynamics_summary_rows: Optional[int] = None
    if dynamics_callback is not None:
        dynamics_df = dynamics_callback.finalize()
        if dynamics_df is not None:
            dynamics_summary_rows = int(len(dynamics_df))
            print(f"training_dynamics.csv rows={dynamics_summary_rows}")

    # Evaluate on train + test (distill-style full report)
    model.to(device)
    model.eval()
    infer_t0 = time.perf_counter()
    train_labels = combined_train["train_label"].astype(int).values
    train_probs = _predict_probs_with_collator(model, dataset, device, args.eval_batch_size)
    best_train_t, train_metrics, _train_curve_unused = _sweep_thresholds_macro_f1(train_labels, train_probs)
    train_metrics["n_samples"] = int(len(train_labels))

    eval_labels = eval_df["binary_label"].astype(int).values
    eval_probs = _predict_probs_with_collator(model, eval_dataset, device, args.eval_batch_size)
    posthoc_inference_wall_sec = float(time.perf_counter() - infer_t0)
    n_pred_samples = int(len(train_labels) + len(eval_labels))
    pred_samples_per_sec = (
        (n_pred_samples / posthoc_inference_wall_sec) if posthoc_inference_wall_sec > 0 else None
    )
    test_metrics = _metrics_from_preds(eval_labels, (eval_probs >= best_train_t).astype(int), eval_probs)
    test_metrics["n_samples"] = int(len(eval_labels))

    total_pipeline_wall_sec = float(time.perf_counter() - pipeline_t0)
    timing_detail: Dict[str, Any] = {
        "pipeline_started_at_utc": pipeline_started_at_utc,
        "setup_wall_sec": round(setup_wall_sec, 4),
        "trainer_train_wall_sec": round(train_wall_sec, 4),
        "posthoc_inference_wall_sec": round(posthoc_inference_wall_sec, 4),
        "total_pipeline_wall_sec": round(total_pipeline_wall_sec, 4),
        "posthoc_prediction_rows": n_pred_samples,
        "posthoc_prediction_rows_per_sec": (
            round(pred_samples_per_sec, 4) if pred_samples_per_sec is not None else None
        ),
        "huggingface_trainer_metrics": hf_train_metrics,
        "notes": (
            "Mech distill: trainer_train_wall_sec is HF training; "
            "posthoc_inference_wall_sec is batched forward on train+eval for metrics."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    distill_cfg: Dict[str, Any] = {
        "approach": "mech",
        "model_name": args.model_name,
        "lambda_mech": float(args.lambda_mech),
        "lambda_mech_ramp_ratio": float(args.lambda_mech_ramp_ratio),
        "mech_aux_mode": str(getattr(args, "mech_aux_mode", "mechanism") or "mechanism"),
        "mech_aux_meta": mech_aux_meta,
        "random_state": int(args.random_state),
        "max_length": int(args.max_length),
        "splitted_data_subdir": str(SPLITTED_DATA_SUBDIR),
        "track_dynamics": bool(getattr(args, "track_dynamics", False)),
        "training_dynamics_rows": dynamics_summary_rows,
        "timing": timing_detail,
        **mechanism_training_metadata(prompt_task),
    }
    distill_cfg["n_mechanism_classes"] = int(n_mech_classes)
    distill_cfg["mech_aux_mode"] = str(getattr(args, "mech_aux_mode", "mechanism") or "mechanism")
    distill_cfg["mech_aux_meta"] = mech_aux_meta
    (output_dir / "encoder_distill_config.json").write_text(
        json.dumps(distill_cfg, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "encoder_model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return {
        "timing": timing_detail,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "threshold_policy": {
            "sweep_min": 0.05,
            "sweep_max": 0.95,
            "sweep_step": 0.05,
            "selection_metric": "f1_macro",
            "best_threshold_train": float(best_train_t),
            "applied_threshold_test": float(best_train_t),
        },
        "class_weighting": f"ce_weight:{args.pos_weight_mode}",
        "pos_weight_mode": str(args.pos_weight_mode),
        "pos_weight_value": None,
        "ce_class_weight": (ce_class_weight.detach().cpu().tolist() if ce_class_weight is not None else None),
        "train_samples": int(len(combined_train)),
        "test_samples": int(len(eval_df)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="EchoHate: encoder training (CE + optional mechanism aux)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--llm_file",
        default=None,
        help=(
            "Mechanism-label file (CSV or JSON) keyed by train row_id. "
            "Rows contribute to the auxiliary loss only when label agrees with gold "
            "and mechanism is in the task inventory. "
            "Omit this when using --accepted_selected_k_csv."
        ),
    )
    p.add_argument(
        "--accepted_selected_k_csv",
        default=None,
        help=(
            "Selected-row supervision CSV with row_id and mechanism. "
            "These rows receive the auxiliary loss without a separate label-agreement check. "
            "row_id must match train.csv order: train_0, train_1, ... unless the split already stores row_id."
        ),
    )
    p.add_argument("--bucket_weight_stable_hard", type=float, default=1.0)
    p.add_argument("--bucket_weight_ambiguous", type=float, default=0.25)
    p.add_argument("--bucket_weight_easy", type=float, default=0.1)
    p.add_argument(
        "--validation_file",
        default=None,
        help=f"Override validation split file. Default: {SPLITTED_DATA_SUBDIR}/validation.csv",
    )
    p.add_argument(
        "--evaluation_split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Evaluation split for reported metrics.",
    )
    p.add_argument(
        "--allow_test_eval_each_epoch",
        action="store_true",
        help="Safety override: allow per-epoch eval on test split (default blocks this to avoid leakage).",
    )
    p.add_argument("--use_llm_as_gt_for_aug", action="store_true", help="If validation_file has source=aug rows, use llm-confirmed label as training GT for those rows.")

    p.add_argument("--model_name", default="roberta-large", help="HF encoder id. Paper runs use roberta-large.")
    p.add_argument("--num_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--random_state", type=int, default=None, help="RNG seed (default: config/echohate_project_config.json seeds.random_state)")
    p.add_argument("--pos_weight_mode", type=str, default="none", choices=["freq", "sqrt", "none"])
    p.add_argument(
        "--class_weight_target",
        type=str,
        default="pos",
        choices=["pos", "neg"],
        help="Which class to weight in BCE: pos(1) or neg(0).",
    )
    p.add_argument(
        "--ce_contrastive_lambda",
        "--ce-contrastive-lambda",
        type=float,
        default=None,
        help="If set in [0,1]: L*CE + (1-L)*contrastive on logits. Omit/None = CE (+ mech) only.",
    )
    p.add_argument(
        "--contrastive_margin",
        "--contrastive-margin",
        type=float,
        default=0.75,
        help="Margin for logit cosine contrastive loss when --ce_contrastive_lambda is set.",
    )

    p.add_argument(
        "--prompt_task",
        default=None,
        choices=["hate", "offensive"],
        help="Mechanism taxonomy (default: infer from dataset via config/echohate_project_config.json).",
    )
    p.add_argument("--lambda_mech", type=float, default=0.05, help="Weight for mechanism auxiliary loss (multi-class).")
    p.add_argument("--lambda_mech_ramp_ratio", type=float, default=0.1, help="Ramp schedule for mechanism loss (fraction of steps).")
    p.add_argument(
        "--mech_aux_mode",
        type=str,
        default="mechanism",
        choices=["mechanism", "binary", "shuffle_within_class"],
        help=(
            "What the aux head predicts on has_mechanism rows (W2 control). "
            "mechanism=8-way LLM mechanism (paper); "
            "binary=gold binary label on the same selected rows (2-way head); "
            "shuffle_within_class=permute LLM mechanism ids within each gold class "
            "(keeps class compatibility + frequencies, destroys within-class semantics)."
        ),
    )
    p.add_argument("--mech_debug_every_steps", type=int, default=200, help="Print mech debug stats every N training steps.")
    p.add_argument(
        "--enforce_mech_per_batch",
        dest="enforce_mech_per_batch",
        action="store_true",
        help="Ensure each train batch includes mechanism rows when available.",
    )
    p.add_argument(
        "--no-enforce_mech_per_batch",
        dest="enforce_mech_per_batch",
        action="store_false",
        help="Disable mech-aware batch sampling.",
    )
    p.set_defaults(enforce_mech_per_batch=True)
    p.add_argument(
        "--require_gpu",
        action="store_true",
        help="Exit if CUDA is not available (recommended for cluster GPU jobs).",
    )
    p.add_argument(
        "--track-dynamics",
        action="store_true",
        help="Record per-epoch training dynamics on train split (training_dynamics.csv).",
    )
    args = p.parse_args()
    if args.random_state is None:
        args.random_state = int(default_random_state())

    if args.require_gpu and not torch.cuda.is_available():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise SystemExit(
            f"ERROR: --require_gpu set but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={cvd}. "
            "Use a PyTorch build with CUDA, and set CUDA_VISIBLE_DEVICES before starting Python."
        )

    set_seed(args.random_state)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else (data_root / args.dataset / "mech_distill")
    if args.llm_file:
        llm_path: Optional[Path] = Path(args.llm_file)
    else:
        default_llm = data_root / args.dataset / "llm_classified_data" / "full_output.json"
        llm_path = default_llm if default_llm.is_file() else None
    validation_path = Path(args.validation_file) if args.validation_file else None

    prompt_task = (args.prompt_task or prompt_task_for_dataset(args.dataset)).strip().lower()
    task_mech_to_id = mech_to_id_for_task(prompt_task)
    n_mech_classes = n_mechanism_classes_for_task(prompt_task)
    args.prompt_task = prompt_task
    args.task_mech_to_id = task_mech_to_id
    args.n_mechanism_classes = n_mech_classes
    print(f"Mechanism head: prompt_task={prompt_task} n_classes={n_mech_classes} (paper 8way / offensive 5)")

    combined, test_df, train_df, validation_df, llm_lookup, _ = load_splits_and_llm(
        data_root,
        args.dataset,
        llm_path,
        validation_file=validation_path,
        use_llm_as_gt_for_aug=args.use_llm_as_gt_for_aug,
        mech_to_id=task_mech_to_id,
    )
    if args.accepted_selected_k_csv:
        accepted_df = pd.read_csv(args.accepted_selected_k_csv)
        if "row_id" not in accepted_df.columns:
            raise SystemExit("--accepted_selected_k_csv must include row_id")
        allowed = set(accepted_df["row_id"].astype(str).tolist())
        # Simple mode: all accepted rows already come from selected hard/severe runs.
        # Use uniform mechanism weight to avoid bucket-name dependency.
        accepted_df["mech_weight"] = 1.0
        if "mechanism" in accepted_df.columns:
            mech_map = dict(zip(accepted_df["row_id"].astype(str), accepted_df["mechanism"]))
            combined["llm_mechanism"] = combined["row_id"].astype(str).map(mech_map).where(
                combined["row_id"].astype(str).isin(allowed),
                other=combined["llm_mechanism"],
            )
        weight_map = dict(zip(accepted_df["row_id"].astype(str), accepted_df["mech_weight"]))
        combined["mech_weight"] = combined["row_id"].astype(str).map(weight_map).fillna(0.0).astype(float)
        combined["has_mechanism"] = combined["row_id"].astype(str).isin(allowed)
    else:
        combined["mech_weight"] = combined["has_mechanism"].astype(float)
    mech_aux_total = int(combined["has_mechanism"].sum())
    mech_rel = int(((combined["source"].astype(str).str.lower() == "validation") & combined["has_mechanism"]).sum())
    mech_aug = int(((combined["source"].astype(str).str.lower() == "aug") & combined["has_mechanism"]).sum())
    print(
        f"Mech distill: lambda_mech={args.lambda_mech}  "
        f"has_mechanism_total={mech_aux_total}  source=validation={mech_rel}  source=aug={mech_aug}"
    )
    eval_df = validation_df if args.evaluation_split == "validation" else test_df
    if args.evaluation_split == "test" and not bool(args.allow_test_eval_each_epoch):
        raise SystemExit(
            "Refusing per-epoch eval on test split to avoid leakage. "
            "Use --evaluation_split validation (recommended), or pass --allow_test_eval_each_epoch to override."
        )
    print(f"Evaluation split: {args.evaluation_split} (samples={len(eval_df)})")
    res = train_one(combined, eval_df, args, out_dir)
    summary = {
        "model_info": {
            "model_name": args.model_name,
            "output_dir": str(out_dir),
            "training_date": datetime.now().isoformat(),
        },
        "timing": res["timing"],
        "dataset_info": {
            "train_combined_samples": int(res["train_samples"]),
            "test_samples": int(res["test_samples"]),
            "splitted_data_subdir": str(SPLITTED_DATA_SUBDIR),
        },
        "training_config": {
            "num_epochs": int(args.num_epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_length": int(args.max_length),
            "random_state": int(args.random_state),
            "pos_weight_mode": str(args.pos_weight_mode),
            "class_weight_target": str(args.class_weight_target),
            "ce_contrastive_lambda": (
                None if args.ce_contrastive_lambda is None else float(args.ce_contrastive_lambda)
            ),
            "contrastive_margin": float(args.contrastive_margin),
            "lambda_mech": float(args.lambda_mech),
            "lambda_mech_ramp_ratio": float(args.lambda_mech_ramp_ratio),
            **mechanism_training_metadata(prompt_task),
            "eval_each_epoch": True,
            "evaluation_split": str(args.evaluation_split),
            "class_weighting": f"ce_weight:{args.pos_weight_mode}",
        },
        "threshold_policy": res["threshold_policy"],
        "train_metrics": res["train_metrics"],
        "test_metrics": res["test_metrics"],
    }
    with open(out_dir / "metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_dir}")
    tm = summary["test_metrics"]
    print(f"Test: acc={tm['accuracy']:.4f} f1_macro={tm['f1_macro']:.4f} auc={tm['auc']:.4f}")


if __name__ == "__main__":
    main()

