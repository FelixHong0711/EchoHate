#!/usr/bin/env python3
"""Row-id helpers for hard-case selection (var25 / rand)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


def _synthetic_row_id_prefix(input_csv: Path) -> str:
    """Match ``roberta_baseline`` / split-CSV row_id conventions."""
    name = input_csv.name.lower()
    if name == "train.csv":
        return "train"
    if name == "validation.csv":
        return "validation"
    if name == "test.csv":
        return "test"
    return "train"


def map_row_ids_for_inference(selected: pd.DataFrame, input_csv: Path) -> pd.DataFrame:
    """Map dynamics row_ids (e.g. train_combined_N) to ids used in the split CSV for LLM infer."""
    with open(input_csv, newline="", encoding="utf-8") as f:
        has_row_id = "row_id" in (csv.DictReader(f).fieldnames or [])
    if has_row_id:
        return selected[["row_id"]].drop_duplicates()
    prefix = _synthetic_row_id_prefix(input_csv)
    rows = []
    for rid in selected.get("row_id", []):
        s = str(rid).strip()
        m = re.match(r"^(?:train|validation|test|train_combined)_(\d+)$", s)
        rows.append(f"{prefix}_{m.group(1)}" if m else s)
    return pd.DataFrame({"row_id": rows}).drop_duplicates()
