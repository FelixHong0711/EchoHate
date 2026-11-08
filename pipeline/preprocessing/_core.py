"""Shared text/label helpers for preprocess pipeline."""

from __future__ import annotations

import pandas as pd

HATE_LABELS = frozenset({"hate", "explicit_hate", "implicit_hate"})


def sanitize_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    s = s.strip()
    if s and s[0] in "-=+@\t":
        return "'" + s
    return s


def sanitize_text_for_csv(s: str) -> str:
    return sanitize_text(s if isinstance(s, str) else str(s))


def label_to_class(s: pd.Series) -> pd.Series:
    normalized = s.astype(str).str.strip().str.lower()
    return normalized.isin(HATE_LABELS).astype(int)


def binary_class(labels: pd.Series, dataset: str) -> pd.Series:
    """Map label column to binary class=1 (positive) per dataset vocabulary."""
    norm = labels.astype(str).str.strip().str.lower()
    ds = (dataset or "").strip()
    if ds == "DynaHate":
        return norm.eq("hate").astype(int)
    return norm.isin(HATE_LABELS).astype(int)


def drop_blank_and_duplicate_text(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=[text_col], keep="first")
    return df.reset_index(drop=True)
