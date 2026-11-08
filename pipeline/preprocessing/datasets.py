#!/usr/bin/env python3
"""Dataset preprocessing for the EchoHate 7-dataset 811 split set.

Kept datasets: IHC, HatEval, DynaHate, Ethos, drop_offensive, toxigen_new, SBIC.
HateXplain path: ``hatexplain-random-split-variants --variants drop_offensive``
(raw ``dataset.json`` → preprocessed → stratified 80/10/10).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore

from pipeline.common.paths import (
    ECHOHATE_ROOT,
    SPLITTED_DATA_SUBDIR,
    TEST_FRAC,
    TRAIN_FRAC,
    VALIDATION_FRAC,
    get_data_root,
)
from pipeline.common.project_config import default_split_random_state
from pipeline.preprocessing._core import (
    binary_class,
    drop_blank_and_duplicate_text,
    sanitize_text_for_csv,
)
from pipeline.preprocessing.split import run_dataset, run_split_subprocess


def resolve_data_root(cli_value: str | None) -> Path:
    """Parent directory of per-dataset folders under data/."""
    if cli_value and str(cli_value).strip():
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("ECHOHATE_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return get_data_root().resolve()


def _split_fracs() -> tuple[float, float, float, str]:
    """Train/val/test fractions and ``splitted_data_*`` subdir from ``pipeline.common.paths``."""
    return float(TRAIN_FRAC), float(VALIDATION_FRAC), float(TEST_FRAC), SPLITTED_DATA_SUBDIR



# --- dynahate ---

DATASET_NAME_DYNAHATE = "DynaHate"
_DEFAULT_RAW_NAMES = (
    "DynaHate_v0.2.2.csv",
    "Dynamically Generated Hate Dataset v0.2.2.csv",
)
_SPLIT_OUT = (
    ("train", "train.csv"),
    ("dev", "validation.csv"),
    ("test", "test.csv"),
)
_NOT_HATE_LABELS = frozenset({"not_hate", "nothate", "no_hate", "normal", "non_hate"})

def _resolve_raw_csv(load_dir: Path, raw_name: str | None) -> Path:
    load_dir = load_dir.expanduser().resolve()
    if not load_dir.is_dir():
        raise FileNotFoundError(load_dir)

    if raw_name:
        path = load_dir / raw_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    for name in _DEFAULT_RAW_NAMES:
        path = load_dir / name
        if path.is_file():
            return path

    matches = sorted(load_dir.glob("*.csv"))
    if len(matches) == 1:
        return matches[0]
    v022 = [p for p in matches if "0.2.2" in p.name or "v0.2.2" in p.name.lower()]
    if len(v022) == 1:
        return v022[0]

    raise FileNotFoundError(
        f"No DynaHate CSV under {load_dir}. Pass --raw-csv or place one of: {_DEFAULT_RAW_NAMES}"
    )

def _normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in _NOT_HATE_LABELS:
        return "not_hate"
    if s == "hate":
        return "hate"
    return s

def preprocess_dynahate(
    data_root: Path,
    dataset_name: str = DATASET_NAME_DYNAHATE,
    raw_csv: str | None = None,
) -> pd.DataFrame:
    load_dir = data_root / dataset_name / "raw_data"
    raw_path = _resolve_raw_csv(load_dir, raw_csv)

    df = pd.read_csv(raw_path)
    if df.shape[1] > 0 and str(df.columns[0]).strip() in ("", "Unnamed: 0"):
        df = df.drop(columns=[df.columns[0]])

    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"{raw_path}: expected columns text, label; got {list(df.columns)}")

    df["text"] = df["text"].astype(str).map(sanitize_text_for_csv)
    df["label"] = df["label"].map(_normalize_label)
    before = len(df)
    df = df[df["label"].isin({"hate", "not_hate"})].copy()
    dropped_labels = before - len(df)
    df["class"] = binary_class(df["label"], dataset_name)
    df = drop_blank_and_duplicate_text(df, text_col="text")
    df = df[["text", "label", "class"]].reset_index(drop=True)

    out_dir = data_root / dataset_name / "preprocessed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.csv"
    df.to_csv(out_path, index=False)

    vc = df["label"].value_counts()
    print(f"Raw: {raw_path}", flush=True)
    print(f"  dropped unknown labels: {dropped_labels}", flush=True)
    print(f"  labels: {vc.to_dict()}", flush=True)
    print(f"  wrote {len(df)} rows -> {out_path}", flush=True)
    return df

def _read_and_normalize_raw(
    raw_path: Path,
    dataset_name: str,
    *,
    dedupe: str = "global",
) -> pd.DataFrame:
    """
  Normalize labels/text from raw CSV.

  ``dedupe``:
    - ``global``: drop blank/duplicate text on the full frame (stratified 811 path).
    - ``none``: keep all rows (official splits; no blank/dedupe removal).
    """
    df = pd.read_csv(raw_path)
    if df.shape[1] > 0 and str(df.columns[0]).strip() in ("", "Unnamed: 0"):
        df = df.drop(columns=[df.columns[0]])
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"{raw_path}: expected columns text, label; got {list(df.columns)}")
    if dedupe == "none" and "split" not in df.columns:
        raise ValueError(f"{raw_path}: missing 'split' column; got {list(df.columns)}")

    df = df.copy()
    df["text"] = df["text"].astype(str).map(sanitize_text_for_csv)
    df["label"] = df["label"].map(_normalize_label)
    before = len(df)
    df = df[df["label"].isin({"hate", "not_hate"})].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  dropped unknown labels: {dropped}", flush=True)
    df["class"] = binary_class(df["label"], dataset_name)
    if dedupe == "global":
        df = drop_blank_and_duplicate_text(df, text_col="text")
    return df

def export_official_splits_echohate(
    data_root: Path,
    dataset_name: str,
    raw_csv: str | None = None,
    split_subdir: str = "splitted_data_80_10_10",
) -> dict[str, int]:
    """
    Official DynaHate splits → EchoHate layout (text, label, class only).

    train → train.csv, dev → validation.csv, test → test.csv.
    """
    from pipeline.common.paths import SPLITTED_DATA_SUBDIR

    sub = split_subdir or SPLITTED_DATA_SUBDIR
    load_dir = data_root / dataset_name / "raw_data"
    raw_path = _resolve_raw_csv(load_dir, raw_csv)
    df = _read_and_normalize_raw(raw_path, dataset_name, dedupe="none")

    out_dir = data_root / dataset_name / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    parts: list[pd.DataFrame] = []

    for split_name, out_name in _SPLIT_OUT:
        part = df.loc[df["split"] == split_name, ["text", "label", "class"]].copy()
        part.to_csv(out_dir / out_name, index=False)
        counts[out_name] = len(part)
        parts.append(part)

    unknown = df.loc[~df["split"].isin({s for s, _ in _SPLIT_OUT})]
    if len(unknown):
        print(
            f"WARN: {len(unknown)} rows with split not in train/dev/test "
            f"(values: {sorted(unknown['split'].astype(str).unique())[:8]})",
            file=sys.stderr,
        )

    pre_dir = data_root / dataset_name / "preprocessed_data"
    pre_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(pre_dir / "preprocessed.csv", index=False)

    manifest = {
        "official_splits": True,
        "stratified": False,
        "split_mode": "official",
        "source": str(raw_path),
        "subdir": sub,
        "mapping": {"train": "train.csv", "dev": "validation.csv", "test": "test.csv"},
        "counts": counts,
        "preprocessed_rows": len(combined),
        "note": "Paper train/dev/test from raw 'split' column; dev → validation.csv. Not re-drawn 80/10/10.",
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Raw: {raw_path}", flush=True)
    print(f"  official splits -> {out_dir}", flush=True)
    for out_name, n in counts.items():
        print(f"    {out_name}: {n} rows", flush=True)
    print(f"  preprocessed union ({len(combined)} rows) -> {pre_dir / 'preprocessed.csv'}", flush=True)
    return counts

def export_official_splits(raw_path: Path, out_dir: Path) -> dict[str, int]:
    """Legacy: write full raw rows (all columns) to ``official_splits/``."""
    df = pd.read_csv(raw_path)
    if df.shape[1] > 0:
        df = df.drop(columns=[df.columns[0]])
    if "split" not in df.columns:
        raise ValueError(f"{raw_path}: missing 'split' column; got {list(df.columns)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, out_name in _SPLIT_OUT:
        part = df.loc[df["split"] == split_name]
        out_path = out_dir / out_name
        part.to_csv(out_path, index=False)
        counts[out_name] = len(part)

    unknown = df.loc[~df["split"].isin({s for s, _ in _SPLIT_OUT})]
    if len(unknown):
        print(
            f"WARN: {len(unknown)} rows with split not in train/dev/test "
            f"(values: {sorted(unknown['split'].astype(str).unique())[:8]})",
            file=sys.stderr,
        )
    return counts

def _raw_csv_has_split_column(load_dir: Path, raw_csv: str | None) -> bool:
    raw_path = _resolve_raw_csv(load_dir, raw_csv)
    header = pd.read_csv(raw_path, nrows=0)
    return "split" in header.columns

def main_dynahate(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="DynaHate v0.2.2: official train/dev/test or stratified 80/10/10."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Parent of {DATASET_NAME_DYNAHATE}/ (default: project config data seed dir).",
    )
    ap.add_argument("--dataset", type=str, default=DATASET_NAME_DYNAHATE)
    ap.add_argument(
        "--load-dir",
        type=Path,
        default=None,
        help="Raw CSV directory (default: <data-root>/<dataset>/raw_data).",
    )
    ap.add_argument("--raw-csv", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Split seed (default: create_split_random_state).")
    ap.add_argument("--no-split", action="store_true", help="Only write preprocessed.csv.")
    ap.add_argument(
        "--balanced-validation-test",
        action="store_true",
        help="Balanced validation/test from holdout (optional).",
    )
    ap.add_argument(
        "--official-splits",
        action="store_true",
        help="Use dataset train/dev/test → splitted_data_80_10_10/.",
    )
    ap.add_argument(
        "--stratified-811",
        action="store_true",
        help="Force sklearn 80/10/10 stratified split (ignores official split column).",
    )
    ap.add_argument(
        "--official-splits-only",
        action="store_true",
        help="Write raw-column official splits to official_splits/ (legacy/debug).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Official splits output dir (with --official-splits-only).",
    )
    args = ap.parse_args(argv)

    data_root = resolve_data_root(str(args.data_root) if args.data_root else None)
    dataset_name = args.dataset.strip() or DATASET_NAME_DYNAHATE
    if dataset_name != DATASET_NAME_DYNAHATE:
        print(
            f"WARN: --dataset {dataset_name!r} ignored; using {DATASET_NAME_DYNAHATE!r}",
            file=sys.stderr,
        )
        dataset_name = DATASET_NAME_DYNAHATE
    load_dir = args.load_dir or (data_root / dataset_name / "raw_data")
    raw_path = _resolve_raw_csv(load_dir, args.raw_csv)

    use_official = bool(args.official_splits) or (
        not args.stratified_811 and _raw_csv_has_split_column(load_dir, args.raw_csv)
    )

    if use_official and not args.official_splits_only:
        export_official_splits_echohate(
            data_root=data_root,
            dataset_name=dataset_name,
            raw_csv=args.raw_csv,
        )
        print(f"Done. EchoHate splits: {data_root / dataset_name / 'splitted_data_80_10_10'}", flush=True)
        return

    if args.official_splits_only:
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir
            else (data_root / dataset_name / "official_splits").resolve()
        )
        counts = export_official_splits(raw_path, out_dir)
        print(f"Source: {raw_path}  (rows after drop col0: {sum(counts.values())})")
        for out_name, n in counts.items():
            print(f"  Wrote {out_dir / out_name}  rows={n}")
        print(f"Done. Output dir: {out_dir}")
        return

    preprocess_dynahate(data_root=data_root, dataset_name=dataset_name, raw_csv=args.raw_csv)

    if args.no_split:
        return

    seed = int(args.seed) if args.seed is not None else int(default_split_random_state())
    env = os.environ.copy()
    env["ECHOHATE_DATA_ROOT"] = str(data_root)
    run_split_subprocess(
        dataset_name,
        seed,
        balanced_validation_test=bool(args.balanced_validation_test),
        env=env,
    )
    print(
        f"Split done under {data_root / dataset_name / 'splitted_data_80_10_10'}",
        flush=True,
    )


# --- hatexplain-random-split-variants (drop_offensive 811) ---

HATEXPLAIN_RANDOM_SPLIT_SOURCE = "HateXplain_random_split"
HATEXPLAIN_RANDOM_SPLIT_INPUT_REL = "raw_data/dataset.json"
SPLIT_SEED = int(default_split_random_state())

HATEXPLAIN_RANDOM_SPLIT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "drop_offensive": {
        "description": "drop offensive; hatespeech->hate, normal->not_hate",
        "keep_raw_label": lambda lab: lab in ("hatespeech", "normal"),
        "map_binary": lambda lab: ("hate", 1) if lab == "hatespeech" else ("not_hate", 0),
    },
}

@dataclass(frozen=True)
class ProcRow:
    post_id: str
    text: str
    label: str
    cls: int

def majority_label(annotators: List[Dict[str, Any]], tie_policy: str = "drop") -> Optional[str]:
    labels = [a.get("label") for a in annotators if a.get("label") is not None]
    if not labels:
        return None
    counts = Counter(labels)
    max_count = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return str(winners[0]).strip().lower()
    if tie_policy == "drop":
        return None
    if tie_policy == "first":
        return sorted(str(w).strip().lower() for w in winners)[0]
    raise ValueError(f"Unknown tie_policy={tie_policy}")

def build_text(post_tokens: Iterable[Any]) -> str:
    tokens = [t if isinstance(t, str) else str(t) for t in post_tokens]
    return sanitize_text_for_csv(" ".join(tokens)).strip()

def iter_rows_from_json_data(
    data: Dict[str, Any],
    *,
    tie_policy: str,
    keep_raw_label: Callable[[str], bool],
    map_binary: Callable[[str], tuple[str, int]],
) -> Iterator[ProcRow]:
    for post_key, post in data.items():
        annotators = post.get("annotators", []) or []
        raw = majority_label(annotators, tie_policy=tie_policy)
        if raw is None or not keep_raw_label(raw):
            continue
        text = build_text(post.get("post_tokens", []) or [])
        if not text:
            continue
        out_label, out_class = map_binary(raw)
        post_id = str(post.get("post_id", post_key))
        yield ProcRow(post_id=post_id, text=text, label=out_label, cls=out_class)

def write_preprocessed(rows: List[ProcRow], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "class"])
        w.writeheader()
        for r in rows:
            w.writerow({"text": r.text, "label": r.label, "class": r.cls})
    return len(rows)

def load_json_data(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_hatexplain_random_split_variant(
    data_root: Path,
    variant_name: str,
    data: Dict[str, Any],
    input_json: Path,
    *,
    tie_policy: str,
    split_seed: int,
    link_raw_json: bool,
) -> None:
    if variant_name not in HATEXPLAIN_RANDOM_SPLIT_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_name!r}; choose from {sorted(HATEXPLAIN_RANDOM_SPLIT_VARIANTS)}"
        )
    cfg = HATEXPLAIN_RANDOM_SPLIT_VARIANTS[variant_name]
    dataset_dir = data_root / variant_name
    raw_dir = dataset_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if link_raw_json:
        dest = raw_dir / "dataset.json"
        if not dest.exists():
            try:
                dest.symlink_to(input_json.resolve())
            except OSError:
                import shutil

                shutil.copy2(input_json, dest)

    rows = list(
        iter_rows_from_json_data(
            data,
            tie_policy=tie_policy,
            keep_raw_label=cfg["keep_raw_label"],
            map_binary=cfg["map_binary"],
        )
    )
    preprocessed_path = dataset_dir / "preprocessed_data" / "preprocessed.csv"
    n = write_preprocessed(rows, preprocessed_path)

    if not run_dataset(
        variant_name,
        split_seed,
        preprocessed_subdir="preprocessed_data",
        preprocessed_file="preprocessed.csv",
        data_root=data_root,
    ):
        raise RuntimeError(f"Stratified split failed for {variant_name!r} under {data_root}")

    split_dir = dataset_dir / SPLITTED_DATA_SUBDIR
    manifest = {
        "variant": variant_name,
        "binary_mapping": cfg["description"],
        "source_json": str(input_json.resolve()),
        "source_dataset_folder": HATEXPLAIN_RANDOM_SPLIT_SOURCE,
        "split_mode": "stratified_811",
        "split_seed": int(split_seed),
        "tie_policy": tie_policy,
        "preprocessed_rows": n,
        "splitted_data_subdir": SPLITTED_DATA_SUBDIR,
    }
    (split_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"{variant_name}: preprocessed={n} -> {preprocessed_path}; "
        f"split(seed={split_seed}) -> {split_dir}",
        flush=True,
    )

def resolve_input_json(data_root: Path, cli_path: str) -> Path:
    if cli_path.strip():
        p = Path(cli_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    p = data_root / HATEXPLAIN_RANDOM_SPLIT_SOURCE / HATEXPLAIN_RANDOM_SPLIT_INPUT_REL
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}; pass --input-json or place dataset.json under "
            f"{data_root / HATEXPLAIN_RANDOM_SPLIT_SOURCE / 'raw_data'}"
        )
    return p.resolve()

def main_hatexplain_random_split_variants(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "HateXplain JSON -> drop_offensive binary labels + stratified 811 split."
        )
    )
    p.add_argument("--data-root", type=str, default="", help="e.g. data/seed_0_model (default: ECHOHATE_DATA_ROOT)")
    p.add_argument(
        "--input-json",
        type=str,
        default="",
        help=(
            "dataset.json (default: "
            f"<data-root>/{HATEXPLAIN_RANDOM_SPLIT_SOURCE}/{HATEXPLAIN_RANDOM_SPLIT_INPUT_REL})"
        ),
    )
    p.add_argument(
        "--variants",
        type=str,
        default="drop_offensive",
        help="Comma-separated variant folder names to build",
    )
    p.add_argument("--tie-policy", type=str, default="drop", choices=["drop", "first"])
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument("--no-link-raw", action="store_true", help="Do not symlink/copy dataset.json into variant raw_data/")
    args = p.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root.strip() else get_data_root()
    input_json = resolve_input_json(data_root, args.input_json)
    names = [s.strip() for s in args.variants.split(",") if s.strip()]
    unknown = [n for n in names if n not in HATEXPLAIN_RANDOM_SPLIT_VARIANTS]
    if unknown:
        raise SystemExit(
            f"Unknown variants: {unknown}; valid: {sorted(HATEXPLAIN_RANDOM_SPLIT_VARIANTS)}"
        )

    print(f"Data root: {data_root}", flush=True)
    print(f"Input JSON: {input_json}", flush=True)
    data = load_json_data(input_json)
    print(f"Loaded {len(data)} posts from JSON", flush=True)
    for name in names:
        build_hatexplain_random_split_variant(
            data_root,
            name,
            data,
            input_json,
            tie_policy=args.tie_policy,
            split_seed=int(args.split_seed),
            link_raw_json=not args.no_link_raw,
        )
    print("Done.", flush=True)


# --- hateval ---

DATA_ROOT = str(get_data_root())

DATASET_NAME_HATEVAL = "HatEval"
RAW_DIR = os.path.join(DATA_ROOT, DATASET_NAME_HATEVAL, "raw_data")
INTERMEDIATE_DIR = os.path.join(DATA_ROOT, DATASET_NAME_HATEVAL, "intermediate_data")
PREPROCESSED_DIR = os.path.join(DATA_ROOT, DATASET_NAME_HATEVAL, "preprocessed_data")

def normalize_language(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def pick_text(example: Dict[str, Any]) -> str:
    for k in ("tweet", "text", "sentence"):
        if k in example and example[k] is not None:
            return str(example[k])
    raise KeyError(
        f"Cannot find text column in HatEval example. Expected one of: tweet,text,sentence. "
        f"Available keys: {list(example.keys())}"
    )

def pick_language(example: Dict[str, Any]) -> str:
    for k in ("language", "lang"):
        if k in example and example[k] is not None:
            return normalize_language(example[k])
    raise KeyError(f"Cannot find language column. Expected one of: language,lang. Keys: {list(example.keys())}")

def pick_hs(example: Dict[str, Any]) -> Optional[int]:
    if "HS" not in example:
        return None
    v = example.get("HS")
    if v is None:
        return None
    # Handle bool/int/str
    if isinstance(v, bool):
        return int(v)
    try:
        return int(float(v))
    except Exception:
        return None

def write_split_csv(split_name: str, dataset_split: Iterable[Dict[str, Any]], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["HS", "language", "text"])
        writer.writeheader()
        for ex in dataset_split:
            hs = pick_hs(ex)
            # Keep rows even if HS missing; they will be dropped later.
            lang = pick_language(ex)
            text = pick_text(ex)
            writer.writerow({"HS": hs, "language": lang, "text": text})
            n += 1
    return n

def main_hateval() -> None:
    parser = argparse.ArgumentParser(description="Download HatEval and preprocess/split it.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for splitting (default: config/echohate_seeds.json random_state)",
    )
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(default_split_random_state())

    ds = load_dataset("valeriobasile/HatEval")
    if not hasattr(ds, "keys"):
        raise ValueError("Expected a DatasetDict from load_dataset('valeriobasile/HatEval').")

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # 1) Save raw splits
    for split_name in ds.keys():
        out_path = os.path.join(RAW_DIR, f"{split_name}.csv")
        n = write_split_csv(split_name, ds[split_name], out_path)
        print(f"Saved raw split {split_name}: {n} rows -> {out_path}", flush=True)

    # 2) Combine into intermediate_data/intermediate.csv
    intermediate_path = os.path.join(INTERMEDIATE_DIR, "intermediate.csv")
    kept_intermediate = 0
    with open(intermediate_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["HS", "language", "text"])
        writer.writeheader()
        for split_name in ds.keys():
            for ex in ds[split_name]:
                hs = pick_hs(ex)
                lang = pick_language(ex)
                text = pick_text(ex)
                writer.writerow({"HS": hs, "language": lang, "text": text})
                kept_intermediate += 1

    print(f"Saved intermediate combined: {kept_intermediate} rows -> {intermediate_path}", flush=True)

    # 3) Filter language=en and map HS -> label
    preprocessed_path = os.path.join(PREPROCESSED_DIR, "preprocessed.csv")
    kept = 0
    dropped = 0
    with open(intermediate_path, "r", encoding="utf-8", newline="") as f_in, open(
        preprocessed_path, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["text", "label", "class"])
        writer.writeheader()

        for row in reader:
            lang = normalize_language(row.get("language"))
            if lang != "en":
                dropped += 1
                continue

            hs_raw = row.get("HS")
            try:
                hs = int(float(hs_raw)) if hs_raw not in (None, "", "None") else None
            except Exception:
                hs = None

            if hs not in (0, 1):
                dropped += 1
                continue

            label = "hate" if hs == 1 else "not_hate"
            cls = 1 if hs == 1 else 0

            text = (row.get("text") or "").strip()
            if not text:
                dropped += 1
                continue

            writer.writerow({"text": text, "label": label, "class": cls})
            kept += 1

    print(f"Saved preprocessed: kept={kept} dropped={dropped} -> {preprocessed_path}", flush=True)

    env = os.environ.copy()
    env["ECHOHATE_DATA_ROOT"] = DATA_ROOT
    run_split_subprocess(DATASET_NAME_HATEVAL, int(args.seed), env=env)


# --- ihc ---

def _merge_stg1(stg1_ids: pd.DataFrame, stg1_posts: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(left=stg1_ids, right=stg1_posts, left_index=True, right_index=True, how="inner")
    if "class_y" in out.columns:
        out = out.drop(columns=["class_y"])
    if "class_x" in out.columns:
        out = out.rename(columns={"class_x": "class"})
    return out

def _merge_stg2(stg2_ids: pd.DataFrame, stg2_posts: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(left=stg2_ids, right=stg2_posts, left_index=True, right_index=True, how="inner")
    drop_cols = [
        c
        for c in out.columns
        if c
        in (
            "implicit_class_x",
            "extra_implicit_class_x",
            "implicit_class_y",
            "extra_implicit_class_y",
        )
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out

def _merge_stg3(stg3_ids: pd.DataFrame, stg3_posts: pd.DataFrame) -> pd.DataFrame:
    out = pd.merge(left=stg3_ids, right=stg3_posts, left_index=True, right_index=True, how="inner")
    drop_cols = [c for c in ("target_x", "target_y", "implied_statement_y") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if "implied_statement_x" in out.columns:
        out = out.rename(columns={"implied_statement_x": "implied_statement"})
    return out

def _pick_post_column(df: pd.DataFrame) -> pd.Series:
    if "post" in df.columns:
        return df["post"].astype(str)
    post_cols = [c for c in df.columns if c.startswith("post")]
    if not post_cols:
        raise ValueError(f"No post column in frame columns={list(df.columns)}")
    return df[post_cols[0]].astype(str)

def _finalize_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Export only text/label/class (drop row_id and any other legacy columns)."""
    core = ["text", "label", "class"]
    missing = [c for c in core if c not in df.columns]
    if missing:
        raise ValueError(f"Split frame missing required columns: {missing}")
    return df[core].copy()

def build_pure_set(raw_dir: Path) -> pd.DataFrame:
    stg1_ids = pd.read_csv(raw_dir / "implicit_hate_v1_stg1.tsv", sep="\t", header=0)
    stg1_posts = pd.read_csv(raw_dir / "implicit_hate_v1_stg1_posts.tsv", sep="\t", header=0)

    stg2_ids = pd.read_csv(raw_dir / "implicit_hate_v1_stg2.tsv", sep="\t", header=0)
    stg2_posts = pd.read_csv(raw_dir / "implicit_hate_v1_stg2_posts.tsv", sep="\t", header=0)

    stg3_ids = pd.read_csv(raw_dir / "implicit_hate_v1_stg3.tsv", sep="\t", header=0)
    stg3_posts = pd.read_csv(raw_dir / "implicit_hate_v1_stg3_posts.tsv", sep="\t", header=0)
    stg3_posts = stg3_posts.dropna(axis=0).reset_index(drop=True)
    stg3_ids = stg3_ids.dropna(axis=0).reset_index(drop=True)

    stg1_total = _merge_stg1(stg1_ids, stg1_posts)
    stg2_total = _merge_stg2(stg2_ids, stg2_posts)
    stg3_total = _merge_stg3(stg3_ids, stg3_posts)

    stg1_total["class"] = stg1_total["class"].astype(str).str.strip().str.lower()

    mask_implicit = stg1_total["class"] == "implicit_hate"
    stg1_implicit = stg1_total.loc[mask_implicit, :].copy()

    stg2_feats = stg2_total.drop(columns=["post"], errors="ignore")
    stg3_feats = stg3_total.drop(columns=["post"], errors="ignore")
    pure_implicit = pd.merge(stg1_implicit, stg2_feats, on="ID", how="inner")
    pure_implicit = pd.merge(pure_implicit, stg3_feats, on="ID", how="inner")

    mask_not_hate = stg1_total["class"] == "not_hate"
    stg1_not_hate = stg1_total.loc[mask_not_hate, :].copy()

    ids_stage23 = set(stg2_total["ID"].astype(str)) | set(stg3_total["ID"].astype(str))
    pure_not_mask = ~stg1_not_hate["ID"].astype(str).isin(ids_stage23)
    pure_not_hate = stg1_not_hate.loc[pure_not_mask, :].copy()

    pure_set = pd.concat([pure_implicit, pure_not_hate], join="outer", ignore_index=True)
    pure_set["class"] = pure_set["class"].astype(str).str.strip().str.lower()
    keep = {"implicit_hate", "not_hate"}
    before = len(pure_set)
    pure_set = pure_set[pure_set["class"].isin(keep)].copy()
    dropped = before - len(pure_set)
    if dropped:
        print(f"Dropped {dropped} rows with class outside {{implicit_hate, not_hate}}", flush=True)

    if "post" not in pure_set.columns:
        pure_set["post"] = _pick_post_column(pure_set)

    pure_set["text"] = pure_set["post"].astype(str)
    pure_set["label"] = pure_set["class"].map(
        {"implicit_hate": "implicit_hate", "not_hate": "not_hate"}
    )
    pure_set["class"] = (pure_set["label"] == "implicit_hate").astype(int)

    return _finalize_split_columns(pure_set)

def main_ihc() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=ECHOHATE_ROOT / "data" / "IHC" / "raw_data",
        help="Directory with implicit_hate_v1_stg{1,2,3}*.tsv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/IHC/<splitted_data_XX_YY_ZZ>/ from split fracs)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stratified split (default: config create_split_random_state)",
    )
    args = ap.parse_args()
    if args.seed is None:
        args.seed = int(default_split_random_state())

    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing raw_dir: {raw_dir}")

    train_frac, validation_frac, test_frac, split_subdir = _split_fracs()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else (ECHOHATE_ROOT / "data" / "IHC" / split_subdir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_pure_set(raw_dir)
    if len(df) == 0:
        raise SystemExit("Empty pure set after filtering.")

    labels = df["class"].astype(int)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        random_state=args.seed,
        stratify=labels,
    )
    validation_frac_of_temp = validation_frac / (validation_frac + test_frac)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - validation_frac_of_temp),
        random_state=args.seed,
        stratify=temp_df["class"].astype(int),
    )

    train_path = out_dir / "train.csv"
    validation_path = out_dir / "validation.csv"
    test_path = out_dir / "test.csv"

    for path, part in ((train_path, train_df), (validation_path, validation_df), (test_path, test_df)):
        _finalize_split_columns(part).to_csv(path, index=False)

    meta = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "split_subdir": split_subdir,
        "seed": args.seed,
        "train_frac": train_frac,
        "validation_frac": validation_frac,
        "test_frac": test_frac,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "label_counts_total": df.groupby("label").size().astype(int).to_dict(),
        "class_counts_total": df.groupby("class").size().astype(int).to_dict(),
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"Wrote {train_path}  {validation_path}  {test_path}")


# --- ethos ---

DEFAULT_DATA_ROOT = get_data_root()


DEFAULT_RAW_FILE = "Ethos_Dataset_Binary.csv"
DEFAULT_HATE_THRESHOLD = 0.5

def preprocess_ethos(
    dataset_name: str,
    data_root: Path,
    raw_file: str,
    hate_threshold: float,
) -> pd.DataFrame:
    raw_path = data_root / dataset_name / "raw_data" / raw_file
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {raw_path}")

    df = pd.read_csv(raw_path, sep=";", engine="python")
    for col in ("comment", "isHate"):
        if col not in df.columns:
            raise ValueError(f"{raw_path}: expected columns comment, isHate; got {list(df.columns)}")

    df = df.rename(columns={"comment": "text"})
    df["text"] = df["text"].astype(str).map(sanitize_text_for_csv)
    df["isHate"] = pd.to_numeric(df["isHate"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["isHate"]).copy()
    dropped_nan = before - len(df)

    df["label"] = df["isHate"].apply(
        lambda v: "hate" if float(v) >= hate_threshold else "not_hate"
    )
    df["class"] = (df["label"] == "hate").astype(int)
    df = drop_blank_and_duplicate_text(df, text_col="text")
    df = df[["text", "label", "class"]].reset_index(drop=True)

    out_dir = data_root / dataset_name / "preprocessed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.csv"
    df.to_csv(out_path, index=False)

    vc = df["label"].value_counts()
    print(f"Raw: {raw_path}", flush=True)
    print(f"  dropped non-numeric isHate: {dropped_nan}", flush=True)
    print(f"  hate threshold: >={hate_threshold}", flush=True)
    print(f"  labels: {vc.to_dict()}", flush=True)
    print(f"  wrote {len(df)} rows -> {out_path}", flush=True)
    return df

def main_ethos(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess Ethos binary CSV for EchoHate.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Ethos",
        help="Dataset folder under data root (default: Ethos).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Parent of dataset dirs (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default=DEFAULT_RAW_FILE,
        help="CSV filename under <dataset>/raw_data/.",
    )
    parser.add_argument(
        "--hate-threshold",
        type=float,
        default=DEFAULT_HATE_THRESHOLD,
        help="Rows with isHate >= threshold are hate (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed (default: echohate_project_config create_split_random_state).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Only write preprocessed.csv; do not run stratified split.",
    )
    parser.add_argument(
        "--balanced-validation-test",
        action="store_true",
        help="Pass through to ``python -m preprocess split`` (optional for small n).",
    )
    args = parser.parse_args(argv)
    seed = int(args.seed) if args.seed is not None else int(default_split_random_state())
    data_root = resolve_data_root(args.data_root)
    dataset_name = args.dataset.strip()

    preprocess_ethos(
        dataset_name=dataset_name,
        data_root=data_root,
        raw_file=args.raw_file,
        hate_threshold=float(args.hate_threshold),
    )

    if args.no_split:
        return

    env = os.environ.copy()
    env["ECHOHATE_DATA_ROOT"] = str(data_root)
    split_argv = ["--datasets", dataset_name, "--seed", str(seed)]
    if args.balanced_validation_test:
        split_argv.append("--balanced-validation-test")
    run_split_subprocess(dataset_name, seed, env=env)
    print(f"Split done under {data_root / dataset_name}", flush=True)


# --- toxigen ---

# --- toxigen (ToxiGen annotated HF subset) ---

DATASET_NAME_TOXIGEN = "ToxiGen"
DEFAULT_TARGET_TOTAL = 6776
_TOXIGEN_HF = ("toxigen/toxigen-data", "annotated")

def _load_toxigen_annotated() -> pd.DataFrame:
    if load_dataset is None:
        raise ImportError("Install `datasets` to load ToxiGen from Hugging Face.")
    ds = load_dataset(*_TOXIGEN_HF)
    rows: list[dict[str, Any]] = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            rows.append(dict(ex))
    return pd.DataFrame(rows)

def _binarize_toxigen(df: pd.DataFrame) -> pd.DataFrame:
    """
    ToxiGen paper mapping (HF columns ``toxicity_ai`` / ``toxicity_human``):

    score = max(HARMFULIFAI, HARMFULIFHUMAN)
    - score < 3 → non-toxic  → ``not_hate`` / class 0
    - score = 3 → ambiguous  → dropped
    - score > 3 → toxic      → ``hate`` / class 1
    """
    out = df.copy()
    out["text"] = out["text"].astype(str).map(sanitize_text_for_csv)
    out = out.dropna(subset=["toxicity_human", "toxicity_ai"])
    scores = out[["toxicity_human", "toxicity_ai"]].astype(float).max(axis=1)
    out = out.loc[scores != 3.0].copy()
    out["class"] = (scores.loc[out.index] > 3.0).astype(int)
    out["label"] = out["class"].map({0: "not_hate", 1: "hate"})
    out = drop_blank_and_duplicate_text(out[["text", "label", "class"]], text_col="text")
    return out[["text", "label", "class"]].reset_index(drop=True)

def _stratified_subsample(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")
    if len(df) < target_n:
        raise ValueError(
            f"ToxiGen pool has {len(df)} rows after binarization; "
            f"cannot subsample to {target_n}."
        )
    if len(df) == target_n:
        return df.reset_index(drop=True)
    _, sample = train_test_split(
        df,
        test_size=target_n,
        stratify=df["class"],
        random_state=int(seed),
    )
    return sample.reset_index(drop=True)

def preprocess_toxigen(
    dataset_name: str,
    data_root: Path,
    *,
    target_total: int = DEFAULT_TARGET_TOTAL,
    sample_seed: int | None = None,
) -> tuple[pd.DataFrame, int]:
    raw = _load_toxigen_annotated()
    pool = _binarize_toxigen(raw)
    seed = int(sample_seed if sample_seed is not None else default_split_random_state())
    df = _stratified_subsample(pool, int(target_total), seed)

    out_dir = data_root / dataset_name / "preprocessed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.csv"
    df.to_csv(out_path, index=False)

    vc = df["label"].value_counts()
    print(f"HF: {_TOXIGEN_HF[0]} config={_TOXIGEN_HF[1]!r}", flush=True)
    print(f"  pool after binarization: {len(pool)} rows", flush=True)
    print(f"  stratified subsample (seed={seed}) -> {len(df)} rows", flush=True)
    print(f"  labels: {vc.to_dict()}", flush=True)
    print(f"  wrote -> {out_path}", flush=True)
    return df, len(pool)

def main_toxigen(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load ToxiGen annotated subset from Hugging Face, binarize toxicity scores, "
            "stratified subsample to a target size, and stratified 80/10/10 split."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="toxigen_new",
        help="Dataset folder under data root (default: toxigen_new).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Parent of dataset dirs (default: {get_data_root()}).",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=DEFAULT_TARGET_TOTAL,
        help=f"Stratified subsample size before 80/10/10 split (default: {DEFAULT_TARGET_TOTAL}).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for stratified subsample (default: project split seed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed (default: project create_split_random_state).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Only write preprocessed.csv; do not run stratified split.",
    )
    parser.add_argument(
        "--balanced-validation-test",
        action="store_true",
        help="Pass through to stratified split (optional for small n).",
    )
    args = parser.parse_args(argv)
    seed = int(args.seed) if args.seed is not None else int(default_split_random_state())
    sample_seed = int(args.sample_seed) if args.sample_seed is not None else seed
    data_root = resolve_data_root(args.data_root)
    dataset_name = args.dataset.strip()

    df, pool_rows = preprocess_toxigen(
        dataset_name=dataset_name,
        data_root=data_root,
        target_total=int(args.target_total),
        sample_seed=sample_seed,
    )

    if args.no_split:
        return

    env = os.environ.copy()
    env["ECHOHATE_DATA_ROOT"] = str(data_root)
    run_split_subprocess(
        dataset_name,
        seed,
        balanced_validation_test=bool(args.balanced_validation_test),
        env=env,
    )
    split_dir = data_root / dataset_name / "splitted_data_80_10_10"
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "source_hf": {"repo": _TOXIGEN_HF[0], "config": _TOXIGEN_HF[1]},
                "binary_mapping": (
                    "score=max(HARMFULIFAI,HARMFULIFHUMAN) via max(toxicity_ai,toxicity_human); "
                    "<3 non-toxic->not_hate, =3 ambiguous (dropped), >3 toxic->hate"
                ),
                "pool_rows_after_binarization": int(pool_rows),
                "target_total_after_subsample": int(args.target_total),
                "sample_seed": int(sample_seed),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split done under {split_dir}", flush=True)


# --- sbic ---

DEFAULT_DATA_ROOT = get_data_root()

DATASET_NAME_SBIC = "SBIC"

_RAW_SPLIT_FILES = {
    "trn": "SBIC.v2.trn.csv",
    "dev": "SBIC.v2.dev.csv",
    "tst": "SBIC.v2.tst.csv",
}
_OFFICIAL_OUT = (
    ("trn", "train.csv"),
    ("dev", "validation.csv"),
    ("tst", "test.csv"),
)

def aggregate_sbic_annotations(
    split: str,
    raw_path: Path,
    *,
    aggregate_seed: int,
) -> pd.DataFrame:
    """
    Aggregate worker-level SBIC rows to one row per unique ``post``.

    Ported from ``experiment/LAHN/prepare_sbic.py`` (AllenAI FEB aggregation rules).
    Uses ``groupby("post")`` for linear-time aggregation (LAHN's row scan is O(n^2)).
    """
    df = pd.read_csv(raw_path)
    required = {"post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{raw_path}: missing columns {missing}")

    columns = ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
    aggregated: list[list] = []
    rng = random.Random(int(aggregate_seed))
    is_eval = split in ("dev", "tst")

    for post, rows in df.groupby("post", sort=False):
        n = len(rows)
        offensive_frac = float(rows["offensiveYN"].sum()) / float(n)
        offensive_label = 1.0 if offensive_frac >= 0.5 else 0.0

        who_target_frac = float(rows["whoTarget"].sum()) / float(n)
        who_target_label = 1.0 if who_target_frac >= 0.5 else 0.0

        if who_target_label != 1.0:
            aggregated.append([post, offensive_label, who_target_label, None, None])
            continue

        minorities = rows["targetMinority"]
        stereotypes = rows["targetStereotype"]
        if is_eval:
            minority_labels: list[str] = []
            stereotype_labels: list[str] = []
            for m, s in zip(minorities, stereotypes):
                if not pd.isna(s):
                    minority_labels.append(str(m))
                    stereotype_labels.append(str(s))
            aggregated.append(
                [
                    post,
                    offensive_label,
                    who_target_label,
                    " [SEP] ".join(minority_labels),
                    " [SEP] ".join(stereotype_labels),
                ]
            )
            continue

        temp: list[list] = []
        for m, s in zip(minorities, stereotypes):
            if not pd.isna(s):
                temp.append([post, offensive_label, who_target_label, m, s])
        if temp:
            aggregated.append(rng.choice(temp))
        else:
            aggregated.append([post, offensive_label, who_target_label, None, None])

    return pd.DataFrame(aggregated, columns=columns)

def _label_maps(label_mode: str) -> tuple[dict[float, str], dict[str, int]]:
    mode = (label_mode or "offensive").strip().lower()
    if mode == "hate":
        return (
            {0.0: "not_hate", 1.0: "hate"},
            {"not_hate": 0, "hate": 1},
        )
    if mode == "offensive":
        return (
            {0.0: "not_offensive", 1.0: "offensive"},
            {"not_offensive": 0, "offensive": 1},
        )
    raise ValueError(f"Unsupported --label-mode {label_mode!r}; use offensive or hate.")

def aggregated_to_echohate(
    agg: pd.DataFrame,
    *,
    label_mode: str,
    dedupe: bool = True,
) -> pd.DataFrame:
    yn_to_label, _ = _label_maps(label_mode)
    out = pd.DataFrame()
    out["text"] = agg["post"].astype(str).map(sanitize_text_for_csv)
    out["label"] = agg["offensiveYN"].astype(float).map(yn_to_label)
    out["class"] = (agg["offensiveYN"].astype(float) >= 0.5).astype(int)
    if dedupe:
        out = drop_blank_and_duplicate_text(out, text_col="text")
    else:
        out = out.reset_index(drop=True)
    return out[["text", "label", "class"]]

def _resolve_raw_csv(raw_dir: Path, split: str, override: str | None) -> Path:
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = raw_dir / p
        p = p.resolve()
    else:
        p = raw_dir / _RAW_SPLIT_FILES[split]
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing SBIC raw CSV for split {split!r}: {p}\n"
            f"Place {_RAW_SPLIT_FILES[split]} under {raw_dir} or pass --raw-{split}."
        )
    return p

def export_official_splits(
    data_root: Path,
    dataset_name: str,
    raw_dir: Path,
    *,
    label_mode: str,
    aggregate_seed: int,
    dedupe: bool = True,
    split_subdir: str = SPLITTED_DATA_SUBDIR,
    raw_overrides: dict[str, str | None] | None = None,
) -> dict[str, int]:
    """Write official trn/dev/tst splits to EchoHate layout."""
    overrides = raw_overrides or {}
    out_dir = data_root / dataset_name / split_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    parts: list[pd.DataFrame] = []

    for split_key, out_name in _OFFICIAL_OUT:
        raw_path = _resolve_raw_csv(raw_dir, split_key, overrides.get(split_key))
        agg = aggregate_sbic_annotations(split_key, raw_path, aggregate_seed=aggregate_seed)
        part = aggregated_to_echohate(agg, label_mode=label_mode, dedupe=dedupe)
        part.to_csv(out_dir / out_name, index=False)
        counts[out_name] = len(part)
        parts.append(part)
        print(f"  {raw_path.name} -> {out_name}: {len(part)} rows", flush=True)

    pre_dir = data_root / dataset_name / "preprocessed_data"
    pre_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(pre_dir / "preprocessed.csv", index=False)

    manifest = {
        "official_splits": True,
        "stratified": False,
        "split_mode": "official",
        "label_mode": label_mode,
        "aggregate_seed": int(aggregate_seed),
        "dedupe_text": bool(dedupe),
        "raw_dir": str(raw_dir.resolve()),
        "subdir": split_subdir,
        "mapping": {"trn": "train.csv", "dev": "validation.csv", "tst": "test.csv"},
        "counts": counts,
        "preprocessed_rows": len(combined),
        "note": "SBIC v2 official trn/dev/tst; dev→validation.csv, tst→test.csv.",
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    vc = combined["label"].value_counts()
    print(f"Raw dir: {raw_dir}", flush=True)
    print(f"  label_mode: {label_mode}", flush=True)
    print(f"  dedupe_text: {dedupe}", flush=True)
    print(f"  combined labels: {vc.to_dict()}", flush=True)
    print(f"  splits -> {out_dir}", flush=True)
    print(f"  preprocessed union ({len(combined)} rows) -> {pre_dir / 'preprocessed.csv'}", flush=True)
    return counts

def build_combined_preprocessed(
    data_root: Path,
    dataset_name: str,
    raw_dir: Path,
    *,
    label_mode: str,
    aggregate_seed: int,
    dedupe: bool = True,
    raw_overrides: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    overrides = raw_overrides or {}
    parts: list[pd.DataFrame] = []
    for split_key in _RAW_SPLIT_FILES:
        raw_path = _resolve_raw_csv(raw_dir, split_key, overrides.get(split_key))
        agg = aggregate_sbic_annotations(split_key, raw_path, aggregate_seed=aggregate_seed)
        parts.append(aggregated_to_echohate(agg, label_mode=label_mode, dedupe=dedupe))
    combined = pd.concat(parts, ignore_index=True)
    if dedupe:
        combined = drop_blank_and_duplicate_text(combined, text_col="text")

    out_dir = data_root / dataset_name / "preprocessed_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.csv"
    combined.to_csv(out_path, index=False)

    vc = combined["label"].value_counts()
    print(f"Raw dir: {raw_dir}", flush=True)
    print(f"  label_mode: {label_mode}", flush=True)
    print(f"  dedupe_text: {dedupe}", flush=True)
    print(f"  combined labels: {vc.to_dict()}", flush=True)
    print(f"  wrote {len(combined)} rows -> {out_path}", flush=True)
    return combined

def main_sbic(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess SBIC v2 for EchoHate.")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME_SBIC, help="Dataset folder under data root.")
    parser.add_argument("--data-root", type=str, default=None, help=f"Parent of dataset dirs (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory with SBIC.v2.*.csv (default: <data-root>/<dataset>/raw_data).",
    )
    parser.add_argument("--raw-trn", type=str, default=None, help="Override train raw CSV filename or path.")
    parser.add_argument("--raw-dev", type=str, default=None, help="Override dev raw CSV filename or path.")
    parser.add_argument("--raw-tst", type=str, default=None, help="Override test raw CSV filename or path.")
    parser.add_argument(
        "--label-mode",
        choices=("offensive", "hate"),
        default="offensive",
        help="Label vocabulary for EchoHate CSVs (default: offensive/not_offensive).",
    )
    parser.add_argument(
        "--aggregate-seed",
        type=int,
        default=None,
        help="RNG seed for train stereotype row selection (default: project split seed).",
    )
    parser.add_argument(
        "--stratified-split",
        action="store_true",
        help="Pool all splits and run stratified 80/10/10 (default: keep official trn/dev/tst).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Stratified split seed (default: project seed).")
    parser.add_argument(
        "--split-subdir",
        type=str,
        default=SPLITTED_DATA_SUBDIR,
        help=f"Output split folder name (default: {SPLITTED_DATA_SUBDIR}).",
    )
    parser.add_argument(
        "--balanced-validation-test",
        action="store_true",
        help="Balanced 50/50 validation/test (stratified mode only).",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep all aggregated rows (LAHN parity; default drops blank/duplicate text).",
    )
    args = parser.parse_args(argv)
    dedupe = not args.no_dedupe

    seed = int(args.seed) if args.seed is not None else int(default_split_random_state())
    aggregate_seed = (
        int(args.aggregate_seed) if args.aggregate_seed is not None else seed
    )
    data_root = resolve_data_root(args.data_root)
    dataset_name = args.dataset.strip()
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else (data_root / dataset_name / "raw_data")
    overrides = {"trn": args.raw_trn, "dev": args.raw_dev, "tst": args.raw_tst}

    if args.stratified_split:
        build_combined_preprocessed(
            data_root=data_root,
            dataset_name=dataset_name,
            raw_dir=raw_dir,
            label_mode=args.label_mode,
            aggregate_seed=aggregate_seed,
            dedupe=dedupe,
            raw_overrides=overrides,
        )
        env = os.environ.copy()
        env["ECHOHATE_DATA_ROOT"] = str(data_root)
        if args.balanced_validation_test:
            import subprocess

            cmd = [
                sys.executable,
                "-m",
                "pipeline.preprocessing",
                "split",
                "--datasets",
                dataset_name,
                "--seed",
                str(seed),
                "--balanced-validation-test",
            ]
            subprocess.run(cmd, check=True, cwd=str(ECHOHATE_ROOT), env=env)
        else:
            run_split_subprocess(
                dataset_name,
                seed,
                preprocessed_subdir="preprocessed_data",
                preprocessed_file="preprocessed.csv",
                env=env,
            )
        print(f"Stratified split done under {data_root / dataset_name / args.split_subdir}", flush=True)
        return

    export_official_splits(
        data_root=data_root,
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        label_mode=args.label_mode,
        aggregate_seed=aggregate_seed,
        dedupe=dedupe,
        split_subdir=args.split_subdir,
        raw_overrides=overrides,
    )
    print("Done.", flush=True)


_COMMANDS: dict[str, Callable[..., None]] = {
    "dynahate": main_dynahate,
    "hatexplain-random-split-variants": main_hatexplain_random_split_variants,
    "hateval": main_hateval,
    "ihc": main_ihc,
    "ethos": main_ethos,
    "toxigen": main_toxigen,
    "sbic": main_sbic,
}


def run_command(command: str, argv: list[str] | None = None) -> None:
    import inspect

    fn = _COMMANDS.get(command)
    if fn is None:
        raise SystemExit(f"Unknown dataset command {command!r}")
    rest = list(argv or [])
    old_argv = sys.argv
    sys.argv = [command, *rest]
    try:
        params = inspect.signature(fn).parameters
        if params:
            fn(rest)
        else:
            fn()
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        p = argparse.ArgumentParser(
            description="Dataset preprocessing (7 datasets, 811 splits)."
        )
        p.add_argument("command", nargs="?", choices=sorted(_COMMANDS))
        p.print_help()
        print("\nCommands:", ", ".join(sorted(_COMMANDS)))
        return
    run_command(argv[0], argv[1:])


if __name__ == "__main__":
    main()
