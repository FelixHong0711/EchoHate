#!/usr/bin/env python3
"""Stratified train / validation / test splits (see pipeline.common.paths)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.common.paths import (
    ECHOHATE_ROOT,
    SPLITTED_DATA_SUBDIR,
    TEST_FRAC,
    TRAIN_FRAC,
    VALIDATION_CSV,
    VALIDATION_FRAC,
    get_data_root,
)
from pipeline.common.project_config import default_split_random_state
from pipeline.preprocessing._core import label_to_class

if str(ECHOHATE_ROOT) not in sys.path:
    sys.path.insert(0, str(ECHOHATE_ROOT))

SEED = int(default_split_random_state())
STRATIFY_BY_LABEL_DATASETS: set[str] = set()


def split_stratified(df: pd.DataFrame, stratify_col: str = "class", seed: int = SEED):
    train, rest = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df[stratify_col], random_state=seed
    )
    validation, test = train_test_split(
        rest, train_size=0.5, stratify=rest[stratify_col], random_state=seed
    )
    return train, validation, test


def split_stratified_balanced_validation_test(
    df: pd.DataFrame,
    stratify_col: str = "class",
    seed: int = SEED,
):
    train, rest = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df[stratify_col], random_state=seed
    )

    if stratify_col != "class":
        validation, test = train_test_split(
            rest, train_size=0.5, stratify=rest[stratify_col], random_state=seed
        )
        return train, validation, test, 0

    if "class" not in rest.columns:
        raise ValueError("balanced split requires 'class' column in rest dataframe")
    rest = rest.copy()
    rest["class"] = pd.to_numeric(rest["class"], errors="coerce")
    rest = rest.dropna(subset=["class"]).copy()
    rest["class"] = rest["class"].astype(int)
    rest = rest[rest["class"].isin([0, 1])].copy()

    c0 = rest[rest["class"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    c1 = rest[rest["class"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    k = min(len(c0), len(c1)) // 2
    if k <= 0:
        raise ValueError(
            f"Not enough holdout samples to make balanced validation/test. "
            f"class0={len(c0)} class1={len(c1)}"
        )

    validation = pd.concat([c0.iloc[:k], c1.iloc[:k]], ignore_index=True)
    test = pd.concat([c0.iloc[k : 2 * k], c1.iloc[k : 2 * k]], ignore_index=True)

    validation = validation.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    dropped = int(len(rest) - (len(validation) + len(test)))
    return train, validation, test, dropped


def run_dataset(
    dataset_name: str,
    seed: int,
    preprocessed_subdir: str = "preprocessed_data",
    preprocessed_file: str = "preprocessed.csv",
    balanced_validation_test: bool = False,
    data_root: os.PathLike | str | None = None,
) -> bool:
    root = Path(data_root) if data_root is not None else get_data_root()
    path = root / dataset_name / preprocessed_subdir / preprocessed_file
    path = str(path)
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path)
    for col in ["text", "label", "class"]:
        if col not in df.columns:
            print(f"  {dataset_name}: skip (missing column '{col}')")
            return False
    dropped_for_balance = 0
    stratify_col = "label" if dataset_name in STRATIFY_BY_LABEL_DATASETS else "class"
    if balanced_validation_test:
        train, validation, test, dropped_for_balance = split_stratified_balanced_validation_test(
            df, stratify_col="class", seed=seed
        )
    else:
        train, validation, test = split_stratified(df, stratify_col=stratify_col, seed=seed)
    if dataset_name in STRATIFY_BY_LABEL_DATASETS:
        for part in (train, validation, test):
            part["class"] = label_to_class(part["label"])
    out_dir = os.path.join(str(root), dataset_name, SPLITTED_DATA_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    validation.to_csv(os.path.join(out_dir, VALIDATION_CSV), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    manifest = {
        "train_frac": TRAIN_FRAC,
        "validation_frac": VALIDATION_FRAC,
        "test_frac": TEST_FRAC,
        "stratified": True,
        "stratify": (
            "sklearn.train_test_split(..., stratify=). "
            "Stratify on class (label-stratified datasets: none in default preprocess set)."
        ),
        "subdir": SPLITTED_DATA_SUBDIR,
        "seed": int(seed),
        "balanced_validation_test": bool(balanced_validation_test),
        "dropped_for_balance": int(dropped_for_balance),
        "input_preprocessed_file": preprocessed_file,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(
        f"  {dataset_name}: {len(df)} rows -> train={len(train)}, "
        f"validation={len(validation)}, test={len(test)}, "
        f"dropped_for_balance={dropped_for_balance} -> {out_dir}"
    )
    return True


def run_split_subprocess(
    dataset_name: str,
    seed: int,
    *,
    preprocessed_subdir: str = "preprocessed_data",
    preprocessed_file: str = "preprocessed.csv",
    balanced_validation_test: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Run stratified split for one dataset (used by dataset preprocess CLIs)."""
    old_env = os.environ.copy()
    if env:
        os.environ.update(env)
    try:
        if not run_dataset(
            dataset_name,
            seed,
            preprocessed_subdir=preprocessed_subdir,
            preprocessed_file=preprocessed_file,
            balanced_validation_test=balanced_validation_test,
        ):
            raise RuntimeError(
                f"Split failed for {dataset_name!r} under {get_data_root()} "
                f"(missing {preprocessed_subdir}/{preprocessed_file}?)"
            )
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split preprocessed data into train/validation/test (stratified)"
    )
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preprocessed_subdir", type=str, default="preprocessed_data")
    parser.add_argument("--preprocessed_file", type=str, default="preprocessed.csv")
    parser.add_argument("--balanced_validation_test", action="store_true")
    parser.add_argument(
        "--balanced-validation-test",
        action="store_true",
        dest="balanced_validation_test",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.datasets:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        data_root = get_data_root()
        names = [
            d
            for d in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, d))
            and os.path.isfile(
                os.path.join(data_root, d, args.preprocessed_subdir, "preprocessed.csv")
            )
        ]

    print(f"Data root: {get_data_root()}", flush=True)
    print(
        f"Splits: train {TRAIN_FRAC:.0%}, validation {VALIDATION_FRAC:.0%}, test {TEST_FRAC:.0%} "
        f"balanced_validation_test={bool(args.balanced_validation_test)}",
        flush=True,
    )
    for name in names:
        if not run_dataset(
            name,
            args.seed,
            args.preprocessed_subdir,
            args.preprocessed_file,
            args.balanced_validation_test,
        ):
            print(f"  {name}: skipped")
    print("Done.")


if __name__ == "__main__":
    main()
