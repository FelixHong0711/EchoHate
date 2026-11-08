#!/usr/bin/env python3
"""
Select training rows for mechanism-supervised encoder training.

Writes::

    <run_dir>/<selection_subdir>/
        selected_hard_rows.csv
        selected_hard_rows_for_inference.csv
        hard_train_for_toy_prompt.csv
        selection_summary.json

Methods
-------
balanced_random (alias ``rand50``)
    Stratified random ``sample_frac`` per class. Needs ``train.csv`` only.

variability_top25 (alias ``var25`` / ``var50``)
    Top ``top_frac`` per class by ``p_gold_std`` from ``training_dynamics.csv``.

``selection_gmm/`` mirror
-------------------------
By default **does not** copy into ``selection_gmm/`` (keeps rand50 vs var25 side by side).
Pass ``--mirror-selection-gmm`` to also copy the selection under ``selection_gmm/``.

Examples::

    # 50% balanced random on eacl_seed0_seq (like rand50 bootstrap)
    python -m pipeline.stage1.select_hard_cases_for_mech \\
      --method rand50 --seed-dir eacl_seed0_seq \\
      --mirror-selection-gmm

    # Top-25% variability (needs baseline training_dynamics.csv)
    python -m pipeline.stage1.select_hard_cases_for_mech \\
      --method var25 --seed-dir eacl_seed0_seq \\
      --baselines roberta_large_baseline_wCL \\
      --mirror-selection-gmm

Then train with ``python -m pipeline stage2 mech_distill`` and a mechanism-label CSV.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from pipeline.stage1.selection_balanced_random import (
    DEFAULT_BASELINES as BR_BASELINES,
    DEFAULT_DATASETS as BR_DATASETS,
    DEFAULT_SAMPLE_FRAC,
    load_train_frame,
    process_run_dir as process_balanced_random_run,
    process_seed_baselines as process_balanced_random_seed,
    resolve_training_random_state,
    subdir_for_sample_frac,
)
from pipeline.stage1.selection_variability_top25 import (
    DEFAULT_TOP_FRAC,
    process_run_dir as process_variability_run,
    subdir_for_top_frac,
)

MIRROR_SUBDIR = "selection_gmm"

METHOD_ALIASES: Dict[str, str] = {
    "rand50": "balanced_random",
    "rand25": "balanced_random",
    "random25": "balanced_random",
    "random50": "balanced_random",
    "balanced_random": "balanced_random",
    "balanced_random25": "balanced_random",
    "balanced_random50": "balanced_random",
    "var25": "variability_top25",
    "var50": "variability_top25",
    "variability": "variability_top25",
    "variability_top25": "variability_top25",
    "variability_top50": "variability_top25",
}

# Default fraction when method alias implies a size (overridable via --sample-frac / --top-frac).
METHOD_DEFAULT_FRAC: Dict[str, float] = {
    "rand25": 0.25,
    "random25": 0.25,
    "balanced_random25": 0.25,
    "rand50": 0.5,
    "random50": 0.5,
    "balanced_random50": 0.5,
    "var25": 0.25,
    "variability_top25": 0.25,
    "var50": 0.5,
    "variability_top50": 0.5,
}


def _normalize_method(name: str) -> str:
    key = (name or "").strip().lower()
    if key not in METHOD_ALIASES:
        choices = ", ".join(sorted(set(METHOD_ALIASES.keys())))
        raise SystemExit(f"Unknown --method {name!r}. Choose from: {choices}")
    return METHOD_ALIASES[key]


def _default_train_csv(data_root: Path, seed_dir: str, dataset: str, split_subdir: str) -> Path:
    return data_root / seed_dir / dataset / split_subdir / "train.csv"


def _hard_train_for_toy_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", "text", "class", "gold_label"]
    for c in cols:
        if c not in frame.columns:
            raise ValueError(f"hard_train_for_toy_prompt missing column {c}")
    hard = frame[cols].copy()
    if "label" in frame.columns:
        hard["label"] = frame["label"]
    else:
        hard["label"] = hard["gold_label"].map({0: "not_hate", 1: "hate"}).astype(str)
    return hard


def _join_selected_with_train(selected_ids: pd.DataFrame, train_csv: Path) -> pd.DataFrame:
    train_df = load_train_frame(train_csv)
    ids = set(selected_ids["row_id"].astype(str).tolist())
    out = train_df[train_df["row_id"].astype(str).isin(ids)].copy()
    if len(out) != len(ids):
        missing = len(ids) - len(out)
        raise ValueError(
            f"{train_csv}: {missing} selected row_id(s) not found in train split "
            f"(selected={len(ids)} matched={len(out)})"
        )
    return out


def _mirror_selection(
    run_dir: Path,
    source_subdir: str,
    *,
    method: str,
    summary: Dict[str, Any],
    train_csv: Optional[Path] = None,
) -> Dict[str, Path]:
    """Copy canonical hard CSVs into selection_gmm/ for toy_prompt + adapt loaders."""
    src = run_dir / source_subdir
    mirror = run_dir / MIRROR_SUBDIR
    mirror.mkdir(parents=True, exist_ok=True)

    hard_src = src / "selected_hard_rows.csv"
    infer_src = src / "selected_hard_rows_for_inference.csv"
    toy_src = src / "hard_train_for_toy_prompt.csv"

    if not hard_src.is_file():
        raise FileNotFoundError(f"missing {hard_src}")
    if not infer_src.is_file():
        raise FileNotFoundError(f"missing {infer_src}")

    hard = pd.read_csv(hard_src)
    infer = pd.read_csv(infer_src)
    hard.to_csv(mirror / "selected_hard_rows.csv", index=False)
    infer.to_csv(mirror / "selected_hard_rows_for_inference.csv", index=False)

    written: Dict[str, Path] = {
        "mirror_hard": mirror / "selected_hard_rows.csv",
        "mirror_infer": mirror / "selected_hard_rows_for_inference.csv",
    }

    if toy_src.is_file():
        toy = pd.read_csv(toy_src)
    elif train_csv is not None and train_csv.is_file():
        toy = _hard_train_for_toy_prompt(_join_selected_with_train(hard[["row_id"]], train_csv))
    else:
        toy = None

    if toy is not None:
        toy.to_csv(mirror / "hard_train_for_toy_prompt.csv", index=False)
        written["mirror_toy_hard"] = mirror / "hard_train_for_toy_prompt.csv"

    (mirror / "selection_method.json").write_text(
        json.dumps(
            {
                "method": method,
                "source_subdir": source_subdir,
                "mirrored_at": datetime.now(timezone.utc).isoformat(),
                "n_selected": int(len(hard)),
                "summary": {k: summary.get(k) for k in ("sample_frac", "top_frac", "random_state_base")},
                "note": (
                    "Active mirror for toy_prompt infer and adapt_toy_prompt_hard_for_mech "
                    "(selection_gmm/)."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["mirror_meta"] = mirror / "selection_method.json"
    return written


def _ensure_hard_train_toy_csv(run_dir: Path, subdir: str, train_csv: Path) -> Path:
    """Write hard_train_for_toy_prompt.csv under subdir if missing (var25 / gmm)."""
    out_dir = run_dir / subdir
    toy_path = out_dir / "hard_train_for_toy_prompt.csv"
    if toy_path.is_file() and toy_path.stat().st_size > 0:
        return toy_path
    hard_path = out_dir / "selected_hard_rows.csv"
    if not hard_path.is_file():
        raise FileNotFoundError(f"missing {hard_path}")
    hard = pd.read_csv(hard_path)
    if "text" in hard.columns and "class" in hard.columns:
        frame = hard
    else:
        frame = _join_selected_with_train(hard[["row_id"]], train_csv)
    toy = _hard_train_for_toy_prompt(frame)
    toy.to_csv(toy_path, index=False)
    return toy_path



def update_seed_manifest(
    data_root: Path,
    seed_dir: str,
    *,
    method: str,
    selection_subdir: str,
    mirrored_to: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest_path = data_root / seed_dir / "seed_manifest.json"
    doc: Dict[str, Any] = {}
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    block: Dict[str, Any] = {
        "method": method,
        "subdir": selection_subdir,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if mirrored_to:
        block["mirrored_to"] = mirrored_to
    if extra:
        block.update(extra)
    doc["hard_case_selection"] = block
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _selection_subdir_for_method(method: str, *, sample_frac: float, top_frac: float) -> str:
    if method == "balanced_random":
        return subdir_for_sample_frac(sample_frac)
    if method == "variability_top25":
        return subdir_for_top_frac(top_frac)
    return MIRROR_SUBDIR


def run_selection(
    *,
    method: str,
    data_root: Path,
    seed_dir: str,
    datasets: Sequence[str],
    baselines: Sequence[str],
    split_subdir: str,
    run_dir: Optional[Path],
    train_csv: Optional[Path],
    sample_frac: float,
    top_frac: float,
    random_state: Optional[int],
    mirror_selection_gmm: bool,
    update_manifest: bool,
) -> List[Dict[str, Any]]:
    method = _normalize_method(method)
    rs = resolve_training_random_state(data_root, seed_dir, random_state)
    results: List[Dict[str, Any]] = []

    if method == "balanced_random":
        if run_dir is not None:
            rd = run_dir.expanduser().resolve()
            if train_csv is None:
                parts = rd.parts
                try:
                    idx = parts.index(seed_dir)
                    dataset = parts[idx + 1]
                except (ValueError, IndexError) as exc:
                    raise SystemExit("cannot infer dataset from --run-dir; pass --train-csv") from exc
                train_csv = _default_train_csv(data_root, seed_dir, dataset, split_subdir)
            info = process_balanced_random_run(
                rd,
                Path(train_csv),
                sample_frac=sample_frac,
                random_state=rs,
                mirror_to_selection_gmm=mirror_selection_gmm,
            )
            results.append(info)
        else:
            results = process_balanced_random_seed(
                data_root=data_root,
                seed_dir=seed_dir,
                datasets=datasets,
                baselines=baselines,
                sample_frac=sample_frac,
                random_state=rs,
                split_subdir=split_subdir,
                mirror_to_selection_gmm=mirror_selection_gmm,
            )
        manifest_extra = {"sample_frac": sample_frac, "random_state_base": rs}

    elif method == "variability_top25":
        subdir = _selection_subdir_for_method(
            method, sample_frac=sample_frac, top_frac=top_frac
        )
        if run_dir is not None:
            rd = run_dir.expanduser().resolve()
            tc = train_csv
            if tc is None:
                parts = rd.parts
                try:
                    idx = parts.index(seed_dir)
                    dataset = parts[idx + 1]
                except (ValueError, IndexError) as exc:
                    raise SystemExit("cannot infer dataset from --run-dir; pass --train-csv") from exc
                tc = _default_train_csv(data_root, seed_dir, dataset, split_subdir)
            info = process_variability_run(
                rd,
                top_frac=top_frac,
                input_csv=Path(tc),
                mirror_to_selection_gmm=False,
                selection_subdir=subdir,
            )
            _ensure_hard_train_toy_csv(rd, subdir, Path(tc))
            if mirror_selection_gmm:
                _mirror_selection(
                    rd, subdir, method=method, summary=info, train_csv=Path(tc)
                )
            results.append(info)
        else:
            for ds in datasets:
                tc = _default_train_csv(data_root, seed_dir, ds, split_subdir)
                for bl in baselines:
                    rd = data_root / seed_dir / ds / bl
                    if not (rd / "training_dynamics.csv").is_file():
                        results.append(
                            {
                                "run_dir": str(rd),
                                "skipped": True,
                                "reason": "missing training_dynamics.csv",
                            }
                        )
                        continue
                    info = process_variability_run(
                        rd,
                        top_frac=top_frac,
                        input_csv=tc if tc.is_file() else None,
                        mirror_to_selection_gmm=False,
                        selection_subdir=subdir,
                    )
                    if tc.is_file():
                        _ensure_hard_train_toy_csv(rd, subdir, tc)
                    if mirror_selection_gmm and tc.is_file():
                        _mirror_selection(
                            rd, subdir, method=method, summary=info, train_csv=tc
                        )
                    info["dataset"] = ds
                    info["baseline"] = bl
                    results.append(info)
        manifest_extra = {"top_frac": top_frac}

    else:
        raise SystemExit(f"Unsupported method {method!r} (use balanced_random or variability_top25)")

    if update_manifest:
        subdir = _selection_subdir_for_method(
            method, sample_frac=sample_frac, top_frac=top_frac
        )
        update_seed_manifest(
            data_root,
            seed_dir,
            method=method,
            selection_subdir=subdir,
            mirrored_to=MIRROR_SUBDIR if mirror_selection_gmm else None,
            extra=manifest_extra,
        )

    return results


def _print_next_steps(method: str, mirror: bool) -> None:
    print("\n--- Next steps (mechanism-supervised encoder) ---")
    print("Attach one mechanism label per selected train row, then train:")
    print("  python -m pipeline stage2 mech_distill \\")
    print("    --dataset <DATASET> --data_root data/<SEED> \\")
    print("    --accepted_selected_k_csv <mechanism_labels.csv> \\")
    print("    --lambda_mech 0.05 --ce_contrastive_lambda 0.5")
    print("row_id in that CSV is train_<index> in train.csv order.")
    if not mirror:
        print(
            "\nNote: --mirror-selection-gmm was not set. "
            "The selection still lives under the method subdirectory of the baseline run."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select high-variability or class-matched random training rows for mechanism supervision."
    )
    p.add_argument(
        "--method",
        required=True,
        help="rand25|rand50|var25|var50 (or balanced_random / variability_top25)",
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--seed-dir", default="eacl_seed0_seq")
    p.add_argument("--datasets", default=",".join(BR_DATASETS))
    p.add_argument("--baselines", default=",".join(BR_BASELINES))
    p.add_argument("--split-subdir", default="splitted_data_80_10_10")
    p.add_argument("--run-dir", type=Path, default=None, help="Single baseline run dir")
    p.add_argument("--train-csv", type=Path, default=None, help="Train split CSV (single run)")
    p.add_argument(
        "--sample-frac",
        type=float,
        default=DEFAULT_SAMPLE_FRAC,
        help="balanced_random: per-class fraction (default 0.5)",
    )
    p.add_argument(
        "--top-frac",
        type=float,
        default=DEFAULT_TOP_FRAC,
        help="variability_top25: per-class top fraction (default 0.25)",
    )
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument(
        "--mirror-selection-gmm",
        action="store_true",
        help="Also copy this selection into selection_gmm/.",
    )
    p.add_argument(
        "--update-seed-manifest",
        action="store_true",
        help="Write hard_case_selection block to data/<seed-dir>/seed_manifest.json",
    )
    p.add_argument(
        "--no-next-steps",
        action="store_true",
        help="Do not print mech-only workflow reminders",
    )
    args = p.parse_args()

    method_key = (args.method or "").strip().lower()
    method = _normalize_method(args.method)
    sample_frac = float(args.sample_frac)
    top_frac = float(args.top_frac)
    # Alias sizes (rand25/var50/…) win when CLI still has method-family defaults.
    if method_key in METHOD_DEFAULT_FRAC:
        implied = float(METHOD_DEFAULT_FRAC[method_key])
        if method == "balanced_random" and sample_frac == float(DEFAULT_SAMPLE_FRAC):
            sample_frac = implied
        if method == "variability_top25" and top_frac == float(DEFAULT_TOP_FRAC):
            top_frac = implied
    if method == "balanced_random" and not (0.0 < sample_frac <= 1.0):
        raise SystemExit("--sample-frac must be in (0, 1]")
    if method == "variability_top25" and not (0.0 < top_frac <= 1.0):
        raise SystemExit("--top-frac must be in (0, 1]")

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    baselines = [x.strip() for x in str(args.baselines).split(",") if x.strip()]

    print(
        f"method={method} seed_dir={args.seed_dir} "
        f"sample_frac={sample_frac} top_frac={top_frac} "
        f"mirror_gmm={bool(args.mirror_selection_gmm)}"
    )
    results = run_selection(
        method=method,
        data_root=args.data_root.expanduser().resolve(),
        seed_dir=str(args.seed_dir),
        datasets=datasets,
        baselines=baselines,
        split_subdir=str(args.split_subdir),
        run_dir=args.run_dir,
        train_csv=args.train_csv,
        sample_frac=sample_frac,
        top_frac=top_frac,
        random_state=args.random_state,
        mirror_selection_gmm=bool(args.mirror_selection_gmm),
        update_manifest=bool(args.update_seed_manifest),
    )

    ok = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    for r in ok:
        ds = r.get("dataset", "")
        bl = r.get("baseline", "")
        prefix = f"{ds}/{bl}: " if ds else ""
        n_sel = r.get("n_selected", "?")
        n_train = r.get("n_train", "?")
        print(f"[ok] {prefix}{r.get('run_dir')} selected={n_sel}/{n_train}")
    for r in skipped:
        print(f"[skip] {r.get('run_dir')}: {r.get('reason')}")
    print(f"done: ok={len(ok)} skipped={len(skipped)}")

    if not args.no_next_steps and ok:
        _print_next_steps(method, bool(args.mirror_selection_gmm))


if __name__ == "__main__":
    main()
