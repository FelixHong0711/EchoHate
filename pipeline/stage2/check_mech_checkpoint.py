#!/usr/bin/env python3
"""
Check mech run-dir checkpoints against the current mechanism taxonomy.

Exit codes (for Slurm / submit skip logic):
  0 — artifacts present and mechanism-head compatible (safe to skip training)
  1 — artifacts present but stale / incompatible (retrain)
  2 — required artifacts missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from pipeline.stage2.mech_distill import checkpoint_mechanism_compatible


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mech_artifacts_ok(run_dir: Path) -> bool:
    ckpt = run_dir / "encoder_model.pt"
    return ckpt.is_file() and ckpt.stat().st_size > 0


def _load_cfg(run_dir: Path) -> Dict[str, Any]:
    for name in ("encoder_distill_config.json", "metric_summary.json"):
        cfg = _read_json(run_dir / name)
        if cfg:
            return cfg
    return {}


def _load_state(run_dir: Path) -> Dict[str, Any]:
    ckpt = run_dir / "encoder_model.pt"
    if not ckpt.is_file():
        return {}
    return torch.load(ckpt, map_location="cpu", weights_only=True)


def _expect_mech_head(cfg: Dict[str, Any]) -> bool:
    tc = cfg.get("training_config")
    if isinstance(tc, dict):
        return float(tc.get("lambda_mech", 0.0) or 0.0) > 0.0
    return float(cfg.get("lambda_mech", 0.05) or 0.0) > 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check mechanism-head checkpoint compatibility")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--approach",
        choices=("mech",),
        default="mech",
        help="Mechanism-head checkpoint check.",
    )
    ap.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Accepted for callers; artifacts are always required.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()

    if not _mech_artifacts_ok(run_dir):
        print(f"MISSING artifacts under {run_dir}", file=sys.stderr)
        return 2

    cfg = _load_cfg(run_dir)
    state = _load_state(run_dir)
    expect = _expect_mech_head(cfg)
    ok, msg = checkpoint_mechanism_compatible(cfg=cfg, state_dict=state, expect_mech_head=expect)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
