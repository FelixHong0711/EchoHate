#!/usr/bin/env python
"""Run a pipeline module with foreign co2hate checkouts scrubbed from sys.path.

Keeps ``ECHOHATE_ROOT`` as the *logical* path when possible (``.../echohate`` even if
that path is a symlink to ``.../co2hate``) so __file__ / metrics do not pollute with
``/co2hate/`` path strings.

Usage:
  ECHOHATE_ROOT=... python echohate_run_module.py pipeline.stage2.roberta_baseline [args...]
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _same_tree(a: Path, b: Path) -> bool:
    try:
        ar = a.resolve()
        br = b.resolve()
        return ar == br or br in ar.parents or ar in br.parents
    except OSError:
        return False


def _echohate_root() -> str:
    env = os.environ.get("ECHOHATE_ROOT", "").strip()
    if env:
        p = Path(env)
        # Do NOT Path.resolve() here — keep .../echohate when it is a symlink to co2hate.
        if p.exists():
            return str(p)
        return str(p.resolve())
    # job_submission/stage2/roberta_baseline/ → repo root
    here = Path(__file__).resolve().parents[3]
    # Prefer sibling/self named echohate when this file lives under co2hate alias.
    cand = here.parent / "echohate"
    if cand.exists() and _same_tree(cand, here):
        return str(cand)
    return str(here)


def _scrub_sys_path(root: str) -> None:
    root_p = Path(root)
    try:
        root_phys = root_p.resolve()
    except OSError:
        root_phys = root_p
    cleaned = []
    for p in sys.path:
        if not p:
            continue
        norm = p.replace("\\", "/")
        if "/co2hate/" in norm or norm.rstrip("/").endswith("/co2hate"):
            try:
                pp = Path(p).resolve()
            except OSError:
                continue
            # Keep the alias path only when it is *this* checkout.
            if pp != root_phys and root_phys not in pp.parents and pp not in root_phys.parents:
                continue
        cleaned.append(p)
    sys.path[:] = cleaned
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for k in list(sys.modules):
        if k != "pipeline" and not k.startswith("pipeline."):
            continue
        mod = sys.modules.get(k)
        f = getattr(mod, "__file__", None) or ""
        if "/co2hate/" not in f.replace("\\", "/"):
            continue
        try:
            fp = Path(f).resolve()
        except OSError:
            del sys.modules[k]
            continue
        if fp != root_phys and root_phys not in fp.parents:
            del sys.modules[k]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: echohate_run_module.py <module> [args...]\n"
            "example: echohate_run_module.py pipeline.stage2.roberta_baseline --dataset IHC"
        )
    root = _echohate_root()
    os.environ["ECHOHATE_ROOT"] = root
    _scrub_sys_path(root)
    mod = sys.argv[1]
    sys.argv = [mod, *sys.argv[2:]]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
