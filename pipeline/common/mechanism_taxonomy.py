"""Mechanism taxonomies (8-way hate, 5-way offensive).

Closed mechanism inventories used by the mechanism-supervised encoder.

Hate = 6 hate mechs + 2 not-hate mechs. Offensive = 3 + 2.
"""

from __future__ import annotations

from typing import Dict, Tuple

# --- Hate (8-way) ---

HATE_MECHANISM_LABELS = [
    "Dehumanization",
    "Inferiority",
    "Group Vilification",
    "Threatening and Incitement",
    "In-group Victimhood Framing",
    "Support for Hateful Entities",
]

NOT_HATE_MECHANISM_LABELS = [
    "Not Offensive",
    "Offensive but No Protected-Group Attack Intent",
]

# Tie-break when several hate mechs still equally fit (mirrors IHC B>F>A>C>E).
HATE_MECHANISM_PRIORITY = [
    "Support for Hateful Entities",
    "Threatening and Incitement",
    "In-group Victimhood Framing",
    "Dehumanization",
    "Inferiority",
    "Group Vilification",
]

# --- Offensive (5-way) ---

OFFENSIVE_MECHANISM_LABELS = [
    "Stereotyping and Implied Bias",
    "Lewdness",
    "Explicit Rudeness or Toxicity",
]

NOT_OFFENSIVE_MECHANISM_LABELS = [
    "Neutral or Benign Speech",
    "Counter Speech",
]

OFFENSIVE_MECHANISM_PRIORITY = [
    "Stereotyping and Implied Bias",
    "Lewdness",
    "Explicit Rudeness or Toxicity",
]


def mechanism_labels_for_task(task: str) -> Tuple[str, ...]:
    """Per-task label tuple: hate=8, offensive=5."""
    t = (task or "hate").strip().lower()
    if t == "offensive":
        return tuple(OFFENSIVE_MECHANISM_LABELS) + tuple(NOT_OFFENSIVE_MECHANISM_LABELS)
    return tuple(HATE_MECHANISM_LABELS) + tuple(NOT_HATE_MECHANISM_LABELS)


def mech_to_id_for_task(task: str) -> Dict[str, int]:
    return {m: i for i, m in enumerate(mechanism_labels_for_task(task))}


def n_mechanism_classes_for_task(task: str) -> int:
    return len(mechanism_labels_for_task(task))


N_HATE_MECHANISM_CLASSES = len(HATE_MECHANISM_LABELS) + len(NOT_HATE_MECHANISM_LABELS)
N_OFFENSIVE_MECHANISM_CLASSES = len(OFFENSIVE_MECHANISM_LABELS) + len(
    NOT_OFFENSIVE_MECHANISM_LABELS
)
