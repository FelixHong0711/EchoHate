"""EchoHate encoder pipeline (preprocessing, variability selection, mechanism-supervised training).

Partial release for the ARR October submission. The LLM teacher (mechanism
annotation) and uncertainty-guided deferral are withheld and will be released
upon acceptance. Datasets, mechanism labels, and cluster jobs are not included.
"""

from pipeline.common.paths import ECHOHATE_ROOT

__all__ = ["ECHOHATE_ROOT"]
