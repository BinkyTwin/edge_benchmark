"""
Edge SLM Benchmark Framework
============================

Framework de benchmark pour évaluer des SLMs sur Apple Silicon
dans un contexte bancaire réglementé.

Modules:
    - lmstudio_client: Client REST API LM Studio
    - reproducibility: Gestion des seeds et de l'environnement
    - statistics: Intervalles de confiance et tests de significativité
    - performance: Benchmarks de performance (TTFT, tokens/s, RAM)
    - capability: Benchmarks de capacités (Banking77, coding, etc.)
    - compliance: Analyse de conformité NIST/OWASP
"""

__version__ = "0.1.0"
__author__ = "Edge Benchmark Team"

from .lmstudio_client import LMStudioClient
from .reproducibility import ReproducibilityManager, set_deterministic_mode
from .statistics import StatisticalAnalyzer, ConfidenceInterval, ModelComparison

__all__ = [
    "LMStudioClient",
    "ReproducibilityManager",
    "set_deterministic_mode",
    "StatisticalAnalyzer",
    "ConfidenceInterval",
    "ModelComparison",
]


