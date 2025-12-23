"""
Edge SLM Benchmark Framework
============================

Benchmark framework for evaluating SLMs on Apple Silicon
in a regulated banking context.

Modules:
    - lmstudio_client: LM Studio REST API client
    - reproducibility: Seed and environment management
    - statistics: Confidence intervals and significance tests
    - performance: Performance benchmarks (TTFT, tokens/s, RAM)
    - capability: Capability benchmarks (Banking77, coding, etc.)
    - compliance: NIST/OWASP compliance analysis
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


