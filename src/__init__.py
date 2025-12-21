"""
Edge SLM Benchmark Framework
============================

Framework de benchmark pour évaluer des SLMs sur Apple Silicon
dans un contexte bancaire réglementé.

Modules:
    - lmstudio_client: Client REST API LM Studio
    - performance: Benchmarks de performance (TTFT, tokens/s, RAM)
    - capability: Benchmarks de capacités (Banking77, coding, etc.)
    - compliance: Analyse de conformité NIST/OWASP
"""

__version__ = "0.1.0"
__author__ = "Edge Benchmark Team"

from .lmstudio_client import LMStudioClient

__all__ = ["LMStudioClient"]

