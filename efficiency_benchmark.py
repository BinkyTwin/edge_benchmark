"""
Efficiency Benchmark - Week 1 (LM Studio)
Mesure latence, throughput, RAM (système) pour comparer des modèles sur une machine 16GB.

Usage (CLI):
  python efficiency_benchmark.py --model-index 0 --runs 2 --max-tokens 100

Notes:
- Par défaut on utilise l'endpoint OpenAI-compatible de LM Studio: /v1/completions
- La RAM mesurée ici est la RAM système (psutil.virtual_memory), pas la RAM du process LM Studio.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import requests


DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234"
DEFAULT_COMPLETIONS_URL = f"{DEFAULT_LMSTUDIO_BASE_URL}/v1/completions"
DEFAULT_MODELS_URL = f"{DEFAULT_LMSTUDIO_BASE_URL}/v1/models"

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


# Liste simple: tu choisis l'index selon le modèle que tu as chargé dans LM Studio.
# (Tu peux modifier cette liste dans le notebook sans toucher au script.)
MODELS: List[str] = [
    "qwen/qwen3-vl-8b",
    "google/gemma-3n-e4b",
    "mistralai/ministral-3-14b-reasoning",
    "google/gemma-3-12b",
]


def get_system_memory_info() -> Dict[str, float]:
    """RAM système + disponibilité (en GB)."""
    vm = psutil.virtual_memory()
    return {
        "memory_percent": round(vm.percent, 1),
        "used_gb": round(vm.used / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
    }


def list_lmstudio_models(models_url: str = DEFAULT_MODELS_URL, timeout: int = 10) -> List[str]:
    """
    Retourne les IDs exposés par LM Studio via /v1/models (OpenAI compatible).
    Si LM Studio ne supporte pas cet endpoint dans ta config, renvoie [].
    """
    try:
        r = requests.get(models_url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = []
        for it in items:
            mid = it.get("id")
            if mid:
                ids.append(mid)
        return ids
    except Exception:
        return []


def estimate_tokens_from_response(text: str, usage: Optional[Dict[str, Any]] = None) -> int:
    """
    Essaie d'utiliser usage.completion_tokens si présent, sinon estimation grossière (~4 chars/token).
    """
    if isinstance(usage, dict):
        ct = usage.get("completion_tokens")
        if isinstance(ct, int) and ct > 0:
            return ct
    return max(1, len(text) // 4)


@dataclass
class InferenceResult:
    timestamp: str
    model: str
    status: str
    latency_sec: Optional[float] = None
    tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    mem_before_percent: Optional[float] = None
    mem_after_percent: Optional[float] = None
    mem_used_gb: Optional[float] = None
    mem_available_gb: Optional[float] = None
    error: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "status": self.status,
            "latency_sec": self.latency_sec,
            "tokens": self.tokens,
            "tokens_per_sec": self.tokens_per_sec,
            "mem_before_percent": self.mem_before_percent,
            "mem_after_percent": self.mem_after_percent,
            "mem_used_gb": self.mem_used_gb,
            "mem_available_gb": self.mem_available_gb,
            "error": self.error,
        }


class EfficiencyBenchmark:
    def __init__(self, completions_url: str = DEFAULT_COMPLETIONS_URL, results_dir: Path = RESULTS_DIR):
        self.completions_url = completions_url
        self.results_dir = results_dir
        self.results: List[InferenceResult] = []

    def run_inference(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 180,
    ) -> InferenceResult:
        mem_before = get_system_memory_info()
        start = time.time()

        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        try:
            r = requests.post(self.completions_url, json=payload, timeout=timeout)
            r.raise_for_status()
            elapsed = time.time() - start
            mem_after = get_system_memory_info()

            data = r.json() if r.content else {}
            choice0 = (data.get("choices") or [{}])[0] if isinstance(data, dict) else {}
            text = choice0.get("text") or ""
            usage = data.get("usage") if isinstance(data, dict) else None

            tokens = estimate_tokens_from_response(text, usage)
            tps = (tokens / elapsed) if elapsed > 0 else 0.0

            return InferenceResult(
                timestamp=datetime.now().isoformat(),
                model=model_id,
                status="success",
                latency_sec=round(elapsed, 3),
                tokens=tokens,
                tokens_per_sec=round(tps, 2),
                mem_before_percent=mem_before["memory_percent"],
                mem_after_percent=mem_after["memory_percent"],
                mem_used_gb=mem_after["used_gb"],
                mem_available_gb=mem_after["available_gb"],
            )
        except Exception as e:
            return InferenceResult(
                timestamp=datetime.now().isoformat(),
                model=model_id,
                status="error",
                error=str(e),
            )

    def run_suite(
        self,
        model_id: str,
        runs: int = 2,
        prompt: str = "What is artificial intelligence? Explain briefly in 1 paragraph.",
        max_tokens: int = 100,
        sleep_sec: float = 2.0,
    ) -> List[InferenceResult]:
        for _ in range(runs):
            res = self.run_inference(model_id=model_id, prompt=prompt, max_tokens=max_tokens)
            self.results.append(res)
            time.sleep(sleep_sec)
        return self.results

    def save_csv(self, filename: str = "efficiency_results.csv") -> Path:
        csv_path = self.results_dir / filename
        successful = [r.to_row() for r in self.results if r.status == "success"]
        if not successful:
            return csv_path
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(successful[0].keys()))
            writer.writeheader()
            writer.writerows(successful)
        return csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LM Studio efficiency benchmark")
    p.add_argument("--base-url", default=DEFAULT_LMSTUDIO_BASE_URL, help="LM Studio base URL (default: http://localhost:1234)")
    p.add_argument("--model-index", type=int, default=0, help="Index dans la liste MODELS")
    p.add_argument("--runs", type=int, default=2, help="Nombre de runs")
    p.add_argument("--max-tokens", type=int, default=100, help="max_tokens pour /v1/completions")
    p.add_argument("--prompt", default="What is artificial intelligence? Explain briefly in 1 paragraph.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    completions_url = f"{args.base_url.rstrip('/')}/v1/completions"

    if not (0 <= args.model_index < len(MODELS)):
        raise SystemExit(f"model-index invalide: {args.model_index} (0..{len(MODELS)-1})")

    model_id = MODELS[args.model_index]
    bench = EfficiencyBenchmark(completions_url=completions_url)
    bench.run_suite(model_id=model_id, runs=args.runs, prompt=args.prompt, max_tokens=args.max_tokens)
    csv_path = bench.save_csv()

    ok = [r for r in bench.results if r.status == "success"]
    if ok:
        avg_lat = sum(r.latency_sec or 0 for r in ok) / len(ok)
        avg_tps = sum(r.tokens_per_sec or 0 for r in ok) / len(ok)
        peak_mem = max(r.mem_used_gb or 0 for r in ok)
        print(f"Model: {model_id}")
        print(f"Runs: {len(ok)}/{len(bench.results)}")
        print(f"Avg latency: {avg_lat:.2f}s | Avg throughput: {avg_tps:.2f} tok/s | Peak RAM used: {peak_mem:.2f} GB")
        print(f"Saved: {csv_path}")
    else:
        print("Aucun run réussi. Vérifie que LM Studio tourne et que le modèle est chargé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


