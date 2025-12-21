"""
Metrics Collector
=================

Module pour la collecte et l'agrégation des métriques de performance:
- Timing (TTFT, total time)
- Throughput (tokens/s)
- Memory (RAM usage via psutil)
- Statistics (mean, std, percentiles)
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


@dataclass
class MemorySnapshot:
    """Snapshot de l'utilisation mémoire."""
    
    timestamp: float
    rss_mb: float           # Resident Set Size
    vms_mb: float           # Virtual Memory Size
    percent: float          # Pourcentage de RAM utilisée
    
    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """Capture un snapshot de la mémoire actuelle."""
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        
        return cls(
            timestamp=time.time(),
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=mem_percent,
        )


@dataclass
class AggregatedMetrics:
    """Métriques agrégées sur plusieurs runs."""
    
    # TTFT
    ttft_mean_ms: float = 0.0
    ttft_std_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    
    # Total time
    total_time_mean_ms: float = 0.0
    total_time_std_ms: float = 0.0
    total_time_p50_ms: float = 0.0
    total_time_p95_ms: float = 0.0
    
    # Output tokens/s
    output_tps_mean: float = 0.0
    output_tps_std: float = 0.0
    output_tps_p50: float = 0.0
    output_tps_p95: float = 0.0
    
    # Prompt tokens/s
    prompt_tps_mean: float = 0.0
    prompt_tps_std: float = 0.0
    
    # Memory
    peak_ram_mb: float = 0.0
    avg_ram_mb: float = 0.0
    
    # Tokens
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    
    # Metadata
    num_runs: int = 0
    num_successful: int = 0
    success_rate: float = 0.0
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "ttft": {
                "mean_ms": round(self.ttft_mean_ms, 2),
                "std_ms": round(self.ttft_std_ms, 2),
                "p50_ms": round(self.ttft_p50_ms, 2),
                "p95_ms": round(self.ttft_p95_ms, 2),
                "min_ms": round(self.ttft_min_ms, 2),
                "max_ms": round(self.ttft_max_ms, 2),
            },
            "total_time": {
                "mean_ms": round(self.total_time_mean_ms, 2),
                "std_ms": round(self.total_time_std_ms, 2),
                "p50_ms": round(self.total_time_p50_ms, 2),
                "p95_ms": round(self.total_time_p95_ms, 2),
            },
            "output_tokens_per_sec": {
                "mean": round(self.output_tps_mean, 2),
                "std": round(self.output_tps_std, 2),
                "p50": round(self.output_tps_p50, 2),
                "p95": round(self.output_tps_p95, 2),
            },
            "prompt_tokens_per_sec": {
                "mean": round(self.prompt_tps_mean, 2),
                "std": round(self.prompt_tps_std, 2),
            },
            "memory": {
                "peak_ram_mb": round(self.peak_ram_mb, 2),
                "avg_ram_mb": round(self.avg_ram_mb, 2),
            },
            "tokens": {
                "avg_prompt": round(self.avg_prompt_tokens, 1),
                "avg_completion": round(self.avg_completion_tokens, 1),
            },
            "runs": {
                "total": self.num_runs,
                "successful": self.num_successful,
                "success_rate": round(self.success_rate, 4),
            },
        }


class MetricsCollector:
    """
    Collecteur de métriques pour les benchmarks de performance.
    
    Collecte les métriques de timing, throughput et mémoire
    sur plusieurs runs et calcule les statistiques agrégées.
    """
    
    def __init__(self):
        """Initialise le collecteur."""
        self.runs: list[dict] = []
        self.memory_snapshots: list[MemorySnapshot] = []
        self._monitoring = False
    
    def reset(self):
        """Réinitialise le collecteur."""
        self.runs = []
        self.memory_snapshots = []
        self._monitoring = False
    
    def add_run(
        self,
        ttft_ms: float,
        total_time_ms: float,
        output_tokens_per_sec: float,
        prompt_tokens_per_sec: float,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
        error: Optional[str] = None,
        **extra,
    ):
        """
        Ajoute les métriques d'un run.
        
        Args:
            ttft_ms: Time to first token (ms)
            total_time_ms: Temps total (ms)
            output_tokens_per_sec: Débit de génération
            prompt_tokens_per_sec: Vitesse d'ingestion
            prompt_tokens: Nombre de tokens du prompt
            completion_tokens: Nombre de tokens générés
            success: Si le run a réussi
            error: Message d'erreur si échec
            **extra: Métriques supplémentaires
        """
        run_data = {
            "ttft_ms": ttft_ms,
            "total_time_ms": total_time_ms,
            "output_tokens_per_sec": output_tokens_per_sec,
            "prompt_tokens_per_sec": prompt_tokens_per_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "success": success,
            "error": error,
            "timestamp": time.time(),
            **extra,
        }
        self.runs.append(run_data)
        
        # Capturer la mémoire à chaque run
        self.memory_snapshots.append(MemorySnapshot.capture())
    
    def add_from_completion_metrics(self, metrics: "CompletionMetrics", success: bool = True):
        """
        Ajoute les métriques depuis un objet CompletionMetrics.
        
        Args:
            metrics: Métriques de complétion
            success: Si le run a réussi
        """
        self.add_run(
            ttft_ms=metrics.ttft_ms,
            total_time_ms=metrics.total_time_ms,
            output_tokens_per_sec=metrics.output_tokens_per_sec,
            prompt_tokens_per_sec=metrics.prompt_tokens_per_sec,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            success=success,
            finish_reason=metrics.finish_reason,
        )
    
    def aggregate(self) -> AggregatedMetrics:
        """
        Calcule les métriques agrégées sur tous les runs.
        
        Returns:
            AggregatedMetrics avec statistiques
        """
        if not self.runs:
            return AggregatedMetrics()
        
        # Filtrer les runs réussis pour les stats
        successful_runs = [r for r in self.runs if r["success"]]
        
        if not successful_runs:
            return AggregatedMetrics(
                num_runs=len(self.runs),
                num_successful=0,
                success_rate=0.0,
            )
        
        # Extraire les valeurs
        ttfts = [r["ttft_ms"] for r in successful_runs]
        total_times = [r["total_time_ms"] for r in successful_runs]
        output_tps = [r["output_tokens_per_sec"] for r in successful_runs if r["output_tokens_per_sec"] > 0]
        prompt_tps = [r["prompt_tokens_per_sec"] for r in successful_runs if r["prompt_tokens_per_sec"] > 0]
        prompt_tokens = [r["prompt_tokens"] for r in successful_runs]
        completion_tokens = [r["completion_tokens"] for r in successful_runs]
        
        # RAM
        ram_values = [s.rss_mb for s in self.memory_snapshots]
        
        return AggregatedMetrics(
            # TTFT
            ttft_mean_ms=statistics.mean(ttfts),
            ttft_std_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            ttft_p50_ms=statistics.median(ttfts),
            ttft_p95_ms=self._percentile(ttfts, 0.95),
            ttft_min_ms=min(ttfts),
            ttft_max_ms=max(ttfts),
            
            # Total time
            total_time_mean_ms=statistics.mean(total_times),
            total_time_std_ms=statistics.stdev(total_times) if len(total_times) > 1 else 0,
            total_time_p50_ms=statistics.median(total_times),
            total_time_p95_ms=self._percentile(total_times, 0.95),
            
            # Output tokens/s
            output_tps_mean=statistics.mean(output_tps) if output_tps else 0,
            output_tps_std=statistics.stdev(output_tps) if len(output_tps) > 1 else 0,
            output_tps_p50=statistics.median(output_tps) if output_tps else 0,
            output_tps_p95=self._percentile(output_tps, 0.95) if output_tps else 0,
            
            # Prompt tokens/s
            prompt_tps_mean=statistics.mean(prompt_tps) if prompt_tps else 0,
            prompt_tps_std=statistics.stdev(prompt_tps) if len(prompt_tps) > 1 else 0,
            
            # Memory
            peak_ram_mb=max(ram_values) if ram_values else 0,
            avg_ram_mb=statistics.mean(ram_values) if ram_values else 0,
            
            # Tokens
            avg_prompt_tokens=statistics.mean(prompt_tokens) if prompt_tokens else 0,
            avg_completion_tokens=statistics.mean(completion_tokens) if completion_tokens else 0,
            
            # Runs
            num_runs=len(self.runs),
            num_successful=len(successful_runs),
            success_rate=len(successful_runs) / len(self.runs),
        )
    
    def get_raw_data(self) -> list[dict]:
        """Retourne les données brutes de tous les runs."""
        return self.runs.copy()
    
    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        """Calcule le percentile p d'une liste de valeurs."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class MemoryMonitor:
    """
    Moniteur de mémoire pour capturer le pic de RAM pendant l'inférence.
    
    Utilise un thread séparé pour sampler la mémoire périodiquement.
    """
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Args:
            sample_interval: Intervalle d'échantillonnage en secondes
        """
        self.sample_interval = sample_interval
        self.snapshots: list[MemorySnapshot] = []
        self._running = False
        self._thread = None
    
    def start(self):
        """Démarre le monitoring."""
        import threading
        
        self.snapshots = []
        self._running = True
        
        def _sample():
            while self._running:
                self.snapshots.append(MemorySnapshot.capture())
                time.sleep(self.sample_interval)
        
        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()
    
    def stop(self) -> dict:
        """
        Arrête le monitoring et retourne les statistiques.
        
        Returns:
            Dictionnaire avec peak_ram_mb, avg_ram_mb, num_samples
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.snapshots:
            return {"peak_ram_mb": 0, "avg_ram_mb": 0, "num_samples": 0}
        
        ram_values = [s.rss_mb for s in self.snapshots]
        return {
            "peak_ram_mb": max(ram_values),
            "avg_ram_mb": statistics.mean(ram_values),
            "min_ram_mb": min(ram_values),
            "num_samples": len(ram_values),
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


