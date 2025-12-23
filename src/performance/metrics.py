"""
Metrics Collector
=================

Module for collection and aggregation of performance metrics:
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
    """Memory usage snapshot."""
    
    timestamp: float
    rss_mb: float           # Resident Set Size
    vms_mb: float           # Virtual Memory Size
    percent: float          # RAM percentage used
    
    @classmethod
    def capture(cls) -> "MemorySnapshot":
        """Captures a snapshot of current memory."""
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
    """Aggregated metrics across multiple runs."""
    
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
        """Converts to dictionary."""
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
    Metrics collector for performance benchmarks.
    
    Collects timing, throughput and memory metrics
    across multiple runs and calculates aggregated statistics.
    """
    
    def __init__(self):
        """Initializes the collector."""
        self.runs: list[dict] = []
        self.memory_snapshots: list[MemorySnapshot] = []
        self._monitoring = False
    
    def reset(self):
        """Resets the collector."""
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
        Adds metrics from a run.
        
        Args:
            ttft_ms: Time to first token (ms)
            total_time_ms: Total time (ms)
            output_tokens_per_sec: Generation throughput
            prompt_tokens_per_sec: Ingestion speed
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of generated tokens
            success: Whether the run succeeded
            error: Error message if failed
            **extra: Additional metrics
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
        
        # Capture memory at each run
        self.memory_snapshots.append(MemorySnapshot.capture())
    
    def add_from_completion_metrics(self, metrics: "CompletionMetrics", success: bool = True):
        """
        Adds metrics from a CompletionMetrics object.
        
        Args:
            metrics: Completion metrics
            success: Whether the run succeeded
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
        Calculates aggregated metrics across all runs.
        
        Returns:
            AggregatedMetrics with statistics
        """
        if not self.runs:
            return AggregatedMetrics()
        
        # Filter successful runs for stats
        successful_runs = [r for r in self.runs if r["success"]]
        
        if not successful_runs:
            return AggregatedMetrics(
                num_runs=len(self.runs),
                num_successful=0,
                success_rate=0.0,
            )
        
        # Extract values
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
        """Returns raw data from all runs."""
        return self.runs.copy()
    
    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        """Calculates the p percentile of a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class MemoryMonitor:
    """
    Memory monitor to capture peak RAM during inference.
    
    Uses a separate thread to sample memory periodically.
    """
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Args:
            sample_interval: Sampling interval in seconds
        """
        self.sample_interval = sample_interval
        self.snapshots: list[MemorySnapshot] = []
        self._running = False
        self._thread = None
    
    def start(self):
        """Starts the monitoring."""
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
        Stops the monitoring and returns statistics.
        
        Returns:
            Dictionary with peak_ram_mb, avg_ram_mb, num_samples
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


