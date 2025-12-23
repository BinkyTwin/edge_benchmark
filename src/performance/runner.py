"""
Performance Runner
==================

Main orchestrator for performance benchmarks.
Executes scenarios with metrics collection and logging.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, CompletionResult
from ..statistics import StatisticalAnalyzer, ConfidenceInterval
from ..reproducibility import ReproducibilityManager
from .metrics import MetricsCollector, MemoryMonitor, AggregatedMetrics
from .scenarios import ScenarioExecutor, ScenarioConfig


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    warmup_runs: int = 3
    benchmark_runs: int = 20
    cooldown_seconds: float = 2.0
    stream: bool = True
    temperature: float = 0
    top_p: float = 1
    seed: int = 42  # Seed for reproducibility


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark."""
    
    model_id: str
    scenario_name: str
    config: BenchmarkConfig
    metrics: AggregatedMetrics
    raw_results: list[dict]
    timestamp: str
    duration_seconds: float
    
    # Model metadata (format, quantization)
    model_format: str = "unknown"           # mlx, gguf
    model_quantization: str = "unknown"     # 4bit, Q4_K_M, etc.
    model_config_key: str = ""              # Unique key in models.yaml
    model_display_name: str = ""            # Display name
    
    # Scenario-specific metrics
    json_valid_rate: Optional[float] = None
    
    # Advanced statistics (95% CI)
    confidence_intervals: Optional[dict] = None
    
    # Environment for reproducibility
    environment: Optional[dict] = None
    
    def get_filename_suffix(self) -> str:
        """Return suffix for filenames (e.g., MLX_4bit)."""
        return f"{self.model_format.upper()}_{self.model_quantization}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_format": self.model_format,
            "model_quantization": self.model_quantization,
            "model_config_key": self.model_config_key,
            "model_display_name": self.model_display_name,
            "scenario_name": self.scenario_name,
            "config": {
                "warmup_runs": self.config.warmup_runs,
                "benchmark_runs": self.config.benchmark_runs,
                "cooldown_seconds": self.config.cooldown_seconds,
                "stream": self.config.stream,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
            },
            "metrics": self.metrics.to_dict(),
            "confidence_intervals": self.confidence_intervals,
            "json_valid_rate": self.json_valid_rate,
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_raw_results": len(self.raw_results),
            "environment": self.environment,
        }


class PerformanceRunner:
    """
    Performance benchmark orchestrator.
    
    Manages scenario execution, metrics collection,
    and results logging.
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        scenarios_config_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        """
        Args:
            client: LM Studio client
            scenarios_config_path: Path to scenarios.yaml
            results_dir: Directory for results
        """
        self.client = client
        self.scenario_executor = ScenarioExecutor(scenarios_config_path)
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_scenario(
        self,
        model_id: str,
        scenario_name: str,
        config: Optional[BenchmarkConfig] = None,
        progress_bar: bool = True,
        model_info: Optional[dict] = None,
    ) -> BenchmarkResult:
        """
        Execute a benchmark scenario for a model.
        
        Args:
            model_id: LM Studio model ID
            scenario_name: Name of the scenario to execute
            config: Benchmark configuration
            progress_bar: Display a progress bar
            model_info: Model metadata (format, quantization, etc.)
            
        Returns:
            BenchmarkResult with aggregated metrics
        """
        config = config or BenchmarkConfig()
        scenario = self.scenario_executor.get_scenario(scenario_name)
        model_info = model_info or {}
        
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Display model information
        display_name = model_info.get("display_name", model_id)
        model_format = model_info.get("format", "unknown").upper()
        quantization = model_info.get("quantization", "unknown")
        
        print(f"\n{'='*60}")
        print(f"Benchmark: {scenario.name}")
        print(f"Model: {display_name}")
        print(f"Format: {model_format} | Quantization: {quantization}")
        print(f"Category: {scenario.category}")
        print(f"{'='*60}")
        
        # Preliminary check that the model can respond
        print(f"\n[Check] Testing model availability...")
        test_result = self.client.complete(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            stream=False,
        )
        if not test_result.success:
            error_msg = test_result.error or "Unknown error"
            print(f"[FATAL] Model '{model_id}' is not responding!")
            print(f"[FATAL] Error: {error_msg}")
            print(f"\n⚠️  Please make sure the model is LOADED in LM Studio:")
            print(f"    1. Open LM Studio")
            print(f"    2. Go to 'Chat' or 'Server' tab")
            print(f"    3. Select and load model: {model_id}")
            print(f"    4. Wait for it to finish loading (100%)")
            print(f"    5. Re-run this benchmark")
            raise RuntimeError(f"Model not available: {error_msg}")
        print(f"[Check] Model is responding ✓")
        
        start_time = time.time()
        collector = MetricsCollector()
        raw_results = []
        json_valid_count = 0
        
        # Get prompts
        prompts = self.scenario_executor.get_prompts(
            scenario_name, 
            num_prompts=config.benchmark_runs
        )
        
        # Warm-up
        print(f"\n[Warm-up] Running {config.warmup_runs} warm-up requests...")
        self.client.warmup(model_id, num_requests=config.warmup_runs)
        print("[Warm-up] Complete")
        
        # Benchmark runs
        print(f"\n[Benchmark] Running {config.benchmark_runs} benchmark requests...")
        
        iterator = tqdm(prompts, desc="Benchmarking") if progress_bar else prompts
        
        for prompt_data in iterator:
            # Memory monitoring
            with MemoryMonitor(sample_interval=0.05) as mem_monitor:
                result = self.client.complete(
                    model=model_id,
                    messages=prompt_data["messages"],
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=prompt_data.get("max_tokens", 512),
                    stream=config.stream,
                    response_format=prompt_data.get("response_format"),
                )
                mem_stats = mem_monitor.stop()
            
            # Log error if failure
            if not result.success:
                error_msg = result.error or "Unknown error"
                tqdm.write(f"[ERROR] Run failed: {error_msg[:100]}")
            
            # Collect metrics
            collector.add_run(
                ttft_ms=result.metrics.ttft_ms,
                total_time_ms=result.metrics.total_time_ms,
                output_tokens_per_sec=result.metrics.output_tokens_per_sec,
                prompt_tokens_per_sec=result.metrics.prompt_tokens_per_sec,
                prompt_tokens=result.metrics.prompt_tokens,
                completion_tokens=result.metrics.completion_tokens,
                success=result.success,
                error=result.error,
                peak_ram_mb=mem_stats.get("peak_ram_mb", 0),
            )
            
            # Validate JSON if applicable
            if prompt_data.get("response_format"):
                validation = self.scenario_executor.validate_json_output(
                    result.content,
                    prompt_data.get("expected_fields", [])
                )
                if validation["is_valid"]:
                    json_valid_count += 1
            
            # Store raw result
            raw_results.append({
                "content_preview": result.content[:200] if result.content else "",
                "metrics": result.metrics.to_dict(),
                "success": result.success,
                "error": result.error,
                "json_valid": result.json_valid if prompt_data.get("response_format") else None,
            })
            
            # Cooldown
            time.sleep(config.cooldown_seconds)
        
        # Aggregate metrics
        aggregated = collector.aggregate()
        duration = time.time() - start_time
        
        # Calculate valid JSON rate
        json_valid_rate = None
        if scenario.use_structured_output:
            json_valid_rate = json_valid_count / len(prompts) if prompts else 0
        
        # Calculate confidence intervals (95% CI)
        confidence_intervals = self._compute_confidence_intervals(collector.get_raw_data())
        
        result = BenchmarkResult(
            model_id=model_id,
            scenario_name=scenario_name,
            config=config,
            metrics=aggregated,
            raw_results=raw_results,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            model_format=model_info.get("format", "unknown"),
            model_quantization=model_info.get("quantization", "unknown"),
            model_config_key=model_info.get("config_key", ""),
            model_display_name=model_info.get("display_name", model_id),
            json_valid_rate=json_valid_rate,
            confidence_intervals=confidence_intervals,
        )
        
        # Display summary
        self._print_summary(result)
        
        return result
    
    def run_all_scenarios(
        self,
        model_id: str,
        scenarios: Optional[list[str]] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> list[BenchmarkResult]:
        """
        Execute all scenarios for a model.
        
        Args:
            model_id: Model ID
            scenarios: List of scenarios (None = all)
            config: Benchmark configuration
            
        Returns:
            List of BenchmarkResult
        """
        scenarios = scenarios or self.scenario_executor.list_scenarios()
        results = []
        
        for scenario_name in scenarios:
            result = self.run_scenario(model_id, scenario_name, config)
            results.append(result)
            self.save_result(result)
        
        return results
    
    def run_model_comparison(
        self,
        model_ids: list[str],
        scenario_name: str,
        config: Optional[BenchmarkConfig] = None,
    ) -> dict[str, BenchmarkResult]:
        """
        Compare multiple models on a scenario.
        
        Args:
            model_ids: List of model IDs
            scenario_name: Scenario name
            config: Benchmark configuration
            
        Returns:
            Dictionary model_id -> BenchmarkResult
        """
        results = {}
        
        for model_id in model_ids:
            print(f"\n{'#'*60}")
            print(f"# Model: {model_id}")
            print(f"{'#'*60}")
            
            result = self.run_scenario(model_id, scenario_name, config)
            results[model_id] = result
            self.save_result(result)
        
        # Display comparison
        self._print_comparison(results, scenario_name)
        
        return results
    
    def save_result(self, result: BenchmarkResult, filename: Optional[str] = None):
        """
        Save a result in JSONL format.
        
        The filename includes the format (MLX/GGUF) to distinguish results.
        Format: perf_{model}_{FORMAT}_{quantization}_{scenario}_{timestamp}.jsonl
        
        Args:
            result: Result to save
            filename: Filename (auto-generated if None)
        """
        if filename is None:
            model_name = result.model_id.replace("/", "_")
            format_suffix = result.get_filename_suffix()  # Ex: MLX_4bit, GGUF_Q4_K_M
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perf_{model_name}_{format_suffix}_{result.scenario_name}_{timestamp}.jsonl"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        
        print(f"\n[Saved] Results saved to {filepath}")
    
    def save_comparison_csv(
        self,
        results: dict[str, BenchmarkResult],
        filename: str = "comparison.csv",
    ):
        """
        Save a model comparison as CSV.
        
        Args:
            results: Dictionary model_id -> BenchmarkResult
            filename: CSV filename
        """
        import pandas as pd
        
        rows = []
        for model_id, result in results.items():
            metrics = result.metrics
            row = {
                "model": model_id,
                "scenario": result.scenario_name,
                "ttft_mean_ms": metrics.ttft_mean_ms,
                "ttft_p95_ms": metrics.ttft_p95_ms,
                "output_tps_mean": metrics.output_tps_mean,
                "output_tps_p95": metrics.output_tps_p95,
                "peak_ram_mb": metrics.peak_ram_mb,
                "success_rate": metrics.success_rate,
                "json_valid_rate": result.json_valid_rate,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\n[Saved] Comparison saved to {filepath}")
    
    def _compute_confidence_intervals(self, raw_data: list[dict]) -> dict:
        """
        Calculate 95% confidence intervals for key metrics.
        
        Uses bootstrap for robustness (no normality assumption).
        
        Args:
            raw_data: Raw data from runs
            
        Returns:
            Dictionary with CI for each metric
        """
        if not raw_data:
            return {}
        
        # Extract values from successful runs
        successful = [r for r in raw_data if r.get("success", True)]
        if len(successful) < 3:
            return {"error": "insufficient_data"}
        
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        ci_results = {}
        
        # TTFT
        ttft_values = np.array([r["ttft_ms"] for r in successful])
        ci_ttft = analyzer.confidence_interval_bootstrap(ttft_values)
        ci_results["ttft_ms"] = ci_ttft.to_dict()
        
        # Output tokens/s
        tps_values = np.array([r["output_tokens_per_sec"] for r in successful if r["output_tokens_per_sec"] > 0])
        if len(tps_values) > 2:
            ci_tps = analyzer.confidence_interval_bootstrap(tps_values)
            ci_results["output_tokens_per_sec"] = ci_tps.to_dict()
        
        # Total time
        time_values = np.array([r["total_time_ms"] for r in successful])
        ci_time = analyzer.confidence_interval_bootstrap(time_values)
        ci_results["total_time_ms"] = ci_time.to_dict()
        
        return ci_results
    
    def compare_models_statistically(
        self,
        results: dict[str, BenchmarkResult],
        metric: str = "ttft_ms",
    ) -> list:
        """
        Statistically compare multiple models.
        
        Args:
            results: Dict model_id -> BenchmarkResult
            metric: Metric to compare
            
        Returns:
            List of comparisons with significance tests
        """
        analyzer = StatisticalAnalyzer()
        
        # Extract data by model
        models_data = {}
        for model_id, result in results.items():
            raw = result.raw_results
            if metric == "ttft_ms":
                values = [r["metrics"]["ttft_ms"] for r in raw if r.get("success", True)]
            elif metric == "output_tokens_per_sec":
                values = [r["metrics"]["output_tokens_per_sec"] for r in raw if r.get("success", True)]
            else:
                continue
            models_data[model_id] = np.array(values)
        
        if len(models_data) < 2:
            return []
        
        # Compare all models
        comparisons = analyzer.compare_all_models(
            models_data=models_data,
            metric_name=metric,
            paired=True,  # Same prompts used
        )
        
        return comparisons
    
    def _print_summary(self, result: BenchmarkResult):
        """Display a results summary."""
        metrics = result.metrics
        ci = result.confidence_intervals or {}
        
        print(f"\n{'─'*40}")
        print("RESULTS SUMMARY")
        print(f"{'─'*40}")
        
        # TTFT with CI
        ttft_ci = ci.get("ttft_ms", {})
        if ttft_ci and "lower" in ttft_ci:
            print(f"TTFT:           {metrics.ttft_mean_ms:.1f} ms [95% CI: {ttft_ci['lower']:.1f}, {ttft_ci['upper']:.1f}]")
        else:
            print(f"TTFT:           {metrics.ttft_mean_ms:.1f} ± {metrics.ttft_std_ms:.1f} ms")
        
        print(f"TTFT (p95):     {metrics.ttft_p95_ms:.1f} ms")
        
        # Tokens/s with CI
        tps_ci = ci.get("output_tokens_per_sec", {})
        if tps_ci and "lower" in tps_ci:
            print(f"Output t/s:     {metrics.output_tps_mean:.1f} [95% CI: {tps_ci['lower']:.1f}, {tps_ci['upper']:.1f}]")
        else:
            print(f"Output t/s:     {metrics.output_tps_mean:.1f} ± {metrics.output_tps_std:.1f}")
        
        print(f"Peak RAM:       {metrics.peak_ram_mb:.1f} MB")
        print(f"Success rate:   {metrics.success_rate*100:.1f}%")
        
        if result.json_valid_rate is not None:
            print(f"JSON valid:     {result.json_valid_rate*100:.1f}%")
        
        print(f"Duration:       {result.duration_seconds:.1f}s")
        print(f"{'─'*40}")
    
    def _print_comparison(self, results: dict[str, BenchmarkResult], scenario_name: str):
        """Display a model comparison."""
        print(f"\n{'='*60}")
        print(f"COMPARISON - {scenario_name}")
        print(f"{'='*60}")
        
        # Header
        print(f"{'Model':<30} {'TTFT (ms)':<12} {'t/s':<10} {'RAM (MB)':<10}")
        print("-" * 60)
        
        for model_id, result in results.items():
            m = result.metrics
            print(f"{model_id:<30} {m.ttft_mean_ms:<12.1f} {m.output_tps_mean:<10.1f} {m.peak_ram_mb:<10.1f}")
        
        print("=" * 60)


