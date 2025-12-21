"""
Performance Runner
==================

Orchestrateur principal pour les benchmarks de performance.
Exécute les scénarios avec collecte de métriques et logging.
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
    """Configuration d'un benchmark."""
    
    warmup_runs: int = 3
    benchmark_runs: int = 20
    cooldown_seconds: float = 2.0
    stream: bool = True
    temperature: float = 0
    top_p: float = 1
    seed: int = 42  # Seed pour reproductibilité


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark complet."""
    
    model_id: str
    scenario_name: str
    config: BenchmarkConfig
    metrics: AggregatedMetrics
    raw_results: list[dict]
    timestamp: str
    duration_seconds: float
    
    # Métriques spécifiques au scénario
    json_valid_rate: Optional[float] = None
    
    # Statistiques avancées (IC 95%)
    confidence_intervals: Optional[dict] = None
    
    # Environnement pour reproductibilité
    environment: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "model_id": self.model_id,
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
    Orchestrateur de benchmarks de performance.
    
    Gère l'exécution des scénarios, la collecte des métriques,
    et le logging des résultats.
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        scenarios_config_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ):
        """
        Args:
            client: Client LM Studio
            scenarios_config_path: Chemin vers scenarios.yaml
            results_dir: Dossier pour les résultats
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
    ) -> BenchmarkResult:
        """
        Exécute un scénario de benchmark pour un modèle.
        
        Args:
            model_id: ID du modèle LM Studio
            scenario_name: Nom du scénario à exécuter
            config: Configuration du benchmark
            progress_bar: Afficher une barre de progression
            
        Returns:
            BenchmarkResult avec métriques agrégées
        """
        config = config or BenchmarkConfig()
        scenario = self.scenario_executor.get_scenario(scenario_name)
        
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        print(f"\n{'='*60}")
        print(f"Benchmark: {scenario.name}")
        print(f"Model: {model_id}")
        print(f"Category: {scenario.category}")
        print(f"{'='*60}")
        
        start_time = time.time()
        collector = MetricsCollector()
        raw_results = []
        json_valid_count = 0
        
        # Obtenir les prompts
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
            # Monitoring mémoire
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
            
            # Collecter les métriques
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
            
            # Valider JSON si applicable
            if prompt_data.get("response_format"):
                validation = self.scenario_executor.validate_json_output(
                    result.content,
                    prompt_data.get("expected_fields", [])
                )
                if validation["is_valid"]:
                    json_valid_count += 1
            
            # Stocker le résultat brut
            raw_results.append({
                "content_preview": result.content[:200] if result.content else "",
                "metrics": result.metrics.to_dict(),
                "success": result.success,
                "error": result.error,
                "json_valid": result.json_valid if prompt_data.get("response_format") else None,
            })
            
            # Cooldown
            time.sleep(config.cooldown_seconds)
        
        # Agréger les métriques
        aggregated = collector.aggregate()
        duration = time.time() - start_time
        
        # Calculer le taux de JSON valides
        json_valid_rate = None
        if scenario.use_structured_output:
            json_valid_rate = json_valid_count / len(prompts) if prompts else 0
        
        # Calculer les intervalles de confiance (IC 95%)
        confidence_intervals = self._compute_confidence_intervals(collector.get_raw_data())
        
        result = BenchmarkResult(
            model_id=model_id,
            scenario_name=scenario_name,
            config=config,
            metrics=aggregated,
            raw_results=raw_results,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            json_valid_rate=json_valid_rate,
            confidence_intervals=confidence_intervals,
        )
        
        # Afficher le résumé
        self._print_summary(result)
        
        return result
    
    def run_all_scenarios(
        self,
        model_id: str,
        scenarios: Optional[list[str]] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> list[BenchmarkResult]:
        """
        Exécute tous les scénarios pour un modèle.
        
        Args:
            model_id: ID du modèle
            scenarios: Liste de scénarios (None = tous)
            config: Configuration du benchmark
            
        Returns:
            Liste de BenchmarkResult
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
        Compare plusieurs modèles sur un scénario.
        
        Args:
            model_ids: Liste des IDs de modèles
            scenario_name: Nom du scénario
            config: Configuration du benchmark
            
        Returns:
            Dictionnaire model_id -> BenchmarkResult
        """
        results = {}
        
        for model_id in model_ids:
            print(f"\n{'#'*60}")
            print(f"# Model: {model_id}")
            print(f"{'#'*60}")
            
            result = self.run_scenario(model_id, scenario_name, config)
            results[model_id] = result
            self.save_result(result)
        
        # Afficher la comparaison
        self._print_comparison(results, scenario_name)
        
        return results
    
    def save_result(self, result: BenchmarkResult, filename: Optional[str] = None):
        """
        Sauvegarde un résultat en JSONL.
        
        Args:
            result: Résultat à sauvegarder
            filename: Nom du fichier (auto-généré si None)
        """
        if filename is None:
            model_name = result.model_id.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perf_{model_name}_{result.scenario_name}_{timestamp}.jsonl"
        
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
        Sauvegarde une comparaison de modèles en CSV.
        
        Args:
            results: Dictionnaire model_id -> BenchmarkResult
            filename: Nom du fichier CSV
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
        Calcule les intervalles de confiance 95% pour les métriques clés.
        
        Utilise bootstrap pour robustesse (pas d'hypothèse de normalité).
        
        Args:
            raw_data: Données brutes des runs
            
        Returns:
            Dictionnaire avec IC pour chaque métrique
        """
        if not raw_data:
            return {}
        
        # Extraire les valeurs des runs réussis
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
        Compare statistiquement plusieurs modèles.
        
        Args:
            results: Dict model_id -> BenchmarkResult
            metric: Métrique à comparer
            
        Returns:
            Liste de comparaisons avec tests de significativité
        """
        analyzer = StatisticalAnalyzer()
        
        # Extraire les données par modèle
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
        
        # Comparer tous les modèles
        comparisons = analyzer.compare_all_models(
            models_data=models_data,
            metric_name=metric,
            paired=True,  # Mêmes prompts utilisés
        )
        
        return comparisons
    
    def _print_summary(self, result: BenchmarkResult):
        """Affiche un résumé des résultats."""
        metrics = result.metrics
        ci = result.confidence_intervals or {}
        
        print(f"\n{'─'*40}")
        print("RESULTS SUMMARY")
        print(f"{'─'*40}")
        
        # TTFT avec IC
        ttft_ci = ci.get("ttft_ms", {})
        if ttft_ci and "lower" in ttft_ci:
            print(f"TTFT:           {metrics.ttft_mean_ms:.1f} ms [95% CI: {ttft_ci['lower']:.1f}, {ttft_ci['upper']:.1f}]")
        else:
            print(f"TTFT:           {metrics.ttft_mean_ms:.1f} ± {metrics.ttft_std_ms:.1f} ms")
        
        print(f"TTFT (p95):     {metrics.ttft_p95_ms:.1f} ms")
        
        # Tokens/s avec IC
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
        """Affiche une comparaison des modèles."""
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


