#!/usr/bin/env python3
"""
Run Performance Benchmarks
==========================

Script pour exécuter les benchmarks de performance sur les modèles SLM.

Usage:
    python scripts/run_performance.py --models all --scenarios all
    python scripts/run_performance.py --models gemma_3n_e4b_gguf --scenarios interactive_assistant
    python scripts/run_performance.py --compare --scenario interactive_assistant
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.lmstudio_client import LMStudioClient
from src.performance.runner import PerformanceRunner, BenchmarkConfig


def load_models_config(config_path: Path) -> dict:
    """Charge la configuration des modèles."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_ids(models_config: dict, selection: str) -> list[str]:
    """
    Récupère les IDs des modèles selon la sélection.
    
    Args:
        models_config: Configuration des modèles
        selection: 'all', 'gguf', 'mlx', ou un ID spécifique
        
    Returns:
        Liste des IDs de modèles LM Studio
    """
    models = models_config.get("models", {})
    groups = models_config.get("model_groups", {})
    
    if selection == "all":
        return [m["id"] for m in models.values()]
    elif selection == "gguf":
        model_keys = groups.get("gguf_only", [])
        return [models[k]["id"] for k in model_keys if k in models]
    elif selection == "mlx":
        model_keys = groups.get("mlx_only", [])
        return [models[k]["id"] for k in model_keys if k in models]
    elif selection in models:
        return [models[selection]["id"]]
    else:
        # Supposer que c'est un ID LM Studio direct
        return [selection]


def main():
    parser = argparse.ArgumentParser(
        description="Run SLM performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to benchmark: 'all', 'gguf', 'mlx', or specific model key/ID",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Scenarios: 'all' or specific scenario name",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison mode (all models on one scenario)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark runs per scenario (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown between runs in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio API base URL",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "configs",
        help="Path to configs directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    
    args = parser.parse_args()
    
    # Charger la configuration
    models_config_path = args.config_dir / "models.yaml"
    scenarios_config_path = args.config_dir / "scenarios.yaml"
    
    if not models_config_path.exists():
        print(f"Error: Models config not found at {models_config_path}")
        sys.exit(1)
    
    models_config = load_models_config(models_config_path)
    model_ids = get_model_ids(models_config, args.models)
    
    print("=" * 60)
    print("EDGE SLM PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Models: {model_ids}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Runs per scenario: {args.runs}")
    print(f"LM Studio URL: {args.base_url}")
    print("=" * 60)
    
    # Configuration du benchmark
    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        cooldown_seconds=args.cooldown,
        stream=True,
        temperature=0,
        top_p=1,
    )
    
    # Initialiser le client et le runner
    with LMStudioClient(base_url=args.base_url) as client:
        # Vérifier la santé du serveur
        health = client.health_check()
        if health["status"] != "healthy":
            print(f"Error: LM Studio server not healthy: {health}")
            sys.exit(1)
        
        print("\n[OK] LM Studio server is healthy")
        
        # Lister les modèles disponibles
        available_models = client.list_models()
        print(f"[OK] Available models: {len(available_models)}")
        
        runner = PerformanceRunner(
            client=client,
            scenarios_config_path=scenarios_config_path,
            results_dir=args.results_dir,
        )
        
        if args.compare:
            # Mode comparaison: tous les modèles sur un scénario
            scenario = args.scenarios if args.scenarios != "all" else "interactive_assistant"
            results = runner.run_model_comparison(
                model_ids=model_ids,
                scenario_name=scenario,
                config=config,
            )
            runner.save_comparison_csv(results)
            
        else:
            # Mode standard: chaque modèle sur les scénarios sélectionnés
            scenarios = None if args.scenarios == "all" else [args.scenarios]
            
            all_results = []
            for model_id in model_ids:
                results = runner.run_all_scenarios(
                    model_id=model_id,
                    scenarios=scenarios,
                    config=config,
                )
                all_results.extend(results)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()


