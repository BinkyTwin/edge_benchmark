#!/usr/bin/env python3
"""
Run Performance Benchmarks
==========================

Script to run performance benchmarks on SLM models.

Features:
- Model-by-model or batch execution
- Automatic checkpoint after each model/scenario
- Automatic resume after crash (--resume)
- Error protection (continues even if a model fails)

Usage:
    # All models, all scenarios
    python scripts/run_performance.py --models all --scenarios all
    
    # Single model
    python scripts/run_performance.py --models gemma_3n_e4b_gguf --scenarios interactive_assistant
    
    # Resume after crash
    python scripts/run_performance.py --resume
    
    # Comparison mode
    python scripts/run_performance.py --compare --scenario interactive_assistant
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.lmstudio_client import LMStudioClient
from src.performance.runner import PerformanceRunner, BenchmarkConfig
from src.checkpoint import CheckpointManager, run_with_checkpoint


from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Complete model information for benchmarking."""
    id: str                    # ID LM Studio (ex: google/gemma-3n-e4b)
    config_key: str            # Clé dans models.yaml (ex: gemma_3n_e4b_gguf)
    display_name: str          # Nom d'affichage (ex: Gemma 3n E4B (GGUF Q4_K_M))
    format: str                # Format: "mlx" ou "gguf"
    quantization: str          # Quantization: "4bit", "Q4_K_M", etc.
    
    def get_filename_suffix(self) -> str:
        """Return suffix for filenames (e.g., MLX_4bit, GGUF_Q4_K_M)."""
        return f"{self.format.upper()}_{self.quantization}"
    
    def __str__(self) -> str:
        return f"{self.id} ({self.format.upper()} {self.quantization})"


def load_models_config(config_path: Path) -> dict:
    """Load models configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_models(models_config: dict, selection: str) -> list[ModelInfo]:
    """
    Get models with their metadata based on selection.
    
    Returns:
        List of ModelInfo with all necessary metadata
    """
    models = models_config.get("models", {})
    groups = models_config.get("model_groups", {})
    
    def create_model_info(config_key: str, model_data: dict) -> ModelInfo:
        return ModelInfo(
            id=model_data["id"],
            config_key=config_key,
            display_name=model_data.get("display_name", model_data["id"]),
            format=model_data.get("format", "unknown"),
            quantization=model_data.get("quantization", "unknown"),
        )
    
    if selection == "all":
        return [create_model_info(k, m) for k, m in models.items()]
    elif selection == "gguf":
        model_keys = groups.get("gguf_only", [])
        return [create_model_info(k, models[k]) for k in model_keys if k in models]
    elif selection == "mlx":
        model_keys = groups.get("mlx_only", [])
        return [create_model_info(k, models[k]) for k in model_keys if k in models]
    elif selection in models:
        return [create_model_info(selection, models[selection])]
    else:
            # Assume it's a direct LM Studio ID - unknown format
        return [ModelInfo(
            id=selection,
            config_key=selection,
            display_name=selection,
            format="unknown",
            quantization="unknown",
        )]


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint system",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    models_config_path = args.config_dir / "models.yaml"
    scenarios_config_path = args.config_dir / "scenarios.yaml"
    
    if not models_config_path.exists():
        print(f"Error: Models config not found at {models_config_path}")
        sys.exit(1)
    
    models_config = load_models_config(models_config_path)
    models = get_models(models_config, args.models)
    
    # Determine scenarios
    from src.performance.scenarios import ScenarioExecutor
    scenario_executor = ScenarioExecutor(scenarios_config_path)
    if args.scenarios == "all":
        scenarios = scenario_executor.list_scenarios()
    else:
        scenarios = [args.scenarios]
    
    print("=" * 60)
    print("EDGE SLM PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  - {m.display_name} [{m.format.upper()}]")
    print(f"Scenarios: {scenarios}")
    print(f"Runs per scenario: {args.runs}")
    print(f"Seed: {args.seed}")
    print(f"LM Studio URL: {args.base_url}")
    print("=" * 60)
    
    # Benchmark configuration
    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        cooldown_seconds=args.cooldown,
        stream=True,
        temperature=0,
        top_p=1,
        seed=args.seed,
    )
    
    # Initialize checkpoint manager
    # Note: We use config_key for checkpoints as it's unique (includes format)
    model_keys = [m.config_key for m in models]
    
    checkpoint = None
    if not args.no_checkpoint:
        checkpoint = CheckpointManager(results_dir=args.results_dir)
        
        if args.resume:
            state = checkpoint.resume_latest()
            if state:
                # Filter models according to checkpoint
                remaining_keys = set(state.planned_models) - set(k for k, _, _ in state.completed)
                models = [m for m in models if m.config_key in remaining_keys or m.config_key in state.planned_models]
                scenarios = state.planned_scenarios
            else:
                # Nouveau checkpoint
                checkpoint.start_experiment(
                    task_type="performance",
                    models=model_keys,
                    scenarios=scenarios,
                )
        else:
            checkpoint.start_experiment(
                task_type="performance",
                models=model_keys,
                scenarios=scenarios,
            )
    
    # Initialize client and runner
    with LMStudioClient(base_url=args.base_url) as client:
        # Check server health
        health = client.health_check()
        if health["status"] != "healthy":
            print(f"Error: LM Studio server not healthy: {health}")
            sys.exit(1)
        
        print("\n[OK] LM Studio server is healthy")
        
        # List available models
        available_models = client.list_models()
        print(f"[OK] Available models: {len(available_models)}")
        
        runner = PerformanceRunner(
            client=client,
            scenarios_config_path=scenarios_config_path,
            results_dir=args.results_dir,
        )
        
        all_results = {}
        failed_count = 0
        
        if args.compare:
            # Comparison mode: all models on one scenario
            scenario = scenarios[0] if scenarios else "interactive_assistant"
            
            for model in models:
                if checkpoint and checkpoint.should_skip(model.config_key, scenario):
                    print(f"\n[Skip] {model.config_key}/{scenario} already completed")
                    continue
                
                try:
                    print(f"\n{'#'*60}")
                    print(f"# Model: {model.display_name}")
                    print(f"# Format: {model.format.upper()} | Quantization: {model.quantization}")
                    print(f"{'#'*60}")
                    
                    # Display instructions for the user
                    print(f"\n⚠️  IMPORTANT: Make sure the following model is loaded in LM Studio:")
                    print(f"    ID: {model.id}")
                    print(f"    Expected format: {model.format.upper()} {model.quantization}")
                    print("")
                    
                    result = runner.run_scenario(
                        model_id=model.id,
                        scenario_name=scenario,
                        config=config,
                        model_info={
                            "config_key": model.config_key,
                            "format": model.format,
                            "quantization": model.quantization,
                            "display_name": model.display_name,
                        }
                    )
                    runner.save_result(result)
                    all_results[model.config_key] = result
                    
                    if checkpoint:
                        checkpoint.mark_completed(model.config_key, scenario, 
                            f"perf_{model.id.replace('/', '_')}_{model.get_filename_suffix()}_{scenario}.jsonl")
                    
                except KeyboardInterrupt:
                    print("\n[Interrupted] Checkpoint saved. Use --resume to continue.")
                    sys.exit(1)
                    
                except Exception as e:
                    print(f"\n[Error] {model.config_key}: {e}")
                    failed_count += 1
                    if checkpoint:
                        checkpoint.mark_failed(model.config_key, scenario, str(e))
                    # Continue with the next model
                    continue
            
            # Display comparison
            if all_results:
                runner._print_comparison(all_results, scenario)
                runner.save_comparison_csv(all_results)
        
        else:
            # Standard mode: each model on selected scenarios
            for model in models:
                for scenario_name in scenarios:
                    if checkpoint and checkpoint.should_skip(model.config_key, scenario_name):
                        print(f"\n[Skip] {model.config_key}/{scenario_name} already completed")
                        continue
                    
                    try:
                        print(f"\n{'#'*60}")
                        print(f"# Model: {model.display_name}")
                        print(f"# Format: {model.format.upper()} | Quantization: {model.quantization}")
                        print(f"# Scenario: {scenario_name}")
                        print(f"{'#'*60}")
                        
                        # Display instructions for the user
                        print(f"\n⚠️  IMPORTANT: Make sure the following model is loaded in LM Studio:")
                        print(f"    ID: {model.id}")
                        print(f"    Expected format: {model.format.upper()} {model.quantization}")
                        print("")
                        
                        result = runner.run_scenario(
                            model_id=model.id,
                            scenario_name=scenario_name,
                            config=config,
                            model_info={
                                "config_key": model.config_key,
                                "format": model.format,
                                "quantization": model.quantization,
                                "display_name": model.display_name,
                            }
                        )
                        runner.save_result(result)
                        
                        if checkpoint:
                            checkpoint.mark_completed(model.config_key, scenario_name, 
                                f"perf_{model.id.replace('/', '_')}_{model.get_filename_suffix()}_{scenario_name}.jsonl")
                        
                    except KeyboardInterrupt:
                        print("\n[Interrupted] Checkpoint saved. Use --resume to continue.")
                        sys.exit(1)
                        
                    except Exception as e:
                        print(f"\n[Error] {model.config_key}/{scenario_name}: {e}")
                        failed_count += 1
                        if checkpoint:
                            checkpoint.mark_failed(model.config_key, scenario_name, str(e))
                        # Continue with the next one
                        continue
    
    # Final summary
    if checkpoint:
        checkpoint.print_summary()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    if failed_count > 0:
        print(f"⚠️  {failed_count} model(s)/scenario(s) failed (see checkpoint for details)")
    print(f"Results saved to: {args.results_dir}")
    
    if checkpoint and checkpoint.state:
        remaining = checkpoint.state.get_remaining()
        if remaining:
            print(f"\n💡 To complete remaining tasks, run:")
            print(f"   python scripts/run_performance.py --resume")


if __name__ == "__main__":
    main()
