#!/usr/bin/env python3
"""
Run Capability Benchmarks
=========================

Script pour exécuter les benchmarks de capacités sur les modèles SLM.

Features:
- Exécution modèle par modèle
- Checkpoint automatique après chaque tâche
- Reprise automatique en cas de crash (--resume)
- Protection contre les erreurs

Usage:
    # Banking evaluations (FOCUS PRINCIPAL)
    python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b
    python scripts/run_capability.py --task financial_phrasebank --model google/gemma-3n-e4b
    python scripts/run_capability.py --task banking_all --model google/gemma-3n-e4b
    
    # Tous les modèles sur une tâche
    python scripts/run_capability.py --task banking77 --all-models
    
    # Realistic scenarios
    python scripts/run_capability.py --task realistic --model google/gemma-3n-e4b
    
    # Coding
    python scripts/run_capability.py --task coding --model google/gemma-3n-e4b
    
    # Mini harness (MMLU + GSM8K)
    python scripts/run_capability.py --task harness --gguf-path /path/to/model.gguf
    
    # Reprendre après un crash
    python scripts/run_capability.py --resume
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.lmstudio_client import LMStudioClient
from src.checkpoint import CheckpointManager


def load_models_config(config_path: Path) -> dict:
    """Charge la configuration des modèles."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_all_model_ids(config_path: Path) -> list[str]:
    """Récupère tous les IDs de modèles depuis la config."""
    models_config = load_models_config(config_path)
    models = models_config.get("models", {})
    return [m["id"] for m in models.values()]


def main():
    parser = argparse.ArgumentParser(
        description="Run SLM capability benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="banking77",
        choices=[
            "banking77",
            "financial_phrasebank", 
            "banking_all",
            "realistic",
            "coding",
            "harness",
            "all",
        ],
        help="Task to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3n-e4b",
        help="Model ID for LM Studio",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run on all models from config",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        help="Path to GGUF file (for harness task)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples (None = full dataset)",
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
    
    # Déterminer les modèles à évaluer
    if args.all_models:
        models_config_path = args.config_dir / "models.yaml"
        model_ids = get_all_model_ids(models_config_path)
    else:
        model_ids = [args.model]
    
    # Déterminer les tâches
    if args.task == "all":
        tasks = ["banking77", "financial_phrasebank", "realistic", "coding"]
    else:
        tasks = [args.task]
    
    print("=" * 60)
    print("EDGE SLM CAPABILITY BENCHMARK")
    print("=" * 60)
    print(f"Task(s): {tasks}")
    print(f"Model(s): {model_ids}")
    print(f"LM Studio URL: {args.base_url}")
    print("=" * 60)
    
    # Initialiser le checkpoint manager
    checkpoint = None
    if not args.no_checkpoint:
        checkpoint = CheckpointManager(results_dir=args.results_dir)
        
        if args.resume:
            state = checkpoint.resume_latest()
            if state:
                model_ids = state.planned_models
                tasks = state.planned_tasks
        else:
            checkpoint.start_experiment(
                task_type="capability",
                models=model_ids,
                tasks=tasks,
            )
    
    # Initialiser le client LM Studio
    client = LMStudioClient(base_url=args.base_url)
    
    # Vérifier la santé du serveur (sauf pour harness)
    if args.task != "harness":
        health = client.health_check()
        if health["status"] != "healthy":
            print(f"Error: LM Studio server not healthy: {health}")
            sys.exit(1)
        print("\n[OK] LM Studio server is healthy")
    
    failed_count = 0
    
    try:
        for model_id in model_ids:
            for task in tasks:
                # Vérifier si déjà complété
                if checkpoint and checkpoint.should_skip(model_id, task):
                    print(f"\n[Skip] {model_id}/{task} already completed")
                    continue
                
                try:
                    print(f"\n{'#'*60}")
                    print(f"# Model: {model_id} | Task: {task}")
                    print(f"{'#'*60}")
                    
                    result_file = run_single_task(
                        task=task,
                        model_id=model_id,
                        client=client,
                        args=args,
                    )
                    
                    if checkpoint:
                        checkpoint.mark_completed(model_id, task, result_file)
                    
                except KeyboardInterrupt:
                    print("\n[Interrupted] Checkpoint saved. Use --resume to continue.")
                    if checkpoint:
                        checkpoint.mark_failed(model_id, task, "KeyboardInterrupt")
                    raise
                    
                except Exception as e:
                    print(f"\n[Error] {model_id}/{task}: {e}")
                    failed_count += 1
                    if checkpoint:
                        checkpoint.mark_failed(model_id, task, str(e))
                    # Continue avec le prochain
                    continue
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    
    finally:
        client.close()
    
    # Résumé final
    if checkpoint:
        checkpoint.print_summary()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    if failed_count > 0:
        print(f"⚠️  {failed_count} task(s) failed (see checkpoint for details)")
    print(f"Results saved to: {args.results_dir}")
    
    if checkpoint and checkpoint.state:
        remaining = checkpoint.state.get_remaining()
        if remaining:
            print(f"\n💡 To complete remaining tasks, run:")
            print(f"   python scripts/run_capability.py --resume")


def run_single_task(task: str, model_id: str, client: LMStudioClient, args) -> str:
    """
    Exécute une seule tâche et retourne le nom du fichier de résultat.
    """
    result_file = ""
    
    if task == "banking77":
        from src.capability.banking_eval import BankingEvaluator
        
        evaluator = BankingEvaluator(client, results_dir=args.results_dir)
        result = evaluator.evaluate_banking77(
            model_id=model_id,
            sample_size=args.sample_size,
            progress_bar=not args.no_progress,
        )
        evaluator.save_result(result)
        result_file = f"eval_banking77_{model_id.replace('/', '_')}.json"
        
    elif task == "financial_phrasebank":
        from src.capability.banking_eval import BankingEvaluator
        
        evaluator = BankingEvaluator(client, results_dir=args.results_dir)
        result = evaluator.evaluate_financial_phrasebank(
            model_id=model_id,
            sample_size=args.sample_size or 1000,
            progress_bar=not args.no_progress,
        )
        evaluator.save_result(result)
        result_file = f"eval_financial_phrasebank_{model_id.replace('/', '_')}.json"
        
    elif task == "banking_all":
        from src.capability.banking_eval import BankingEvaluator
        
        evaluator = BankingEvaluator(client, results_dir=args.results_dir)
        results = evaluator.evaluate_all(
            model_id=model_id,
            banking77_samples=args.sample_size,
            phrasebank_samples=args.sample_size or 1000,
        )
        result_file = f"eval_banking_all_{model_id.replace('/', '_')}.json"
        
    elif task == "realistic":
        from src.capability.realistic_scenarios import RealisticScenariosEvaluator
        
        evaluator = RealisticScenariosEvaluator(client, results_dir=args.results_dir)
        results = evaluator.run_all_scenarios(model_id=model_id)
        result_file = f"realistic_scenarios_{model_id.replace('/', '_')}.json"
        
    elif task == "coding":
        from src.capability.coding_eval import CodingEvaluator
        
        evaluator = CodingEvaluator(client, results_dir=args.results_dir)
        result = evaluator.evaluate_humaneval(
            model_id=model_id,
            sample_size=args.sample_size or 30,
            progress_bar=not args.no_progress,
        )
        evaluator.save_result(result)
        result_file = f"coding_humaneval_{model_id.replace('/', '_')}.json"
        
    elif task == "harness":
        from src.capability.harness_runner import HarnessRunner
        
        if not args.gguf_path:
            raise ValueError("--gguf-path required for harness task")
        
        runner = HarnessRunner(results_dir=args.results_dir)
        results = runner.run_mini_benchmark(
            model_path=str(args.gguf_path),
            tasks=["mmlu", "gsm8k"],
        )
        result_file = f"harness_{args.gguf_path.stem}.json"
    
    return result_file


if __name__ == "__main__":
    main()
