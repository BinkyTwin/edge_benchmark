#!/usr/bin/env python3
"""
Run Capability Benchmarks
=========================

Script pour exécuter les benchmarks de capacités sur les modèles SLM.

Usage:
    # Banking evaluations (FOCUS PRINCIPAL)
    python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b
    python scripts/run_capability.py --task financial_phrasebank --model google/gemma-3n-e4b
    python scripts/run_capability.py --task banking_all --model google/gemma-3n-e4b
    
    # Realistic scenarios
    python scripts/run_capability.py --task realistic --model google/gemma-3n-e4b
    
    # Coding
    python scripts/run_capability.py --task coding --model google/gemma-3n-e4b
    
    # Mini harness (MMLU + GSM8K)
    python scripts/run_capability.py --task harness --gguf-path /path/to/model.gguf
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lmstudio_client import LMStudioClient


def main():
    parser = argparse.ArgumentParser(
        description="Run SLM capability benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
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
    
    print("=" * 60)
    print("EDGE SLM CAPABILITY BENCHMARK")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"LM Studio URL: {args.base_url}")
    print("=" * 60)
    
    # Initialiser le client LM Studio
    client = LMStudioClient(base_url=args.base_url)
    
    # Vérifier la santé du serveur (sauf pour harness)
    if args.task != "harness":
        health = client.health_check()
        if health["status"] != "healthy":
            print(f"Error: LM Studio server not healthy: {health}")
            sys.exit(1)
        print("\n[OK] LM Studio server is healthy")
    
    try:
        if args.task == "banking77":
            from src.capability.banking_eval import BankingEvaluator
            
            evaluator = BankingEvaluator(client, results_dir=args.results_dir)
            result = evaluator.evaluate_banking77(
                model_id=args.model,
                sample_size=args.sample_size,
                progress_bar=not args.no_progress,
            )
            evaluator.save_result(result)
            
        elif args.task == "financial_phrasebank":
            from src.capability.banking_eval import BankingEvaluator
            
            evaluator = BankingEvaluator(client, results_dir=args.results_dir)
            result = evaluator.evaluate_financial_phrasebank(
                model_id=args.model,
                sample_size=args.sample_size or 1000,
                progress_bar=not args.no_progress,
            )
            evaluator.save_result(result)
            
        elif args.task == "banking_all":
            from src.capability.banking_eval import BankingEvaluator
            
            evaluator = BankingEvaluator(client, results_dir=args.results_dir)
            results = evaluator.evaluate_all(
                model_id=args.model,
                banking77_samples=args.sample_size,
                phrasebank_samples=args.sample_size or 1000,
            )
            
        elif args.task == "realistic":
            from src.capability.realistic_scenarios import RealisticScenariosEvaluator
            
            evaluator = RealisticScenariosEvaluator(client, results_dir=args.results_dir)
            results = evaluator.run_all_scenarios(model_id=args.model)
            
        elif args.task == "coding":
            from src.capability.coding_eval import CodingEvaluator
            
            evaluator = CodingEvaluator(client, results_dir=args.results_dir)
            result = evaluator.evaluate_humaneval(
                model_id=args.model,
                sample_size=args.sample_size or 30,
                progress_bar=not args.no_progress,
            )
            evaluator.save_result(result)
            
        elif args.task == "harness":
            from src.capability.harness_runner import HarnessRunner
            
            if not args.gguf_path:
                print("Error: --gguf-path required for harness task")
                sys.exit(1)
            
            runner = HarnessRunner(results_dir=args.results_dir)
            results = runner.run_mini_benchmark(
                model_path=str(args.gguf_path),
                tasks=["mmlu", "gsm8k"],
            )
            
        elif args.task == "all":
            # Exécuter tous les benchmarks (sauf harness qui nécessite GGUF)
            from src.capability.banking_eval import BankingEvaluator
            from src.capability.realistic_scenarios import RealisticScenariosEvaluator
            from src.capability.coding_eval import CodingEvaluator
            
            print("\n[1/3] Banking Evaluations...")
            banking_eval = BankingEvaluator(client, results_dir=args.results_dir)
            banking_results = banking_eval.evaluate_all(model_id=args.model)
            
            print("\n[2/3] Realistic Scenarios...")
            realistic_eval = RealisticScenariosEvaluator(client, results_dir=args.results_dir)
            realistic_results = realistic_eval.run_all_scenarios(model_id=args.model)
            
            print("\n[3/3] Coding Evaluation...")
            coding_eval = CodingEvaluator(client, results_dir=args.results_dir)
            coding_result = coding_eval.evaluate_humaneval(model_id=args.model)
            coding_eval.save_result(coding_result)
    
    finally:
        client.close()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()

