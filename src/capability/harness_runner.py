"""
LM Evaluation Harness Runner
============================

Mini-benchmark via lm-evaluation-harness for quick validation.
Executes MMLU and GSM8K subsets directly on GGUF files.

Objective: quick capability validation (~30min-1h per model),
not an exhaustive benchmark.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class HarnessConfig:
    """Configuration for lm-evaluation-harness."""
    
    task: str
    num_fewshot: int = 5
    limit: Optional[int] = None
    batch_size: int = 4
    device: str = "mps"  # Apple Silicon
    
    def to_args(self) -> list[str]:
        """Converts to CLI arguments."""
        args = [
            "--tasks", self.task,
            "--num_fewshot", str(self.num_fewshot),
            "--batch_size", str(self.batch_size),
            "--device", self.device,
        ]
        if self.limit:
            args.extend(["--limit", str(self.limit)])
        return args


@dataclass
class HarnessResult:
    """Result of a harness evaluation."""
    
    task: str
    model_path: str
    
    # Metrics
    accuracy: float = 0.0
    accuracy_stderr: float = 0.0
    
    # Other metrics (task-specific)
    metrics: dict = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    duration_seconds: float = 0.0
    timestamp: str = ""
    
    # Details
    config: dict = field(default_factory=dict)
    raw_output: str = ""
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model_path": self.model_path,
            "accuracy": round(self.accuracy, 4),
            "accuracy_stderr": round(self.accuracy_stderr, 4),
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
            "config": self.config,
        }


class HarnessRunner:
    """
    Runner for lm-evaluation-harness.
    
    Executes academic benchmarks (MMLU, GSM8K) directly
    on GGUF files, bypassing LM Studio logprobs limitation.
    """
    
    # Supported tasks with their default configurations
    SUPPORTED_TASKS = {
        "mmlu": {
            "task_name": "mmlu",
            "num_fewshot": 5,
            "limit": 200,  # Subset pour rapidité
            "description": "Massive Multitask Language Understanding",
        },
        "gsm8k": {
            "task_name": "gsm8k",
            "num_fewshot": 5,
            "limit": 100,
            "description": "Grade School Math 8K",
        },
        "arc_challenge": {
            "task_name": "arc_challenge",
            "num_fewshot": 25,
            "limit": 100,
            "description": "AI2 Reasoning Challenge",
        },
        "hellaswag": {
            "task_name": "hellaswag",
            "num_fewshot": 10,
            "limit": 200,
            "description": "HellaSwag commonsense",
        },
    }
    
    def __init__(
        self,
        results_dir: Optional[Path] = None,
        gguf_models_dir: Optional[Path] = None,
    ):
        """
        Args:
            results_dir: Directory for results
            gguf_models_dir: Directory containing GGUF files
        """
        self.results_dir = results_dir or Path("results")
        self.gguf_models_dir = gguf_models_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def check_harness_installed(self) -> bool:
        """Checks if lm-evaluation-harness is installed."""
        try:
            result = subprocess.run(
                ["lm_eval", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_evaluation(
        self,
        model_path: str,
        task: str,
        config: Optional[HarnessConfig] = None,
        verbose: bool = True,
    ) -> HarnessResult:
        """
        Executes a harness evaluation on a GGUF model.
        
        Args:
            model_path: Path to the GGUF file
            task: Task name (mmlu, gsm8k, etc.)
            config: Harness configuration
            verbose: Display output
            
        Returns:
            HarnessResult with metrics
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task '{task}' not supported. Choose from: {list(self.SUPPORTED_TASKS.keys())}")
        
        task_info = self.SUPPORTED_TASKS[task]
        
        if config is None:
            config = HarnessConfig(
                task=task_info["task_name"],
                num_fewshot=task_info["num_fewshot"],
                limit=task_info["limit"],
            )
        
        print(f"\n{'='*60}")
        print(f"LM-EVALUATION-HARNESS")
        print(f"Task: {task} ({task_info['description']})")
        print(f"Model: {model_path}")
        print(f"Limit: {config.limit} samples")
        print(f"{'='*60}")
        
        start_time = time.time()
        
            # Build the command
        cmd = [
            "lm_eval",
            "--model", "gguf",
            "--model_args", f"pretrained={model_path}",
            "--output_path", str(self.results_dir / "harness_output"),
            "--log_samples",
        ]
        cmd.extend(config.to_args())
        
        if verbose:
            print(f"\n[CMD] {' '.join(cmd)}")
        
            # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 heure max
            )
            
            duration = time.time() - start_time
            
            if result.returncode != 0:
                print(f"[ERROR] Harness failed: {result.stderr}")
                return HarnessResult(
                    task=task,
                    model_path=model_path,
                    raw_output=result.stderr,
                    timestamp=datetime.now().isoformat(),
                )
            
                # Parse results
            harness_result = self._parse_output(
                result.stdout,
                task,
                model_path,
                duration,
                config,
            )
            
            if verbose:
                self._print_results(harness_result)
            
            return harness_result
            
        except subprocess.TimeoutExpired:
            print("[ERROR] Harness timed out after 1 hour")
            return HarnessResult(
                task=task,
                model_path=model_path,
                raw_output="Timeout",
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            print(f"[ERROR] Harness failed: {e}")
            return HarnessResult(
                task=task,
                model_path=model_path,
                raw_output=str(e),
                timestamp=datetime.now().isoformat(),
            )
    
    def run_mini_benchmark(
        self,
        model_path: str,
        tasks: Optional[list[str]] = None,
    ) -> dict[str, HarnessResult]:
        """
        Executes the complete mini-benchmark (MMLU + GSM8K subsets).
        
        Args:
            model_path: Path to the GGUF file
            tasks: Task list (default: mmlu, gsm8k)
            
        Returns:
            Dictionary task -> HarnessResult
        """
        tasks = tasks or ["mmlu", "gsm8k"]
        results = {}
        
        print(f"\n{'#'*60}")
        print(f"# MINI-BENCHMARK LM-EVALUATION-HARNESS")
        print(f"# Model: {model_path}")
        print(f"# Tasks: {tasks}")
        print(f"{'#'*60}")
        
        # Check installation
        if not self.check_harness_installed():
            print("\n[ERROR] lm-evaluation-harness is not installed!")
            print("Install with: pip install lm-eval")
            return results
        
        for task in tasks:
            print(f"\n[{tasks.index(task)+1}/{len(tasks)}] Running {task}...")
            result = self.run_evaluation(model_path, task)
            results[task] = result
            self.save_result(result)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _parse_output(
        self,
        output: str,
        task: str,
        model_path: str,
        duration: float,
        config: HarnessConfig,
    ) -> HarnessResult:
        """Parses the harness output."""
        result = HarnessResult(
            task=task,
            model_path=model_path,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat(),
            config={"limit": config.limit, "num_fewshot": config.num_fewshot},
            raw_output=output[-2000:],  # Garder les 2000 derniers caractères
        )
        
        # Parse metrics from the output
        # Typical format is:
        # |    Tasks    |Version|Filter|n-shot|Metric|Value |   |Stderr|
        # |-------------|-------|------|------|------|------|---|------|
        # |mmlu         |      1|none  |     5|acc   |0.5123|±  |0.0234|
        
        lines = output.split('\n')
        for line in lines:
            if '|' in line and task.lower() in line.lower():
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 8:
                    try:
                        # Find the Value column
                        for i, part in enumerate(parts):
                            if part.replace('.', '').replace('-', '').isdigit():
                                result.accuracy = float(part)
                                # Find stderr
                                if i + 2 < len(parts) and parts[i+1] == '±':
                                    result.accuracy_stderr = float(parts[i+2])
                                break
                    except (ValueError, IndexError):
                        pass
        
        # Find the number of samples
        for line in lines:
            if "samples" in line.lower() or "examples" in line.lower():
                import re
                match = re.search(r'(\d+)\s*(samples|examples)', line.lower())
                if match:
                    result.num_samples = int(match.group(1))
                    break
        
        if result.num_samples == 0 and config.limit:
            result.num_samples = config.limit
        
        return result
    
    def save_result(self, result: HarnessResult, filename: Optional[str] = None):
        """Saves a result."""
        if filename is None:
            model_name = Path(result.model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"harness_{result.task}_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"[Saved] Results saved to {filepath}")
    
    def _print_results(self, result: HarnessResult):
        """Displays the results."""
        print(f"\n{'─'*40}")
        print("RESULTS")
        print(f"{'─'*40}")
        print(f"Task:       {result.task}")
        print(f"Accuracy:   {result.accuracy*100:.2f}% ± {result.accuracy_stderr*100:.2f}%")
        print(f"Samples:    {result.num_samples}")
        print(f"Duration:   {result.duration_seconds:.1f}s")
        print(f"{'─'*40}")
    
    def _print_summary(self, results: dict[str, HarnessResult]):
        """Displays the mini-benchmark summary."""
        print(f"\n{'='*60}")
        print("MINI-BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n{'Task':<20} {'Accuracy':<15} {'Stderr':<10} {'Duration':<10}")
        print("-" * 55)
        
        for task, result in results.items():
            print(f"{task:<20} {result.accuracy*100:.2f}%{'':<7} ±{result.accuracy_stderr*100:.2f}%{'':<3} {result.duration_seconds:.0f}s")
        
        total_duration = sum(r.duration_seconds for r in results.values())
        avg_accuracy = sum(r.accuracy for r in results.values()) / len(results) if results else 0
        
        print("-" * 55)
        print(f"{'Average':<20} {avg_accuracy*100:.2f}%")
        print(f"{'Total time':<20} {total_duration/60:.1f} minutes")
        print("=" * 60)


# === ALTERNATIVE: Evaluation without harness (via LM Studio API) ===

class SimplifiedHarnessEval:
    """
    Simplified version of harness evaluation via LM Studio API.
    
    For cases where lm-evaluation-harness is not installed
    or for faster evaluation.
    """
    
    def __init__(self, client):
        """
        Args:
            client: LMStudioClient
        """
        from ..lmstudio_client import LMStudioClient
        self.client = client
    
    def evaluate_mmlu_sample(
        self,
        model_id: str,
        num_samples: int = 50,
    ) -> dict:
        """
        Evaluates an MMLU sample via forced-choice.
        
        Note: This method is less precise than the harness
        as it does not use logprobs.
        """
        from datasets import load_dataset
        from ..lmstudio_client import format_messages
        from tqdm import tqdm
        import random
        
        # Charger MMLU
        dataset = load_dataset("cais/mmlu", "all", split="test")
        samples = random.sample(list(dataset), min(num_samples, len(dataset)))
        
        correct = 0
        
        for sample in tqdm(samples, desc="MMLU"):
            question = sample["question"]
            choices = sample["choices"]
            answer = sample["answer"]  # Index 0-3
            
            prompt = f"""Question: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with only the letter (A, B, C, or D):"""
            
            result = self.client.complete(
                model=model_id,
                messages=format_messages(prompt),
                temperature=0,
                max_tokens=5,
            )
            
            if result.success:
                pred = result.content.strip().upper()
                expected = ["A", "B", "C", "D"][answer]
                if expected in pred:
                    correct += 1
        
        return {
            "task": "mmlu_simplified",
            "accuracy": correct / len(samples) if samples else 0,
            "num_samples": len(samples),
            "method": "forced_choice (not logprobs)",
        }

