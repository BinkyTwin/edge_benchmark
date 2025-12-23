"""
Coding Evaluation Module
========================

Mini coding test with HumanEval (subset).
Objective: verify if SLMs can assist with basic 
scripting/automation tasks.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, format_messages


@dataclass
class CodingResult:
    """Result of a coding evaluation."""
    
    model_id: str
    dataset_name: str
    
    # Main metrics
    pass_at_1: float = 0.0
    syntax_valid_rate: float = 0.0
    
    # Details
    num_samples: int = 0
    num_passed: int = 0
    num_syntax_valid: int = 0
    
    # Performance
    avg_latency_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # Metadata
    timestamp: str = ""
    results: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "pass_at_1": round(self.pass_at_1, 4),
            "syntax_valid_rate": round(self.syntax_valid_rate, 4),
            "num_samples": self.num_samples,
            "num_passed": self.num_passed,
            "num_syntax_valid": self.num_syntax_valid,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "timestamp": self.timestamp,
        }


class CodingEvaluator:
    """
    Evaluator for code generation tasks.
    
    Uses HumanEval (subset) to evaluate coding
    capabilities of SLMs.
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        results_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.client = client
        self.results_dir = results_dir or Path("results")
        self.cache_dir = cache_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_humaneval(
        self,
        sample_size: int = 30,
        seed: int = 42,
    ) -> list[dict]:
        """
        Load a subset of HumanEval.
        
        Args:
            sample_size: Number of problems to load
            seed: Seed for sampling
            
        Returns:
            List of problems
        """
        import random
        
        print(f"[HumanEval] Loading dataset...")
        dataset = load_dataset(
            "openai/openai_humaneval",
            split="test",
            cache_dir=self.cache_dir,
        )
        
        # Convert to list
        samples = list(dataset)
        
        # Select a subset
        if sample_size and sample_size < len(samples):
            random.seed(seed)
            samples = random.sample(samples, sample_size)
        
        print(f"[HumanEval] Loaded {len(samples)} problems")
        return samples
    
    def evaluate_humaneval(
        self,
        model_id: str,
        sample_size: int = 30,
        progress_bar: bool = True,
    ) -> CodingResult:
        """
        Evaluate a model on HumanEval (subset).
        
        Args:
            model_id: LM Studio model ID
            sample_size: Number of problems
            progress_bar: Display progress
            
        Returns:
            CodingResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"HUMANEVAL EVALUATION (Mini)")
        print(f"Model: {model_id}")
        print(f"Sample size: {sample_size}")
        print(f"{'='*60}")
        
        samples = self.load_humaneval(sample_size=sample_size)
        
        system_prompt = """You are an expert Python programmer.
Complete the given function according to its docstring.
Write only the function body, no explanations.
Ensure the code is syntactically correct."""
        
        results = []
        latencies = []
        passed = 0
        syntax_valid = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            prompt = sample["prompt"]
            test_code = sample["test"]
            entry_point = sample["entry_point"]
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=256,
                stream=False,
                stop=["\ndef ", "\nclass ", "\nif __name__"],
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            completion = result.content if result.success else ""
            
            # Clean completion
            completion = self._clean_completion(completion, prompt)
            
            # Check syntax
            full_code = prompt + completion
            is_syntax_valid = self._check_syntax(full_code)
            if is_syntax_valid:
                syntax_valid += 1
            
            # Run tests (if syntax is valid)
            test_passed = False
            error_msg = None
            
            if is_syntax_valid:
                test_passed, error_msg = self._run_tests(
                    full_code,
                    test_code,
                    entry_point,
                )
                if test_passed:
                    passed += 1
            
            results.append({
                "task_id": sample["task_id"],
                "entry_point": entry_point,
                "completion": completion[:200],  # Truncate for log
                "syntax_valid": is_syntax_valid,
                "test_passed": test_passed,
                "error": error_msg,
                "latency_ms": result.metrics.total_time_ms,
            })
        
        total_time = time.time() - start_time
        
        coding_result = CodingResult(
            model_id=model_id,
            dataset_name="humaneval_mini",
            pass_at_1=passed / len(samples) if samples else 0,
            syntax_valid_rate=syntax_valid / len(samples) if samples else 0,
            num_samples=len(samples),
            num_passed=passed,
            num_syntax_valid=syntax_valid,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            timestamp=datetime.now().isoformat(),
            results=results,
        )
        
        self._print_results(coding_result)
        return coding_result
    
    def _clean_completion(self, completion: str, prompt: str) -> str:
        """Clean model completion."""
        # Remove markdown code markers
        completion = re.sub(r'```python\s*', '', completion)
        completion = re.sub(r'```\s*', '', completion)
        
        # Remove prompt if repeated
        if prompt.strip() in completion:
            completion = completion.replace(prompt.strip(), "")
        
        # Remove duplicate function definitions
        lines = completion.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('def ') and 'def ' in prompt:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _run_tests(
        self,
        code: str,
        test_code: str,
        entry_point: str,
        timeout: float = 5.0,
    ) -> tuple[bool, Optional[str]]:
        """
        Run tests on generated code.
        
        Note: Execution in a restricted environment.
        
        Args:
            code: Full code (prompt + completion)
            test_code: Test code
            entry_point: Function name
            timeout: Timeout in seconds
            
        Returns:
            Tuple (passed, error_message)
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        # Prepare full code
        full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            # Create restricted execution environment
            exec_globals = {
                "__builtins__": {
                    "range": range,
                    "len": len,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "bool": bool,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "sorted": sorted,
                    "reversed": reversed,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "any": any,
                    "all": all,
                    "isinstance": isinstance,
                    "type": type,
                    "round": round,
                    "print": lambda *args, **kwargs: None,  # Silence print
                    "AssertionError": AssertionError,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                    "IndexError": IndexError,
                    "True": True,
                    "False": False,
                    "None": None,
                }
            }
            
            exec(full_code, exec_globals)
            signal.alarm(0)  # Cancel timeout
            return True, None
            
        except TimeoutError:
            return False, "Timeout"
        except AssertionError as e:
            signal.alarm(0)
            return False, f"AssertionError: {e}"
        except Exception as e:
            signal.alarm(0)
            return False, f"{type(e).__name__}: {e}"
    
    def save_result(self, result: CodingResult, filename: Optional[str] = None):
        """Save result."""
        if filename is None:
            model_name = result.model_id.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coding_{result.dataset_name}_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n[Saved] Results saved to {filepath}")
    
    def _print_results(self, result: CodingResult):
        """Display results."""
        print(f"\n{'─'*40}")
        print("RESULTS")
        print(f"{'─'*40}")
        print(f"Pass@1:          {result.pass_at_1*100:.1f}%")
        print(f"Syntax valid:    {result.syntax_valid_rate*100:.1f}%")
        print(f"Passed:          {result.num_passed}/{result.num_samples}")
        print(f"Avg latency:     {result.avg_latency_ms:.1f} ms")
        print(f"Total time:      {result.total_time_seconds:.1f}s")
        print(f"{'─'*40}")


