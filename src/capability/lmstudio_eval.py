"""
LM Studio Evaluator
===================

Generic evaluator for tasks via LM Studio API.
Support for forced-choice generation (A/B/C/D).
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, format_messages


@dataclass
class EvalSample:
    """Evaluation sample."""
    
    prompt: str
    expected: str
    context: Optional[str] = None
    metadata: dict = None


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    
    accuracy: float = 0.0
    num_correct: int = 0
    num_total: int = 0
    avg_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class LMStudioEvaluator:
    """
    Generic evaluator using LM Studio API.
    
    Supports:
    - Forced-choice (A/B/C/D)
    - Classification
    - Extraction
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        results_dir: Optional[Path] = None,
    ):
        """
        Args:
            client: LM Studio client
            results_dir: Directory for results
        """
        self.client = client
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_forced_choice(
        self,
        model_id: str,
        samples: list[dict],
        system_prompt: str,
        choices: list[str] = ["A", "B", "C", "D"],
        progress_bar: bool = True,
    ) -> EvalMetrics:
        """
        Evaluates with forced-choice (answer among A/B/C/D).
        
        Args:
            model_id: Model ID
            samples: List of samples with 'prompt' and 'expected'
            system_prompt: System prompt
            choices: Valid choices
            progress_bar: Display progress
            
        Returns:
            EvalMetrics
        """
        predictions = []
        latencies = []
        
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            messages = format_messages(sample["prompt"], system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=5,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            if result.success:
                pred = self._parse_choice(result.content, choices)
                predictions.append(pred)
            else:
                predictions.append(None)
        
        # Calculate accuracy
        correct = sum(
            1 for pred, sample in zip(predictions, samples)
            if pred and pred == sample.get("expected")
        )
        
        return EvalMetrics(
            accuracy=correct / len(samples) if samples else 0,
            num_correct=correct,
            num_total=len(samples),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        )
    
    def evaluate_classification(
        self,
        model_id: str,
        samples: list[dict],
        system_prompt: str,
        labels: list[str],
        parse_fn: Optional[Callable[[str], str]] = None,
        progress_bar: bool = True,
    ) -> EvalMetrics:
        """
        Evaluates a classification task.
        
        Args:
            model_id: Model ID
            samples: List of samples with 'text' and 'label'
            system_prompt: System prompt
            labels: Possible labels
            parse_fn: Function to parse the response
            progress_bar: Display progress
            
        Returns:
            EvalMetrics
        """
        predictions = []
        latencies = []
        
        if parse_fn is None:
            parse_fn = lambda x: self._parse_label(x, labels)
        
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            prompt = f"Text: {sample['text']}\n\nLabel:"
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=20,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            if result.success:
                pred = parse_fn(result.content)
                predictions.append(pred)
            else:
                predictions.append(None)
        
        # Calculate accuracy
        correct = sum(
            1 for pred, sample in zip(predictions, samples)
            if pred and pred == sample.get("label")
        )
        
        return EvalMetrics(
            accuracy=correct / len(samples) if samples else 0,
            num_correct=correct,
            num_total=len(samples),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        )
    
    def evaluate_json_extraction(
        self,
        model_id: str,
        samples: list[dict],
        system_prompt: str,
        required_fields: list[str],
        progress_bar: bool = True,
    ) -> dict:
        """
        Evaluates JSON extraction.
        
        Args:
            model_id: Model ID
            samples: List of samples with 'document' and 'expected'
            system_prompt: System prompt
            required_fields: Required fields
            progress_bar: Display progress
            
        Returns:
            Dictionary with metrics
        """
        valid_json_count = 0
        field_accuracy = {field: 0 for field in required_fields}
        latencies = []
        
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            prompt = sample.get("prompt", sample.get("document", ""))
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=300,
                stream=False,
                response_format={"type": "json_object"},
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            if result.success and result.json_valid:
                valid_json_count += 1
                parsed = result.parsed_json
                expected = sample.get("expected", {})
                
                for field in required_fields:
                    if field in parsed and field in expected:
                        if parsed[field] == expected[field]:
                            field_accuracy[field] += 1
        
        total = len(samples)
        return {
            "json_valid_rate": valid_json_count / total if total else 0,
            "field_accuracy": {
                field: count / total if total else 0
                for field, count in field_accuracy.items()
            },
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "num_samples": total,
        }
    
    def _parse_choice(self, content: str, choices: list[str]) -> Optional[str]:
        """Parses a choice among the options."""
        content = content.strip().upper()
        
        for choice in choices:
            if choice.upper() in content:
                return choice.upper()
        
        return None
    
    def _parse_label(self, content: str, labels: list[str]) -> Optional[str]:
        """Parses a label among the options."""
        content = content.strip().lower()
        
        for label in labels:
            if label.lower() in content:
                return label
        
        return None
    
    def save_results(
        self,
        results: dict,
        task_name: str,
        model_id: str,
    ):
        """Saves the results."""
        model_name = model_id.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{task_name}_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[Saved] Results saved to {filepath}")


