"""
Checkpoint Module
=================

Checkpoint system for resuming benchmarks in case of interruption.
Saves state after each model/scenario.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Set


@dataclass
class CheckpointState:
    """Checkpoint state."""
    
    experiment_id: str
    task_type: str  # "performance" or "capability"
    
    # Planned models/scenarios
    planned_models: list[str] = field(default_factory=list)
    planned_scenarios: list[str] = field(default_factory=list)
    planned_tasks: list[str] = field(default_factory=list)
    
    # Completed
    completed: list[dict] = field(default_factory=list)
    
    # Failed (with error)
    failed: list[dict] = field(default_factory=list)
    
    # Timestamps
    started_at: str = ""
    last_updated: str = ""
    
    def get_completed_keys(self) -> Set[str]:
        """Return keys of completed tasks."""
        return {f"{c['model']}|{c.get('scenario', c.get('task', ''))}" for c in self.completed}
    
    def get_failed_keys(self) -> Set[str]:
        """Return keys of failed tasks."""
        return {f"{f['model']}|{f.get('scenario', f.get('task', ''))}" for f in self.failed}
    
    def is_completed(self, model: str, scenario_or_task: str) -> bool:
        """Check if a model/scenario combination is already completed."""
        key = f"{model}|{scenario_or_task}"
        return key in self.get_completed_keys()
    
    def add_completed(self, model: str, scenario_or_task: str, result_file: str):
        """Mark a task as completed."""
        self.completed.append({
            "model": model,
            "scenario": scenario_or_task,
            "result_file": result_file,
            "completed_at": datetime.now().isoformat(),
        })
        self.last_updated = datetime.now().isoformat()
    
    def add_failed(self, model: str, scenario_or_task: str, error: str):
        """Mark a task as failed."""
        self.failed.append({
            "model": model,
            "scenario": scenario_or_task,
            "error": error,
            "failed_at": datetime.now().isoformat(),
        })
        self.last_updated = datetime.now().isoformat()
    
    def get_remaining(self) -> list[tuple[str, str]]:
        """Return remaining model/scenario combinations."""
        completed_keys = self.get_completed_keys()
        failed_keys = self.get_failed_keys()
        done_keys = completed_keys | failed_keys
        
        remaining = []
        for model in self.planned_models:
            for scenario in self.planned_scenarios or self.planned_tasks or [""]:
                key = f"{model}|{scenario}"
                if key not in done_keys:
                    remaining.append((model, scenario))
        
        return remaining
    
    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "task_type": self.task_type,
            "planned_models": self.planned_models,
            "planned_scenarios": self.planned_scenarios,
            "planned_tasks": self.planned_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "progress": {
                "total": len(self.planned_models) * max(len(self.planned_scenarios), len(self.planned_tasks), 1),
                "completed": len(self.completed),
                "failed": len(self.failed),
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        return cls(
            experiment_id=data["experiment_id"],
            task_type=data["task_type"],
            planned_models=data.get("planned_models", []),
            planned_scenarios=data.get("planned_scenarios", []),
            planned_tasks=data.get("planned_tasks", []),
            completed=data.get("completed", []),
            failed=data.get("failed", []),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
        )


class CheckpointManager:
    """
    Checkpoint manager for benchmarks.
    
    Allows:
    - Saving state after each model/scenario
    - Resuming where we left off with --resume
    - Protection against crashes (try/except with save)
    """
    
    def __init__(
        self,
        results_dir: Path,
        experiment_id: Optional[str] = None,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_file = self.results_dir / f"checkpoint_{self.experiment_id}.json"
        self.state: Optional[CheckpointState] = None
    
    def start_experiment(
        self,
        task_type: str,
        models: list[str],
        scenarios: Optional[list[str]] = None,
        tasks: Optional[list[str]] = None,
    ) -> CheckpointState:
        """
        Start a new experiment or load an existing checkpoint.
        
        Args:
            task_type: "performance" or "capability"
            models: List of models to evaluate
            scenarios: List of scenarios (for performance)
            tasks: List of tasks (for capability)
            
        Returns:
            CheckpointState
        """
        # Check if a checkpoint exists
        if self.checkpoint_file.exists():
            print(f"[Checkpoint] Found existing checkpoint: {self.checkpoint_file}")
            self.state = self.load()
            
            remaining = self.state.get_remaining()
            if remaining:
                print(f"[Checkpoint] Resuming from checkpoint. {len(remaining)} tasks remaining.")
            else:
                print("[Checkpoint] All tasks already completed!")
            
            return self.state
        
        # Create a new checkpoint
        self.state = CheckpointState(
            experiment_id=self.experiment_id,
            task_type=task_type,
            planned_models=models,
            planned_scenarios=scenarios or [],
            planned_tasks=tasks or [],
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
        
        self.save()
        print(f"[Checkpoint] Started new experiment: {self.experiment_id}")
        
        return self.state
    
    def save(self):
        """Save the checkpoint."""
        if self.state:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
    
    def load(self) -> CheckpointState:
        """Load an existing checkpoint."""
        with open(self.checkpoint_file) as f:
            data = json.load(f)
        return CheckpointState.from_dict(data)
    
    def mark_completed(self, model: str, scenario_or_task: str, result_file: str):
        """Mark a task as completed and save."""
        if self.state:
            self.state.add_completed(model, scenario_or_task, result_file)
            self.save()
            print(f"[Checkpoint] Saved: {model} / {scenario_or_task}")
    
    def mark_failed(self, model: str, scenario_or_task: str, error: str):
        """Mark a task as failed and save."""
        if self.state:
            self.state.add_failed(model, scenario_or_task, error)
            self.save()
            print(f"[Checkpoint] Failed (saved): {model} / {scenario_or_task} - {error[:50]}")
    
    def should_skip(self, model: str, scenario_or_task: str) -> bool:
        """Check if a task should be skipped (already completed)."""
        if self.state:
            return self.state.is_completed(model, scenario_or_task)
        return False
    
    def print_summary(self):
        """Display checkpoint summary."""
        if not self.state:
            return
        
        s = self.state
        total = len(s.planned_models) * max(len(s.planned_scenarios), len(s.planned_tasks), 1)
        
        print(f"\n{'='*50}")
        print("CHECKPOINT SUMMARY")
        print(f"{'='*50}")
        print(f"Experiment: {s.experiment_id}")
        print(f"Type: {s.task_type}")
        print(f"Progress: {len(s.completed)}/{total} completed")
        
        if s.failed:
            print(f"Failed: {len(s.failed)}")
            for f in s.failed:
                print(f"  - {f['model']}/{f.get('scenario', f.get('task', 'N/A'))}: {f['error'][:40]}...")
        
        remaining = s.get_remaining()
        if remaining:
            print(f"Remaining: {len(remaining)}")
            for model, scenario in remaining[:5]:
                print(f"  - {model}/{scenario}")
            if len(remaining) > 5:
                print(f"  ... and {len(remaining) - 5} more")
        
        print(f"{'='*50}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint in the results directory."""
        checkpoints = list(self.results_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        
        # Sort by modification date
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def resume_latest(self) -> Optional[CheckpointState]:
        """Resume from the latest available checkpoint."""
        latest = self.find_latest_checkpoint()
        if latest:
            self.checkpoint_file = latest
            self.state = self.load()
            print(f"[Checkpoint] Resuming from: {latest.name}")
            return self.state
        
        print("[Checkpoint] No checkpoint found to resume.")
        return None


def run_with_checkpoint(
    func,
    model: str,
    scenario_or_task: str,
    checkpoint_manager: CheckpointManager,
    **kwargs,
) -> tuple[bool, any]:
    """
    Execute a function with checkpoint protection.
    
    Args:
        func: Function to execute
        model: Model ID
        scenario_or_task: Scenario or task name
        checkpoint_manager: Checkpoint manager
        **kwargs: Arguments for the function
        
    Returns:
        Tuple (success, result)
    """
    # Check if already completed
    if checkpoint_manager.should_skip(model, scenario_or_task):
        print(f"[Skip] {model}/{scenario_or_task} already completed")
        return True, None
    
    try:
        print(f"\n[Running] {model} / {scenario_or_task}")
        result = func(**kwargs)
        
        # Mark as completed
        result_file = f"result_{model.replace('/', '_')}_{scenario_or_task}.json"
        checkpoint_manager.mark_completed(model, scenario_or_task, result_file)
        
        return True, result
        
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving checkpoint before exit...")
        checkpoint_manager.mark_failed(model, scenario_or_task, "KeyboardInterrupt")
        raise
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Error] {model}/{scenario_or_task}: {error_msg}")
        checkpoint_manager.mark_failed(model, scenario_or_task, error_msg)
        return False, None

