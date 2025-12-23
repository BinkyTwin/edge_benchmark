"""
Reproducibility Module
======================

Scientific reproducibility management:
- Global seed and determinism
- Environment capture (OS, hardware, versions)
- Configuration serialization
- Reproducibility protocol
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HardwareInfo:
    """Hardware information."""
    
    machine: str = ""
    processor: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""
    memory_gb: float = 0.0
    
    # Apple Silicon specific
    apple_silicon: bool = False
    chip_model: str = ""  # M1, M2, M3, etc.
    
    @classmethod
    def capture(cls) -> "HardwareInfo":
        """Captures hardware information."""
        import psutil
        
        info = cls(
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count() or 0,
            memory_gb=psutil.virtual_memory().total / (1024**3),
        )
        
        # Detect Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info.apple_silicon = True
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                info.chip_model = result.stdout.strip()
            except Exception:
                info.chip_model = "Apple Silicon (unknown)"
        
        # CPU brand for x86
        if not info.apple_silicon:
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                info.cpu_brand = result.stdout.strip()
            except Exception:
                info.cpu_brand = platform.processor()
        
        return info


@dataclass
class SoftwareInfo:
    """Software environment information."""
    
    os_name: str = ""
    os_version: str = ""
    os_release: str = ""
    python_version: str = ""
    
    # Key package versions
    package_versions: dict = field(default_factory=dict)
    
    # LM Studio
    lmstudio_version: str = ""
    lmstudio_api_version: str = ""
    
    @classmethod
    def capture(cls, lmstudio_client=None) -> "SoftwareInfo":
        """Captures software information."""
        info = cls(
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=sys.version,
        )
        
        # Key package versions
        packages_to_check = [
            "openai", "httpx", "datasets", "pandas", "numpy",
            "scikit-learn", "psutil", "lm-eval", "tqdm"
        ]
        
        for pkg in packages_to_check:
            try:
                import importlib.metadata
                info.package_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                info.package_versions[pkg] = "not installed"
        
        # LM Studio version (if client provided)
        if lmstudio_client:
            try:
                # Attempt to retrieve version via API
                response = lmstudio_client.http_client.get("/api/version")
                if response.status_code == 200:
                    data = response.json()
                    info.lmstudio_version = data.get("version", "unknown")
            except Exception:
                info.lmstudio_version = "unknown"
        
        return info


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Identifiers
    experiment_id: str = ""
    experiment_name: str = ""
    timestamp: str = ""
    
    # Seeds
    global_seed: int = 42
    numpy_seed: int = 42
    random_seed: int = 42
    
    # Hardware & Software
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    software: SoftwareInfo = field(default_factory=SoftwareInfo)
    
    # Benchmark configurations
    benchmark_config: dict = field(default_factory=dict)
    model_configs: dict = field(default_factory=dict)
    sampling_params: dict = field(default_factory=dict)
    
    # Hash for integrity verification
    config_hash: str = ""
    
    def compute_hash(self) -> str:
        """Computes a configuration hash for verification."""
        config_str = json.dumps({
            "global_seed": self.global_seed,
            "benchmark_config": self.benchmark_config,
            "model_configs": self.model_configs,
            "sampling_params": self.sampling_params,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Converts to serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "seeds": {
                "global_seed": self.global_seed,
                "numpy_seed": self.numpy_seed,
                "random_seed": self.random_seed,
            },
            "hardware": asdict(self.hardware),
            "software": asdict(self.software),
            "benchmark_config": self.benchmark_config,
            "model_configs": self.model_configs,
            "sampling_params": self.sampling_params,
            "config_hash": self.config_hash,
        }


class ReproducibilityManager:
    """
    Scientific reproducibility manager.
    
    Ensures:
    - Determinism via global seeds
    - Complete environment capture
    - Configuration serialization
    - Documented reproducibility protocol
    """
    
    def __init__(
        self,
        seed: int = 42,
        experiment_name: str = "edge_slm_benchmark",
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            seed: Global seed for reproducibility
            experiment_name: Experiment name
            output_dir: Output directory
        """
        self.seed = seed
        self.experiment_name = experiment_name
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config: Optional[ExperimentConfig] = None
    
    def set_global_seed(self, seed: Optional[int] = None):
        """
        Sets the global seed for reproducibility.
        
        Affects:
        - random (Python stdlib)
        - numpy.random
        - os.environ PYTHONHASHSEED
        """
        seed = seed or self.seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Hash seed (for dict reproducibility, etc.)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        print(f"[Reproducibility] Global seed set to {seed}")
        
        return seed
    
    def capture_environment(self, lmstudio_client=None) -> ExperimentConfig:
        """
        Captures the complete experiment environment.
        
        Args:
            lmstudio_client: LM Studio client to retrieve its version
            
        Returns:
            Complete ExperimentConfig
        """
        timestamp = datetime.now().isoformat()
        experiment_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            timestamp=timestamp,
            global_seed=self.seed,
            numpy_seed=self.seed,
            random_seed=self.seed,
            hardware=HardwareInfo.capture(),
            software=SoftwareInfo.capture(lmstudio_client),
        )
        
        config.config_hash = config.compute_hash()
        self.config = config
        
        return config
    
    def load_configs(
        self,
        models_config_path: Optional[Path] = None,
        scenarios_config_path: Optional[Path] = None,
        sampling_config_path: Optional[Path] = None,
    ):
        """
        Loads and integrates YAML configurations.
        
        Args:
            models_config_path: Path to models.yaml
            scenarios_config_path: Path to scenarios.yaml
            sampling_config_path: Path to sampling_params.yaml
        """
        import yaml
        
        if self.config is None:
            self.capture_environment()
        
        if models_config_path and models_config_path.exists():
            with open(models_config_path) as f:
                self.config.model_configs = yaml.safe_load(f)
        
        if scenarios_config_path and scenarios_config_path.exists():
            with open(scenarios_config_path) as f:
                self.config.benchmark_config = yaml.safe_load(f)
        
        if sampling_config_path and sampling_config_path.exists():
            with open(sampling_config_path) as f:
                self.config.sampling_params = yaml.safe_load(f)
        
        # Recompute the hash
        self.config.config_hash = self.config.compute_hash()
    
    def save_experiment_config(self, filename: Optional[str] = None) -> Path:
        """
        Saves the complete experiment configuration.
        
        Args:
            filename: Filename (auto-generated if None)
            
        Returns:
            Path of the saved file
        """
        if self.config is None:
            self.capture_environment()
        
        if filename is None:
            filename = f"experiment_config_{self.config.experiment_id}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        print(f"[Reproducibility] Experiment config saved to {filepath}")
        return filepath
    
    def generate_reproducibility_report(self) -> str:
        """
        Generates a reproducibility report for the paper.
        
        Returns:
            Markdown text of the report
        """
        if self.config is None:
            self.capture_environment()
        
        c = self.config
        
        report = f"""
## Reproducibility Protocol

### Environment Specification

**Hardware:**
- Machine: {c.hardware.chip_model or c.hardware.processor}
- CPU Cores: {c.hardware.cpu_count}
- Memory: {c.hardware.memory_gb:.1f} GB unified memory
- Apple Silicon: {c.hardware.apple_silicon}

**Software:**
- OS: {c.software.os_name} {c.software.os_release}
- Python: {c.software.python_version.split()[0]}
- LM Studio: {c.software.lmstudio_version}

**Key Package Versions:**
"""
        for pkg, version in c.software.package_versions.items():
            if version != "not installed":
                report += f"- {pkg}: {version}\n"
        
        report += f"""
### Determinism Settings

- Global seed: {c.global_seed}
- NumPy seed: {c.numpy_seed}
- Python random seed: {c.random_seed}
- PYTHONHASHSEED: {c.global_seed}

### Benchmark Protocol

- Warm-up runs: 3 (discarded)
- Benchmark runs: 20 per scenario
- Temperature: 0 (deterministic sampling)
- Top-p: 1 (no nucleus sampling)
- Machine state: Plugged in, power saving disabled

### Configuration Hash

`{c.config_hash}`

This hash can be used to verify that the exact same configuration was used.

### Replication Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure LM Studio is running with the same models
4. Run: `python scripts/run_performance.py --seed {c.global_seed}`
5. Compare results using the configuration hash
"""
        return report
    
    def print_environment_summary(self):
        """Prints an environment summary."""
        if self.config is None:
            self.capture_environment()
        
        c = self.config
        
        print("\n" + "=" * 60)
        print("EXPERIMENT ENVIRONMENT")
        print("=" * 60)
        print(f"Experiment ID: {c.experiment_id}")
        print(f"Timestamp: {c.timestamp}")
        print(f"\nHardware:")
        print(f"  Chip: {c.hardware.chip_model or c.hardware.processor}")
        print(f"  Memory: {c.hardware.memory_gb:.1f} GB")
        print(f"  Apple Silicon: {c.hardware.apple_silicon}")
        print(f"\nSoftware:")
        print(f"  OS: {c.software.os_name} {c.software.os_release}")
        print(f"  Python: {c.software.python_version.split()[0]}")
        print(f"\nSeeds:")
        print(f"  Global: {c.global_seed}")
        print(f"\nConfig Hash: {c.config_hash}")
        print("=" * 60)


def set_deterministic_mode(seed: int = 42) -> ReproducibilityManager:
    """
    Configures global deterministic mode.
    
    Utility function for quick setup.
    
    Args:
        seed: Global seed
        
    Returns:
        Configured ReproducibilityManager
    """
    manager = ReproducibilityManager(seed=seed)
    manager.set_global_seed()
    manager.capture_environment()
    return manager

