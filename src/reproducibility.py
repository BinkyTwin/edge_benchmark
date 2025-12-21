"""
Reproducibility Module
======================

Gestion de la reproductibilité scientifique:
- Seed global et déterminisme
- Capture de l'environnement (OS, hardware, versions)
- Sérialisation des configurations
- Protocole de reproductibilité
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
    """Informations sur le hardware."""
    
    machine: str = ""
    processor: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""
    memory_gb: float = 0.0
    
    # Apple Silicon spécifique
    apple_silicon: bool = False
    chip_model: str = ""  # M1, M2, M3, etc.
    
    @classmethod
    def capture(cls) -> "HardwareInfo":
        """Capture les informations hardware."""
        import psutil
        
        info = cls(
            machine=platform.machine(),
            processor=platform.processor(),
            cpu_count=os.cpu_count() or 0,
            memory_gb=psutil.virtual_memory().total / (1024**3),
        )
        
        # Détecter Apple Silicon
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
        
        # CPU brand pour x86
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
    """Informations sur l'environnement logiciel."""
    
    os_name: str = ""
    os_version: str = ""
    os_release: str = ""
    python_version: str = ""
    
    # Versions des packages clés
    package_versions: dict = field(default_factory=dict)
    
    # LM Studio
    lmstudio_version: str = ""
    lmstudio_api_version: str = ""
    
    @classmethod
    def capture(cls, lmstudio_client=None) -> "SoftwareInfo":
        """Capture les informations logicielles."""
        info = cls(
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            python_version=sys.version,
        )
        
        # Versions des packages clés
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
        
        # LM Studio version (si client fourni)
        if lmstudio_client:
            try:
                # Tenter de récupérer la version via l'API
                response = lmstudio_client.http_client.get("/api/version")
                if response.status_code == 200:
                    data = response.json()
                    info.lmstudio_version = data.get("version", "unknown")
            except Exception:
                info.lmstudio_version = "unknown"
        
        return info


@dataclass
class ExperimentConfig:
    """Configuration complète d'une expérience."""
    
    # Identifiants
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
    
    # Configurations du benchmark
    benchmark_config: dict = field(default_factory=dict)
    model_configs: dict = field(default_factory=dict)
    sampling_params: dict = field(default_factory=dict)
    
    # Hash pour vérification d'intégrité
    config_hash: str = ""
    
    def compute_hash(self) -> str:
        """Calcule un hash de la configuration pour vérification."""
        config_str = json.dumps({
            "global_seed": self.global_seed,
            "benchmark_config": self.benchmark_config,
            "model_configs": self.model_configs,
            "sampling_params": self.sampling_params,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire sérialisable."""
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
    Gestionnaire de reproductibilité scientifique.
    
    Assure:
    - Déterminisme via seeds globaux
    - Capture complète de l'environnement
    - Sérialisation des configurations
    - Protocole de reproductibilité documenté
    """
    
    def __init__(
        self,
        seed: int = 42,
        experiment_name: str = "edge_slm_benchmark",
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            seed: Seed global pour reproductibilité
            experiment_name: Nom de l'expérience
            output_dir: Dossier de sortie
        """
        self.seed = seed
        self.experiment_name = experiment_name
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config: Optional[ExperimentConfig] = None
    
    def set_global_seed(self, seed: Optional[int] = None):
        """
        Définit le seed global pour reproductibilité.
        
        Affecte:
        - random (Python stdlib)
        - numpy.random
        - os.environ PYTHONHASHSEED
        """
        seed = seed or self.seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Hash seed (pour reproductibilité des dicts, etc.)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        print(f"[Reproducibility] Global seed set to {seed}")
        
        return seed
    
    def capture_environment(self, lmstudio_client=None) -> ExperimentConfig:
        """
        Capture l'environnement complet de l'expérience.
        
        Args:
            lmstudio_client: Client LM Studio pour récupérer sa version
            
        Returns:
            ExperimentConfig complet
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
        Charge et intègre les configurations YAML.
        
        Args:
            models_config_path: Chemin vers models.yaml
            scenarios_config_path: Chemin vers scenarios.yaml
            sampling_config_path: Chemin vers sampling_params.yaml
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
        
        # Recalculer le hash
        self.config.config_hash = self.config.compute_hash()
    
    def save_experiment_config(self, filename: Optional[str] = None) -> Path:
        """
        Sauvegarde la configuration complète de l'expérience.
        
        Args:
            filename: Nom du fichier (auto-généré si None)
            
        Returns:
            Chemin du fichier sauvegardé
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
        Génère un rapport de reproductibilité pour le papier.
        
        Returns:
            Texte Markdown du rapport
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
        """Affiche un résumé de l'environnement."""
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
    Configure le mode déterministe global.
    
    Fonction utilitaire pour une configuration rapide.
    
    Args:
        seed: Seed global
        
    Returns:
        ReproducibilityManager configuré
    """
    manager = ReproducibilityManager(seed=seed)
    manager.set_global_seed()
    manager.capture_environment()
    return manager

