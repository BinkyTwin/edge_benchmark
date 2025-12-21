# Edge SLM Benchmark Framework

**Local Generative AI on Enterprise Edge Devices in Regulated Banking**

Framework de benchmark pour évaluer des Small Language Models (SLMs) sur Apple Silicon dans un contexte bancaire réglementé.

## Problématique

> Measuring SLM Utility Under Latency/Privacy Constraints and Assessing Residual Compliance Risk on Consumer-Grade Macs

## Modèles Évalués

| Model | Format | Quantization | Context |
|-------|--------|--------------|---------|
| Gemma 3n E4B | MLX / GGUF | 4bit / Q4_K_M | 32K |
| Qwen3-VL 4B | MLX / GGUF | 4bit / Q4_K_M | 262K |
| Ministral 3 3B | GGUF | Q4_K_M | 262K |

## Structure du Projet

```
edge_benchmark/
├── configs/           # Fichiers de configuration YAML
├── src/               # Code source
│   ├── lmstudio_client.py
│   ├── reproducibility.py  # Gestion seeds et environnement
│   ├── statistics.py       # IC 95% et tests de significativité
│   ├── performance/   # Benchmarks performance
│   ├── capability/    # Benchmarks capacités
│   └── compliance/    # Analyse conformité
├── prompts/           # Templates de prompts
├── scripts/           # Entry points
└── results/           # Résultats générés
```

## Installation

```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

## Prérequis

- **LM Studio** installé et configuré avec les modèles
- Serveur LM Studio démarré sur `http://localhost:1234`
- macOS avec Apple Silicon (M1/M2/M3)
- Python 3.10+

## Utilisation

### 1. Benchmark de Performance

```bash
python scripts/run_performance.py --models all --scenarios all --seed 42
```

### 2. Benchmark de Capacités (Banking)

```bash
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b
python scripts/run_capability.py --task financial_phrasebank --model google/gemma-3n-e4b
```

### 3. Mini-benchmark Harness

```bash
python scripts/run_capability.py --task harness --gguf-path /path/to/model.gguf
```

### 4. Génération de Rapport

```bash
python scripts/generate_report.py --input results/ --output report/ --compliance
```

## Benchmarks

### Performance
- **TTFT** (Time To First Token) avec IC 95%
- **Output tokens/s** avec IC 95%
- **Peak RAM**
- 3 scénarios: Interactive, Summarization, JSON

### Capacités
- **Banking77**: Intent classification (77 classes)
- **Financial PhraseBank**: Sentiment analysis
- **HumanEval Mini**: Code generation
- **MMLU/GSM8K**: Validation rapide

---

## Reproducibility Protocol

### Determinism Settings

Ce framework implémente un protocole de reproductibilité rigoureux :

```python
from src.reproducibility import set_deterministic_mode

# Configure le mode déterministe global
manager = set_deterministic_mode(seed=42)
manager.capture_environment()
manager.save_experiment_config()
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Global seed | 42 | Reproductibilité Python random |
| NumPy seed | 42 | Reproductibilité sampling |
| PYTHONHASHSEED | 42 | Déterminisme des dicts |
| Temperature | 0 | Sampling déterministe |
| Top-p | 1 | Pas de nucleus sampling |

### Environment Capture

Chaque expérience sérialise automatiquement :
- **Hardware** : chip model, CPU cores, memory
- **Software** : OS version, Python version, package versions
- **LM Studio** : version, API version
- **Configs** : models.yaml, scenarios.yaml, sampling_params.yaml
- **Config hash** : SHA256 pour vérification d'intégrité

### Benchmark Protocol

1. **Warm-up** : 3 requêtes (non comptées)
2. **Benchmark runs** : 20 par scénario
3. **Cooldown** : 2s entre chaque run
4. **Machine state** : branchée, mode économie désactivé
5. **Logging** : JSONL complet avec timestamps

---

## Statistical Analysis

### Confidence Intervals (IC 95%)

Les métriques sont rapportées avec intervalles de confiance 95% calculés par **bootstrap** (10,000 itérations) :

```
TTFT: 245.3 ms [95% CI: 238.1, 252.7]
Output t/s: 42.8 [95% CI: 40.2, 45.1]
```

Le bootstrap est préféré à la t-distribution car :
- Ne suppose pas de normalité
- Robuste pour petits échantillons (n=20)

### Significance Testing

Pour les comparaisons entre modèles :

| Test | Conditions | Usage |
|------|------------|-------|
| Paired t-test | Données normales, n≥20 | Comparaison sur mêmes prompts |
| Wilcoxon signed-rank | Non-paramétrique | Petit échantillon ou non-normal |
| Holm correction | Comparaisons multiples | Contrôle FWER |

Tailles d'effet rapportées :
- **Cohen's d** (t-test) : 0.2 petit, 0.5 moyen, 0.8 grand
- **Rank-biserial r** (Wilcoxon) : mêmes seuils

### Example Output

```
| Model A | Model B | Metric | Diff (%) | p-value | Sig. | Effect |
|---------|---------|--------|----------|---------|------|--------|
| Gemma   | Qwen    | TTFT   | -12.3%   | 0.0023  | ✓    | 0.67   |
| Gemma   | Mistral | TTFT   | +5.2%    | 0.1842  |      | 0.21   |
```

---

## Conformité

Framework d'analyse basé sur:
- **NIST AI RMF 1.0** (AI Risk Management Framework)
- **NIST AI 600-1** (Generative AI Profile)
- **OWASP Top 10 for LLM Applications**

### License Audit

| Model | License | Commercial Use |
|-------|---------|----------------|
| Gemma 3n E4B | Gemma Terms of Use | Restricted |
| Qwen3-VL 4B | Apache 2.0 | Allowed |
| Ministral 3 3B | Apache 2.0 | Allowed |

---

## Replication Instructions

Pour répliquer exactement une expérience :

1. **Clone** le repository
2. **Install** : `pip install -r requirements.txt`
3. **Vérifier** le config hash dans le fichier `experiment_config_*.json`
4. **Exécuter** avec le même seed :
   ```bash
   python scripts/run_performance.py --seed 42 --models all
   ```
5. **Comparer** les résultats avec le hash de configuration

## Citation

Si vous utilisez ce framework dans vos travaux :

```bibtex
@software{edge_slm_benchmark,
  title = {Edge SLM Benchmark Framework},
  year = {2024},
  note = {Local Generative AI on Enterprise Edge Devices}
}
```

## Licence

Projet de recherche - Usage interne
