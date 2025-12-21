# Edge SLM Benchmark Framework

**Local Generative AI on Enterprise Edge Devices in Regulated Banking**

Framework de benchmark pour évaluer des Small Language Models (SLMs) sur Apple Silicon dans un contexte bancaire réglementé.

---

## Table des Matières

1. [Problématique](#problématique)
2. [Modèles Évalués](#modèles-évalués)
3. [Installation](#installation)
4. [Guide Rapide](#guide-rapide)
5. [Utilisation Détaillée](#utilisation-détaillée)
6. [Système de Checkpoint](#système-de-checkpoint)
7. [Benchmarks Disponibles](#benchmarks-disponibles)
8. [Reproductibilité](#reproductibilité)
9. [Analyse Statistique](#analyse-statistique)
10. [Conformité](#conformité)
11. [Structure du Projet](#structure-du-projet)

---

## Problématique

> **Measuring SLM Utility Under Latency/Privacy Constraints and Assessing Residual Compliance Risk on Consumer-Grade Macs**

Ce framework permet de répondre à la question : **"Les SLMs peuvent-ils être déployés efficacement sur des laptops Apple Silicon dans un contexte bancaire réglementé ?"**

Les axes d'évaluation :
- **Performance** : Latence (TTFT), débit (tokens/s), consommation mémoire
- **Capacités** : Classification bancaire, analyse de sentiment, génération de code
- **Conformité** : Analyse de risques NIST/OWASP, audit des licences

---

## Modèles Évalués

| Model | Publisher | Format | Quantization | Context | Type |
|-------|-----------|--------|--------------|---------|------|
| Gemma 3n E4B | Google | MLX / GGUF | 4bit / Q4_K_M | 32K | VLM |
| Qwen3-VL 4B | Alibaba | MLX / GGUF | 4bit / Q4_K_M | 262K | VLM |
| Ministral 3 3B | Mistral AI | GGUF | Q4_K_M | 262K | VLM |

---

## Installation

### Prérequis

- **macOS** avec Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **LM Studio** installé avec les modèles téléchargés
- **16 GB RAM minimum** (recommandé)

### Étapes d'installation

```bash
# 1. Cloner le repository (ou naviguer vers le dossier)
cd /path/to/edge_benchmark

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "from src import LMStudioClient; print('OK')"
```

### Configuration de LM Studio

1. Ouvrir **LM Studio**
2. Télécharger les modèles (Gemma, Qwen, Ministral)
3. Démarrer le serveur local : **Developer** → **Start Server**
4. Vérifier que le serveur est accessible sur `http://localhost:1234`

---

## Guide Rapide

### Lancer un benchmark en 3 commandes

```bash
# Activer l'environnement
source venv/bin/activate

# Lancer un benchmark Banking77 sur Gemma
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b

# Générer le rapport
python scripts/generate_report.py --input results/ --output report/
```

---

## Utilisation Détaillée

### 1. Benchmark de Performance

Évalue la latence, le débit et la consommation mémoire.

```bash
# Tous les modèles, tous les scénarios
python scripts/run_performance.py --models all --scenarios all

# Un seul modèle, un seul scénario
python scripts/run_performance.py --models gemma_3n_e4b_gguf --scenarios interactive_assistant

# Comparaison de tous les modèles sur un scénario
python scripts/run_performance.py --compare --scenarios interactive_assistant

# Personnaliser le nombre de runs
python scripts/run_performance.py --models all --runs 10 --warmup 2
```

**Options disponibles :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--models` | `all`, `gguf`, `mlx`, ou ID spécifique | `all` |
| `--scenarios` | `all`, `interactive_assistant`, `long_form_summarization`, `structured_json_output` | `all` |
| `--runs` | Nombre de runs par scénario | 20 |
| `--warmup` | Requêtes de warm-up | 3 |
| `--cooldown` | Pause entre runs (secondes) | 2.0 |
| `--seed` | Seed pour reproductibilité | 42 |
| `--resume` | Reprendre depuis le checkpoint | - |

### 2. Benchmark de Capacités (Banking)

Évalue les capacités des modèles sur des tâches bancaires.

```bash
# Banking77 - Classification d'intents (77 classes)
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b

# Financial PhraseBank - Analyse de sentiment
python scripts/run_capability.py --task financial_phrasebank --model google/gemma-3n-e4b

# Les deux tâches banking
python scripts/run_capability.py --task banking_all --model google/gemma-3n-e4b

# Scénarios réalistes (FAQ, extraction, résumé)
python scripts/run_capability.py --task realistic --model google/gemma-3n-e4b

# Test de codage (HumanEval mini)
python scripts/run_capability.py --task coding --model google/gemma-3n-e4b

# Tous les modèles sur une tâche
python scripts/run_capability.py --task banking77 --all-models

# Limiter le nombre d'échantillons (pour tests rapides)
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b --sample-size 100
```

**Tâches disponibles :**

| Tâche | Description | Métriques |
|-------|-------------|-----------|
| `banking77` | Classification d'intents bancaires | Accuracy, Macro-F1 |
| `financial_phrasebank` | Sentiment sur news financières | Accuracy, Macro-F1 |
| `banking_all` | Les deux tâches ci-dessus | - |
| `realistic` | FAQ, sentiment, extraction, résumé | Diverses |
| `coding` | HumanEval (30 problèmes) | Pass@1 |
| `harness` | MMLU + GSM8K via lm-eval | Accuracy |
| `all` | Toutes les tâches | - |

### 3. Mini-benchmark Harness (MMLU/GSM8K)

Pour les benchmarks académiques nécessitant des logprobs :

```bash
# Nécessite le chemin vers le fichier GGUF
python scripts/run_capability.py --task harness --gguf-path ~/.cache/lm-studio/models/gemma-3n-e4b-q4_k_m.gguf
```

### 4. Génération de Rapports

```bash
# Rapport Markdown + HTML + CSV
python scripts/generate_report.py --input results/ --output report/ --format all

# Avec rapports de conformité
python scripts/generate_report.py --input results/ --output report/ --compliance
```

---

## Système de Checkpoint

Le framework sauvegarde automatiquement l'état après chaque modèle/scénario.

### En cas de crash ou interruption

```bash
# Le script affiche :
[Interrupted] Checkpoint saved. Use --resume to continue.

# Pour reprendre exactement là où tu t'es arrêté :
python scripts/run_performance.py --resume
python scripts/run_capability.py --resume
```

### Résumé du checkpoint

À la fin de chaque exécution :

```
==================================================
CHECKPOINT SUMMARY
==================================================
Experiment: exp_20241221_143052
Type: performance
Progress: 4/6 completed
Failed: 1
  - qwen/qwen3-vl-4b/summarization: Connection timeout...
Remaining: 1
  - mistralai/ministral-3-3b/interactive_assistant
==================================================

💡 To complete remaining tasks, run:
   python scripts/run_performance.py --resume
```

### Désactiver les checkpoints

```bash
python scripts/run_performance.py --no-checkpoint
```

---

## Benchmarks Disponibles

### Performance (3 scénarios)

| Scénario | Input | Output | Focus |
|----------|-------|--------|-------|
| Interactive Assistant | 200-400 tokens | 128 tokens | TTFT, latence |
| Long-form Summarization | 2000-4000 tokens | 256-512 tokens | Débit, RAM |
| Structured JSON Output | 500-1000 tokens | JSON | Taux de validité |

### Capacités (tâches banking)

| Dataset | Source | Taille | Tâche |
|---------|--------|--------|-------|
| Banking77 | Hugging Face | ~3K test | Intent classification |
| Financial PhraseBank | Hugging Face | ~5K | Sentiment analysis |
| HumanEval | OpenAI | 30 subset | Code generation |
| MMLU | Eleuther | 200 subset | Multi-task |
| GSM8K | Eleuther | 100 subset | Math reasoning |

---

## Reproductibilité

### Protocole de déterminisme

```python
from src.reproducibility import set_deterministic_mode

# Configure le mode déterministe global
manager = set_deterministic_mode(seed=42)
manager.capture_environment()
manager.save_experiment_config()
```

### Paramètres fixés

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Global seed | 42 | Python random |
| NumPy seed | 42 | Sampling |
| Temperature | 0 | Déterministe |
| Top-p | 1 | Pas de nucleus |

### Capture d'environnement

Chaque expérience génère un fichier `experiment_config_*.json` contenant :
- Hardware (chip, RAM, cores)
- Software (OS, Python, packages)
- Configs (models.yaml, scenarios.yaml)
- **Config hash SHA256** pour vérification

---

## Analyse Statistique

### Intervalles de Confiance (IC 95%)

Les métriques sont rapportées avec bootstrap (10,000 itérations) :

```
TTFT: 245.3 ms [95% CI: 238.1, 252.7]
Output t/s: 42.8 [95% CI: 40.2, 45.1]
```

### Tests de Significativité

Pour comparer les modèles :

| Test | Conditions | Usage |
|------|------------|-------|
| Paired t-test | Données normales, n≥20 | Mêmes prompts |
| Wilcoxon | Non-paramétrique | Petit échantillon |
| Holm correction | Comparaisons multiples | FWER |

### Tailles d'effet

- **Cohen's d** : 0.2 petit, 0.5 moyen, 0.8 grand
- **Rank-biserial r** : mêmes seuils

---

## Conformité

### Framework d'analyse

- **NIST AI RMF 1.0** - Risk Management Framework
- **NIST AI 600-1** - Generative AI Profile  
- **OWASP Top 10 LLM** - Risques spécifiques LLM

### Audit des Licences

| Model | License | Commercial |
|-------|---------|------------|
| Gemma 3n E4B | Gemma Terms of Use | Restricted |
| Qwen3-VL 4B | Apache 2.0 | ✓ Allowed |
| Ministral 3 3B | Apache 2.0 | ✓ Allowed |

### Générer les rapports de conformité

```bash
python scripts/generate_report.py --compliance
```

---

## Structure du Projet

```
edge_benchmark/
├── configs/                    # Configuration YAML
│   ├── models.yaml            # Définition des modèles
│   ├── scenarios.yaml         # Scénarios de performance
│   ├── eval_tasks.yaml        # Tâches d'évaluation
│   └── sampling_params.yaml   # Paramètres d'échantillonnage
├── src/
│   ├── __init__.py
│   ├── lmstudio_client.py     # Client API LM Studio
│   ├── checkpoint.py          # Système de checkpoint
│   ├── reproducibility.py     # Seeds et environnement
│   ├── statistics.py          # IC et tests statistiques
│   ├── performance/           # Module performance
│   │   ├── runner.py          # Orchestrateur
│   │   ├── metrics.py         # Collecte métriques
│   │   └── scenarios.py       # Définition scénarios
│   ├── capability/            # Module capacités
│   │   ├── banking_eval.py    # Banking77 + PhraseBank
│   │   ├── coding_eval.py     # HumanEval
│   │   ├── harness_runner.py  # lm-evaluation-harness
│   │   └── realistic_scenarios.py
│   └── compliance/            # Module conformité
│       ├── risk_analysis.py   # NIST/OWASP
│       └── license_audit.py   # Audit licences
├── prompts/                   # Templates de prompts
├── scripts/                   # Points d'entrée
│   ├── run_performance.py
│   ├── run_capability.py
│   └── generate_report.py
├── results/                   # Résultats générés
├── requirements.txt
├── README.md
└── TUTORIAL.md               # Guide pas à pas
```

---

## Citation

```bibtex
@software{edge_slm_benchmark,
  title = {Edge SLM Benchmark Framework},
  year = {2024},
  note = {Local Generative AI on Enterprise Edge Devices in Regulated Banking}
}
```

## Licence

Projet de recherche - Usage académique
