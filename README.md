# Edge SLM Benchmark Framework

**Local Generative AI on Edge Devices in Regulated Banking Environments**

A multi-dimensional benchmarking framework for evaluating Small Language Models (SLMs) on Apple Silicon in regulated banking contexts.

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## English

### Overview

This framework addresses the question: **"Can SLMs be effectively deployed on Apple Silicon laptops in regulated banking contexts?"**

We evaluate three dimensions:
- **Performance**: Latency (TTFT), throughput (tokens/s), memory consumption
- **Capability**: Banking intent classification, financial sentiment, code generation
- **Risk Analysis**: NIST AI RMF 1.0 / OWASP Top 10 LLM assessment, license audit

### Key Results

| Metric | Range | Notes |
|--------|-------|-------|
| TTFT (Interactive) | 440–737 ms | GGUF faster for first token |
| Throughput | 7–41 tokens/s | MLX better for sustained throughput |
| Banking77 Accuracy | 59–64% | Zero-shot baseline |
| Financial PhraseBank | 72–73% | Zero-shot baseline |
| HumanEval Pass@1 | 40–57% | Full benchmark (164 problems) |

### Models Evaluated

| Model | Publisher | Format | Quantization | Parameters |
|-------|-----------|--------|--------------|------------|
| Gemma 3n E4B | Google | MLX / GGUF | 4-bit / Q4_K_M | 4B |
| Qwen3-VL 4B | Alibaba | MLX / GGUF | 4-bit / Q4_K_M | 4B |
| Ministral 3 3B | Mistral AI | GGUF | Q4_K_M | 3B |

### Hardware & Software Stack

**Hardware**: MacBook Air M4 (2024), 16 GB unified memory

**Software**:
- LM Studio 0.3.9 (build 9)
- llama.cpp b4547
- MLX 0.22.0 / mlx-lm 0.21.0
- Python 3.14.0a2

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/BinkyTwin/edge_benchmark.git
cd edge_benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start LM Studio server on localhost:1234

# 3. Run benchmarks
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b
python scripts/run_performance.py --models all --scenarios interactive_assistant

# 4. Generate report
python scripts/generate_report.py --input results/ --output report/ --compliance
```

### Benchmark Protocol

- **Performance**: n=200 runs, 3-15 warmup runs (excluded), seed=42, temperature=0, top_p=1.0
- **Capability**: Zero-shot evaluation, full datasets (Banking77: 3,076 samples, FPB: 970, HumanEval: 164)
- **Checkpoint system**: Resume after interruption with `--resume`

### Important Limitations

> **What this study does NOT prove:**
> - Not a formal compliance certification (structured risk analysis, not audit)
> - Not production-ready validation (zero-shot results suited for triage, not autonomous decisions)
> - Not security testing (controls not empirically tested against adversarial attacks)
> - Not full memory measurement (RSS of benchmark process, not total LM Studio footprint)

### Project Structure

```
edge_benchmark/
├── configs/           # YAML configuration files
├── src/               # Core framework code
│   ├── performance/   # Performance benchmarking
│   ├── capability/    # Capability evaluation
│   └── compliance/    # Risk analysis (NIST/OWASP)
├── scripts/           # Entry points
├── results/           # Benchmark outputs
├── prompts/           # Prompt templates
└── v1.tex             # Research paper (LaTeX)
```

### Citation

```bibtex
@misc{djeddou2025edgeslm,
  title={Local Generative AI on Edge Devices in Regulated Banking Environments:
         Benchmarking SLM Performance and Compliance Implications on Consumer Hardware},
  author={Djeddou, Abdelatif and Bouda, Manissa},
  year={2025},
  note={GitHub: https://github.com/BinkyTwin/edge_benchmark}
}
```

---

## Français

### Présentation

Ce framework répond à la question : **« Les SLMs peuvent-ils être déployés efficacement sur des laptops Apple Silicon dans un contexte bancaire réglementé ? »**

Trois axes d'évaluation :
- **Performance** : Latence (TTFT), débit (tokens/s), consommation mémoire
- **Capacités** : Classification bancaire, analyse de sentiment, génération de code
- **Analyse de risques** : Évaluation NIST AI RMF 1.0 / OWASP Top 10 LLM, audit des licences

### Résultats Clés

| Métrique | Plage | Notes |
|----------|-------|-------|
| TTFT (Interactif) | 440–737 ms | GGUF plus rapide pour le premier token |
| Débit | 7–41 tokens/s | MLX meilleur pour le débit soutenu |
| Banking77 Accuracy | 59–64% | Baseline zero-shot |
| Financial PhraseBank | 72–73% | Baseline zero-shot |
| HumanEval Pass@1 | 40–57% | Benchmark complet (164 problèmes) |

### Installation

#### Prérequis

- **macOS** avec Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **LM Studio** avec les modèles téléchargés
- **16 Go RAM minimum**

#### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/BinkyTwin/edge_benchmark.git
cd edge_benchmark

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Configurer LM Studio
# Ouvrir LM Studio → Developer → Start Server (localhost:1234)
```

### Utilisation

#### Benchmark de Performance

```bash
# Tous les modèles, tous les scénarios
python scripts/run_performance.py --models all --scenarios all

# Un seul modèle
python scripts/run_performance.py --models gemma_3n_e4b_gguf --scenarios interactive_assistant

# Reprendre après interruption
python scripts/run_performance.py --resume
```

#### Benchmark de Capacités

```bash
# Banking77 - Classification d'intents
python scripts/run_capability.py --task banking77 --model google/gemma-3n-e4b

# Financial PhraseBank - Sentiment
python scripts/run_capability.py --task financial_phrasebank --model google/gemma-3n-e4b

# HumanEval - Code
python scripts/run_capability.py --task coding --model google/gemma-3n-e4b

# Tous les modèles
python scripts/run_capability.py --task banking77 --all-models
```

#### Génération de Rapports

```bash
python scripts/generate_report.py --input results/ --output report/ --format all --compliance
```

### Protocole Expérimental

| Paramètre | Valeur |
|-----------|--------|
| Runs par scénario | n = 200 |
| Warmup | 3 (GGUF) / 15 (MLX), exclus |
| Seed | 42 |
| Temperature | 0 |
| Top-p | 1.0 |
| Streaming | Activé |

### Limitations Importantes

> **Ce que cette étude NE prouve PAS :**
> - Pas une certification de conformité (analyse de risques structurée, pas un audit formel)
> - Pas une validation production-ready (résultats zero-shot adaptés au triage, pas aux décisions autonomes)
> - Pas de tests de sécurité empiriques (contrôles non testés contre attaques adversariales)
> - Pas de mesure mémoire complète (RSS du processus benchmark, pas l'empreinte LM Studio)

### Licences des Modèles

| Modèle | Licence | Usage Commercial |
|--------|---------|------------------|
| Gemma 3n E4B | Gemma Terms of Use | Restreint |
| Qwen3-VL 4B | Apache 2.0 | Autorisé |
| Ministral 3 3B | Apache 2.0 | Autorisé |

### Analyse de Risques

Le framework applique les cadres NIST AI RMF 1.0 et OWASP Top 10 LLM pour une évaluation structurée des risques :
- 7 risques identifiés
- 9 contrôles proposés
- 1 seul risque résiduel « medium » (prompt injection)

**Note** : Cette analyse ne constitue pas un audit de conformité formel.

### Structure du Projet

```
edge_benchmark/
├── configs/                    # Configuration YAML
│   ├── models.yaml            # Définition des modèles
│   ├── scenarios.yaml         # Scénarios de performance
│   └── eval_tasks.yaml        # Tâches d'évaluation
├── src/
│   ├── lmstudio_client.py     # Client API LM Studio
│   ├── checkpoint.py          # Système de checkpoint
│   ├── reproducibility.py     # Seeds et environnement
│   ├── statistics.py          # IC et tests statistiques
│   ├── performance/           # Module performance
│   ├── capability/            # Module capacités
│   └── compliance/            # Module analyse de risques
├── scripts/                   # Points d'entrée
│   ├── run_performance.py
│   ├── run_capability.py
│   └── generate_report.py
├── results/                   # Résultats générés
├── prompts/                   # Templates de prompts
├── v1.tex                     # Article de recherche (LaTeX)
└── requirements.txt
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Abdelatif Djeddou** - abdelatif.djeddou@edu.devinci.fr
- **Manissa Bouda** - manissa.bouda@edu.devinci.fr

École de Management Léonard de Vinci
