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
python scripts/run_performance.py --models all --scenarios all
```

### 2. Benchmark de Capacités (Banking)

```bash
python scripts/run_capability.py --task banking77 --models all
python scripts/run_capability.py --task financial_phrasebank --models all
```

### 3. Mini-benchmark Harness

```bash
python scripts/run_capability.py --task mmlu_mini --use-harness
```

### 4. Génération de Rapport

```bash
python scripts/generate_report.py --input results/ --output report/
```

## Benchmarks

### Performance
- **TTFT** (Time To First Token)
- **Output tokens/s**
- **Peak RAM**
- 3 scénarios: Interactive, Summarization, JSON

### Capacités
- **Banking77**: Intent classification (77 classes)
- **Financial PhraseBank**: Sentiment analysis
- **HumanEval Mini**: Code generation
- **MMLU/GSM8K**: Validation rapide

## Méthodologie

- Temperature = 0 (déterministe)
- 20 runs par scénario
- Warm-up de 3 requêtes
- Machine branchée, mode éco désactivé

## Conformité

Framework d'analyse basé sur:
- **NIST AI RMF 1.0**
- **OWASP Top 10 for LLM Applications**

## Licence

Projet de recherche - Usage interne

