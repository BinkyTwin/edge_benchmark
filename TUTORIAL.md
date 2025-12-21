# Tutoriel : Edge SLM Benchmark Framework

## Guide Pas à Pas pour Réaliser ton Projet de Recherche

Ce tutoriel te guide étape par étape pour exécuter tous les benchmarks et produire les résultats nécessaires à ton article de recherche.

---

## Table des Matières

1. [Phase 1 : Préparation de l'environnement](#phase-1--préparation-de-lenvironnement)
2. [Phase 2 : Configuration de LM Studio](#phase-2--configuration-de-lm-studio)
3. [Phase 3 : Benchmarks de Performance](#phase-3--benchmarks-de-performance)
4. [Phase 4 : Benchmarks de Capacités Banking](#phase-4--benchmarks-de-capacités-banking)
5. [Phase 5 : Analyse de Conformité](#phase-5--analyse-de-conformité)
6. [Phase 6 : Génération des Rapports](#phase-6--génération-des-rapports)
7. [Phase 7 : Interprétation des Résultats](#phase-7--interprétation-des-résultats)
8. [Troubleshooting](#troubleshooting)

---

## Phase 1 : Préparation de l'environnement

### Étape 1.1 : Vérifier les prérequis

Avant de commencer, assure-toi d'avoir :

```bash
# Vérifier la version de Python (3.10+ requis)
python3 --version

# Vérifier que tu es sur Apple Silicon
uname -m
# Doit afficher : arm64
```

### Étape 1.2 : Créer l'environnement virtuel

```bash
# Naviguer vers le dossier du projet
cd /Users/lotfi/Documents/Projet/edge_benchmark

# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate

# Vérifier que tu es dans le bon environnement
which python
# Doit afficher : /Users/lotfi/Documents/Projet/edge_benchmark/venv/bin/python
```

### Étape 1.3 : Installer les dépendances

```bash
# Installer toutes les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "from src import LMStudioClient; print('✅ Installation OK')"
```

**Temps estimé :** 5-10 minutes

---

## Phase 2 : Configuration de LM Studio

### Étape 2.1 : Télécharger les modèles

1. Ouvrir **LM Studio**
2. Aller dans **Discover** (recherche de modèles)
3. Télécharger les modèles suivants :

#### Modèles GGUF (format universel)

| Modèle | Recherche | Format à choisir |
|--------|-----------|------------------|
| Gemma 3n E4B | `gemma 3n e4b` | **Q4_K_M (GGUF)** |
| Qwen3-VL 4B | `qwen3 vl 4b` | **Q4_K_M (GGUF)** |
| Ministral 3 3B | `ministral 3b` | **Q4_K_M (GGUF)** |

#### Modèles MLX (optimisé Apple Silicon)

| Modèle | Recherche | Format à choisir |
|--------|-----------|------------------|
| Gemma 3n E4B | `gemma 3n e4b mlx` | **4bit (MLX)** |
| Qwen3-VL 4B | `qwen3 vl 4b mlx` | **4bit (MLX)** |

> **⚠️ IMPORTANT : Formats MLX vs GGUF**
> 
> LM Studio utilise le **même ID** pour un modèle, peu importe le format (MLX ou GGUF).
> Par exemple, `google/gemma-3n-e4b` peut être la version MLX ou GGUF.
> 
> **Tu dois charger manuellement le bon format** dans LM Studio avant chaque benchmark.
> Le script t'indiquera quel format est attendu avant de lancer les tests.

### Étape 2.2 : Démarrer le serveur LM Studio

1. Aller dans l'onglet **Developer** (ou **Local Server**)
2. Sélectionner un modèle (commence par Gemma)
3. Cliquer sur **Start Server**
4. Vérifier que le serveur est démarré sur `http://localhost:1234`

### Étape 2.3 : Tester la connexion

```bash
# Dans ton terminal (avec venv activé)
python -c "
from src.lmstudio_client import LMStudioClient
client = LMStudioClient()
health = client.health_check()
print(f'Status: {health}')
models = client.list_models()
print(f'Modèles disponibles: {len(models)}')
for m in models:
    print(f'  - {m}')
"
```

**Tu dois voir :** Le statut "healthy" et la liste des modèles.

**Temps estimé :** 10-30 minutes (selon ta connexion pour télécharger les modèles)

---

## Phase 3 : Benchmarks de Performance

### Objectif

Mesurer TTFT (Time To First Token), tokens/s, et consommation RAM pour chaque modèle.

### Étape 3.1 : Test rapide (vérifier que tout fonctionne)

Avant de lancer un benchmark complet, fais un test rapide :

**1. Dans LM Studio :** Charge le modèle `Gemma 3n E4B` en format **GGUF Q4_K_M**

**2. Lance le test :**

```bash
# Test rapide : 5 runs seulement
python scripts/run_performance.py \
    --models gemma_3n_e4b_gguf \
    --scenarios interactive_assistant \
    --runs 5 \
    --warmup 1
```

**Ce que tu dois voir :**
```
============================================================
EDGE SLM PERFORMANCE BENCHMARK
============================================================
Models (1):
  - Gemma 3n E4B (GGUF Q4_K_M) [GGUF]
Scenarios: ['interactive_assistant']
...

⚠️  IMPORTANT: Assurez-vous que le modèle suivant est chargé dans LM Studio:
    ID: google/gemma-3n-e4b
    Format attendu: GGUF Q4_K_M

[Check] Model is responding ✓
...

RESULTS SUMMARY
────────────────────────────────────
TTFT:           XXX.X ms [95% CI: XXX.X, XXX.X]
Output t/s:     XX.X [95% CI: XX.X, XX.X]
Peak RAM:       XXXX.X MB
Success rate:   100.0%
Duration:       XX.Xs
────────────────────────────────────

[Saved] Results saved to .../perf_google_gemma-3n-e4b_GGUF_Q4_K_M_interactive_assistant_XXXXXX.jsonl
```

> **Note :** Le nom du fichier de résultats inclut maintenant le format (GGUF/MLX) et la quantization pour distinguer les tests.

### Étape 3.2 : Benchmark complet - Un modèle à la fois

**Important :** Lance les modèles un par un pour éviter les problèmes. LM Studio ne peut charger qu'un modèle à la fois.

---

#### PARTIE A : Modèles GGUF

##### Modèle 1 : Gemma 3n E4B (GGUF)

1. Dans LM Studio : Charger `Gemma 3n E4B` au format **GGUF Q4_K_M**
2. Vérifier que le serveur est démarré
3. Lancer le benchmark :

```bash
python scripts/run_performance.py \
    --models gemma_3n_e4b_gguf \
    --scenarios all \
    --runs 20 \
    --seed 42
```

**Fichiers générés :** `perf_google_gemma-3n-e4b_GGUF_Q4_K_M_*.jsonl`

**Temps estimé :** 30-45 minutes

##### Modèle 2 : Qwen3-VL 4B (GGUF)

1. Dans LM Studio : **Stopper** le serveur, charger `Qwen3-VL 4B` au format **GGUF Q4_K_M**, **redémarrer** le serveur
2. Lancer le benchmark :

```bash
python scripts/run_performance.py \
    --models qwen3_vl_4b_gguf \
    --scenarios all \
    --runs 20 \
    --seed 42
```

**Fichiers générés :** `perf_qwen_qwen3-vl-4b_GGUF_Q4_K_M_*.jsonl`

##### Modèle 3 : Ministral 3 3B (GGUF)

1. Dans LM Studio : Changer pour `Ministral 3 3B` au format **GGUF Q4_K_M**
2. Lancer le benchmark :

```bash
python scripts/run_performance.py \
    --models ministral_3_3b_gguf \
    --scenarios all \
    --runs 20 \
    --seed 42
```

**Fichiers générés :** `perf_mistralai_ministral-3-3b_GGUF_Q4_K_M_*.jsonl`

---

#### PARTIE B : Modèles MLX (pour comparaison GGUF vs MLX)

> **Objectif :** Comparer les performances GGUF vs MLX sur Apple Silicon

##### Modèle 4 : Gemma 3n E4B (MLX)

1. Dans LM Studio : Charger `Gemma 3n E4B` au format **MLX 4bit**
2. Lancer le benchmark :

```bash
python scripts/run_performance.py \
    --models gemma_3n_e4b_mlx \
    --scenarios all \
    --runs 20 \
    --seed 42
```

**Fichiers générés :** `perf_google_gemma-3n-e4b_MLX_4bit_*.jsonl`

##### Modèle 5 : Qwen3-VL 4B (MLX)

1. Dans LM Studio : Charger `Qwen3-VL 4B` au format **MLX 4bit**
2. Lancer le benchmark :

```bash
python scripts/run_performance.py \
    --models qwen3_vl_4b_mlx \
    --scenarios all \
    --runs 20 \
    --seed 42
```

**Fichiers générés :** `perf_qwen_qwen3-vl-4b_MLX_4bit_*.jsonl`

---

> **Récapitulatif des 5 benchmarks à lancer :**
> 
> | # | Modèle | Format | Clé config |
> |---|--------|--------|------------|
> | 1 | Gemma 3n E4B | GGUF Q4_K_M | `gemma_3n_e4b_gguf` |
> | 2 | Qwen3-VL 4B | GGUF Q4_K_M | `qwen3_vl_4b_gguf` |
> | 3 | Ministral 3 3B | GGUF Q4_K_M | `ministral_3_3b_gguf` |
> | 4 | Gemma 3n E4B | MLX 4bit | `gemma_3n_e4b_mlx` |
> | 5 | Qwen3-VL 4B | MLX 4bit | `qwen3_vl_4b_mlx` |

### Étape 3.3 : Vérifier les résultats

```bash
# Lister les fichiers de résultats
ls -la results/perf_*.jsonl

# Afficher un aperçu
head -n 5 results/perf_*.jsonl
```

### En cas de crash

Si le benchmark s'interrompt :

```bash
# Reprendre là où tu t'es arrêté
python scripts/run_performance.py --resume
```

---

## Phase 4 : Benchmarks de Capacités Banking

### Objectif

Évaluer la précision des modèles sur des tâches bancaires réelles.

### Étape 4.1 : Banking77 (Classification d'intents)

C'est le **benchmark principal** pour ton article.

```bash
# Pour chaque modèle, faire :

# 1. Gemma (charger dans LM Studio d'abord)
python scripts/run_capability.py \
    --task banking77 \
    --model google/gemma-3n-e4b

# 2. Qwen (changer le modèle dans LM Studio)
python scripts/run_capability.py \
    --task banking77 \
    --model qwen/qwen3-vl-4b

# 3. Ministral (changer le modèle dans LM Studio)
python scripts/run_capability.py \
    --task banking77 \
    --model mistralai/ministral-3-3b
```

**Temps estimé :** 20-40 minutes par modèle (3000+ échantillons)

### Étape 4.2 : Financial PhraseBank (Sentiment)

```bash
# Pour chaque modèle :
python scripts/run_capability.py \
    --task financial_phrasebank \
    --model google/gemma-3n-e4b \
    --sample-size 1000
```

**Temps estimé :** 15-25 minutes par modèle

### Étape 4.3 : Test de codage (optionnel mais recommandé)

```bash
python scripts/run_capability.py \
    --task coding \
    --model google/gemma-3n-e4b \
    --sample-size 30
```

**Temps estimé :** 10-15 minutes par modèle

### Étape 4.4 : Scénarios réalistes banking

```bash
python scripts/run_capability.py \
    --task realistic \
    --model google/gemma-3n-e4b
```

### Vérifier les résultats

```bash
# Lister les résultats
ls -la results/eval_*.json results/realistic_*.json

# Voir un résumé
cat results/eval_banking77_*.json | python -m json.tool | head -30
```

---

## Phase 5 : Analyse de Conformité

### Objectif

Générer les rapports d'analyse de risques et d'audit des licences.

### Étape 5.1 : Générer les rapports de conformité

```bash
# Cette commande génère automatiquement :
# - risk_analysis_*.json (analyse NIST/OWASP)
# - license_audit_*.json (audit des licences)

python scripts/generate_report.py --compliance --output report/
```

### Étape 5.2 : Examiner les résultats

```bash
# Voir le résumé des risques
cat report/compliance/risk_analysis_*.json | python -m json.tool | head -50

# Voir l'audit des licences
cat report/compliance/license_audit_*.json | python -m json.tool
```

---

## Phase 6 : Génération des Rapports

### Étape 6.1 : Générer le rapport complet

```bash
python scripts/generate_report.py \
    --input results/ \
    --output report/ \
    --format all \
    --compliance
```

### Étape 6.2 : Fichiers générés

Après cette commande, tu auras dans `report/` :

| Fichier | Description |
|---------|-------------|
| `benchmark_report_*.md` | Rapport Markdown complet |
| `benchmark_report_*.html` | Rapport HTML (visualisation navigateur) |
| `performance_results.csv` | Tableau des performances |
| `capability_results.csv` | Tableau des capacités |
| `compliance/risk_analysis_*.json` | Analyse de risques |
| `compliance/license_audit_*.json` | Audit des licences |

### Étape 6.3 : Visualiser le rapport HTML

```bash
# Ouvrir le rapport dans le navigateur
open report/benchmark_report_*.html
```

---

## Phase 7 : Interprétation des Résultats

### 7.1 Métriques de Performance

| Métrique | Bon | Moyen | Mauvais | Interprétation |
|----------|-----|-------|---------|----------------|
| TTFT | < 300ms | 300-600ms | > 600ms | Temps avant première réponse |
| tokens/s | > 40 | 20-40 | < 20 | Vitesse de génération |
| Peak RAM | < 8GB | 8-12GB | > 12GB | Sur laptop 16GB |

### 7.2 Métriques de Capacités

| Métrique | Bon | Acceptable | Insuffisant |
|----------|-----|------------|-------------|
| Banking77 Accuracy | > 70% | 50-70% | < 50% |
| Macro-F1 | > 60% | 40-60% | < 40% |
| Financial Sentiment Acc | > 75% | 60-75% | < 60% |

### 7.3 Ce qui compte pour ton article

1. **Comparaison GGUF vs MLX** : Montre les différences de performance sur Apple Silicon
2. **Trade-off Performance vs Accuracy** : Le modèle le plus rapide est-il le plus précis ?
3. **Viabilité banking** : Les modèles atteignent-ils un niveau acceptable pour le contexte bancaire ?
4. **Conformité** : Quels risques résiduels ? Quelles licences sont compatibles ?

---

## Troubleshooting

### Problème : "LM Studio server not healthy"

**Solution :**
1. Vérifier que LM Studio est démarré
2. Vérifier que le serveur local est activé
3. Vérifier le port : `curl http://localhost:1234/v1/models`

### Problème : "Out of memory"

**Solution :**
1. Fermer les autres applications
2. Utiliser un modèle plus petit (Q4 au lieu de Q8)
3. Réduire `--runs` à 10

### Problème : "Model not found" ou toutes les métriques à 0

**Solution :**
1. Vérifier que le **bon format** est chargé (MLX vs GGUF)
2. Le script affiche maintenant le format attendu :
   ```
   ⚠️  IMPORTANT: Assurez-vous que le modèle suivant est chargé dans LM Studio:
       ID: google/gemma-3n-e4b
       Format attendu: GGUF Q4_K_M
   ```
3. Vérifier l'ID exact du modèle : `curl http://localhost:1234/v1/models`
4. S'assurer que le modèle est bien **chargé** (pas juste téléchargé)

### Problème : Confusion MLX vs GGUF

**Solution :**
- Les fichiers de résultats incluent maintenant le format dans leur nom :
  - `perf_..._GGUF_Q4_K_M_...jsonl` → Version GGUF
  - `perf_..._MLX_4bit_...jsonl` → Version MLX
- Les résultats JSON contiennent aussi `model_format` et `model_quantization`

### Problème : Benchmark interrompu

**Solution :**
```bash
# Reprendre là où tu t'es arrêté
python scripts/run_performance.py --resume
python scripts/run_capability.py --resume
```

### Problème : Résultats incohérents

**Solution :**
1. Vérifier que le bon modèle est chargé dans LM Studio
2. S'assurer que la machine est branchée
3. Désactiver le mode économie d'énergie
4. Fermer les applications lourdes (navigateur, etc.)

---

## Checklist Finale

### Avant de rédiger ton article, vérifie :

- [ ] Benchmark performance exécuté pour les 3 modèles GGUF
- [ ] Benchmark performance exécuté pour les 2 modèles MLX (comparaison)
- [ ] Banking77 exécuté pour tous les modèles
- [ ] Financial PhraseBank exécuté pour tous les modèles
- [ ] Rapports de conformité générés
- [ ] CSV exportés pour les tableaux de l'article
- [ ] Screenshots/visualisations prêts

### Vérifier que les fichiers de résultats sont complets :

```bash
# Lister tous les résultats avec le format visible
ls results/perf_*_GGUF_*.jsonl  # Résultats GGUF
ls results/perf_*_MLX_*.jsonl   # Résultats MLX
```

### Fichiers à inclure dans l'article

1. **Tableau Performance** : `report/performance_results.csv`
2. **Tableau Capacités** : `report/capability_results.csv`
3. **Analyse de risques** : `report/compliance/risk_analysis_*.json`
4. **Audit licences** : `report/compliance/license_audit_*.json`

### Pour la reproductibilité

Inclure dans le paper :
- Le **config hash** (trouvé dans `results/experiment_config_*.json`)
- Les **seeds** utilisés (42 par défaut)
- La **version LM Studio**
- Les **spécifications hardware** (capturées automatiquement)

---

## Timeline Recommandée

| Phase | Durée estimée |
|-------|---------------|
| Phase 1 : Setup | 15-30 min |
| Phase 2 : LM Studio | 30-60 min (téléchargement modèles) |
| Phase 3 : Performance GGUF | 2-3 heures (3 modèles × 3 scénarios) |
| Phase 3 : Performance MLX | 1-2 heures (2 modèles × 3 scénarios) |
| Phase 4 : Capacités | 3-4 heures (3 modèles × 2-3 tâches) |
| Phase 5 : Conformité | 10 min |
| Phase 6 : Rapports | 5 min |
| **Total** | **8-10 heures** |

**Note :** Tu peux étaler sur plusieurs jours grâce au système de checkpoint !

### Workflow recommandé pour changer de modèle/format

1. **Arrêter** le serveur LM Studio (bouton Stop)
2. **Sélectionner** le nouveau modèle/format dans l'onglet "My Models"
3. **Démarrer** le serveur (bouton Start)
4. **Attendre** que le modèle soit complètement chargé (barre de progression à 100%)
5. **Lancer** le benchmark correspondant

---

## Aide Supplémentaire

Si tu as des questions :
1. Vérifie les logs dans le terminal
2. Consulte les fichiers de résultats dans `results/`
3. Regarde les checkpoints : `cat results/checkpoint_*.json`

Bon benchmark ! 🚀

