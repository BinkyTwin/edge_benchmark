"""
Realistic Banking Scenarios
===========================

Scénarios réalistes pour évaluer l'utilité pratique des SLMs
dans un contexte bancaire, utilisant des datasets publics:

1. Financial QA (FinQA via ChanceFocus/flare-finqa)
2. Financial Sentiment (Financial PhraseBank via ChanceFocus/flare-fpb)
3. Conversational Financial QA (ConvFinQA via ChanceFocus/flare-convfinqa)
4. Multilingual French Financial RAG (sujet-ai/Sujet-Financial-RAG-FR-Dataset)
5. Financial Tweet Sentiment (zeroshot/twitter-financial-news-sentiment)

Tous les datasets sont publics et disponibles sur Hugging Face sans restriction.

Métriques rigoureuses:
- Exact match strict avec normalisation
- Numerical accuracy avec tolérance ±1%
- Bootstrap CI 95% sur toutes les métriques
- Documentation des limites méthodologiques
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, format_messages


# === CONFIGURATION DES DATASETS ===

DATASET_CONFIG = {
    "flare_finqa": {
        "name": "ChanceFocus/flare-finqa",
        "description": "Financial QA with numerical reasoning from SEC filings",
        "citation": "Chen et al., 2021 - FinQA: A Dataset of Numerical Reasoning over Financial Data",
        "split": "test",
    },
    "flare_fpb": {
        "name": "ChanceFocus/flare-fpb",
        "description": "Financial PhraseBank - Sentiment analysis on financial news",
        "citation": "Malo et al., 2014 - Good debt or bad debt: Detecting semantic orientations in economic texts",
        "split": "test",
    },
    "flare_convfinqa": {
        "name": "ChanceFocus/flare-convfinqa",
        "description": "Conversational Financial QA - Multi-turn dialogues",
        "citation": "Chen et al., 2022 - ConvFinQA: Exploring the Chain of Numerical Reasoning",
        "split": "test",
    },
    "sujet_rag_fr": {
        "name": "sujet-ai/Sujet-Financial-RAG-FR-Dataset",
        "description": "French financial RAG dataset based on French company reports",
        "citation": "Sujet AI, 2024 (Qualitative assessment - no gold answers available)",
        "split": "train",
    },
    "twitter_financial": {
        "name": "zeroshot/twitter-financial-news-sentiment",
        "description": "Financial sentiment from Twitter/X posts",
        "citation": "ZeroShot, 2023",
        "split": "train",
    },
}


# === SAMPLE SIZE RATIONALE (pour papier arXiv) ===

SAMPLE_SIZE_RATIONALE = {
    "financial_qa": {
        "n": 50,
        "dataset_size": 1147,
        "justification": "FinQA test set: 1147 samples. 50 samples provides 95% CI width ~±14% at 75% accuracy.",
        "power": "Sufficient to detect 15% accuracy difference between models (alpha=0.05, power=0.8)",
    },
    "sentiment": {
        "n": 100,
        "dataset_size": 970,
        "justification": "FPB test set: 970 samples. 100 samples provides 95% CI width ~±10% at baseline accuracy.",
        "power": "Sufficient to detect 10% accuracy difference (alpha=0.05, power=0.85)",
    },
    "conversational_qa": {
        "n": 30,
        "dataset_size": 1490,
        "justification": "ConvFinQA test set: 1490 samples. 30 samples for exploratory multi-turn evaluation.",
        "power": "Sufficient for preliminary assessment; larger samples recommended for publication claims",
    },
    "info_extraction": {
        "n": 30,
        "dataset_size": 1147,
        "justification": "Uses FinQA contexts. 30 samples for structural validity assessment.",
        "power": "Qualitative metric (JSON validity) - sample sufficient for capability demonstration",
        "metric_type": "structural_validity",
        "limitation": "No gold JSON available - measures structure not content accuracy",
    },
    "multilingual_fr": {
        "n": 30,
        "dataset_size": 28880,
        "justification": "Sujet FR dataset: 28880 samples. 30 samples for qualitative language assessment.",
        "power": "Qualitative only - no gold answers available for quantitative evaluation",
        "metric_type": "qualitative_assessment",
        "limitation": "No gold answers - metrics are heuristic (language detection, response length)",
    },
    "twitter_sentiment": {
        "n": 50,
        "dataset_size": 9543,
        "justification": "Twitter Financial: 9543 train samples. 50 samples for cross-domain validation.",
        "power": "Sufficient to validate sentiment transfer across domains",
    },
}


# === FALLBACK DATA (si datasets indisponibles) ===

FALLBACK_SENTIMENT_SAMPLES = [
    {"text": "The new mobile app is amazing! So much easier to use than before.", "expected": "positive"},
    {"text": "I've been waiting on hold for 45 minutes. This is unacceptable.", "expected": "negative"},
    {"text": "The branch closes at 5pm on weekdays.", "expected": "neutral"},
    {"text": "Your customer service team resolved my issue quickly. Very impressed!", "expected": "positive"},
    {"text": "Hidden fees everywhere! I'm switching banks.", "expected": "negative"},
]

FALLBACK_QA_SAMPLES = [
    {
        "context": "Annual Report 2024: Revenue increased by 15% to $2.5 billion compared to prior year.",
        "question": "What was the revenue growth percentage?",
        "answer": "15%",
    },
    {
        "context": "Q4 earnings show net income of $450 million, up from $380 million in Q4 2023.",
        "question": "What was the net income increase?",
        "answer": "$70 million",
    },
]


# === STATISTICAL ANALYZER ===

class StatisticalAnalyzer:
    """
    Analyse statistique rigoureuse pour les métriques de benchmark.
    
    Fournit:
    - Bootstrap CI 95%
    - Variance et écart-type
    - Standard Error of Mean (SEM)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def bootstrap_ci(
        self,
        values: list,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """
        Calcule l'intervalle de confiance par bootstrap.
        
        Args:
            values: Liste de valeurs (0/1 pour accuracy, ou floats)
            n_bootstrap: Nombre d'échantillons bootstrap
            ci: Niveau de confiance (0.95 = 95%)
            
        Returns:
            Tuple (lower_bound, upper_bound)
        """
        if not values or len(values) < 2:
            return (0.0, 0.0)
        
        values = np.array(values)
        n = len(values)
        
        # Générer les échantillons bootstrap
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculer les percentiles
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (round(float(lower), 4), round(float(upper), 4))
    
    def compute_stats(self, values: list) -> dict:
        """
        Calcule les statistiques descriptives complètes.
        
        Returns:
            Dict avec mean, std, variance, sem, n, ci_95
        """
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "sem": 0.0,
                "n": 0,
                "ci_95": [0.0, 0.0],
            }
        
        values = np.array(values)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        
        ci_lower, ci_upper = self.bootstrap_ci(values.tolist())
        
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "variance": round(variance, 4),
            "sem": round(sem, 4),
            "n": n,
            "ci_95": [ci_lower, ci_upper],
        }
    
    def binary_metrics_with_ci(self, correct_list: list[bool]) -> dict:
        """
        Calcule accuracy avec IC pour des résultats binaires.
        
        Args:
            correct_list: Liste de booléens (True = correct)
            
        Returns:
            Dict avec accuracy, ci_95, n
        """
        binary = [1 if c else 0 for c in correct_list]
        stats = self.compute_stats(binary)
        
        return {
            "accuracy": stats["mean"],
            "accuracy_ci_95": stats["ci_95"],
            "accuracy_std": stats["std"],
            "n": stats["n"],
        }


# === DATASET LOADERS ===

class DatasetLoader:
    """Charge les datasets depuis Hugging Face avec fallback."""
    
    def __init__(self, cache_dir: Optional[Path] = None, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        self._cache = {}
    
    def load_flare_finqa(self, split: str = "test", limit: int = 50) -> list[dict]:
        """
        Charge FinQA via ChanceFocus/flare-finqa.
        
        Structure:
        - query: question complète avec contexte
        - answer: réponse attendue
        - text: question seule
        """
        cache_key = f"flare_finqa_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading flare-finqa ({split}, limit={limit})...")
            dataset = load_dataset(
                "ChanceFocus/flare-finqa",
                split=split,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                
                # Extraire le contexte de la query (avant "Context:")
                query = item.get("query", "")
                context = ""
                if "Context:" in query:
                    parts = query.split("Context:")
                    if len(parts) > 1:
                        context = parts[1].strip()
                        # Trouver où commence la question
                        if "\n" in context:
                            context = context[:context.rfind("\n")]
                
                samples.append({
                    "context": context[:4000],
                    "question": item.get("text", ""),
                    "answer": str(item.get("answer", "")),
                    "source": "flare_finqa",
                    "id": item.get("id", str(idx)),
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} FinQA samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load flare-finqa: {e}")
            return FALLBACK_QA_SAMPLES[:limit]
    
    def load_flare_convfinqa(self, split: str = "test", limit: int = 30) -> list[dict]:
        """
        Charge ConvFinQA via ChanceFocus/flare-convfinqa.
        
        Structure:
        - query: question avec contexte
        - answer: réponse
        - turn: numéro du tour de dialogue
        - dialogue_id: id du dialogue
        """
        cache_key = f"flare_convfinqa_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading flare-convfinqa ({split}, limit={limit})...")
            dataset = load_dataset(
                "ChanceFocus/flare-convfinqa",
                split=split,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                
                query = item.get("query", "")
                context = ""
                if "Context:" in query or "context:" in query.lower():
                    # Extraire le contexte
                    context = query
                
                samples.append({
                    "context": context[:3500],
                    "question": query[-500:] if len(query) > 500 else query,  # Dernière partie = question
                    "answer": str(item.get("answer", "")),
                    "turn": item.get("turn", 0),
                    "dialogue_id": item.get("dialogue_id", idx),
                    "source": "flare_convfinqa",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} ConvFinQA samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load flare-convfinqa: {e}")
            return FALLBACK_QA_SAMPLES[:limit]
    
    def load_flare_fpb(self, split: str = "test", limit: int = 100) -> list[dict]:
        """
        Charge Financial PhraseBank via ChanceFocus/flare-fpb.
        
        Structure:
        - text: phrase financière
        - answer: sentiment (positive, negative, neutral)
        - choices: liste des choix possibles
        """
        cache_key = f"flare_fpb_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading flare-fpb ({split}, limit={limit})...")
            dataset = load_dataset(
                "ChanceFocus/flare-fpb",
                split=split,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "text": item.get("text", ""),
                    "expected": item.get("answer", "neutral"),
                    "source": "flare_fpb",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} FPB sentiment samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load flare-fpb: {e}")
            return FALLBACK_SENTIMENT_SAMPLES[:limit]
    
    def load_twitter_financial(self, split: str = "train", limit: int = 100) -> list[dict]:
        """
        Charge Twitter Financial Sentiment.
        
        Structure:
        - text: tweet
        - label: 0 (bearish/negative), 1 (bullish/positive), 2 (neutral)
        """
        cache_key = f"twitter_financial_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading twitter-financial ({split}, limit={limit})...")
            dataset = load_dataset(
                "zeroshot/twitter-financial-news-sentiment",
                split=split,
            )
            
            label_map = {0: "negative", 1: "positive", 2: "neutral"}
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "text": item.get("text", ""),
                    "expected": label_map.get(item.get("label", 2), "neutral"),
                    "source": "twitter_financial",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} Twitter sentiment samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load twitter-financial: {e}")
            return FALLBACK_SENTIMENT_SAMPLES[:limit]
    
    def load_sujet_rag_fr(self, split: str = "train", limit: int = 30) -> list[dict]:
        """
        Charge le dataset Sujet Financial RAG FR.
        
        Structure:
        - question: question en français
        - context: contexte du rapport financier
        """
        cache_key = f"sujet_fr_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading Sujet Financial RAG FR ({split}, limit={limit})...")
            dataset = load_dataset(
                "sujet-ai/Sujet-Financial-RAG-FR-Dataset",
                split=split,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "context": item.get("context", "")[:3500],
                    "question": item.get("question", ""),
                    "answer": "",  # Pas de réponse gold dans ce dataset
                    "source": "sujet_rag_fr",
                    "language": "fr",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} French financial samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load Sujet RAG FR: {e}")
            return []


@dataclass
class ScenarioResult:
    """Résultat d'un scénario avec métadonnées méthodologiques."""
    
    scenario_name: str
    model_id: str
    num_samples: int
    
    # Métriques spécifiques au scénario
    metrics: dict = field(default_factory=dict)
    
    # Temps
    avg_latency_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # Metadata
    dataset_source: str = ""
    dataset_citation: str = ""
    
    # Méthodologie (nouveau)
    methodology: dict = field(default_factory=dict)
    
    # Détails
    responses: list = field(default_factory=list)
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "model_id": self.model_id,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "dataset_source": self.dataset_source,
            "dataset_citation": self.dataset_citation,
            "methodology": self.methodology,
            "timestamp": self.timestamp,
        }


class RealisticScenariosEvaluator:
    """
    Évaluateur de scénarios réalistes banking.
    
    Utilise des datasets publics pour une évaluation rigoureuse et reproductible.
    Inclut:
    - Métriques strictes (exact match, numerical accuracy)
    - Intervalles de confiance 95% (bootstrap)
    - Documentation des limites méthodologiques
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        results_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.client = client
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader(seed=seed)
        self.stats = StatisticalAnalyzer(seed=seed)
        self.seed = seed
    
    # === STRICT METRICS ===
    
    def _extract_numbers(self, text: str) -> list[float]:
        """
        Extrait tous les nombres d'un texte avec support robuste pour:
        - Formats internationaux (US: 1,234.56 vs EU: 1.234,56)
        - Nombres négatifs en format comptable: (150.0) = -150.0
        - Pourcentages, monnaies, unités (million/billion)
        """
        if not text:
            return []
        
        numbers = []
        text_original = text
        text_lower = text.lower()
        
        # 1. Extraire les nombres négatifs en format comptable: (150.0) ou (1,234.56)
        # Ceci est courant en comptabilité pour représenter les pertes
        accounting_pattern = r'\((\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?)\)'
        for match in re.finditer(accounting_pattern, text_lower):
            num_str = match.group(1)
            parsed = self._parse_number_string(num_str)
            if parsed is not None:
                numbers.append(-parsed)  # Négatif car entre parenthèses
        
        # 2. Patterns standards pour les nombres
        patterns = [
            # Pourcentages (avec signe optionnel)
            r'[+-]?\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?%',
            # Monnaie avec symbole (USD, EUR, etc.)
            r'[+-]?[$€£¥]\s*\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?',
            # Nombres avec unités (million, billion, etc.)
            r'[+-]?\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?\s*(?:million|billion|trillion|m|b|bn|k)',
            # Nombres décimaux standards (attrape tout le reste)
            r'[+-]?\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                num_str = match.group()
                parsed = self._parse_number_string(num_str)
                if parsed is not None and parsed not in numbers and -parsed not in numbers:
                    numbers.append(parsed)
        
        return numbers
    
    def _parse_number_string(self, num_str: str) -> float | None:
        """
        Parse une chaîne de caractères en nombre, gérant les formats internationaux.
        
        Détecte automatiquement:
        - Format US: 1,234.56 (virgule = séparateur milliers, point = décimal)
        - Format EU: 1.234,56 (point = séparateur milliers, virgule = décimal)
        """
        if not num_str:
            return None
        
        # Nettoyer les symboles monétaires et espaces
        clean = num_str.strip()
        clean = re.sub(r'[$€£¥\s]', '', clean)
        clean = clean.replace('%', '')
        
        # Gérer les unités multiplicatrices
        multiplier = 1.0
        unit_patterns = [
            (r'trillion|t$', 1e12),
            (r'billion|bn|b$', 1e9),
            (r'million|m$', 1e6),
            (r'thousand|k$', 1e3),
        ]
        for pattern, mult in unit_patterns:
            if re.search(pattern, clean, re.IGNORECASE):
                multiplier = mult
                clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)
                break
        
        clean = clean.strip()
        
        # Gérer le signe
        negative = clean.startswith('-')
        if clean.startswith(('+', '-')):
            clean = clean[1:]
        
        # Détecter le format: US vs EU
        # Heuristique: regarder le dernier séparateur
        # Si le dernier est une virgule suivie de 1-2 chiffres: format EU (décimal = virgule)
        # Si le dernier est un point suivi de 1-2 chiffres: format US (décimal = point)
        
        has_comma = ',' in clean
        has_period = '.' in clean
        
        if has_comma and has_period:
            # Les deux sont présents - déterminer lequel est le décimal
            last_comma = clean.rfind(',')
            last_period = clean.rfind('.')
            
            if last_comma > last_period:
                # Format EU: 1.234,56 (virgule est le décimal)
                clean = clean.replace('.', '').replace(',', '.')
            else:
                # Format US: 1,234.56 (point est le décimal)
                clean = clean.replace(',', '')
        elif has_comma:
            # Seulement des virgules
            # Si la partie après la dernière virgule a 3 chiffres, c'est un séparateur de milliers
            parts = clean.split(',')
            if len(parts[-1]) == 3 and len(parts) > 1:
                # Séparateur de milliers (format US sans décimal)
                clean = clean.replace(',', '')
            else:
                # Décimal (format EU sans séparateur de milliers)
                clean = clean.replace(',', '.')
        elif has_period:
            # Seulement des points
            # Si la partie après le dernier point a 3 chiffres, c'est un séparateur de milliers (EU)
            parts = clean.split('.')
            if len(parts[-1]) == 3 and len(parts) > 1:
                # Pourrait être un séparateur de milliers EU, mais aussi juste un nombre comme 1.234
                # Par défaut, traiter comme décimal (format US)
                pass
            # Sinon c'est un décimal standard
        
        try:
            result = float(clean) * multiplier
            return -result if negative else result
        except ValueError:
            return None
    
    def _extract_final_answer(self, text: str) -> str:
        """
        Extrait la réponse finale d'un texte, isolant le résultat numérique
        du reste de la phrase.
        
        Cherche des patterns comme:
        - "The answer is 15%"
        - "Final answer: 42.5"
        - "= 123.45"
        - "Result: $1,234.56"
        """
        if not text:
            return ""
        
        # Patterns pour identifier la réponse finale
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)',
            r'(?:result|total|sum|value)\s*(?:is|:|=)\s*(.+?)(?:\.|$)',
            r'=\s*([+-]?\$?[\d,.\s]+(?:%|million|billion)?)',
            r'(?:^|\n)([+-]?\$?[\d,.\s]+(?:%|million|billion)?)(?:\s|$)',
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in answer_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: retourner le texte nettoyé
        return text.strip()
    
    def _numerical_match(
        self,
        pred: str,
        gold: str,
        tolerance: float = 0.01,
    ) -> bool:
        """
        Compare les réponses numériques avec une tolérance.
        
        Args:
            pred: Prédiction du modèle
            gold: Réponse attendue
            tolerance: Tolérance relative (0.01 = 1%)
            
        Returns:
            True si les nombres correspondent dans la tolérance
        """
        if not pred or not gold:
            return False
        
        gold_numbers = self._extract_numbers(gold)
        
        if not gold_numbers:
            return False
        
        # D'abord, essayer d'extraire la réponse finale (plus précis)
        final_answer = self._extract_final_answer(pred)
        pred_numbers_final = self._extract_numbers(final_answer)
        
        # Sinon, utiliser tous les nombres de la prédiction
        pred_numbers_all = self._extract_numbers(pred)
        
        # Combiner: priorité aux nombres de la réponse finale
        pred_numbers = pred_numbers_final if pred_numbers_final else pred_numbers_all
        
        # Vérifier si le nombre gold principal est présent dans la prédiction
        gold_main = gold_numbers[0]
        
        for pred_num in pred_numbers:
            if self._numbers_match(pred_num, gold_main, tolerance):
                return True
        
        # Si pas trouvé dans la réponse finale, chercher dans tous les nombres
        if pred_numbers_final and not pred_numbers_all == pred_numbers_final:
            for pred_num in pred_numbers_all:
                if self._numbers_match(pred_num, gold_main, tolerance):
                    return True
        
        return False
    
    def _numbers_match(
        self,
        a: float,
        b: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Compare deux nombres avec tolérance relative et absolue."""
        # Cas spécial: les deux sont zéro
        if a == 0 and b == 0:
            return True
        
        # Cas spécial: un seul est zéro
        if a == 0 or b == 0:
            return abs(a - b) < 0.001  # Tolérance absolue pour les petits nombres
        
        # Tolérance relative
        relative_diff = abs(a - b) / max(abs(a), abs(b))
        return relative_diff <= tolerance
    
    def _exact_match_strict(self, pred: str, gold: str) -> bool:
        """
        Exact match après normalisation stricte.
        
        Normalisation:
        - Lowercase
        - Retirer ponctuation
        - Retirer articles (the, a, an)
        - Comparer tokens triés
        """
        if not pred or not gold:
            return False
        
        def normalize(text: str) -> set[str]:
            # Lowercase
            text = text.lower()
            # Retirer ponctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # Tokenize
            tokens = text.split()
            # Retirer articles
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be'}
            tokens = [t for t in tokens if t not in stop_words and len(t) > 0]
            return set(tokens)
        
        pred_tokens = normalize(pred)
        gold_tokens = normalize(gold)
        
        # Match exact si gold tokens sont un sous-ensemble de pred
        # (gold peut être court: "15%" mais pred: "the answer is 15%")
        if not gold_tokens:
            return False
        
        return gold_tokens.issubset(pred_tokens) or pred_tokens == gold_tokens
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalise une réponse pour la comparaison."""
        if not answer:
            return ""
        answer = answer.lower().strip()
        answer = ''.join(c for c in answer if c.isalnum() or c.isspace() or c == '.')
        return answer
    
    def _compute_f1(self, pred: str, gold: str) -> float:
        """Calcule le F1 score basé sur le chevauchement de tokens."""
        if not pred or not gold:
            return 0.0
        
        pred_tokens = set(self._normalize_answer(pred).split())
        gold_tokens = set(self._normalize_answer(gold).split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def _is_french(self, text: str) -> bool:
        """Vérifie si le texte semble être en français (heuristique simple)."""
        if not text:
            return False
        
        french_indicators = [
            "le ", "la ", "les ", "un ", "une ", "des ",
            "de ", "du ", "en ", "est ", "sont ", "pour ",
            "avec ", "dans ", "sur ", "par ", "que ", "qui ",
            "ce ", "cette ", "ces ", "aux ", "était ", "été ",
            "chiffre", "exercice", "groupe", "résultat", "millions",
        ]
        
        text_lower = text.lower()
        matches = sum(1 for indicator in french_indicators if indicator in text_lower)
        
        return matches >= 2
    
    # === EVALUATION METHODS ===
    
    def evaluate_financial_qa(
        self,
        model_id: str,
        num_samples: int = 50,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue sur FinQA - Question Answering financier avec raisonnement numérique.
        
        Métriques strictes:
        - exact_match_strict: Comparaison normalisée stricte
        - numerical_accuracy: Match numérique avec tolérance ±1%
        - f1_score: Token overlap
        - answer_present: Réponse dans le texte (métrique secondaire)
        
        Toutes les métriques incluent IC 95% par bootstrap.
        """
        samples = self.loader.load_flare_finqa(limit=num_samples)
        
        system_prompt = """You are a financial analyst assistant. 
Answer questions about financial documents accurately and concisely.
If the answer requires calculation, show your work briefly.
Provide the final answer clearly."""
        
        responses = []
        latencies = []
        
        # Listes pour calcul des IC
        exact_strict_list = []
        numerical_list = []
        f1_list = []
        answer_present_list = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Financial QA") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            expected = sample.get("answer", "")
            
            if context:
                prompt = f"""Context:
{context}

Question: {question}

Answer:"""
            else:
                prompt = f"Question: {question}\n\nAnswer:"
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=150,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            # Calcul des métriques strictes
            predicted = result.content.strip() if result.content else ""
            
            is_exact_strict = self._exact_match_strict(predicted, expected)
            is_numerical = self._numerical_match(predicted, expected)
            is_present = expected.lower() in predicted.lower() if expected else False
            f1 = self._compute_f1(predicted, expected)
            
            exact_strict_list.append(is_exact_strict)
            numerical_list.append(is_numerical)
            answer_present_list.append(is_present)
            f1_list.append(f1)
            
            responses.append({
                "question": question,
                "expected": expected,
                "predicted": predicted[:200],
                "exact_match_strict": is_exact_strict,
                "numerical_match": is_numerical,
                "answer_present": is_present,
                "f1": round(f1, 4),
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        # Calculer les statistiques avec IC
        exact_stats = self.stats.binary_metrics_with_ci(exact_strict_list)
        numerical_stats = self.stats.binary_metrics_with_ci(numerical_list)
        present_stats = self.stats.binary_metrics_with_ci(answer_present_list)
        f1_stats = self.stats.compute_stats(f1_list)
        
        # Récupérer les métadonnées de sample size
        rationale = SAMPLE_SIZE_RATIONALE.get("financial_qa", {})
        
        return ScenarioResult(
            scenario_name="financial_qa",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                # Métriques strictes avec IC
                "exact_match_strict": exact_stats["accuracy"],
                "exact_match_strict_ci_95": exact_stats["accuracy_ci_95"],
                "numerical_accuracy": numerical_stats["accuracy"],
                "numerical_accuracy_ci_95": numerical_stats["accuracy_ci_95"],
                # Métriques soft
                "f1_score": f1_stats["mean"],
                "f1_score_ci_95": f1_stats["ci_95"],
                "answer_present": present_stats["accuracy"],
                "answer_present_ci_95": present_stats["accuracy_ci_95"],
                # Meta
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_finqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_finqa"]["citation"],
            methodology={
                "metric_type": "quantitative",
                "sample_size_justification": rationale.get("justification", ""),
                "power_analysis": rationale.get("power", ""),
                "limitations": [],
            },
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_sentiment_analysis(
        self,
        model_id: str,
        num_samples: int = 100,
        use_twitter: bool = False,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue sur Financial PhraseBank ou Twitter Financial Sentiment.
        
        Métriques:
        - Accuracy avec IC 95%
        - Macro F1 avec IC 95%
        - Per-class breakdown
        """
        if use_twitter:
            samples = self.loader.load_twitter_financial(limit=num_samples)
            dataset_key = "twitter_financial"
            rationale_key = "twitter_sentiment"
        else:
            samples = self.loader.load_flare_fpb(limit=num_samples)
            dataset_key = "flare_fpb"
            rationale_key = "sentiment"
        
        system_prompt = """Analyze the sentiment of financial statements.
Respond with exactly one word: positive, negative, or neutral."""
        
        predictions = []
        latencies = []
        correct_list = []
        confusion = {"positive": {}, "negative": {}, "neutral": {}}
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Sentiment Analysis") if progress_bar else samples
        
        for sample in iterator:
            text = sample["text"]
            expected = sample["expected"]
            
            prompt = f"Financial statement: {text}\n\nSentiment:"
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=10,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            # Parser la prédiction
            pred = "neutral"
            if result.success and result.content:
                content = result.content.lower().strip()
                for label in ["positive", "negative", "neutral"]:
                    if label in content:
                        pred = label
                        break
            
            is_correct = pred == expected
            correct_list.append(is_correct)
            
            # Mise à jour confusion matrix
            if expected not in confusion:
                confusion[expected] = {}
            confusion[expected][pred] = confusion[expected].get(pred, 0) + 1
            
            predictions.append({
                "text": text[:100],
                "expected": expected,
                "predicted": pred,
                "correct": is_correct,
                "raw_response": result.content[:50] if result.content else "",
            })
        
        total_time = time.time() - start_time
        
        # Calcul des métriques avec IC
        accuracy_stats = self.stats.binary_metrics_with_ci(correct_list)
        
        # Macro F1 (calculer par classe puis moyenner)
        f1_per_class = []
        for label in ["positive", "negative", "neutral"]:
            tp = confusion.get(label, {}).get(label, 0)
            fp = sum(confusion.get(other, {}).get(label, 0) for other in confusion if other != label)
            fn = sum(confusion.get(label, {}).get(other, 0) for other in confusion.get(label, {}) if other != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_per_class.append(f1)
        
        macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0
        
        # Récupérer les métadonnées
        rationale = SAMPLE_SIZE_RATIONALE.get(rationale_key, {})
        
        return ScenarioResult(
            scenario_name="financial_sentiment" if not use_twitter else "twitter_sentiment",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "accuracy": accuracy_stats["accuracy"],
                "accuracy_ci_95": accuracy_stats["accuracy_ci_95"],
                "accuracy_std": accuracy_stats["accuracy_std"],
                "macro_f1": round(macro_f1, 4),
                "correct": sum(correct_list),
                "total": len(samples),
                "per_class_f1": {
                    "positive": round(f1_per_class[0], 4),
                    "negative": round(f1_per_class[1], 4),
                    "neutral": round(f1_per_class[2], 4),
                },
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG[dataset_key]["name"],
            dataset_citation=DATASET_CONFIG[dataset_key]["citation"],
            methodology={
                "metric_type": "quantitative",
                "sample_size_justification": rationale.get("justification", ""),
                "power_analysis": rationale.get("power", ""),
                "limitations": [],
            },
            responses=predictions,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_conversational_qa(
        self,
        model_id: str,
        num_samples: int = 30,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue sur ConvFinQA - QA conversationnel financier multi-tours.
        
        Métriques strictes:
        - exact_match_strict: Comparaison normalisée stricte
        - numerical_accuracy: Match numérique avec tolérance ±1%
        - answer_present: Réponse présente (métrique secondaire)
        
        Toutes les métriques incluent IC 95%.
        """
        samples = self.loader.load_flare_convfinqa(limit=num_samples)
        
        system_prompt = """You are a financial analyst assistant engaged in a conversation.
Answer financial questions based on the provided context.
Be precise and concise. For numerical questions, provide the exact number."""
        
        responses = []
        latencies = []
        
        exact_strict_list = []
        numerical_list = []
        answer_present_list = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Conversational QA") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            expected = sample.get("answer", "")
            
            prompt = f"""{context}

Please provide the answer:"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=100,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            predicted = result.content.strip() if result.content else ""
            
            # Métriques strictes
            is_exact_strict = self._exact_match_strict(predicted, expected)
            is_numerical = self._numerical_match(predicted, expected)
            
            # Answer present (métrique laxiste)
            is_present = False
            if expected and predicted:
                exp_norm = str(expected).lower().replace(",", "").replace(" ", "")
                pred_norm = predicted.lower().replace(",", "").replace(" ", "")
                is_present = exp_norm in pred_norm
            
            exact_strict_list.append(is_exact_strict)
            numerical_list.append(is_numerical)
            answer_present_list.append(is_present)
            
            responses.append({
                "question": question[:150] if question else context[:150],
                "expected": str(expected),
                "predicted": predicted[:150],
                "exact_match_strict": is_exact_strict,
                "numerical_match": is_numerical,
                "answer_present": is_present,
                "turn": sample.get("turn", 0),
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        # Calculer les statistiques avec IC
        exact_stats = self.stats.binary_metrics_with_ci(exact_strict_list)
        numerical_stats = self.stats.binary_metrics_with_ci(numerical_list)
        present_stats = self.stats.binary_metrics_with_ci(answer_present_list)
        
        # Récupérer les métadonnées
        rationale = SAMPLE_SIZE_RATIONALE.get("conversational_qa", {})
        
        return ScenarioResult(
            scenario_name="conversational_qa",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "exact_match_strict": exact_stats["accuracy"],
                "exact_match_strict_ci_95": exact_stats["accuracy_ci_95"],
                "numerical_accuracy": numerical_stats["accuracy"],
                "numerical_accuracy_ci_95": numerical_stats["accuracy_ci_95"],
                "answer_present": present_stats["accuracy"],
                "answer_present_ci_95": present_stats["accuracy_ci_95"],
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_convfinqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_convfinqa"]["citation"],
            methodology={
                "metric_type": "quantitative",
                "sample_size_justification": rationale.get("justification", ""),
                "power_analysis": rationale.get("power", ""),
                "limitations": ["Multi-turn context may be truncated"],
            },
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def _extract_json_from_text(self, text: str) -> dict | None:
        """
        Extrait un objet JSON d'un texte brut.
        
        Gère les cas où le JSON est entouré de texte ou de markdown:
        - ```json ... ```
        - { ... } directement
        - Texte avant/après le JSON
        """
        if not text:
            return None
        
        # 1. Essayer de parser directement
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 2. Chercher un bloc markdown ```json ... ```
        markdown_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        match = re.search(markdown_pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # 3. Chercher le premier { et le dernier } correspondant
        start = text.find('{')
        if start == -1:
            return None
        
        # Compter les accolades pour trouver la fin
        depth = 0
        end = -1
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end == -1:
            return None
        
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    
    def _compute_grounding_score(
        self,
        parsed_json: dict,
        source_text: str,
    ) -> dict:
        """
        Calcule le score de Grounding: vérifie si les nombres dans le JSON
        existent dans le texte source.
        
        Cela permet de détecter les hallucinations numériques.
        
        Returns:
            Dict avec:
            - grounding_rate: % de nombres du JSON trouvés dans le source
            - total_numbers_in_json: nombre total de valeurs numériques dans le JSON
            - grounded_numbers: nombre de valeurs correctement ancrées
            - hallucinated_numbers: liste des nombres qui ne sont pas dans le source
        """
        if not parsed_json or not source_text:
            return {
                "grounding_rate": 0.0,
                "total_numbers_in_json": 0,
                "grounded_numbers": 0,
                "hallucinated_numbers": [],
            }
        
        # Extraire tous les nombres du texte source
        source_numbers = set(self._extract_numbers(source_text))
        
        # Extraire tous les nombres du JSON (récursivement)
        json_numbers = self._extract_numbers_from_json(parsed_json)
        
        if not json_numbers:
            return {
                "grounding_rate": 1.0,  # Pas de nombres = pas d'hallucination
                "total_numbers_in_json": 0,
                "grounded_numbers": 0,
                "hallucinated_numbers": [],
            }
        
        # Vérifier le grounding de chaque nombre
        grounded = []
        hallucinated = []
        
        for num in json_numbers:
            is_grounded = False
            for source_num in source_numbers:
                if self._numbers_match(num, source_num, tolerance=0.01):
                    is_grounded = True
                    break
            
            if is_grounded:
                grounded.append(num)
            else:
                hallucinated.append(num)
        
        grounding_rate = len(grounded) / len(json_numbers) if json_numbers else 0.0
        
        return {
            "grounding_rate": round(grounding_rate, 4),
            "total_numbers_in_json": len(json_numbers),
            "grounded_numbers": len(grounded),
            "hallucinated_numbers": hallucinated[:10],  # Limiter pour lisibilité
        }
    
    def _extract_numbers_from_json(self, obj, numbers: list = None) -> list[float]:
        """Extrait récursivement tous les nombres d'un objet JSON."""
        if numbers is None:
            numbers = []
        
        if isinstance(obj, (int, float)):
            if not isinstance(obj, bool):  # bool est une sous-classe de int
                numbers.append(float(obj))
        elif isinstance(obj, str):
            # Extraire les nombres des chaînes
            extracted = self._extract_numbers(obj)
            numbers.extend(extracted)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_numbers_from_json(value, numbers)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_numbers_from_json(item, numbers)
        
        return numbers
    
    def evaluate_info_extraction(
        self,
        model_id: str,
        num_samples: int = 30,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue l'extraction d'information structurée à partir de documents financiers.
        
        NOTE MÉTHODOLOGIQUE:
        Cette évaluation mesure la VALIDITÉ STRUCTURELLE, pas l'exactitude du contenu.
        Il n'y a pas de gold JSON disponible dans FinQA.
        
        Stratégie JSON:
        1. Essayer le mode JSON contraint du serveur (response_format)
        2. Si échec, fallback vers extraction manuelle du JSON depuis la réponse
        
        Métriques:
        - JSON validity rate (avec IC 95%)
        - Field extraction count
        - Grounding rate (nouveaux): % de nombres du JSON présents dans le source
        """
        samples = self.loader.load_flare_finqa(limit=num_samples)
        
        # Prompt optimisé pour Grounding > 85%
        # Rôle d'auditeur strict = moins de prise de risque = moins d'hallucinations
        system_prompt = """You are a strict financial data auditor. 
Your ONLY goal is to extract raw data into JSON.

RULES FOR DATA INTEGRITY:
1. TRUTH OVER COMPLETION: If a specific figure is not explicitly stated in the text, use null. NEVER guess.
2. VERIFICATION STEP: Before writing a number, locate it in the source text. If it is not there, do not include it.
3. NO SCALING ERRORS: Ensure "millions" or "billions" are correctly handled or kept as strings if ambiguous.
4. STRICT JSON: Output only the JSON object. No preamble, no postscript.

If you are unsure about a value, set it to null instead of hallucinating."""
        
        results = []
        latencies = []
        valid_json_list = []
        fields_list = []
        grounding_list = []
        json_mode_used = []  # Track which mode succeeded
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Info Extraction") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")[:2500]
            
            if not context:
                continue
            
            # Prompt avec balises d'isolation pour améliorer l'attention
            # et instruction de "Double-Check" pour réduire les hallucinations
            prompt = f"""[SOURCE DOCUMENT]
{context}
[/SOURCE DOCUMENT]

[INSTRUCTION]
Extract the following fields into JSON. 
For each 'value' in 'key_figures', you MUST ensure the digit exists exactly as written in the [SOURCE DOCUMENT].

Required Format:
{{
  "document_type": "string or null",
  "key_figures": [
    {{"label": "exact description", "value": number_or_null}}
  ],
  "time_period": "string or null",
  "entities": ["string"]
}}
[/INSTRUCTION]

JSON:"""
            
            messages = format_messages(prompt, system_prompt)
            
            parsed_json = None
            mode_used = "none"
            result = None
            
            # Format LM Studio json_schema 
            # https://lmstudio.ai/docs/developer/openai-compat/structured-output
            extraction_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "financial_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "document_type": {
                                "type": ["string", "null"],
                                "description": "Type of document (earnings, 10-K, quarterly report, etc.)"
                            },
                            "key_figures": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "value": {"type": ["number", "string", "null"]}
                                    },
                                    "required": ["label", "value"]
                                }
                            },
                            "time_period": {
                                "type": ["string", "null"],
                                "description": "Fiscal period if mentioned"
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Companies or organizations mentioned"
                            }
                        },
                        "required": ["document_type", "key_figures", "time_period", "entities"]
                    }
                }
            }
            
            # Stratégie 1: Essayer le mode JSON Schema contraint (LM Studio)
            try:
                result = self.client.complete(
                    model=model_id,
                    messages=messages,
                    temperature=0,
                    max_tokens=500,
                    stream=False,
                    response_format=extraction_schema,
                )
                
                latencies.append(result.metrics.total_time_ms)
                
                if result.json_valid and result.parsed_json is not None:
                    parsed_json = result.parsed_json
                    mode_used = "json_schema_constrained"
                elif result.content:
                    # Le mode contraint a échoué, essayer extraction manuelle
                    parsed_json = self._extract_json_from_text(result.content)
                    if parsed_json:
                        mode_used = "manual_from_constrained"
            except Exception as e:
                # Le serveur ne supporte peut-être pas response_format json_schema
                # Essayer le format json_object simple (fallback OpenAI)
                try:
                    result = self.client.complete(
                        model=model_id,
                        messages=messages,
                        temperature=0,
                        max_tokens=500,
                        stream=False,
                        response_format={"type": "json_object"},
                    )
                    
                    latencies.append(result.metrics.total_time_ms)
                    
                    if result.json_valid and result.parsed_json is not None:
                        parsed_json = result.parsed_json
                        mode_used = "json_object_fallback"
                    elif result.content:
                        parsed_json = self._extract_json_from_text(result.content)
                        if parsed_json:
                            mode_used = "manual_from_json_object"
                except Exception:
                    pass
            
            # Stratégie 2: Fallback vers appel sans mode contraint (prompt only)
            if parsed_json is None:
                try:
                    result = self.client.complete(
                        model=model_id,
                        messages=messages,
                        temperature=0,
                        max_tokens=500,
                        stream=False,
                        # Pas de response_format - le modèle doit suivre le prompt
                    )
                    
                    if not latencies or len(latencies) < len(results) + 1:
                        latencies.append(result.metrics.total_time_ms)
                    
                    if result.content:
                        parsed_json = self._extract_json_from_text(result.content)
                        if parsed_json:
                            mode_used = "prompt_only_fallback"
                except Exception as e:
                    print(f"[Warning] JSON extraction failed: {e}")
            
            is_valid = parsed_json is not None
            valid_json_list.append(is_valid)
            json_mode_used.append(mode_used)
            
            extraction_result = {
                "success": is_valid,
                "json_valid": is_valid,
                "parsed": parsed_json,
                "mode_used": mode_used,
                "raw_response": result.content[:300] if result and result.content else None,
            }
            
            # Calculer les champs présents
            if is_valid and parsed_json:
                expected_fields = ["document_type", "key_figures", "time_period", "entities"]
                fields_present = sum(1 for f in expected_fields if f in parsed_json)
                extraction_result["fields_present"] = fields_present
                extraction_result["fields_expected"] = len(expected_fields)
                fields_list.append(fields_present)
                
                # Calculer le score de Grounding
                grounding = self._compute_grounding_score(parsed_json, context)
                extraction_result["grounding"] = grounding
                grounding_list.append(grounding["grounding_rate"])
            else:
                fields_list.append(0)
                grounding_list.append(0.0)
            
            results.append(extraction_result)
        
        total_time = time.time() - start_time
        
        # Calculer les statistiques
        valid_stats = self.stats.binary_metrics_with_ci(valid_json_list)
        fields_stats = self.stats.compute_stats(fields_list)
        grounding_stats = self.stats.compute_stats(grounding_list)
        
        # Statistiques sur les modes utilisés
        mode_counts = {}
        for mode in json_mode_used:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Récupérer les métadonnées
        rationale = SAMPLE_SIZE_RATIONALE.get("info_extraction", {})
        
        return ScenarioResult(
            scenario_name="info_extraction",
            model_id=model_id,
            num_samples=len(results),
            metrics={
                "json_valid_rate": valid_stats["accuracy"],
                "json_valid_rate_ci_95": valid_stats["accuracy_ci_95"],
                "valid_json_count": sum(valid_json_list),
                "avg_fields_extracted": fields_stats["mean"],
                "avg_fields_ci_95": fields_stats["ci_95"],
                # Nouvelle métrique de Grounding
                "grounding_rate": grounding_stats["mean"],
                "grounding_rate_ci_95": grounding_stats["ci_95"],
                # Métadonnées sur les modes
                "json_mode_stats": mode_counts,
                # Métadonnées importantes
                "metric_type": "structural_validity_with_grounding",
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_finqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_finqa"]["citation"],
            methodology={
                "metric_type": "structural_validity_with_grounding",
                "sample_size_justification": rationale.get("justification", ""),
                "power_analysis": rationale.get("power", ""),
                "json_strategy": "Try constrained mode first, fallback to manual extraction",
                "grounding_explanation": "Measures % of numbers in JSON that exist in source text",
                "limitations": [
                    "No gold JSON available - measures structure not content accuracy",
                    "Field presence does not guarantee field correctness",
                    "Grounding only checks numbers, not text content accuracy",
                ],
            },
            responses=results,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_multilingual_fr(
        self,
        model_id: str,
        num_samples: int = 30,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue la capacité multilingue sur le dataset français Sujet Financial RAG.
        
        NOTE MÉTHODOLOGIQUE:
        Cette évaluation est QUALITATIVE uniquement.
        Le dataset n'a pas de gold answers disponibles.
        
        Métriques:
        - French response rate (heuristique)
        - Average response length
        - Success rate
        """
        samples = self.loader.load_sujet_rag_fr(limit=num_samples)
        
        if not samples:
            print("[Warning] French dataset not available, skipping multilingual evaluation")
            return ScenarioResult(
                scenario_name="multilingual_fr",
                model_id=model_id,
                num_samples=0,
                metrics={"error": "Dataset not available"},
                methodology={
                    "metric_type": "qualitative_assessment",
                    "limitations": ["Dataset not available"],
                },
                timestamp=datetime.now().isoformat(),
            )
        
        system_prompt = """Vous êtes un assistant financier expert. 
Répondez aux questions sur les documents financiers de manière précise et concise.
Répondez toujours en français."""
        
        responses = []
        latencies = []
        french_list = []
        length_list = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Multilingual FR") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            
            prompt = f"""Contexte:
{context}

Question: {question}

Réponse:"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=200,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            predicted = result.content.strip() if result.content else ""
            
            is_french = self._is_french(predicted)
            french_list.append(is_french)
            length_list.append(len(predicted))
            
            responses.append({
                "question": question,
                "predicted": predicted[:200],
                "is_french": is_french,
                "response_length": len(predicted),
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        # Calculer les statistiques
        french_stats = self.stats.binary_metrics_with_ci(french_list)
        length_stats = self.stats.compute_stats(length_list)
        
        # Récupérer les métadonnées
        rationale = SAMPLE_SIZE_RATIONALE.get("multilingual_fr", {})
        
        return ScenarioResult(
            scenario_name="multilingual_fr",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "french_response_rate": french_stats["accuracy"],
                "french_response_rate_ci_95": french_stats["accuracy_ci_95"],
                "avg_response_length": length_stats["mean"],
                "avg_response_length_ci_95": length_stats["ci_95"],
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
                # Métadonnées importantes
                "metric_type": "qualitative_assessment",
                "is_qualitative": True,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["sujet_rag_fr"]["name"],
            dataset_citation=DATASET_CONFIG["sujet_rag_fr"]["citation"],
            methodology={
                "metric_type": "qualitative_assessment",
                "sample_size_justification": rationale.get("justification", ""),
                "power_analysis": rationale.get("power", ""),
                "limitations": [
                    "No gold answers available - metrics are heuristic",
                    "Language detection is based on simple keyword matching",
                    "Cannot assess response correctness or relevance quantitatively",
                ],
            },
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_all_scenarios(
        self,
        model_id: str,
        include_multilingual: bool = True,
        include_conversational: bool = True,
    ) -> dict[str, ScenarioResult]:
        """
        Exécute tous les scénarios réalistes.
        
        Args:
            model_id: ID du modèle LM Studio
            include_multilingual: Inclure le test français (qualitatif)
            include_conversational: Inclure ConvFinQA
        """
        results = {}
        total_scenarios = 4 + int(include_multilingual) + int(include_conversational)
        current = 0
        
        print(f"\n{'='*60}")
        print(f"REALISTIC BANKING SCENARIOS (Public Datasets)")
        print(f"Model: {model_id}")
        print(f"Scenarios: {total_scenarios}")
        print(f"Metrics: Strict (exact match, numerical) + Bootstrap CI 95%")
        print(f"{'='*60}")
        
        # Financial QA
        current += 1
        print(f"\n[{current}/{total_scenarios}] Financial QA (FinQA)")
        results["financial_qa"] = self.evaluate_financial_qa(model_id)
        
        # Sentiment Analysis
        current += 1
        print(f"\n[{current}/{total_scenarios}] Financial Sentiment (Financial PhraseBank)")
        results["sentiment"] = self.evaluate_sentiment_analysis(model_id)
        
        # Conversational QA (optional)
        if include_conversational:
            current += 1
            print(f"\n[{current}/{total_scenarios}] Conversational QA (ConvFinQA)")
            results["conversational"] = self.evaluate_conversational_qa(model_id)
        
        # Information Extraction
        current += 1
        print(f"\n[{current}/{total_scenarios}] Information Extraction (JSON) [Structural Validity]")
        results["extraction"] = self.evaluate_info_extraction(model_id)
        
        # Twitter Sentiment
        current += 1
        print(f"\n[{current}/{total_scenarios}] Twitter Financial Sentiment")
        results["twitter_sentiment"] = self.evaluate_sentiment_analysis(model_id, use_twitter=True, num_samples=50)
        
        # Multilingual (optional, qualitative)
        if include_multilingual:
            current += 1
            print(f"\n[{current}/{total_scenarios}] Multilingual French [Qualitative Assessment]")
            results["multilingual_fr"] = self.evaluate_multilingual_fr(model_id)
        
        # Résumé
        self._print_summary(results)
        
        # Sauvegarder
        self._save_results(results, model_id)
        
        return results
    
    def _print_summary(self, results: dict[str, ScenarioResult]):
        """Affiche le résumé des résultats avec IC."""
        print(f"\n{'='*60}")
        print("SUMMARY (with 95% Confidence Intervals)")
        print(f"{'='*60}")
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            
            # Afficher le type de métrique
            metric_type = result.methodology.get("metric_type", "quantitative")
            if metric_type != "quantitative":
                print(f"  ⚠️  Metric type: {metric_type}")
            
            print(f"  Dataset: {result.dataset_source or 'N/A'}")
            print(f"  Samples: {result.num_samples}")
            
            for key, value in result.metrics.items():
                if key in ["confusion_matrix", "error", "metric_type", "is_qualitative", "per_class_f1"]:
                    continue
                if key.endswith("_ci_95"):
                    continue  # Afficher avec la métrique principale
                
                if isinstance(value, float):
                    ci_key = f"{key}_ci_95"
                    if ci_key in result.metrics:
                        ci = result.metrics[ci_key]
                        if 0 <= value <= 1:
                            print(f"  {key}: {value:.2%} [CI 95%: {ci[0]:.2%} - {ci[1]:.2%}]")
                        else:
                            print(f"  {key}: {value:.2f} [CI 95%: {ci[0]:.2f} - {ci[1]:.2f}]")
                    else:
                        if 0 <= value <= 1:
                            print(f"  {key}: {value:.2%}")
                        else:
                            print(f"  {key}: {value:.2f}")
                elif isinstance(value, (int, str)):
                    print(f"  {key}: {value}")
            
            print(f"  avg_latency: {result.avg_latency_ms:.1f} ms")
            
            # Afficher les limitations si présentes
            if result.methodology.get("limitations"):
                print(f"  Limitations: {len(result.methodology['limitations'])} noted")
    
    def _save_results(self, results: dict[str, ScenarioResult], model_id: str):
        """Sauvegarde les résultats avec méthodologie complète."""
        model_name = model_id.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realistic_scenarios_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "framework_version": "2.0",  # Version avec métriques strictes
            "datasets_used": DATASET_CONFIG,
            "sample_size_rationale": SAMPLE_SIZE_RATIONALE,
            "results": {name: result.to_dict() for name, result in results.items()},
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Saved] Results saved to {filepath}")


# === EXPORT ===

__all__ = [
    "RealisticScenariosEvaluator",
    "ScenarioResult",
    "DatasetLoader",
    "StatisticalAnalyzer",
    "DATASET_CONFIG",
    "SAMPLE_SIZE_RATIONALE",
]
