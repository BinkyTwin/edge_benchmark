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
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        "citation": "Sujet AI, 2024",
        "split": "train",
    },
    "twitter_financial": {
        "name": "zeroshot/twitter-financial-news-sentiment",
        "description": "Financial sentiment from Twitter/X posts",
        "citation": "ZeroShot, 2023",
        "split": "train",
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
    """Résultat d'un scénario."""
    
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
            "timestamp": self.timestamp,
        }


class RealisticScenariosEvaluator:
    """
    Évaluateur de scénarios réalistes banking.
    
    Utilise des datasets publics pour une évaluation rigoureuse et reproductible.
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
        self.seed = seed
    
    def evaluate_financial_qa(
        self,
        model_id: str,
        num_samples: int = 50,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue sur FinQA - Question Answering financier avec raisonnement numérique.
        
        Métriques:
        - Exact Match (EM)
        - F1 Score (token overlap)
        - Answer presence (réponse dans le texte généré)
        """
        samples = self.loader.load_flare_finqa(limit=num_samples)
        
        system_prompt = """You are a financial analyst assistant. 
Answer questions about financial documents accurately and concisely.
If the answer requires calculation, show your work briefly.
Provide the final answer clearly."""
        
        responses = []
        latencies = []
        exact_matches = 0
        answer_present = 0
        f1_scores = []
        
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
            
            # Calcul des métriques
            predicted = result.content.strip() if result.content else ""
            is_exact = self._normalize_answer(predicted) == self._normalize_answer(expected)
            is_present = expected.lower() in predicted.lower() if expected else False
            f1 = self._compute_f1(predicted, expected)
            
            if is_exact:
                exact_matches += 1
            if is_present:
                answer_present += 1
            f1_scores.append(f1)
            
            responses.append({
                "question": question,
                "expected": expected,
                "predicted": predicted[:200],  # Truncate for storage
                "exact_match": is_exact,
                "answer_present": is_present,
                "f1": f1,
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        return ScenarioResult(
            scenario_name="financial_qa",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "exact_match": exact_matches / len(samples) if samples else 0,
                "answer_present": answer_present / len(samples) if samples else 0,
                "f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_finqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_finqa"]["citation"],
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
        - Accuracy
        - Macro F1
        - Per-class breakdown
        """
        if use_twitter:
            samples = self.loader.load_twitter_financial(limit=num_samples)
            dataset_key = "twitter_financial"
        else:
            samples = self.loader.load_flare_fpb(limit=num_samples)
            dataset_key = "flare_fpb"
        
        system_prompt = """Analyze the sentiment of financial statements.
Respond with exactly one word: positive, negative, or neutral."""
        
        predictions = []
        latencies = []
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
            
            # Mise à jour confusion matrix
            if expected not in confusion:
                confusion[expected] = {}
            confusion[expected][pred] = confusion[expected].get(pred, 0) + 1
            
            predictions.append({
                "text": text[:100],  # Truncate
                "expected": expected,
                "predicted": pred,
                "correct": pred == expected,
                "raw_response": result.content[:50] if result.content else "",
            })
        
        total_time = time.time() - start_time
        
        # Calcul des métriques
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / len(samples) if samples else 0
        
        # Macro F1
        f1_scores = []
        for label in ["positive", "negative", "neutral"]:
            tp = confusion.get(label, {}).get(label, 0)
            fp = sum(confusion.get(other, {}).get(label, 0) for other in confusion if other != label)
            fn = sum(confusion.get(label, {}).get(other, 0) for other in confusion.get(label, {}) if other != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        return ScenarioResult(
            scenario_name="financial_sentiment",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "correct": correct,
                "total": len(samples),
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG[dataset_key]["name"],
            dataset_citation=DATASET_CONFIG[dataset_key]["citation"],
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
        
        Métriques:
        - Answer presence
        - Numerical accuracy (si réponse numérique)
        """
        samples = self.loader.load_flare_convfinqa(limit=num_samples)
        
        system_prompt = """You are a financial analyst assistant engaged in a conversation.
Answer financial questions based on the provided context.
Be precise and concise. For numerical questions, provide the exact number."""
        
        responses = []
        latencies = []
        answer_present = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Conversational QA") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            expected = sample.get("answer", "")
            
            # Utiliser le contexte complet de la query
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
            
            # Vérifier si la réponse attendue est présente
            is_present = False
            if expected and predicted:
                # Normaliser pour comparaison
                exp_norm = str(expected).lower().replace(",", "").replace(" ", "")
                pred_norm = predicted.lower().replace(",", "").replace(" ", "")
                is_present = exp_norm in pred_norm
            
            if is_present:
                answer_present += 1
            
            responses.append({
                "question": question[:150] if question else context[:150],
                "expected": str(expected),
                "predicted": predicted[:150],
                "answer_present": is_present,
                "turn": sample.get("turn", 0),
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        return ScenarioResult(
            scenario_name="conversational_qa",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "answer_present": answer_present / len(samples) if samples else 0,
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_convfinqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_convfinqa"]["citation"],
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_info_extraction(
        self,
        model_id: str,
        num_samples: int = 30,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue l'extraction d'information structurée à partir de documents financiers.
        
        Utilise FinQA comme source de contextes financiers réels.
        
        Métriques:
        - JSON validity rate
        - Field extraction count
        """
        samples = self.loader.load_flare_finqa(limit=num_samples)
        
        system_prompt = """You are a financial document parser.
Extract key information from the document and return it as valid JSON.
Always respond with properly formatted JSON, nothing else."""
        
        results = []
        latencies = []
        valid_json_count = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Info Extraction") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")[:2500]
            
            if not context:
                continue
            
            prompt = f"""Extract structured information from this financial document:

{context}

Return a JSON object with:
- document_type: type of document (earnings, 10-K, etc.)
- key_figures: array of numbers found with their labels
- time_period: fiscal period if mentioned
- entities: companies or organizations mentioned

JSON:"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=400,
                stream=False,
                response_format={"type": "json_object"},
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            extraction_result = {
                "success": result.success,
                "json_valid": result.json_valid,
                "parsed": result.parsed_json if result.json_valid else None,
            }
            
            if result.json_valid and result.parsed_json:
                valid_json_count += 1
                
                # Compter les champs extraits
                expected_fields = ["document_type", "key_figures", "time_period", "entities"]
                fields_present = sum(1 for f in expected_fields if f in result.parsed_json)
                extraction_result["fields_present"] = fields_present
                extraction_result["fields_expected"] = len(expected_fields)
            
            results.append(extraction_result)
        
        total_time = time.time() - start_time
        
        # Calcul des métriques
        avg_fields = sum(r.get("fields_present", 0) for r in results) / len(results) if results else 0
        
        return ScenarioResult(
            scenario_name="info_extraction",
            model_id=model_id,
            num_samples=len(results),
            metrics={
                "json_valid_rate": valid_json_count / len(results) if results else 0,
                "valid_json_count": valid_json_count,
                "avg_fields_extracted": avg_fields,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["flare_finqa"]["name"],
            dataset_citation=DATASET_CONFIG["flare_finqa"]["citation"],
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
        
        Métriques:
        - Response quality (length, coherence)
        - Language consistency (réponse en français)
        """
        samples = self.loader.load_sujet_rag_fr(limit=num_samples)
        
        if not samples:
            print("[Warning] French dataset not available, skipping multilingual evaluation")
            return ScenarioResult(
                scenario_name="multilingual_fr",
                model_id=model_id,
                num_samples=0,
                metrics={"error": "Dataset not available"},
                timestamp=datetime.now().isoformat(),
            )
        
        system_prompt = """Vous êtes un assistant financier expert. 
Répondez aux questions sur les documents financiers de manière précise et concise.
Répondez toujours en français."""
        
        responses = []
        latencies = []
        french_responses = 0
        
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
            
            # Vérifier si la réponse est en français
            is_french = self._is_french(predicted)
            if is_french:
                french_responses += 1
            
            responses.append({
                "question": question,
                "predicted": predicted[:200],
                "is_french": is_french,
                "response_length": len(predicted),
                "latency_ms": result.metrics.total_time_ms,
                "success": result.success,
            })
        
        total_time = time.time() - start_time
        
        avg_length = sum(r["response_length"] for r in responses) / len(responses) if responses else 0
        
        return ScenarioResult(
            scenario_name="multilingual_fr",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "french_response_rate": french_responses / len(samples) if samples else 0,
                "avg_response_length": avg_length,
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["sujet_rag_fr"]["name"],
            dataset_citation=DATASET_CONFIG["sujet_rag_fr"]["citation"],
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
            include_multilingual: Inclure le test français
            include_conversational: Inclure ConvFinQA
        """
        results = {}
        total_scenarios = 4 + int(include_multilingual) + int(include_conversational)
        current = 0
        
        print(f"\n{'='*60}")
        print(f"REALISTIC BANKING SCENARIOS (Public Datasets)")
        print(f"Model: {model_id}")
        print(f"Scenarios: {total_scenarios}")
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
        print(f"\n[{current}/{total_scenarios}] Information Extraction (JSON)")
        results["extraction"] = self.evaluate_info_extraction(model_id)
        
        # Twitter Sentiment (alternative)
        current += 1
        print(f"\n[{current}/{total_scenarios}] Twitter Financial Sentiment")
        results["twitter_sentiment"] = self.evaluate_sentiment_analysis(model_id, use_twitter=True, num_samples=50)
        
        # Multilingual (optional)
        if include_multilingual:
            current += 1
            print(f"\n[{current}/{total_scenarios}] Multilingual French (Sujet RAG FR)")
            results["multilingual_fr"] = self.evaluate_multilingual_fr(model_id)
        
        # Résumé
        self._print_summary(results)
        
        # Sauvegarder
        self._save_results(results, model_id)
        
        return results
    
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
    
    def _print_summary(self, results: dict[str, ScenarioResult]):
        """Affiche le résumé des résultats."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Dataset: {result.dataset_source or 'N/A'}")
            print(f"  Samples: {result.num_samples}")
            for key, value in result.metrics.items():
                if key in ["confusion_matrix", "error"]:
                    continue
                if isinstance(value, float):
                    if 0 <= value <= 1:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            print(f"  avg_latency: {result.avg_latency_ms:.1f} ms")
    
    def _save_results(self, results: dict[str, ScenarioResult], model_id: str):
        """Sauvegarde les résultats."""
        model_name = model_id.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realistic_scenarios_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        data = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "datasets_used": DATASET_CONFIG,
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
    "DATASET_CONFIG",
]
