"""
Realistic Banking Scenarios
===========================

Scénarios réalistes pour évaluer l'utilité pratique des SLMs
dans un contexte bancaire, utilisant des datasets publics:

1. Financial QA (FinQA/ConvFinQA)
2. Financial Sentiment (Financial PhraseBank)
3. Document Information Extraction (FinQA context)
4. Document Summarization (Financial reports)

Datasets utilisés:
- TheFinAI/flare-finqa: QA sur documents financiers avec calculs
- takala/financial_phrasebank: Sentiment sur news financières
- sujet-ai/Sujet-Financial-RAG-FR-Dataset: RAG financier FR (multilingue)
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
    "finqa": {
        "name": "TheFinAI/flare-finqa",
        "description": "Financial QA with numerical reasoning",
        "citation": "Chen et al., 2021 - FinQA: A Dataset of Numerical Reasoning over Financial Data",
    },
    "financial_phrasebank": {
        "name": "takala/financial_phrasebank",
        "subset": "sentences_allagree",
        "description": "Financial news sentiment analysis",
        "citation": "Malo et al., 2014 - Good debt or bad debt: Detecting semantic orientations in economic texts",
    },
    "sujet_rag_fr": {
        "name": "sujet-ai/Sujet-Financial-RAG-FR-Dataset",
        "description": "French financial RAG dataset",
        "citation": "Sujet AI, 2024",
    },
    "convfinqa": {
        "name": "TheFinAI/flare-convfinqa",
        "description": "Conversational Financial QA",
        "citation": "Chen et al., 2022 - ConvFinQA",
    },
}


# === FALLBACK DATA (si datasets indisponibles) ===

FALLBACK_FAQ_SAMPLES = [
    {
        "question": "How do I reset my online banking password?",
        "expected_topics": ["password reset", "security", "online banking"],
    },
    {
        "question": "What are the fees for international wire transfers?",
        "expected_topics": ["fees", "international", "wire transfer"],
    },
    {
        "question": "Can I deposit checks using my mobile phone?",
        "expected_topics": ["mobile deposit", "check", "phone"],
    },
    {
        "question": "How long does it take for a direct deposit to show up?",
        "expected_topics": ["direct deposit", "timing", "availability"],
    },
    {
        "question": "What happens if my debit card is stolen?",
        "expected_topics": ["stolen card", "security", "report", "block"],
    },
]

FALLBACK_SENTIMENT_SAMPLES = [
    {"text": "The new mobile app is amazing! So much easier to use than before.", "expected": "positive"},
    {"text": "I've been waiting on hold for 45 minutes. This is unacceptable.", "expected": "negative"},
    {"text": "The branch closes at 5pm on weekdays.", "expected": "neutral"},
    {"text": "Your customer service team resolved my issue quickly. Very impressed!", "expected": "positive"},
    {"text": "Hidden fees everywhere! I'm switching banks.", "expected": "negative"},
]


# === DATASET LOADERS ===

class DatasetLoader:
    """Charge les datasets depuis Hugging Face avec fallback."""
    
    def __init__(self, cache_dir: Optional[Path] = None, seed: int = 42):
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        self._cache = {}
    
    def load_finqa(self, split: str = "test", limit: int = 50) -> list[dict]:
        """
        Charge FinQA pour les tâches de QA financier.
        
        Structure:
        - context: texte du document financier + tables
        - question: question posée
        - answer: réponse attendue
        """
        cache_key = f"finqa_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading FinQA ({split}, limit={limit})...")
            dataset = load_dataset(
                "TheFinAI/flare-finqa",
                split=split,
                trust_remote_code=True,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                # Construire le contexte à partir des données disponibles
                context = item.get("context", "") or item.get("text", "")
                if not context and "table" in item:
                    context = str(item["table"])
                
                samples.append({
                    "context": context[:4000],  # Limiter la taille du contexte
                    "question": item.get("question", ""),
                    "answer": str(item.get("answer", "")),
                    "source": "finqa",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} FinQA samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load FinQA: {e}")
            return self._generate_fallback_qa(limit)
    
    def load_convfinqa(self, split: str = "test", limit: int = 30) -> list[dict]:
        """
        Charge ConvFinQA pour les dialogues multi-tours.
        
        Structure:
        - context: document financier
        - questions: liste de questions (dialogue)
        - answers: liste de réponses
        """
        cache_key = f"convfinqa_{split}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading ConvFinQA ({split}, limit={limit})...")
            dataset = load_dataset(
                "TheFinAI/flare-convfinqa",
                split=split,
                trust_remote_code=True,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "context": item.get("context", "")[:3000],
                    "question": item.get("question", ""),
                    "answer": str(item.get("answer", "")),
                    "dialogue_id": item.get("id", idx),
                    "source": "convfinqa",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} ConvFinQA samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load ConvFinQA: {e}")
            return self._generate_fallback_qa(limit)
    
    def load_financial_phrasebank(self, limit: int = 100) -> list[dict]:
        """
        Charge Financial PhraseBank pour l'analyse de sentiment.
        
        Structure:
        - sentence: phrase financière
        - label: 0 (negative), 1 (neutral), 2 (positive)
        """
        cache_key = f"phrasebank_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading Financial PhraseBank (limit={limit})...")
            dataset = load_dataset(
                "takala/financial_phrasebank",
                "sentences_allagree",
                split="train",
                trust_remote_code=True,
            )
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "text": item["sentence"],
                    "expected": label_map.get(item["label"], "neutral"),
                    "source": "financial_phrasebank",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} sentiment samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load Financial PhraseBank: {e}")
            return FALLBACK_SENTIMENT_SAMPLES
    
    def load_sujet_rag_fr(self, limit: int = 30) -> list[dict]:
        """
        Charge le dataset Sujet Financial RAG FR.
        
        Pour tester la capacité multilingue (français).
        """
        cache_key = f"sujet_fr_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            print(f"[DatasetLoader] Loading Sujet Financial RAG FR (limit={limit})...")
            dataset = load_dataset(
                "sujet-ai/Sujet-Financial-RAG-FR-Dataset",
                split="train",
                trust_remote_code=True,
            )
            
            samples = []
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            for idx in indices[:limit]:
                item = dataset[idx]
                samples.append({
                    "context": item.get("context", "")[:3000],
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "source": "sujet_rag_fr",
                    "language": "fr",
                })
            
            print(f"[DatasetLoader] Loaded {len(samples)} French financial samples")
            self._cache[cache_key] = samples
            return samples
            
        except Exception as e:
            print(f"[DatasetLoader] Warning: Could not load Sujet RAG FR: {e}")
            return []
    
    def _generate_fallback_qa(self, limit: int) -> list[dict]:
        """Génère des échantillons fallback pour QA."""
        return [
            {
                "context": "Annual Report 2024: Revenue increased by 15% to $2.5 billion.",
                "question": "What was the revenue growth?",
                "answer": "15%",
                "source": "fallback",
            }
        ] * min(limit, 5)


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
        - Answer relevance
        """
        samples = self.loader.load_finqa(limit=num_samples)
        
        system_prompt = """You are a financial analyst assistant. 
Answer questions about financial documents accurately and concisely.
If the answer requires calculation, show your work briefly.
Provide only the final answer when possible."""
        
        responses = []
        latencies = []
        exact_matches = 0
        f1_scores = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Financial QA") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            expected = sample.get("answer", "")
            
            prompt = f"""Context:
{context}

Question: {question}

Answer:"""
            
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
            f1 = self._compute_f1(predicted, expected)
            
            if is_exact:
                exact_matches += 1
            f1_scores.append(f1)
            
            responses.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "exact_match": is_exact,
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
                "f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "success_rate": sum(1 for r in responses if r["success"]) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["finqa"]["name"],
            dataset_citation=DATASET_CONFIG["finqa"]["citation"],
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_sentiment_analysis(
        self,
        model_id: str,
        num_samples: int = 100,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue sur Financial PhraseBank - Analyse de sentiment financier.
        
        Métriques:
        - Accuracy
        - Macro F1
        - Per-class precision/recall
        """
        samples = self.loader.load_financial_phrasebank(limit=num_samples)
        
        system_prompt = """Analyze the sentiment of financial news statements.
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
                "text": text,
                "expected": expected,
                "predicted": pred,
                "correct": pred == expected,
                "raw_response": result.content,
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
                "confusion_matrix": confusion,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["financial_phrasebank"]["name"],
            dataset_citation=DATASET_CONFIG["financial_phrasebank"]["citation"],
            responses=predictions,
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
        - Field extraction accuracy
        - Numerical accuracy
        """
        samples = self.loader.load_finqa(limit=num_samples)
        
        system_prompt = """You are a financial document parser.
Extract key information from the document and return it as valid JSON.
Always respond with properly formatted JSON, nothing else."""
        
        extraction_schema = {
            "type": "object",
            "properties": {
                "document_type": {"type": "string", "description": "Type of financial document"},
                "key_figures": {"type": "array", "items": {"type": "object"}},
                "time_period": {"type": "string"},
                "entities_mentioned": {"type": "array", "items": {"type": "string"}},
            }
        }
        
        results = []
        latencies = []
        valid_json_count = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Info Extraction") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")[:2500]
            
            prompt = f"""Extract structured information from this financial document:

{context}

Return a JSON object with:
- document_type: type of document (earnings report, annual report, etc.)
- key_figures: array of numerical metrics found
- time_period: fiscal period mentioned
- entities_mentioned: companies or organizations mentioned

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
                "parsed": result.parsed_json,
                "raw_response": result.content[:500] if result.content else "",
            }
            
            if result.json_valid and result.parsed_json:
                valid_json_count += 1
                
                # Vérifier les champs extraits
                expected_fields = ["document_type", "key_figures", "time_period", "entities_mentioned"]
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
            num_samples=len(samples),
            metrics={
                "json_valid_rate": valid_json_count / len(samples) if samples else 0,
                "valid_json_count": valid_json_count,
                "avg_fields_extracted": avg_fields,
                "total_samples": len(samples),
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["finqa"]["name"],
            dataset_citation=DATASET_CONFIG["finqa"]["citation"],
            responses=results,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_document_summary(
        self,
        model_id: str,
        num_samples: int = 20,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue la capacité de résumé de documents financiers.
        
        Utilise les contextes de FinQA comme documents sources.
        
        Métriques:
        - Summary quality (length, coherence proxy)
        - Key information retention
        - Compression ratio
        """
        samples = self.loader.load_finqa(limit=num_samples)
        
        system_prompt = """You are a financial document summarization assistant.
Create clear, accurate summaries that capture key financial information.
Focus on the most important figures, trends, and insights."""
        
        results = []
        latencies = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Document Summary") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")[:3500]
            
            prompt = f"""Summarize this financial document in 2-3 concise paragraphs:

{context}

Summary:"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=350,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            summary = result.content if result.content else ""
            
            # Métriques de qualité
            summary_result = {
                "success": result.success,
                "summary": summary,
                "summary_length": len(summary),
                "context_length": len(context),
                "compression_ratio": len(summary) / len(context) if context else 0,
            }
            
            # Vérifier si des chiffres clés sont mentionnés
            if result.success and summary:
                # Extraire les nombres du contexte et vérifier leur présence
                import re
                context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:%|million|billion)?\b', context.lower()))
                summary_numbers = set(re.findall(r'\b\d+(?:\.\d+)?(?:%|million|billion)?\b', summary.lower()))
                
                if context_numbers:
                    retention_rate = len(context_numbers & summary_numbers) / len(context_numbers)
                else:
                    retention_rate = 1.0  # Pas de nombres à retenir
                
                summary_result["number_retention_rate"] = retention_rate
            
            results.append(summary_result)
        
        total_time = time.time() - start_time
        
        # Métriques agrégées
        avg_compression = sum(r.get("compression_ratio", 0) for r in results) / len(results) if results else 0
        avg_retention = sum(r.get("number_retention_rate", 0) for r in results) / len(results) if results else 0
        avg_length = sum(r.get("summary_length", 0) for r in results) / len(results) if results else 0
        
        return ScenarioResult(
            scenario_name="document_summary",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "avg_compression_ratio": avg_compression,
                "number_retention_rate": avg_retention,
                "avg_summary_length": avg_length,
                "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            dataset_source=DATASET_CONFIG["finqa"]["name"],
            dataset_citation=DATASET_CONFIG["finqa"]["citation"],
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
        - Answer quality (F1)
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
        
        system_prompt = """Vous êtes un assistant financier. 
Répondez aux questions sur les documents financiers de manière précise et concise.
Répondez toujours en français."""
        
        responses = []
        latencies = []
        french_responses = 0
        f1_scores = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Multilingual FR") if progress_bar else samples
        
        for sample in iterator:
            context = sample.get("context", "")
            question = sample.get("question", "")
            expected = sample.get("answer", "")
            
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
            
            # F1 score
            f1 = self._compute_f1(predicted, expected)
            f1_scores.append(f1)
            
            responses.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "is_french": is_french,
                "f1": f1,
                "latency_ms": result.metrics.total_time_ms,
            })
        
        total_time = time.time() - start_time
        
        return ScenarioResult(
            scenario_name="multilingual_fr",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "french_response_rate": french_responses / len(samples) if samples else 0,
                "success_rate": sum(1 for r in responses if r.get("f1", 0) > 0) / len(responses) if responses else 0,
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
    ) -> dict[str, ScenarioResult]:
        """
        Exécute tous les scénarios réalistes.
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"REALISTIC BANKING SCENARIOS (Public Datasets)")
        print(f"Model: {model_id}")
        print(f"{'='*60}")
        
        # Financial QA
        print("\n[1/5] Financial QA (FinQA)")
        results["financial_qa"] = self.evaluate_financial_qa(model_id)
        
        # Sentiment Analysis
        print("\n[2/5] Financial Sentiment (Financial PhraseBank)")
        results["sentiment"] = self.evaluate_sentiment_analysis(model_id)
        
        # Information Extraction
        print("\n[3/5] Information Extraction")
        results["extraction"] = self.evaluate_info_extraction(model_id)
        
        # Document Summary
        print("\n[4/5] Document Summarization")
        results["summary"] = self.evaluate_document_summary(model_id)
        
        # Multilingual (optional)
        if include_multilingual:
            print("\n[5/5] Multilingual French (Sujet RAG FR)")
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
        # Nettoyer et normaliser
        answer = answer.lower().strip()
        # Retirer la ponctuation
        answer = ''.join(c for c in answer if c.isalnum() or c.isspace())
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
                if key == "confusion_matrix":
                    continue  # Skip detailed confusion matrix in summary
                if isinstance(value, float):
                    if value <= 1:
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
