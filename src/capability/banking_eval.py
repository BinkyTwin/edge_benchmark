"""
Banking Evaluation Module
=========================

Évaluation sur les datasets bancaires:
- Banking77: Classification d'intents (77 classes)
- Financial PhraseBank: Analyse de sentiment (3 classes)

Ce module est le FOCUS PRINCIPAL du benchmark.
"""

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, format_messages


# === LABELS BANKING77 ===
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal"
]

# === LABELS FINANCIAL PHRASEBANK ===
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


@dataclass
class EvaluationResult:
    """Résultat d'une évaluation."""
    
    dataset_name: str
    model_id: str
    
    # Métriques principales
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    
    # Détails
    num_samples: int = 0
    num_correct: int = 0
    
    # Par classe (optionnel)
    per_class_accuracy: dict = field(default_factory=dict)
    confusion_matrix: list = field(default_factory=list)
    
    # Métriques de performance
    avg_latency_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # Métadonnées
    timestamp: str = ""
    errors: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            "dataset_name": self.dataset_name,
            "model_id": self.model_id,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "per_class_accuracy": self.per_class_accuracy,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "timestamp": self.timestamp,
            "num_errors": len(self.errors),
        }


class BankingEvaluator:
    """
    Évaluateur pour les tâches banking.
    
    Supporte Banking77 (intent classification) et 
    Financial PhraseBank (sentiment analysis).
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        results_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            client: Client LM Studio
            results_dir: Dossier pour les résultats
            cache_dir: Dossier cache pour les datasets
        """
        self.client = client
        self.results_dir = results_dir or Path("results")
        self.cache_dir = cache_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_banking77(
        self,
        split: str = "test",
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> list[dict]:
        """
        Charge le dataset Banking77.
        
        Args:
            split: Split à charger ('train' ou 'test')
            sample_size: Nombre d'échantillons (None = tout)
            seed: Seed pour le sampling
            
        Returns:
            Liste de dictionnaires avec 'text' et 'label'
        """
        print(f"[Banking77] Loading {split} split...")
        dataset = load_dataset(
            "PolyAI/banking77",
            split=split,
            cache_dir=self.cache_dir,
        )
        
        samples = []
        for item in dataset:
            samples.append({
                "text": item["text"],
                "label": BANKING77_LABELS[item["label"]],
                "label_id": item["label"],
            })
        
        if sample_size and sample_size < len(samples):
            random.seed(seed)
            samples = random.sample(samples, sample_size)
        
        print(f"[Banking77] Loaded {len(samples)} samples")
        return samples
    
    def load_financial_phrasebank(
        self,
        sample_size: Optional[int] = 1000,
        seed: int = 42,
    ) -> list[dict]:
        """
        Charge le dataset Financial PhraseBank.
        
        Args:
            sample_size: Nombre d'échantillons
            seed: Seed pour le sampling
            
        Returns:
            Liste de dictionnaires avec 'sentence' et 'label'
        """
        print(f"[FinancialPhraseBank] Loading dataset...")
        dataset = load_dataset(
            "takala/financial_phrasebank",
            "sentences_allagree",  # Subset avec accord unanime
            split="train",
            cache_dir=self.cache_dir,
        )
        
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        samples = []
        for item in dataset:
            samples.append({
                "text": item["sentence"],
                "label": label_map[item["label"]],
                "label_id": item["label"],
            })
        
        if sample_size and sample_size < len(samples):
            random.seed(seed)
            samples = random.sample(samples, sample_size)
        
        print(f"[FinancialPhraseBank] Loaded {len(samples)} samples")
        return samples
    
    def evaluate_banking77(
        self,
        model_id: str,
        sample_size: Optional[int] = None,
        few_shot: int = 0,
        progress_bar: bool = True,
    ) -> EvaluationResult:
        """
        Évalue un modèle sur Banking77.
        
        Args:
            model_id: ID du modèle LM Studio
            sample_size: Nombre d'échantillons (None = tout le test set)
            few_shot: Nombre d'exemples few-shot
            progress_bar: Afficher la barre de progression
            
        Returns:
            EvaluationResult avec métriques
        """
        print(f"\n{'='*60}")
        print(f"BANKING77 EVALUATION")
        print(f"Model: {model_id}")
        print(f"{'='*60}")
        
        # Charger les données
        samples = self.load_banking77(split="test", sample_size=sample_size)
        
        # Préparer le prompt système
        system_prompt = """You are a banking intent classifier. 
Given a customer query, identify the intent category from the following list:

{labels}

Respond with ONLY the intent category name, nothing else.""".format(
            labels=", ".join(BANKING77_LABELS)
        )
        
        # Évaluation
        predictions = []
        ground_truth = []
        latencies = []
        errors = []
        
        start_time = time.time()
        
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            user_prompt = f"Customer query: {sample['text']}\n\nIntent category:"
            messages = format_messages(user_prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=50,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            if result.success:
                pred = self._parse_banking77_prediction(result.content)
                predictions.append(pred)
                ground_truth.append(sample["label"])
            else:
                errors.append({
                    "text": sample["text"][:100],
                    "error": result.error,
                })
                predictions.append("unknown")
                ground_truth.append(sample["label"])
        
        total_time = time.time() - start_time
        
        # Calculer les métriques
        accuracy = accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        
        # Accuracy par classe
        per_class = self._compute_per_class_accuracy(ground_truth, predictions)
        
        result = EvaluationResult(
            dataset_name="banking77",
            model_id=model_id,
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            num_samples=len(samples),
            num_correct=sum(1 for p, g in zip(predictions, ground_truth) if p == g),
            per_class_accuracy=per_class,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            timestamp=datetime.now().isoformat(),
            errors=errors,
        )
        
        self._print_results(result)
        return result
    
    def evaluate_financial_phrasebank(
        self,
        model_id: str,
        sample_size: int = 1000,
        progress_bar: bool = True,
    ) -> EvaluationResult:
        """
        Évalue un modèle sur Financial PhraseBank.
        
        Args:
            model_id: ID du modèle LM Studio
            sample_size: Nombre d'échantillons
            progress_bar: Afficher la barre de progression
            
        Returns:
            EvaluationResult avec métriques
        """
        print(f"\n{'='*60}")
        print(f"FINANCIAL PHRASEBANK EVALUATION")
        print(f"Model: {model_id}")
        print(f"{'='*60}")
        
        # Charger les données
        samples = self.load_financial_phrasebank(sample_size=sample_size)
        
        # Préparer le prompt système
        system_prompt = """You are a financial sentiment analyzer.
Analyze the sentiment of financial news sentences.
Respond with exactly one word: positive, negative, or neutral."""
        
        # Évaluation
        predictions = []
        ground_truth = []
        latencies = []
        errors = []
        
        start_time = time.time()
        
        iterator = tqdm(samples, desc="Evaluating") if progress_bar else samples
        
        for sample in iterator:
            user_prompt = f"Sentence: {sample['text']}\n\nSentiment:"
            messages = format_messages(user_prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=10,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            if result.success:
                pred = self._parse_sentiment_prediction(result.content)
                predictions.append(pred)
                ground_truth.append(sample["label"])
            else:
                errors.append({
                    "text": sample["text"][:100],
                    "error": result.error,
                })
                predictions.append("neutral")  # Default
                ground_truth.append(sample["label"])
        
        total_time = time.time() - start_time
        
        # Calculer les métriques
        accuracy = accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=SENTIMENT_LABELS)
        
        # Accuracy par classe
        per_class = self._compute_per_class_accuracy(ground_truth, predictions)
        
        result = EvaluationResult(
            dataset_name="financial_phrasebank",
            model_id=model_id,
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            num_samples=len(samples),
            num_correct=sum(1 for p, g in zip(predictions, ground_truth) if p == g),
            per_class_accuracy=per_class,
            confusion_matrix=cm.tolist(),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            timestamp=datetime.now().isoformat(),
            errors=errors,
        )
        
        self._print_results(result)
        self._print_confusion_matrix(cm, SENTIMENT_LABELS)
        
        return result
    
    def evaluate_all(
        self,
        model_id: str,
        banking77_samples: Optional[int] = None,
        phrasebank_samples: int = 1000,
    ) -> dict[str, EvaluationResult]:
        """
        Évalue un modèle sur tous les datasets banking.
        
        Args:
            model_id: ID du modèle
            banking77_samples: Nombre d'échantillons Banking77
            phrasebank_samples: Nombre d'échantillons PhraseBank
            
        Returns:
            Dictionnaire dataset_name -> EvaluationResult
        """
        results = {}
        
        # Banking77
        results["banking77"] = self.evaluate_banking77(
            model_id=model_id,
            sample_size=banking77_samples,
        )
        self.save_result(results["banking77"])
        
        # Financial PhraseBank
        results["financial_phrasebank"] = self.evaluate_financial_phrasebank(
            model_id=model_id,
            sample_size=phrasebank_samples,
        )
        self.save_result(results["financial_phrasebank"])
        
        return results
    
    def _parse_banking77_prediction(self, content: str) -> str:
        """Parse la prédiction Banking77."""
        content = content.strip().lower()
        
        # Chercher une correspondance exacte
        for label in BANKING77_LABELS:
            if label.lower() in content:
                return label
        
        # Chercher une correspondance partielle
        content_words = set(content.replace("_", " ").split())
        best_match = None
        best_score = 0
        
        for label in BANKING77_LABELS:
            label_words = set(label.replace("_", " ").split())
            score = len(content_words & label_words)
            if score > best_score:
                best_score = score
                best_match = label
        
        return best_match or "unknown"
    
    def _parse_sentiment_prediction(self, content: str) -> str:
        """Parse la prédiction de sentiment."""
        content = content.strip().lower()
        
        for label in SENTIMENT_LABELS:
            if label in content:
                return label
        
        # Fallback: chercher des synonymes
        if any(word in content for word in ["good", "great", "increase", "growth", "profit"]):
            return "positive"
        elif any(word in content for word in ["bad", "poor", "decrease", "loss", "decline"]):
            return "negative"
        
        return "neutral"
    
    def _compute_per_class_accuracy(
        self,
        ground_truth: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Calcule l'accuracy par classe."""
        per_class = {}
        
        for label in set(ground_truth):
            correct = sum(
                1 for g, p in zip(ground_truth, predictions)
                if g == label and p == label
            )
            total = sum(1 for g in ground_truth if g == label)
            per_class[label] = correct / total if total > 0 else 0.0
        
        return per_class
    
    def save_result(self, result: EvaluationResult, filename: Optional[str] = None):
        """Sauvegarde un résultat."""
        if filename is None:
            model_name = result.model_id.replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{result.dataset_name}_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n[Saved] Results saved to {filepath}")
    
    def _print_results(self, result: EvaluationResult):
        """Affiche les résultats."""
        print(f"\n{'─'*40}")
        print("RESULTS")
        print(f"{'─'*40}")
        print(f"Accuracy:       {result.accuracy*100:.2f}%")
        print(f"Macro F1:       {result.macro_f1*100:.2f}%")
        print(f"Weighted F1:    {result.weighted_f1*100:.2f}%")
        print(f"Samples:        {result.num_correct}/{result.num_samples}")
        print(f"Avg latency:    {result.avg_latency_ms:.1f} ms")
        print(f"Total time:     {result.total_time_seconds:.1f}s")
        print(f"Errors:         {len(result.errors)}")
        print(f"{'─'*40}")
    
    def _print_confusion_matrix(self, cm, labels: list[str]):
        """Affiche la matrice de confusion."""
        print(f"\nConfusion Matrix:")
        print(f"{'':>12}", end="")
        for label in labels:
            print(f"{label:>10}", end="")
        print()
        
        for i, label in enumerate(labels):
            print(f"{label:>12}", end="")
            for j in range(len(labels)):
                print(f"{cm[i][j]:>10}", end="")
            print()


