"""
Realistic Banking Scenarios
===========================

Scénarios réalistes pour évaluer l'utilité pratique des SLMs
dans un contexte bancaire:

1. Banking FAQ Assistant
2. Customer Feedback Sentiment
3. Document Information Extraction
4. Internal Document Summary
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..lmstudio_client import LMStudioClient, format_messages


# === DONNÉES DE TEST POUR LES SCÉNARIOS ===

BANKING_FAQ_SAMPLES = [
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
    {
        "question": "How do I set up automatic bill payments?",
        "expected_topics": ["automatic payment", "bills", "setup"],
    },
    {
        "question": "What is the daily ATM withdrawal limit?",
        "expected_topics": ["ATM", "withdrawal", "limit"],
    },
    {
        "question": "Can I open a joint account online?",
        "expected_topics": ["joint account", "online", "open"],
    },
    {
        "question": "How do I dispute a transaction on my credit card?",
        "expected_topics": ["dispute", "transaction", "credit card"],
    },
    {
        "question": "What documents do I need to open a business account?",
        "expected_topics": ["business account", "documents", "requirements"],
    },
]

SENTIMENT_SAMPLES = [
    {
        "text": "The new mobile app is amazing! So much easier to use than before.",
        "expected": "positive",
    },
    {
        "text": "I've been waiting on hold for 45 minutes. This is unacceptable.",
        "expected": "negative",
    },
    {
        "text": "The branch closes at 5pm on weekdays.",
        "expected": "neutral",
    },
    {
        "text": "Your customer service team resolved my issue quickly. Very impressed!",
        "expected": "positive",
    },
    {
        "text": "The ATM ate my card and nobody helped me.",
        "expected": "negative",
    },
    {
        "text": "I received my new credit card in the mail today.",
        "expected": "neutral",
    },
    {
        "text": "Hidden fees everywhere! I'm switching banks.",
        "expected": "negative",
    },
    {
        "text": "The interest rate on savings is competitive.",
        "expected": "positive",
    },
]

EXTRACTION_SAMPLES = [
    {
        "document": """
ACCOUNT STATEMENT - December 2024
Account Holder: Marie Dupont
Account Number: FR76 3000 4000 0500 0012 3456 789
Account Type: Current Account (Compte Courant)

Period: 01/12/2024 - 31/12/2024
Opening Balance: €5,432.10
Closing Balance: €4,891.55

Recent Transactions:
- 05/12/2024: Salary Credit - €2,500.00
- 10/12/2024: Rent Payment - €1,200.00
- 15/12/2024: Grocery Store - €156.78
- 20/12/2024: Utility Bill - €89.45
""",
        "expected": {
            "account_holder": "Marie Dupont",
            "account_type": "Current Account",
            "opening_balance": 5432.10,
            "closing_balance": 4891.55,
        },
        "schema": {
            "account_holder": "string",
            "account_number": "string",
            "account_type": "string",
            "opening_balance": "number",
            "closing_balance": "number",
            "num_transactions": "number",
        }
    },
]

SUMMARY_SAMPLES = [
    {
        "document": """
INTERNAL MEMO - Q4 2024 Performance Review

TO: Branch Managers
FROM: Regional Director
DATE: January 5, 2025
RE: Fourth Quarter Performance Summary

Executive Summary:
The fourth quarter of 2024 demonstrated strong performance across our retail banking network.
Total deposits increased by 8.2% compared to Q3, exceeding our target of 6%.
New account openings reached 4,523 accounts, a 15% increase year-over-year.

Key Highlights:
1. Digital Banking Adoption: Mobile app users grew to 67% of active customers, 
   up from 58% at the start of the year.
2. Customer Satisfaction: NPS scores improved to 42, up from 38 in Q3.
3. Loan Portfolio: Mortgage originations totaled €125M, meeting annual targets.
4. Operational Efficiency: Average transaction processing time reduced by 12%.

Areas for Improvement:
- Cross-selling ratios remain below target at 1.8 products per customer.
- Wait times at peak hours still exceed 10 minutes at 30% of branches.
- Staff turnover in customer service roles increased to 18%.

Action Items for Q1 2025:
1. Implement new queue management system in high-traffic branches.
2. Launch customer loyalty program to improve retention.
3. Conduct training sessions on cross-selling techniques.
4. Review compensation packages for customer service positions.

Please schedule team meetings to discuss these results and action items.
""",
        "expected_points": [
            "deposits increased 8.2%",
            "new accounts 4,523",
            "mobile adoption 67%",
            "NPS improved to 42",
        ]
    },
]


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
            "timestamp": self.timestamp,
        }


class RealisticScenariosEvaluator:
    """
    Évaluateur de scénarios réalistes banking.
    """
    
    def __init__(
        self,
        client: LMStudioClient,
        results_dir: Optional[Path] = None,
    ):
        self.client = client
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_faq_assistant(
        self,
        model_id: str,
        samples: Optional[list[dict]] = None,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue le scénario FAQ Banking Assistant.
        """
        samples = samples or BANKING_FAQ_SAMPLES
        
        system_prompt = """You are a helpful banking customer service assistant.
Answer customer questions accurately and professionally.
If you don't know the answer, say so. Never make up information.
Keep responses concise and actionable."""
        
        responses = []
        latencies = []
        topic_hits = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="FAQ Assistant") if progress_bar else samples
        
        for sample in iterator:
            messages = format_messages(sample["question"], system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=200,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            response_data = {
                "question": sample["question"],
                "response": result.content,
                "success": result.success,
                "latency_ms": result.metrics.total_time_ms,
            }
            
            # Vérifier si les topics attendus sont mentionnés
            if result.success and result.content:
                topics_found = sum(
                    1 for topic in sample.get("expected_topics", [])
                    if topic.lower() in result.content.lower()
                )
                response_data["topics_found"] = topics_found
                if topics_found > 0:
                    topic_hits += 1
            
            responses.append(response_data)
        
        total_time = time.time() - start_time
        
        return ScenarioResult(
            scenario_name="banking_faq",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "topic_relevance_rate": topic_hits / len(samples) if samples else 0,
                "avg_response_length": sum(len(r["response"]) for r in responses) / len(responses) if responses else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            responses=responses,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_sentiment_analysis(
        self,
        model_id: str,
        samples: Optional[list[dict]] = None,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue le scénario Sentiment Analysis.
        """
        samples = samples or SENTIMENT_SAMPLES
        
        system_prompt = """Analyze the sentiment of customer feedback.
Respond with exactly one word: positive, negative, or neutral."""
        
        predictions = []
        latencies = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Sentiment Analysis") if progress_bar else samples
        
        for sample in iterator:
            prompt = f"Customer feedback: {sample['text']}\n\nSentiment:"
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=10,
                stream=False,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            pred = "neutral"
            if result.success and result.content:
                content = result.content.lower().strip()
                for label in ["positive", "negative", "neutral"]:
                    if label in content:
                        pred = label
                        break
            
            predictions.append({
                "text": sample["text"],
                "expected": sample["expected"],
                "predicted": pred,
                "correct": pred == sample["expected"],
            })
        
        total_time = time.time() - start_time
        correct = sum(1 for p in predictions if p["correct"])
        
        return ScenarioResult(
            scenario_name="sentiment_analysis",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "accuracy": correct / len(samples) if samples else 0,
                "correct": correct,
                "total": len(samples),
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            responses=predictions,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_info_extraction(
        self,
        model_id: str,
        samples: Optional[list[dict]] = None,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue le scénario Information Extraction (JSON).
        """
        samples = samples or EXTRACTION_SAMPLES
        
        system_prompt = """You are a document parser. Extract information and return it as valid JSON.
Always respond with properly formatted JSON, nothing else."""
        
        results = []
        latencies = []
        valid_json_count = 0
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Info Extraction") if progress_bar else samples
        
        for sample in iterator:
            schema_str = json.dumps(sample.get("schema", {}), indent=2)
            prompt = f"""Extract information from this document:

{sample['document']}

Return JSON matching this schema:
{schema_str}"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=300,
                stream=False,
                response_format={"type": "json_object"},
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            extraction_result = {
                "success": result.success,
                "json_valid": result.json_valid,
                "parsed": result.parsed_json,
            }
            
            if result.json_valid:
                valid_json_count += 1
                
                # Vérifier les champs extraits
                expected = sample.get("expected", {})
                correct_fields = 0
                for key, value in expected.items():
                    if result.parsed_json and key in result.parsed_json:
                        if str(result.parsed_json[key]).lower() == str(value).lower():
                            correct_fields += 1
                
                extraction_result["correct_fields"] = correct_fields
                extraction_result["total_expected_fields"] = len(expected)
            
            results.append(extraction_result)
        
        total_time = time.time() - start_time
        
        return ScenarioResult(
            scenario_name="info_extraction",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "json_valid_rate": valid_json_count / len(samples) if samples else 0,
                "valid_json_count": valid_json_count,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            responses=results,
            timestamp=datetime.now().isoformat(),
        )
    
    def evaluate_document_summary(
        self,
        model_id: str,
        samples: Optional[list[dict]] = None,
        progress_bar: bool = True,
    ) -> ScenarioResult:
        """
        Évalue le scénario Document Summary.
        """
        samples = samples or SUMMARY_SAMPLES
        
        system_prompt = """You are a document summarization assistant.
Create clear, accurate summaries that capture key information.
Focus on the most important points."""
        
        results = []
        latencies = []
        
        start_time = time.time()
        iterator = tqdm(samples, desc="Document Summary") if progress_bar else samples
        
        for sample in iterator:
            prompt = f"""Summarize this document in 2-3 concise paragraphs:

{sample['document']}

Summary:"""
            
            messages = format_messages(prompt, system_prompt)
            
            result = self.client.complete(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=300,
                stream=True,
            )
            
            latencies.append(result.metrics.total_time_ms)
            
            summary_result = {
                "success": result.success,
                "summary": result.content,
                "length": len(result.content) if result.content else 0,
            }
            
            # Vérifier si les points clés sont mentionnés
            if result.success and result.content:
                points_found = sum(
                    1 for point in sample.get("expected_points", [])
                    if point.lower() in result.content.lower()
                )
                summary_result["key_points_found"] = points_found
                summary_result["total_key_points"] = len(sample.get("expected_points", []))
            
            results.append(summary_result)
        
        total_time = time.time() - start_time
        
        # Calculer le taux de couverture des points clés
        total_points = sum(r.get("total_key_points", 0) for r in results)
        found_points = sum(r.get("key_points_found", 0) for r in results)
        
        return ScenarioResult(
            scenario_name="document_summary",
            model_id=model_id,
            num_samples=len(samples),
            metrics={
                "key_points_coverage": found_points / total_points if total_points else 0,
                "avg_summary_length": sum(r["length"] for r in results) / len(results) if results else 0,
            },
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            total_time_seconds=total_time,
            responses=results,
            timestamp=datetime.now().isoformat(),
        )
    
    def run_all_scenarios(
        self,
        model_id: str,
    ) -> dict[str, ScenarioResult]:
        """
        Exécute tous les scénarios réalistes.
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"REALISTIC BANKING SCENARIOS")
        print(f"Model: {model_id}")
        print(f"{'='*60}")
        
        # FAQ Assistant
        print("\n[1/4] Banking FAQ Assistant")
        results["faq"] = self.evaluate_faq_assistant(model_id)
        
        # Sentiment Analysis
        print("\n[2/4] Sentiment Analysis")
        results["sentiment"] = self.evaluate_sentiment_analysis(model_id)
        
        # Information Extraction
        print("\n[3/4] Information Extraction")
        results["extraction"] = self.evaluate_info_extraction(model_id)
        
        # Document Summary
        print("\n[4/4] Document Summary")
        results["summary"] = self.evaluate_document_summary(model_id)
        
        # Résumé
        self._print_summary(results)
        
        # Sauvegarder
        self._save_results(results, model_id)
        
        return results
    
    def _print_summary(self, results: dict[str, ScenarioResult]):
        """Affiche le résumé des résultats."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2%}" if value <= 1 else f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            print(f"  avg_latency: {result.avg_latency_ms:.1f} ms")
    
    def _save_results(self, results: dict[str, ScenarioResult], model_id: str):
        """Sauvegarde les résultats."""
        model_name = model_id.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realistic_scenarios_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        data = {name: result.to_dict() for name, result in results.items()}
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Saved] Results saved to {filepath}")

