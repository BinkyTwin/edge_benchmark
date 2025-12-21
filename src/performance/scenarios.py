"""
Performance Scenarios
=====================

Définition et exécution des 3 scénarios de benchmark performance:
1. Interactive Assistant (latency-sensitive)
2. Long-form Summarization (throughput-sensitive)
3. Structured JSON Output (reliability-sensitive)
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScenarioConfig:
    """Configuration d'un scénario de performance."""
    
    name: str
    description: str
    category: str
    
    # Paramètres de génération
    input_tokens_min: int
    input_tokens_max: int
    output_tokens: int
    max_tokens: int
    
    # Options
    use_structured_output: bool = False
    response_format: Optional[dict] = None
    
    # Métriques prioritaires
    metrics_focus: list[str] = None
    
    # Seuils de qualité
    quality_thresholds: dict = None


# === PROMPTS PAR SCÉNARIO ===

INTERACTIVE_ASSISTANT_PROMPTS = [
    # Questions bancaires courantes (~200-400 tokens avec contexte)
    """You are a helpful banking assistant. A customer asks:

"I noticed a charge on my credit card that I don't recognize. It says 'AMZN MKTP US' for $47.99. 
I don't remember making this purchase. What should I do? Can you help me dispute this charge?"

Please provide a helpful and accurate response.""",

    """You are a helpful banking assistant. A customer asks:

"I'm trying to set up a recurring transfer from my checking account to my savings account. 
I want to transfer $500 on the 15th of every month. Can you walk me through the process 
and tell me if there are any fees involved?"

Please provide a helpful and accurate response.""",

    """You are a helpful banking assistant. A customer asks:

"My debit card was declined at the grocery store today, but I know I have money in my account. 
I checked my balance online and it shows $1,234.56. This is really embarrassing. 
What could be causing this issue?"

Please provide a helpful and accurate response.""",

    """You are a helpful banking assistant. A customer asks:

"I'm planning to travel to Europe next month for two weeks. I want to use my credit card 
for purchases abroad. What do I need to do to avoid any issues? Are there foreign transaction fees?"

Please provide a helpful and accurate response.""",

    """You are a helpful banking assistant. A customer asks:

"I received an email saying my account has been compromised and I need to click a link 
to verify my identity. The email looks official with your bank's logo. Is this legitimate? 
I'm worried about my account security."

Please provide a helpful and accurate response.""",
]

SUMMARIZATION_PROMPTS = [
    # Documents longs à résumer (~2000-4000 tokens)
    """Please summarize the following financial report:

QUARTERLY FINANCIAL REPORT - Q3 2024

EXECUTIVE SUMMARY
The third quarter of 2024 demonstrated strong performance across all major business segments, 
with consolidated revenues reaching $4.2 billion, representing a 12% increase year-over-year. 
Net income attributable to shareholders was $892 million, or $2.15 per diluted share, 
compared to $756 million, or $1.82 per diluted share, in Q3 2023.

REVENUE BREAKDOWN BY SEGMENT

Consumer Banking Division
The Consumer Banking division generated $1.8 billion in revenue, up 15% from the prior year. 
Key drivers included:
- Net interest income increased by $180 million due to higher average loan balances
- Fee income grew 8% driven by increased debit card usage and ATM transactions
- Digital banking adoption reached 78% of active customers, up from 71% in Q3 2023
- New account openings increased 22% compared to the same period last year

Mortgage lending activity remained robust despite rising interest rates, with $12.4 billion 
in new originations. The refinancing volume declined 35% as expected given the rate environment, 
but purchase mortgages grew 18% year-over-year.

Commercial Banking Division
Commercial Banking revenues totaled $1.4 billion, an increase of 9% year-over-year. 
Notable developments included:
- Commercial loan portfolio grew to $89 billion, up from $82 billion at year-end 2023
- Average commercial deposit balances increased 6% to $45 billion
- Treasury management fees increased 11% driven by new client acquisitions
- Credit quality remained stable with non-performing loans at 0.8% of total loans

Wealth Management Division
The Wealth Management segment contributed $620 million in revenue, up 14% from Q3 2023:
- Assets under management reached $215 billion, a record high
- Net new asset flows were $8.2 billion during the quarter
- Investment management fees grew 16% due to market appreciation and new client assets
- Financial planning services revenue increased 22%

Investment Banking Division
Investment Banking generated $380 million in revenue, compared to $298 million in Q3 2023:
- M&A advisory fees increased 45% with several large transaction completions
- Equity underwriting activity recovered significantly from depressed 2023 levels
- Debt capital markets revenue grew 18% on increased bond issuance

OPERATING EXPENSES AND EFFICIENCY
Total operating expenses were $2.8 billion, up 8% year-over-year. The efficiency ratio 
improved to 58.2% from 59.5% in Q3 2023. Key expense drivers included:
- Technology investments of $340 million focused on digital transformation
- Compensation expense increased 6% reflecting merit increases and hiring
- Occupancy costs decreased 12% due to branch optimization initiatives
- Marketing spend increased 25% to support new product launches

CREDIT QUALITY AND PROVISIONS
The provision for credit losses was $245 million, compared to $198 million in Q3 2023. 
The increase reflects a modest build in reserves due to economic uncertainty. Key metrics:
- Net charge-offs were $178 million, or 0.42% of average loans annualized
- Allowance for credit losses was $3.2 billion, representing 1.45% of total loans
- Non-performing assets were 0.65% of total assets, stable versus prior quarter

CAPITAL AND LIQUIDITY
The company maintained strong capital and liquidity positions:
- Common Equity Tier 1 ratio was 12.8%, well above regulatory minimums
- Total risk-based capital ratio was 15.2%
- Liquidity coverage ratio was 128%, exceeding the 100% requirement
- During Q3, the company repurchased 15 million shares for $675 million

OUTLOOK
Management remains cautiously optimistic about Q4 2024 and full-year results. 
Key expectations include:
- Net interest income expected to be stable to slightly higher
- Fee income growth projected at 5-7% for full year
- Credit costs may increase modestly if economic conditions soften
- The efficiency target of below 58% remains achievable

Provide a concise 2-3 paragraph summary highlighting the key financial metrics and business developments.""",
]

JSON_EXTRACTION_PROMPTS = [
    # Documents pour extraction JSON (~500-1000 tokens)
    {
        "prompt": """Extract the key information from this bank transaction notification and return as JSON:

TRANSACTION NOTIFICATION
========================
Date: December 15, 2024
Time: 14:32:45 EST
Reference Number: TXN-2024-1215-89432

Account Holder: John Michael Smith
Account Number: ****7834
Account Type: Premium Checking

Transaction Details:
- Type: Wire Transfer (Outgoing)
- Amount: $15,750.00 USD
- Recipient: Apex Real Estate Holdings LLC
- Recipient Bank: First National Bank
- Recipient Account: ****4521
- Routing Number: 021000021
- Purpose: Property deposit payment

Status: COMPLETED
Available Balance After Transaction: $42,891.33

Security Verification:
- Two-Factor Authentication: Verified
- IP Address: 192.168.1.xxx (Home Network - Recognized)
- Device: iPhone 14 Pro (Registered Device)

Please contact customer service at 1-800-555-0123 if you did not authorize this transaction.

Extract and return the information as a valid JSON object.""",
        "expected_fields": [
            "transaction_date", "transaction_time", "reference_number",
            "account_holder", "transaction_type", "amount", "recipient_name",
            "status", "remaining_balance"
        ],
        # Format LM Studio: json_schema avec schéma complet
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transaction_extraction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_date": {"type": "string"},
                        "transaction_time": {"type": "string"},
                        "reference_number": {"type": "string"},
                        "account_holder": {"type": "string"},
                        "account_type": {"type": "string"},
                        "transaction_type": {"type": "string"},
                        "amount": {"type": "number"},
                        "currency": {"type": "string"},
                        "recipient_name": {"type": "string"},
                        "recipient_bank": {"type": "string"},
                        "purpose": {"type": "string"},
                        "status": {"type": "string"},
                        "remaining_balance": {"type": "number"},
                        "security_verified": {"type": "boolean"}
                    },
                    "required": ["transaction_date", "transaction_type", "amount", "status"]
                }
            }
        }
    },
]


class ScenarioExecutor:
    """
    Exécuteur de scénarios de performance.
    
    Gère la sélection des prompts et l'exécution des scénarios.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: Chemin vers scenarios.yaml
        """
        self.scenarios: dict[str, ScenarioConfig] = {}
        
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_defaults()
    
    def _load_config(self, config_path: Path):
        """Charge la configuration depuis un fichier YAML."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        for name, scenario_data in config.get("scenarios", {}).items():
            params = scenario_data.get("parameters", {})
            input_range = params.get("input_tokens_range", [200, 400])
            output_range = params.get("output_tokens_range", params.get("output_tokens", 128))
            
            self.scenarios[name] = ScenarioConfig(
                name=scenario_data.get("name", name),
                description=scenario_data.get("description", ""),
                category=scenario_data.get("category", "general"),
                input_tokens_min=input_range[0] if isinstance(input_range, list) else input_range,
                input_tokens_max=input_range[1] if isinstance(input_range, list) else input_range,
                output_tokens=output_range[0] if isinstance(output_range, list) else output_range,
                max_tokens=params.get("max_tokens", 512),
                use_structured_output=params.get("use_structured_output", False),
                metrics_focus=scenario_data.get("metrics_focus", []),
                quality_thresholds=scenario_data.get("quality_thresholds", {}),
            )
    
    def _load_defaults(self):
        """Charge les scénarios par défaut."""
        self.scenarios = {
            "interactive_assistant": ScenarioConfig(
                name="Interactive Assistant",
                description="Banking assistant for customer queries",
                category="latency_sensitive",
                input_tokens_min=200,
                input_tokens_max=400,
                output_tokens=128,
                max_tokens=128,
                metrics_focus=["ttft_ms", "output_tokens_per_sec"],
            ),
            "long_form_summarization": ScenarioConfig(
                name="Long-form Summarization",
                description="Summarization of long documents",
                category="throughput_sensitive",
                input_tokens_min=2000,
                input_tokens_max=4000,
                output_tokens=512,
                max_tokens=512,
                metrics_focus=["output_tokens_per_sec", "peak_ram_mb"],
            ),
            "structured_json_output": ScenarioConfig(
                name="Structured JSON Output",
                description="Information extraction with JSON output",
                category="reliability_sensitive",
                input_tokens_min=500,
                input_tokens_max=1000,
                output_tokens=300,
                max_tokens=300,
                use_structured_output=True,
                # LM Studio utilise json_schema au lieu de json_object
                response_format=None,  # Sera défini par prompt spécifique
                metrics_focus=["json_valid_rate", "output_tokens_per_sec"],
            ),
        }
    
    def get_scenario(self, name: str) -> Optional[ScenarioConfig]:
        """Récupère la configuration d'un scénario."""
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> list[str]:
        """Liste les noms des scénarios disponibles."""
        return list(self.scenarios.keys())
    
    def get_prompts(self, scenario_name: str, num_prompts: int = 20) -> list[dict]:
        """
        Génère les prompts pour un scénario.
        
        Args:
            scenario_name: Nom du scénario
            num_prompts: Nombre de prompts à générer
            
        Returns:
            Liste de dictionnaires avec prompt et paramètres
        """
        scenario = self.scenarios.get(scenario_name)
        if not scenario:
            return []
        
        prompts = []
        
        if scenario_name == "interactive_assistant":
            base_prompts = INTERACTIVE_ASSISTANT_PROMPTS
            for i in range(num_prompts):
                prompt = base_prompts[i % len(base_prompts)]
                prompts.append({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": scenario.max_tokens,
                    "response_format": None,
                })
        
        elif scenario_name == "long_form_summarization":
            base_prompts = SUMMARIZATION_PROMPTS
            for i in range(num_prompts):
                prompt = base_prompts[i % len(base_prompts)]
                prompts.append({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": scenario.max_tokens,
                    "response_format": None,
                })
        
        elif scenario_name == "structured_json_output":
            base_prompts = JSON_EXTRACTION_PROMPTS
            for i in range(num_prompts):
                prompt_data = base_prompts[i % len(base_prompts)]
                # Utiliser le format LM Studio: json_schema (pas json_object)
                prompts.append({
                    "messages": [{"role": "user", "content": prompt_data["prompt"]}],
                    "max_tokens": scenario.max_tokens,
                    "response_format": prompt_data.get("response_format"),
                    "expected_fields": prompt_data.get("expected_fields", []),
                })
        
        return prompts
    
    def validate_json_output(self, content: str, expected_fields: list[str] = None) -> dict:
        """
        Valide une sortie JSON.
        
        Note: Un JSON est considéré valide s'il est syntaxiquement correct.
        Les champs manquants sont reportés mais n'invalident pas le JSON.
        
        Args:
            content: Contenu à valider
            expected_fields: Champs attendus (optionnel, pour info seulement)
            
        Returns:
            Dictionnaire avec is_valid, parsed_json, missing_fields, error
        """
        result = {
            "is_valid": False,
            "parsed_json": None,
            "missing_fields": [],
            "fields_present": 0,
            "fields_expected": 0,
            "error": None,
        }
        
        try:
            parsed = json.loads(content)
            result["is_valid"] = True  # JSON syntaxiquement valide
            result["parsed_json"] = parsed
            
            # Vérifier les champs (informatif, n'invalide pas)
            if expected_fields:
                result["fields_expected"] = len(expected_fields)
                present = [f for f in expected_fields if f in parsed]
                missing = [f for f in expected_fields if f not in parsed]
                result["fields_present"] = len(present)
                result["missing_fields"] = missing
                    
        except json.JSONDecodeError as e:
            result["error"] = str(e)
        
        return result


