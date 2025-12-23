"""
Performance Scenarios
=====================

Definition and execution of 3 performance benchmark scenarios:
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
    """Configuration of a performance scenario."""
    
    name: str
    description: str
    category: str
    
    # Generation parameters
    input_tokens_min: int
    input_tokens_max: int
    output_tokens: int
    max_tokens: int
    
    # Options
    use_structured_output: bool = False
    response_format: Optional[dict] = None
    
    # Priority metrics
    metrics_focus: list[str] = None
    
    # Quality thresholds
    quality_thresholds: dict = None


# === PROMPTS PER SCENARIO ===

INTERACTIVE_ASSISTANT_PROMPTS = [
    # Common banking questions (~200-400 tokens with context)
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
    # Long documents to summarize (~2000-4000 tokens)
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

# System prompt optimized for JSON extraction (strict auditor role)
JSON_EXTRACTION_SYSTEM_PROMPT = """You are a strict financial data auditor.
Your ONLY goal is to extract raw data from documents into JSON.

RULES FOR DATA INTEGRITY:
1. TRUTH OVER COMPLETION: If a value is not explicitly stated, use null. NEVER guess.
2. VERIFICATION: Before writing a value, locate it in the source. If not there, use null.
3. NUMBERS: Extract exact values. Handle currency symbols and formats correctly.
4. STRICT JSON: Output only the JSON object. No preamble, no explanation."""

JSON_EXTRACTION_PROMPTS = [
    # Documents for JSON extraction (~500-1000 tokens)
    {
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
        "prompt": """[SOURCE DOCUMENT]
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
[/SOURCE DOCUMENT]

[INSTRUCTION]
Extract the transaction data into JSON. 
For each value, ensure it exists EXACTLY in the source document. Use null if not found.
[/INSTRUCTION]

JSON:""",
        "expected_fields": [
            "transaction_date", "transaction_time", "reference_number",
            "account_holder", "transaction_type", "amount", "recipient_name",
            "status", "remaining_balance"
        ],
        # LM Studio format: json_schema with strict="true" (STRING, not boolean!)
        # https://lmstudio.ai/docs/developer/openai-compat/structured-output
        # Note: For MLX, Outlines is used. For GGUF, llama.cpp grammar.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transaction_extraction",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transaction_date": {
                            "type": ["string", "null"],
                            "description": "Date of transaction (e.g., December 15, 2024)"
                        },
                        "transaction_time": {
                            "type": ["string", "null"],
                            "description": "Time of transaction with timezone"
                        },
                        "reference_number": {
                            "type": ["string", "null"],
                            "description": "Unique transaction reference"
                        },
                        "account_holder": {
                            "type": ["string", "null"],
                            "description": "Full name of account holder"
                        },
                        "account_type": {
                            "type": ["string", "null"],
                            "description": "Type of account"
                        },
                        "transaction_type": {
                            "type": ["string", "null"],
                            "description": "Type of transaction (Wire Transfer, etc.)"
                        },
                        "amount": {
                            "type": ["number", "null"],
                            "description": "Transaction amount as number"
                        },
                        "currency": {
                            "type": ["string", "null"],
                            "description": "Currency code (USD, EUR, etc.)"
                        },
                        "recipient_name": {
                            "type": ["string", "null"],
                            "description": "Name of recipient"
                        },
                        "recipient_bank": {
                            "type": ["string", "null"],
                            "description": "Recipient's bank name"
                        },
                        "purpose": {
                            "type": ["string", "null"],
                            "description": "Purpose of transaction"
                        },
                        "status": {
                            "type": ["string", "null"],
                            "description": "Transaction status"
                        },
                        "remaining_balance": {
                            "type": ["number", "null"],
                            "description": "Balance after transaction"
                        },
                        "security_verified": {
                            "type": ["boolean", "null"],
                            "description": "Whether security checks passed"
                        }
                    },
                    "required": ["transaction_date", "transaction_type", "amount", "status"]
                }
            }
        }
    },
    # Second example: account statement
    {
        "system_prompt": JSON_EXTRACTION_SYSTEM_PROMPT,
        "prompt": """[SOURCE DOCUMENT]
ACCOUNT STATEMENT SUMMARY
=========================
Statement Period: November 1-30, 2024
Account: Business Checking ****9182
Account Holder: TechStart Solutions Inc.

Opening Balance: $128,456.78
Total Credits: $89,234.00 (12 transactions)
Total Debits: $67,891.45 (34 transactions)
Closing Balance: $149,799.33

Key Transactions:
- Nov 5: Payroll Processing -$45,000.00
- Nov 12: Client Payment +$52,000.00
- Nov 18: Vendor Payment -$12,500.00
- Nov 25: Investment Transfer +$25,000.00

Interest Earned: $234.00 (APY: 2.15%)
Service Fees: $0.00 (waived - business premium)
[/SOURCE DOCUMENT]

[INSTRUCTION]
Extract the account statement data into JSON.
For each value, ensure it exists EXACTLY in the source document. Use null if not found.
[/INSTRUCTION]

JSON:""",
        "expected_fields": [
            "statement_period", "account_type", "account_holder",
            "opening_balance", "closing_balance", "total_credits", "total_debits"
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "account_statement_extraction",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "statement_period": {"type": ["string", "null"]},
                        "account_type": {"type": ["string", "null"]},
                        "account_holder": {"type": ["string", "null"]},
                        "opening_balance": {"type": ["number", "null"]},
                        "closing_balance": {"type": ["number", "null"]},
                        "total_credits": {"type": ["number", "null"]},
                        "total_debits": {"type": ["number", "null"]},
                        "credit_count": {"type": ["integer", "null"]},
                        "debit_count": {"type": ["integer", "null"]},
                        "interest_earned": {"type": ["number", "null"]},
                        "apy_rate": {"type": ["number", "null"]},
                        "service_fees": {"type": ["number", "null"]}
                    },
                    "required": ["statement_period", "opening_balance", "closing_balance"]
                }
            }
        }
    },
]


class ScenarioExecutor:
    """
    Performance scenario executor.
    
    Manages prompt selection and scenario execution.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: Path to scenarios.yaml
        """
        self.scenarios: dict[str, ScenarioConfig] = {}
        
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_defaults()
    
    def _load_config(self, config_path: Path):
        """Load configuration from a YAML file."""
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
        """Load default scenarios."""
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
                # LM Studio uses json_schema instead of json_object
                response_format=None,  # Will be defined by specific prompt
                metrics_focus=["json_valid_rate", "output_tokens_per_sec"],
            ),
        }
    
    def get_scenario(self, name: str) -> Optional[ScenarioConfig]:
        """Get configuration for a scenario."""
        return self.scenarios.get(name)
    
    def list_scenarios(self) -> list[str]:
        """List available scenario names."""
        return list(self.scenarios.keys())
    
    def get_prompts(self, scenario_name: str, num_prompts: int = 20) -> list[dict]:
        """
        Generate prompts for a scenario.
        
        Args:
            scenario_name: Scenario name
            num_prompts: Number of prompts to generate
            
        Returns:
            List of dictionaries with prompt and parameters
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
                
                # Build messages with system prompt if available
                messages = []
                if "system_prompt" in prompt_data:
                    messages.append({"role": "system", "content": prompt_data["system_prompt"]})
                messages.append({"role": "user", "content": prompt_data["prompt"]})
                
                # Use LM Studio format: json_schema with strict=True
                # https://lmstudio.ai/docs/developer/openai-compat/structured-output
                prompts.append({
                    "messages": messages,
                    "max_tokens": scenario.max_tokens,
                    "response_format": prompt_data.get("response_format"),
                    "expected_fields": prompt_data.get("expected_fields", []),
                })
        
        return prompts
    
    def validate_json_output(self, content: str, expected_fields: list[str] = None) -> dict:
        """
        Validate a JSON output.
        
        Note: A JSON is considered valid if it is syntactically correct.
        Missing fields are reported but do not invalidate the JSON.
        
        Args:
            content: Content to validate
            expected_fields: Expected fields (optional, for info only)
            
        Returns:
            Dictionary with is_valid, parsed_json, missing_fields, error
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
            result["is_valid"] = True  # Syntactically valid JSON
            result["parsed_json"] = parsed
            
            # Check fields (informative, does not invalidate)
            if expected_fields:
                result["fields_expected"] = len(expected_fields)
                present = [f for f in expected_fields if f in parsed]
                missing = [f for f in expected_fields if f not in parsed]
                result["fields_present"] = len(present)
                result["missing_fields"] = missing
                    
        except json.JSONDecodeError as e:
            result["error"] = str(e)
        
        return result


