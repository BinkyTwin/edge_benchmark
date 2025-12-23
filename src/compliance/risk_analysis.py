"""
Risk Analysis Module
====================

Risk analysis framework for LLM deployment
in regulated banking environments.

Based on:
- NIST AI RMF 1.0 (AI Risk Management Framework)
- NIST AI 600-1 (Generative AI Profile)
- OWASP Top 10 for LLM Applications
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class RiskLevel(Enum):
    """Risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """OWASP Top 10 LLM risk category."""
    LLM01_PROMPT_INJECTION = "LLM01: Prompt Injection"
    LLM02_INSECURE_OUTPUT = "LLM02: Insecure Output Handling"
    LLM03_TRAINING_DATA_POISONING = "LLM03: Training Data Poisoning"
    LLM04_MODEL_DENIAL_OF_SERVICE = "LLM04: Model Denial of Service"
    LLM05_SUPPLY_CHAIN = "LLM05: Supply Chain Vulnerabilities"
    LLM06_SENSITIVE_INFO_DISCLOSURE = "LLM06: Sensitive Information Disclosure"
    LLM07_INSECURE_PLUGIN = "LLM07: Insecure Plugin Design"
    LLM08_EXCESSIVE_AGENCY = "LLM08: Excessive Agency"
    LLM09_OVERRELIANCE = "LLM09: Overreliance"
    LLM10_MODEL_THEFT = "LLM10: Model Theft"
    
    # On-device specific categories
    DEVICE_LOSS = "Device Loss/Theft"
    DATA_EXFILTRATION = "Data Exfiltration"
    LICENSING = "Licensing/Compliance"


class ControlType(Enum):
    """Control type."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"


@dataclass
class Risk:
    """Risk definition."""
    
    id: str
    category: RiskCategory
    title: str
    description: str
    
    # Assessment
    likelihood: RiskLevel = RiskLevel.MEDIUM
    impact: RiskLevel = RiskLevel.MEDIUM
    inherent_risk: RiskLevel = RiskLevel.MEDIUM
    
    # After controls
    residual_risk: RiskLevel = RiskLevel.LOW
    
    # Impacted assets
    assets: list[str] = field(default_factory=list)
    
    # References
    nist_mapping: list[str] = field(default_factory=list)
    owasp_ref: Optional[str] = None


@dataclass
class Control:
    """Control definition."""
    
    id: str
    title: str
    description: str
    control_type: ControlType
    
    # Mitigated risks
    mitigates_risks: list[str] = field(default_factory=list)
    
    # Implementation
    implementation_status: str = "planned"  # planned, in_progress, implemented
    implementation_notes: str = ""
    
    # Effectiveness
    effectiveness: str = "medium"  # low, medium, high


@dataclass
class ThreatModel:
    """Threat model."""
    
    # Assets
    assets: dict[str, str] = field(default_factory=dict)
    
    # Adversaries
    adversaries: list[dict] = field(default_factory=list)
    
    # Attack surfaces
    attack_surfaces: list[str] = field(default_factory=list)


class RiskAnalyzer:
    """
    Risk analyzer for on-device LLM deployment.
    
    Generates a structured analysis based on NIST AI RMF + OWASP.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/compliance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize risks and controls
        self.risks = self._init_risks()
        self.controls = self._init_controls()
        self.threat_model = self._init_threat_model()
    
    def _init_threat_model(self) -> ThreatModel:
        """Initializes the threat model for on-device LLM in banking."""
        return ThreatModel(
            assets={
                "model": "Quantized SLM (GGUF/MLX) deployed on device",
                "inference_data": "Data processed during inference (prompts, responses)",
                "device": "Apple Silicon laptop with the model",
                "logs": "Inference logs and metrics",
                "credentials": "Potential API keys or tokens",
            },
            adversaries=[
                {
                    "type": "external_attacker",
                    "capability": "medium",
                    "motivation": "Access to banking data via prompt injection",
                },
                {
                    "type": "insider_threat",
                    "capability": "high",
                    "motivation": "Data exfiltration via the model",
                },
                {
                    "type": "device_thief",
                    "capability": "low",
                    "motivation": "Physical access to lost/stolen device",
                },
            ],
            attack_surfaces=[
                "User input (prompts) - injection risk",
                "Documents ingested in context",
                "Model output (generated responses)",
                "Local storage (model, cache, logs)",
                "Model supply chain (origin, integrity)",
            ],
        )
    
    def _init_risks(self) -> list[Risk]:
        """Initializes risks specific to on-device banking context."""
        return [
            # === PROMPT INJECTION ===
            Risk(
                id="R01",
                category=RiskCategory.LLM01_PROMPT_INJECTION,
                title="Prompt Injection via Documents",
                description="""
                An internal or external document containing malicious instructions 
                could manipulate the model to reveal sensitive information 
                or execute unauthorized actions.
                """,
                likelihood=RiskLevel.MEDIUM,
                impact=RiskLevel.HIGH,
                inherent_risk=RiskLevel.HIGH,
                residual_risk=RiskLevel.MEDIUM,
                assets=["inference_data", "model"],
                nist_mapping=["MAP 1.1", "MEASURE 2.7"],
                owasp_ref="LLM01",
            ),
            
            # === SENSITIVE INFO DISCLOSURE ===
            Risk(
                id="R02",
                category=RiskCategory.LLM06_SENSITIVE_INFO_DISCLOSURE,
                title="Sensitive Data Leak via Responses",
                description="""
                The model could include sensitive information (account numbers, 
                client data) in its responses, whether through context memorization 
                or inappropriate generation.
                """,
                likelihood=RiskLevel.MEDIUM,
                impact=RiskLevel.HIGH,
                inherent_risk=RiskLevel.HIGH,
                residual_risk=RiskLevel.LOW,
                assets=["inference_data"],
                nist_mapping=["MAP 1.5", "MEASURE 2.11"],
                owasp_ref="LLM06",
            ),
            
            # === DEVICE LOSS ===
            Risk(
                id="R03",
                category=RiskCategory.DEVICE_LOSS,
                title="Perte ou Vol du Device",
                description="""
                A laptop containing the model and potentially cached data 
                could be lost or stolen, exposing local assets.
                """,
                likelihood=RiskLevel.LOW,
                impact=RiskLevel.HIGH,
                inherent_risk=RiskLevel.MEDIUM,
                residual_risk=RiskLevel.LOW,
                assets=["device", "model", "logs"],
                nist_mapping=["GOVERN 1.2", "MAP 1.3"],
            ),
            
            # === SUPPLY CHAIN ===
            Risk(
                id="R04",
                category=RiskCategory.LLM05_SUPPLY_CHAIN,
                title="Model Integrity (Supply Chain)",
                description="""
                The downloaded model could be modified, corrupted, or come from 
                an untrusted source. Lack of integrity verification.
                """,
                likelihood=RiskLevel.LOW,
                impact=RiskLevel.HIGH,
                inherent_risk=RiskLevel.MEDIUM,
                residual_risk=RiskLevel.LOW,
                assets=["model"],
                nist_mapping=["MAP 3.4", "MEASURE 2.10"],
                owasp_ref="LLM05",
            ),
            
            # === LICENSING ===
            Risk(
                id="R05",
                category=RiskCategory.LICENSING,
                title="Licensing Non-Compliance",
                description="""
                Use of models without verification of license terms,
                potentially leading to intellectual property violations 
                or commercial use restrictions.
                """,
                likelihood=RiskLevel.MEDIUM,
                impact=RiskLevel.MEDIUM,
                inherent_risk=RiskLevel.MEDIUM,
                residual_risk=RiskLevel.LOW,
                assets=["model"],
                nist_mapping=["GOVERN 1.4"],
            ),
            
            # === OVERRELIANCE ===
            Risk(
                id="R06",
                category=RiskCategory.LLM09_OVERRELIANCE,
                title="Over-reliance on Model Responses",
                description="""
                Users could blindly trust the model's responses 
                without verification, leading to errors in critical 
                banking processes.
                """,
                likelihood=RiskLevel.MEDIUM,
                impact=RiskLevel.MEDIUM,
                inherent_risk=RiskLevel.MEDIUM,
                residual_risk=RiskLevel.LOW,
                assets=["inference_data"],
                nist_mapping=["MAP 1.6", "MEASURE 3.3"],
                owasp_ref="LLM09",
            ),
            
            # === DATA EXFILTRATION ===
            Risk(
                id="R07",
                category=RiskCategory.DATA_EXFILTRATION,
                title="Exfiltration via Logs/Cache",
                description="""
                Inference logs or local cache could contain sensitive data
                and be exfiltrated via uncontrolled vectors.
                """,
                likelihood=RiskLevel.LOW,
                impact=RiskLevel.HIGH,
                inherent_risk=RiskLevel.MEDIUM,
                residual_risk=RiskLevel.LOW,
                assets=["logs", "inference_data"],
                nist_mapping=["MAP 1.5", "MEASURE 2.11"],
            ),
        ]
    
    def _init_controls(self) -> list[Control]:
        """Initializes security controls."""
        return [
            # === PREVENTIVE CONTROLS ===
            Control(
                id="C01",
                title="Disk Encryption (FileVault)",
                description="Enable full disk encryption on all devices.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R03", "R07"],
                implementation_status="implemented",
                effectiveness="high",
            ),
            Control(
                id="C02",
                title="Input Validation",
                description="Implement validation and sanitization of user prompts.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R01", "R02"],
                implementation_status="planned",
                effectiveness="medium",
            ),
            Control(
                id="C03",
                title="Model Registry",
                description="Maintain an internal registry of approved models with SHA256 hash.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R04", "R05"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C04",
                title="Minimal Logging Policy",
                description="Do not log sensitive data, regularly purge logs.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R07", "R02"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C05",
                title="Audit des Licences",
                description="Verify and document licenses before deployment.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R05"],
                implementation_status="implemented",
                effectiveness="high",
            ),
            
            # === DETECTIVE CONTROLS ===
            Control(
                id="C06",
                title="Output Monitoring",
                description="Detect sensitive data patterns in responses.",
                control_type=ControlType.DETECTIVE,
                mitigates_risks=["R02", "R06"],
                implementation_status="planned",
                effectiveness="medium",
            ),
            Control(
                id="C07",
                title="Model Integrity Verification",
                description="Verify model hash at startup.",
                control_type=ControlType.DETECTIVE,
                mitigates_risks=["R04"],
                implementation_status="planned",
                effectiveness="high",
            ),
            
            # === CORRECTIVE CONTROLS ===
            Control(
                id="C08",
                title="Remote Wipe Procedure",
                description="Ability to remotely wipe device data (MDM).",
                control_type=ControlType.CORRECTIVE,
                mitigates_risks=["R03"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C09",
                title="Human-in-the-Loop",
                description="Require human validation for critical decisions.",
                control_type=ControlType.CORRECTIVE,
                mitigates_risks=["R06"],
                implementation_status="implemented",
                effectiveness="high",
            ),
        ]
    
    def generate_risk_matrix(self) -> dict:
        """Generates a risk matrix."""
        matrix = {
            "inherent": {},
            "residual": {},
        }
        
        for risk in self.risks:
            matrix["inherent"][risk.id] = {
                "title": risk.title,
                "category": risk.category.value,
                "level": risk.inherent_risk.value,
            }
            matrix["residual"][risk.id] = {
                "title": risk.title,
                "category": risk.category.value,
                "level": risk.residual_risk.value,
            }
        
        return matrix
    
    def generate_control_mapping(self) -> dict:
        """Generates risks -> controls mapping."""
        mapping = {}
        
        for risk in self.risks:
            risk_controls = [
                c for c in self.controls 
                if risk.id in c.mitigates_risks
            ]
            mapping[risk.id] = {
                "risk_title": risk.title,
                "inherent_risk": risk.inherent_risk.value,
                "controls": [
                    {
                        "id": c.id,
                        "title": c.title,
                        "type": c.control_type.value,
                        "status": c.implementation_status,
                        "effectiveness": c.effectiveness,
                    }
                    for c in risk_controls
                ],
                "residual_risk": risk.residual_risk.value,
            }
        
        return mapping
    
    def generate_full_report(self) -> dict:
        """Generates the complete risk analysis report."""
        report = {
            "metadata": {
                "title": "Risk Analysis Report - On-Device LLM Deployment",
                "framework": "NIST AI RMF 1.0 + OWASP Top 10 LLM",
                "generated_at": datetime.now().isoformat(),
                "scope": "Consumer-grade Apple Silicon laptop, banking context",
            },
            "threat_model": {
                "assets": self.threat_model.assets,
                "adversaries": self.threat_model.adversaries,
                "attack_surfaces": self.threat_model.attack_surfaces,
            },
            "risk_taxonomy": [
                {
                    "id": r.id,
                    "category": r.category.value,
                    "title": r.title,
                    "description": r.description.strip(),
                    "likelihood": r.likelihood.value,
                    "impact": r.impact.value,
                    "inherent_risk": r.inherent_risk.value,
                    "residual_risk": r.residual_risk.value,
                    "assets": r.assets,
                    "nist_mapping": r.nist_mapping,
                    "owasp_ref": r.owasp_ref,
                }
                for r in self.risks
            ],
            "controls": [
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "type": c.control_type.value,
                    "mitigates_risks": c.mitigates_risks,
                    "status": c.implementation_status,
                    "effectiveness": c.effectiveness,
                }
                for c in self.controls
            ],
            "risk_matrix": self.generate_risk_matrix(),
            "control_mapping": self.generate_control_mapping(),
            "summary": self._generate_summary(),
        }
        
        return report
    
    def _generate_summary(self) -> dict:
        """Generates the analysis summary."""
        risk_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        
        for risk in self.risks:
            risk_counts[risk.residual_risk.value] += 1
        
        control_counts = {
            "implemented": 0,
            "in_progress": 0,
            "planned": 0,
        }
        
        for control in self.controls:
            control_counts[control.implementation_status] += 1
        
        return {
            "total_risks": len(self.risks),
            "residual_risk_distribution": risk_counts,
            "total_controls": len(self.controls),
            "control_implementation_status": control_counts,
            "key_findings": [
                "On-device deployment reduces network leakage risks",
                "Disk encryption is critical for asset protection",
                "Input validation remains the main security challenge",
                "Model governance (licensing, integrity) must be formalized",
            ],
            "recommendations": [
                "Implement prompt validation before inference",
                "Establish a formal model registry with integrity verification",
                "Define a logging policy minimizing data retention",
                "Train users on SLM limitations (avoid over-reliance)",
            ],
        }
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """Saves the report."""
        report = self.generate_full_report()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_analysis_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[Saved] Risk analysis report saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Prints the analysis summary."""
        summary = self._generate_summary()
        
        print("\n" + "=" * 60)
        print("RISK ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Risks: {summary['total_risks']}")
        print("Residual Risk Distribution:")
        for level, count in summary['residual_risk_distribution'].items():
            print(f"  {level.upper()}: {count}")
        
        print(f"\nTotal Controls: {summary['total_controls']}")
        print("Implementation Status:")
        for status, count in summary['control_implementation_status'].items():
            print(f"  {status}: {count}")
        
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  → {rec}")
        
        print("=" * 60)


