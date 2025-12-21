"""
Risk Analysis Module
====================

Framework d'analyse de risques pour le déploiement de LLMs
en environnement bancaire réglementé.

Basé sur:
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
    """Niveau de risque."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Catégorie de risque OWASP Top 10 LLM."""
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
    
    # Catégories spécifiques on-device
    DEVICE_LOSS = "Device Loss/Theft"
    DATA_EXFILTRATION = "Data Exfiltration"
    LICENSING = "Licensing/Compliance"


class ControlType(Enum):
    """Type de contrôle."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"


@dataclass
class Risk:
    """Définition d'un risque."""
    
    id: str
    category: RiskCategory
    title: str
    description: str
    
    # Évaluation
    likelihood: RiskLevel = RiskLevel.MEDIUM
    impact: RiskLevel = RiskLevel.MEDIUM
    inherent_risk: RiskLevel = RiskLevel.MEDIUM
    
    # Après contrôles
    residual_risk: RiskLevel = RiskLevel.LOW
    
    # Assets impactés
    assets: list[str] = field(default_factory=list)
    
    # Références
    nist_mapping: list[str] = field(default_factory=list)
    owasp_ref: Optional[str] = None


@dataclass
class Control:
    """Définition d'un contrôle."""
    
    id: str
    title: str
    description: str
    control_type: ControlType
    
    # Risques mitigés
    mitigates_risks: list[str] = field(default_factory=list)
    
    # Implémentation
    implementation_status: str = "planned"  # planned, in_progress, implemented
    implementation_notes: str = ""
    
    # Efficacité
    effectiveness: str = "medium"  # low, medium, high


@dataclass
class ThreatModel:
    """Modèle de menaces."""
    
    # Assets
    assets: dict[str, str] = field(default_factory=dict)
    
    # Adversaires
    adversaries: list[dict] = field(default_factory=list)
    
    # Surfaces d'attaque
    attack_surfaces: list[str] = field(default_factory=list)


class RiskAnalyzer:
    """
    Analyseur de risques pour le déploiement LLM on-device.
    
    Génère une analyse structurée basée sur NIST AI RMF + OWASP.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/compliance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser les risques et contrôles
        self.risks = self._init_risks()
        self.controls = self._init_controls()
        self.threat_model = self._init_threat_model()
    
    def _init_threat_model(self) -> ThreatModel:
        """Initialise le modèle de menaces pour LLM on-device en banque."""
        return ThreatModel(
            assets={
                "model": "SLM quantifié (GGUF/MLX) déployé sur device",
                "inference_data": "Données traitées pendant l'inférence (prompts, réponses)",
                "device": "Laptop Apple Silicon avec le modèle",
                "logs": "Logs d'inférence et métriques",
                "credentials": "Éventuelles clés API ou tokens",
            },
            adversaries=[
                {
                    "type": "external_attacker",
                    "capability": "medium",
                    "motivation": "Accès aux données bancaires via prompt injection",
                },
                {
                    "type": "insider_threat",
                    "capability": "high",
                    "motivation": "Exfiltration de données via le modèle",
                },
                {
                    "type": "device_thief",
                    "capability": "low",
                    "motivation": "Accès physique au device perdu/volé",
                },
            ],
            attack_surfaces=[
                "Input utilisateur (prompts) - risque injection",
                "Documents ingérés dans le contexte",
                "Sortie du modèle (réponses générées)",
                "Stockage local (modèle, cache, logs)",
                "Supply chain du modèle (origine, intégrité)",
            ],
        )
    
    def _init_risks(self) -> list[Risk]:
        """Initialise les risques spécifiques au contexte on-device banking."""
        return [
            # === PROMPT INJECTION ===
            Risk(
                id="R01",
                category=RiskCategory.LLM01_PROMPT_INJECTION,
                title="Prompt Injection via Documents",
                description="""
                Un document interne ou externe contenant des instructions malveillantes 
                pourrait manipuler le modèle pour révéler des informations sensibles 
                ou exécuter des actions non autorisées.
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
                title="Fuite de Données Sensibles via Réponses",
                description="""
                Le modèle pourrait inclure des informations sensibles (numéros de compte, 
                données client) dans ses réponses, que ce soit par mémorisation du contexte 
                ou par génération inappropriée.
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
                Un laptop contenant le modèle et potentiellement des données en cache 
                pourrait être perdu ou volé, exposant les assets locaux.
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
                title="Intégrité du Modèle (Supply Chain)",
                description="""
                Le modèle téléchargé pourrait être modifié, corrompu, ou provenir 
                d'une source non fiable. Absence de vérification d'intégrité.
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
                title="Non-Conformité Licensing",
                description="""
                Utilisation de modèles sans vérification des termes de licence, 
                pouvant entraîner des violations de propriété intellectuelle 
                ou des restrictions d'usage commercial.
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
                title="Sur-dépendance aux Réponses du Modèle",
                description="""
                Les utilisateurs pourraient faire confiance aveuglément aux réponses 
                du modèle sans vérification, entraînant des erreurs dans les processus 
                bancaires critiques.
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
                Les logs d'inférence ou le cache local pourraient contenir des données 
                sensibles et être exfiltrés via des vecteurs non contrôlés.
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
        """Initialise les contrôles de sécurité."""
        return [
            # === CONTRÔLES PRÉVENTIFS ===
            Control(
                id="C01",
                title="Chiffrement Disque (FileVault)",
                description="Activer le chiffrement intégral du disque sur tous les devices.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R03", "R07"],
                implementation_status="implemented",
                effectiveness="high",
            ),
            Control(
                id="C02",
                title="Validation des Entrées",
                description="Implémenter une validation et sanitization des prompts utilisateur.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R01", "R02"],
                implementation_status="planned",
                effectiveness="medium",
            ),
            Control(
                id="C03",
                title="Registre des Modèles",
                description="Maintenir un registre interne des modèles approuvés avec hash SHA256.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R04", "R05"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C04",
                title="Politique de Logging Minimale",
                description="Ne pas logger les données sensibles, purger régulièrement les logs.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R07", "R02"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C05",
                title="Audit des Licences",
                description="Vérifier et documenter les licences avant déploiement.",
                control_type=ControlType.PREVENTIVE,
                mitigates_risks=["R05"],
                implementation_status="implemented",
                effectiveness="high",
            ),
            
            # === CONTRÔLES DÉTECTIFS ===
            Control(
                id="C06",
                title="Monitoring des Sorties",
                description="Détecter les patterns de données sensibles dans les réponses.",
                control_type=ControlType.DETECTIVE,
                mitigates_risks=["R02", "R06"],
                implementation_status="planned",
                effectiveness="medium",
            ),
            Control(
                id="C07",
                title="Vérification Intégrité Modèle",
                description="Vérifier le hash du modèle au démarrage.",
                control_type=ControlType.DETECTIVE,
                mitigates_risks=["R04"],
                implementation_status="planned",
                effectiveness="high",
            ),
            
            # === CONTRÔLES CORRECTIFS ===
            Control(
                id="C08",
                title="Procédure de Wipe Distant",
                description="Capacité à effacer les données du device à distance (MDM).",
                control_type=ControlType.CORRECTIVE,
                mitigates_risks=["R03"],
                implementation_status="planned",
                effectiveness="high",
            ),
            Control(
                id="C09",
                title="Human-in-the-Loop",
                description="Exiger validation humaine pour les décisions critiques.",
                control_type=ControlType.CORRECTIVE,
                mitigates_risks=["R06"],
                implementation_status="implemented",
                effectiveness="high",
            ),
        ]
    
    def generate_risk_matrix(self) -> dict:
        """Génère une matrice de risques."""
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
        """Génère le mapping risques -> contrôles."""
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
        """Génère le rapport complet d'analyse de risques."""
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
        """Génère le résumé de l'analyse."""
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
                "Le déploiement on-device réduit les risques de fuite réseau",
                "Le chiffrement disque est critique pour la protection des assets",
                "La validation des entrées reste le principal défi de sécurité",
                "La gouvernance des modèles (licensing, intégrité) doit être formalisée",
            ],
            "recommendations": [
                "Implémenter une validation des prompts avant inférence",
                "Établir un registre formel des modèles avec vérification d'intégrité",
                "Définir une politique de logging minimisant la rétention de données",
                "Former les utilisateurs sur les limites des SLMs (éviter sur-dépendance)",
            ],
        }
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """Sauvegarde le rapport."""
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
        """Affiche le résumé de l'analyse."""
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


