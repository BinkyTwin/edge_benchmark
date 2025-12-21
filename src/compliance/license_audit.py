"""
License Audit Module
====================

Audit des licences des modèles utilisés.
Vérifie la conformité avec les termes d'utilisation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class LicenseType(Enum):
    """Types de licences."""
    APACHE_2_0 = "Apache 2.0"
    MIT = "MIT"
    CUSTOM = "Custom/Proprietary"
    LLAMA = "Llama Community License"
    GEMMA = "Gemma Terms of Use"
    UNKNOWN = "Unknown"


class CommercialUse(Enum):
    """Autorisation d'usage commercial."""
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


@dataclass
class LicenseInfo:
    """Informations sur la licence d'un modèle."""
    
    model_name: str
    publisher: str
    license_type: LicenseType
    license_url: str
    
    # Autorisations
    commercial_use: CommercialUse
    modification_allowed: bool
    redistribution_allowed: bool
    
    # Restrictions
    restrictions: list[str] = field(default_factory=list)
    attribution_required: bool = True
    
    # Notes
    notes: str = ""
    
    # Vérification
    verified: bool = False
    verified_date: Optional[str] = None
    verified_by: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "publisher": self.publisher,
            "license_type": self.license_type.value,
            "license_url": self.license_url,
            "commercial_use": self.commercial_use.value,
            "modification_allowed": self.modification_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "restrictions": self.restrictions,
            "attribution_required": self.attribution_required,
            "notes": self.notes,
            "verified": self.verified,
            "verified_date": self.verified_date,
            "verified_by": self.verified_by,
        }


# === LICENCES DES MODÈLES DU BENCHMARK ===

MODEL_LICENSES = {
    "gemma-3n-e4b": LicenseInfo(
        model_name="Gemma 3n E4B",
        publisher="Google",
        license_type=LicenseType.GEMMA,
        license_url="https://ai.google.dev/gemma/terms",
        commercial_use=CommercialUse.RESTRICTED,
        modification_allowed=True,
        redistribution_allowed=True,
        restrictions=[
            "Prohibited uses include weapons development, surveillance, etc.",
            "Must include attribution in derivative works",
            "Cannot use Google trademarks without permission",
            "Some usage restrictions for high-risk applications",
        ],
        attribution_required=True,
        notes="""
        Gemma Terms of Use is a custom license more restrictive than Apache 2.0.
        Commercial use is allowed but subject to prohibited use cases.
        Requires careful review for regulated industries like banking.
        """,
        verified=True,
        verified_date="2024-12-01",
    ),
    
    "qwen3-vl-4b": LicenseInfo(
        model_name="Qwen3-VL 4B Instruct",
        publisher="Alibaba Cloud (Qwen Team)",
        license_type=LicenseType.APACHE_2_0,
        license_url="https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE",
        commercial_use=CommercialUse.ALLOWED,
        modification_allowed=True,
        redistribution_allowed=True,
        restrictions=[],
        attribution_required=True,
        notes="""
        Standard Apache 2.0 license. Very permissive.
        Commercial use, modification, and redistribution are all allowed.
        Only requirement is attribution and license notice preservation.
        """,
        verified=True,
        verified_date="2024-12-01",
    ),
    
    "ministral-3-3b": LicenseInfo(
        model_name="Ministral 3 3B",
        publisher="Mistral AI",
        license_type=LicenseType.APACHE_2_0,
        license_url="https://mistral.ai/news/mistral-3",
        commercial_use=CommercialUse.ALLOWED,
        modification_allowed=True,
        redistribution_allowed=True,
        restrictions=[],
        attribution_required=True,
        notes="""
        Apache 2.0 license. Fully permissive for commercial use.
        Mistral's edge-oriented model released under open license.
        """,
        verified=True,
        verified_date="2024-12-01",
    ),
}


class LicenseAuditor:
    """
    Auditeur de licences pour les modèles du benchmark.
    
    Vérifie la conformité des modèles avec les exigences
    de l'environnement bancaire réglementé.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/compliance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.licenses = MODEL_LICENSES.copy()
    
    def get_license(self, model_key: str) -> Optional[LicenseInfo]:
        """Récupère les informations de licence d'un modèle."""
        return self.licenses.get(model_key)
    
    def add_license(self, model_key: str, license_info: LicenseInfo):
        """Ajoute une licence au registre."""
        self.licenses[model_key] = license_info
    
    def check_commercial_use(self, model_key: str) -> dict:
        """
        Vérifie si un modèle peut être utilisé commercialement.
        
        Returns:
            Dictionnaire avec statut et détails
        """
        license_info = self.get_license(model_key)
        
        if not license_info:
            return {
                "allowed": False,
                "status": "unknown",
                "message": f"No license information found for {model_key}",
            }
        
        if license_info.commercial_use == CommercialUse.ALLOWED:
            return {
                "allowed": True,
                "status": "allowed",
                "message": f"Commercial use is allowed under {license_info.license_type.value}",
                "restrictions": license_info.restrictions,
            }
        elif license_info.commercial_use == CommercialUse.RESTRICTED:
            return {
                "allowed": True,
                "status": "restricted",
                "message": f"Commercial use is allowed with restrictions under {license_info.license_type.value}",
                "restrictions": license_info.restrictions,
            }
        else:
            return {
                "allowed": False,
                "status": "prohibited",
                "message": f"Commercial use is prohibited under {license_info.license_type.value}",
            }
    
    def check_banking_compatibility(self, model_key: str) -> dict:
        """
        Vérifie la compatibilité d'un modèle pour usage bancaire.
        
        Prend en compte les restrictions spécifiques au secteur régulé.
        """
        license_info = self.get_license(model_key)
        
        if not license_info:
            return {
                "compatible": False,
                "risk_level": "high",
                "issues": ["No license information available"],
                "recommendations": ["Obtain and verify license before use"],
            }
        
        issues = []
        recommendations = []
        risk_level = "low"
        
        # Vérifier le type de licence
        if license_info.license_type == LicenseType.CUSTOM:
            issues.append("Custom license requires legal review")
            recommendations.append("Have legal team review license terms")
            risk_level = "medium"
        
        if license_info.license_type == LicenseType.GEMMA:
            issues.append("Gemma license has specific prohibited uses")
            recommendations.append("Verify banking use case is not in prohibited list")
            risk_level = "medium"
        
        # Vérifier les restrictions
        for restriction in license_info.restrictions:
            if "high-risk" in restriction.lower():
                issues.append("License mentions high-risk application restrictions")
                recommendations.append("Confirm banking assistant is not considered high-risk")
                risk_level = "medium"
        
        # Vérifier la vérification
        if not license_info.verified:
            issues.append("License has not been formally verified")
            recommendations.append("Conduct formal license verification")
            risk_level = "high" if risk_level == "medium" else "medium"
        
        return {
            "compatible": len(issues) == 0 or risk_level != "high",
            "risk_level": risk_level,
            "license_type": license_info.license_type.value,
            "issues": issues if issues else ["No issues identified"],
            "recommendations": recommendations if recommendations else ["No additional actions required"],
            "verified": license_info.verified,
        }
    
    def generate_audit_report(self) -> dict:
        """Génère un rapport d'audit complet des licences."""
        report = {
            "metadata": {
                "title": "License Audit Report",
                "generated_at": datetime.now().isoformat(),
                "num_models": len(self.licenses),
            },
            "models": {},
            "summary": {
                "fully_permissive": 0,
                "restricted": 0,
                "custom": 0,
                "banking_compatible": 0,
            },
        }
        
        for model_key, license_info in self.licenses.items():
            commercial_check = self.check_commercial_use(model_key)
            banking_check = self.check_banking_compatibility(model_key)
            
            report["models"][model_key] = {
                "license_info": license_info.to_dict(),
                "commercial_use_check": commercial_check,
                "banking_compatibility_check": banking_check,
            }
            
            # Mise à jour du résumé
            if license_info.license_type == LicenseType.APACHE_2_0:
                report["summary"]["fully_permissive"] += 1
            elif license_info.license_type in [LicenseType.GEMMA, LicenseType.LLAMA]:
                report["summary"]["restricted"] += 1
            elif license_info.license_type == LicenseType.CUSTOM:
                report["summary"]["custom"] += 1
            
            if banking_check["compatible"]:
                report["summary"]["banking_compatible"] += 1
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """Sauvegarde le rapport d'audit."""
        report = self.generate_audit_report()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"license_audit_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[Saved] License audit report saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Affiche le résumé de l'audit."""
        print("\n" + "=" * 60)
        print("LICENSE AUDIT SUMMARY")
        print("=" * 60)
        
        for model_key, license_info in self.licenses.items():
            banking_check = self.check_banking_compatibility(model_key)
            
            status_icon = "✓" if banking_check["compatible"] else "✗"
            risk_color = {
                "low": "",
                "medium": "[!]",
                "high": "[!!]",
            }
            
            print(f"\n{status_icon} {license_info.model_name}")
            print(f"  Publisher: {license_info.publisher}")
            print(f"  License: {license_info.license_type.value}")
            print(f"  Commercial Use: {license_info.commercial_use.value}")
            print(f"  Banking Risk: {banking_check['risk_level']} {risk_color.get(banking_check['risk_level'], '')}")
            
            if banking_check["issues"] and banking_check["issues"][0] != "No issues identified":
                print("  Issues:")
                for issue in banking_check["issues"]:
                    print(f"    - {issue}")
        
        print("\n" + "=" * 60)
        
        # Résumé
        report = self.generate_audit_report()
        summary = report["summary"]
        
        print(f"\nTotal Models: {len(self.licenses)}")
        print(f"Fully Permissive (Apache 2.0): {summary['fully_permissive']}")
        print(f"Restricted Licenses: {summary['restricted']}")
        print(f"Banking Compatible: {summary['banking_compatible']}/{len(self.licenses)}")
        print("=" * 60)


