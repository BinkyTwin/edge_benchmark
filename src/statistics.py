"""
Statistical Analysis Module
============================

Analyses statistiques rigoureuses pour les benchmarks:
- Intervalles de confiance (IC 95%)
- Bootstrap pour estimations robustes
- Tests de significativité (paired t-test, Wilcoxon)
- Comparaisons multiples avec correction
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Intervalle de confiance."""
    
    mean: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "t-distribution"
    
    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lower:.3f}, {self.upper:.3f}]"
    
    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass
class SignificanceTest:
    """Résultat d'un test de significativité."""
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_size_name: str = ""
    
    # Pour comparaisons multiples
    corrected_p_value: Optional[float] = None
    correction_method: str = ""
    
    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "effect_size_name": self.effect_size_name,
            "corrected_p_value": round(self.corrected_p_value, 6) if self.corrected_p_value else None,
            "correction_method": self.correction_method,
        }


@dataclass
class ModelComparison:
    """Résultat de comparaison entre deux modèles."""
    
    model_a: str
    model_b: str
    metric: str
    
    # Statistiques descriptives
    mean_a: float
    mean_b: float
    diff_mean: float
    diff_percent: float
    
    # Intervalles de confiance
    ci_a: ConfidenceInterval = None
    ci_b: ConfidenceInterval = None
    ci_diff: ConfidenceInterval = None
    
    # Test de significativité
    test_result: SignificanceTest = None
    
    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "diff_mean": round(self.diff_mean, 4),
            "diff_percent": round(self.diff_percent, 2),
            "ci_a": self.ci_a.to_dict() if self.ci_a else None,
            "ci_b": self.ci_b.to_dict() if self.ci_b else None,
            "ci_diff": self.ci_diff.to_dict() if self.ci_diff else None,
            "test_result": self.test_result.to_dict() if self.test_result else None,
        }


class StatisticalAnalyzer:
    """
    Analyseur statistique pour les benchmarks SLM.
    
    Fournit:
    - Intervalles de confiance (t-distribution et bootstrap)
    - Tests de significativité paramétriques et non-paramétriques
    - Correction pour comparaisons multiples
    - Tailles d'effet
    """
    
    def __init__(self, confidence_level: float = 0.95, seed: int = 42):
        """
        Args:
            confidence_level: Niveau de confiance (défaut: 0.95 = 95%)
            seed: Seed pour le bootstrap
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.seed = seed
        np.random.seed(seed)
    
    # =========================================================================
    # INTERVALLES DE CONFIANCE
    # =========================================================================
    
    def confidence_interval_t(
        self,
        data: np.ndarray,
        confidence_level: Optional[float] = None,
    ) -> ConfidenceInterval:
        """
        Calcule l'intervalle de confiance via t-distribution.
        
        Approprié pour échantillons de taille modérée (n >= 20).
        
        Args:
            data: Données (array 1D)
            confidence_level: Niveau de confiance (optionnel)
            
        Returns:
            ConfidenceInterval
        """
        confidence_level = confidence_level or self.confidence_level
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            return ConfidenceInterval(
                mean=float(np.mean(data)),
                lower=float(np.mean(data)),
                upper=float(np.mean(data)),
                confidence_level=confidence_level,
                method="insufficient_data",
            )
        
        mean = np.mean(data)
        se = stats.sem(data)  # Standard error of the mean
        
        # t-value pour le niveau de confiance
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        
        margin = t_value * se
        
        return ConfidenceInterval(
            mean=float(mean),
            lower=float(mean - margin),
            upper=float(mean + margin),
            confidence_level=confidence_level,
            method="t-distribution",
        )
    
    def confidence_interval_bootstrap(
        self,
        data: np.ndarray,
        confidence_level: Optional[float] = None,
        n_bootstrap: int = 10000,
        method: str = "percentile",
    ) -> ConfidenceInterval:
        """
        Calcule l'intervalle de confiance via bootstrap.
        
        Plus robuste, ne suppose pas de distribution normale.
        Recommandé pour petits échantillons ou distributions non-normales.
        
        Args:
            data: Données (array 1D)
            confidence_level: Niveau de confiance
            n_bootstrap: Nombre d'itérations bootstrap
            method: 'percentile' ou 'bca' (bias-corrected accelerated)
            
        Returns:
            ConfidenceInterval
        """
        confidence_level = confidence_level or self.confidence_level
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            return ConfidenceInterval(
                mean=float(np.mean(data)),
                lower=float(np.mean(data)),
                upper=float(np.mean(data)),
                confidence_level=confidence_level,
                method="insufficient_data",
            )
        
        # Bootstrap resampling
        bootstrap_means = np.array([
            np.mean(np.random.choice(data, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])
        
        mean = np.mean(data)
        alpha = 1 - confidence_level
        
        if method == "percentile":
            lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        else:
            # BCa method (plus sophistiqué)
            lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return ConfidenceInterval(
            mean=float(mean),
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method=f"bootstrap_{method}",
        )
    
    # =========================================================================
    # TESTS DE SIGNIFICATIVITÉ
    # =========================================================================
    
    def paired_ttest(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alpha: Optional[float] = None,
    ) -> SignificanceTest:
        """
        Test t apparié (paired t-test).
        
        Utilisé quand les mêmes prompts sont utilisés pour les deux modèles.
        Suppose une distribution normale des différences.
        
        Args:
            data_a: Mesures du modèle A
            data_b: Mesures du modèle B
            alpha: Seuil de significativité
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        if len(data_a) != len(data_b):
            raise ValueError("Les deux échantillons doivent avoir la même taille pour un test apparié")
        
        statistic, p_value = stats.ttest_rel(data_a, data_b)
        
        # Taille d'effet (Cohen's d pour échantillons appariés)
        diff = data_a - data_b
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        return SignificanceTest(
            test_name="paired_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            alpha=alpha,
            effect_size=float(cohens_d),
            effect_size_name="Cohen's d",
        )
    
    def wilcoxon_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alpha: Optional[float] = None,
    ) -> SignificanceTest:
        """
        Test de Wilcoxon signé (non-paramétrique).
        
        Alternative robuste au t-test apparié.
        Ne suppose pas de distribution normale.
        Recommandé pour petits échantillons ou données non-normales.
        
        Args:
            data_a: Mesures du modèle A
            data_b: Mesures du modèle B
            alpha: Seuil de significativité
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        if len(data_a) != len(data_b):
            raise ValueError("Les deux échantillons doivent avoir la même taille")
        
        # Wilcoxon signed-rank test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = stats.wilcoxon(data_a, data_b, alternative='two-sided')
        
        # Taille d'effet (rank-biserial correlation)
        n = len(data_a)
        r = 1 - (2 * statistic) / (n * (n + 1))
        
        return SignificanceTest(
            test_name="wilcoxon_signed_rank",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            alpha=alpha,
            effect_size=float(r),
            effect_size_name="rank-biserial r",
        )
    
    def mann_whitney_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alpha: Optional[float] = None,
    ) -> SignificanceTest:
        """
        Test de Mann-Whitney U (non-paramétrique, échantillons indépendants).
        
        Utilisé quand les prompts sont différents entre modèles.
        
        Args:
            data_a: Mesures du modèle A
            data_b: Mesures du modèle B
            alpha: Seuil de significativité
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        statistic, p_value = stats.mannwhitneyu(
            data_a, data_b, alternative='two-sided'
        )
        
        # Taille d'effet (rank-biserial correlation)
        n1, n2 = len(data_a), len(data_b)
        r = 1 - (2 * statistic) / (n1 * n2)
        
        return SignificanceTest(
            test_name="mann_whitney_u",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < alpha,
            alpha=alpha,
            effect_size=float(r),
            effect_size_name="rank-biserial r",
        )
    
    # =========================================================================
    # COMPARAISONS MULTIPLES
    # =========================================================================
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Correction de Bonferroni pour comparaisons multiples.
        
        Contrôle le FWER (Family-Wise Error Rate).
        Conservatif mais simple.
        
        Args:
            p_values: Liste des p-values
            alpha: Seuil de significativité
            
        Returns:
            Liste de (p_value_corrigée, significatif)
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        corrected_alpha = alpha / n
        
        return [
            (min(p * n, 1.0), p * n < alpha)
            for p in p_values
        ]
    
    def holm_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Correction de Holm-Bonferroni (step-down).
        
        Moins conservatif que Bonferroni, plus puissant.
        
        Args:
            p_values: Liste des p-values
            alpha: Seuil de significativité
            
        Returns:
            Liste de (p_value_corrigée, significatif)
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        
        # Trier les p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # Correction step-down
        corrected_p = np.zeros(n)
        for i, p in enumerate(sorted_p):
            corrected_p[i] = p * (n - i)
        
        # Assurer la monotonie
        for i in range(1, n):
            corrected_p[i] = max(corrected_p[i], corrected_p[i-1])
        
        corrected_p = np.minimum(corrected_p, 1.0)
        
        # Remettre dans l'ordre original
        result = [(0.0, False)] * n
        for i, orig_idx in enumerate(sorted_indices):
            result[orig_idx] = (corrected_p[i], corrected_p[i] < alpha)
        
        return result
    
    # =========================================================================
    # COMPARAISON DE MODÈLES
    # =========================================================================
    
    def compare_models(
        self,
        model_a_name: str,
        model_b_name: str,
        data_a: np.ndarray,
        data_b: np.ndarray,
        metric_name: str,
        paired: bool = True,
        use_bootstrap: bool = True,
    ) -> ModelComparison:
        """
        Compare deux modèles sur une métrique.
        
        Args:
            model_a_name: Nom du modèle A
            model_b_name: Nom du modèle B
            data_a: Mesures du modèle A
            data_b: Mesures du modèle B
            metric_name: Nom de la métrique
            paired: Si True, utilise des tests appariés
            use_bootstrap: Si True, utilise bootstrap pour les IC
            
        Returns:
            ModelComparison complet
        """
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        diff_mean = mean_a - mean_b
        diff_percent = (diff_mean / mean_b) * 100 if mean_b != 0 else 0
        
        # Intervalles de confiance
        if use_bootstrap:
            ci_a = self.confidence_interval_bootstrap(data_a)
            ci_b = self.confidence_interval_bootstrap(data_b)
        else:
            ci_a = self.confidence_interval_t(data_a)
            ci_b = self.confidence_interval_t(data_b)
        
        # IC sur la différence (pour données appariées)
        ci_diff = None
        if paired and len(data_a) == len(data_b):
            diff = data_a - data_b
            if use_bootstrap:
                ci_diff = self.confidence_interval_bootstrap(diff)
            else:
                ci_diff = self.confidence_interval_t(diff)
        
        # Test de significativité
        if paired:
            # Vérifier la normalité des différences
            diff = data_a - data_b
            _, normality_p = stats.shapiro(diff) if len(diff) >= 3 else (0, 0)
            
            if normality_p > 0.05 and len(data_a) >= 20:
                # Distribution normale: t-test
                test_result = self.paired_ttest(data_a, data_b)
            else:
                # Non-normale ou petit échantillon: Wilcoxon
                test_result = self.wilcoxon_test(data_a, data_b)
        else:
            test_result = self.mann_whitney_test(data_a, data_b)
        
        return ModelComparison(
            model_a=model_a_name,
            model_b=model_b_name,
            metric=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            diff_mean=diff_mean,
            diff_percent=diff_percent,
            ci_a=ci_a,
            ci_b=ci_b,
            ci_diff=ci_diff,
            test_result=test_result,
        )
    
    def compare_all_models(
        self,
        models_data: dict[str, np.ndarray],
        metric_name: str,
        paired: bool = True,
    ) -> List[ModelComparison]:
        """
        Compare tous les modèles entre eux.
        
        Args:
            models_data: Dict {model_name: data_array}
            metric_name: Nom de la métrique
            paired: Si True, utilise des tests appariés
            
        Returns:
            Liste de ModelComparison pour chaque paire
        """
        model_names = list(models_data.keys())
        comparisons = []
        p_values = []
        
        # Comparer toutes les paires
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                comparison = self.compare_models(
                    model_a_name=model_names[i],
                    model_b_name=model_names[j],
                    data_a=models_data[model_names[i]],
                    data_b=models_data[model_names[j]],
                    metric_name=metric_name,
                    paired=paired,
                )
                comparisons.append(comparison)
                p_values.append(comparison.test_result.p_value)
        
        # Correction pour comparaisons multiples (Holm)
        if len(p_values) > 1:
            corrected = self.holm_correction(p_values)
            for i, comparison in enumerate(comparisons):
                comparison.test_result.corrected_p_value = corrected[i][0]
                comparison.test_result.correction_method = "holm"
                comparison.test_result.significant = corrected[i][1]
        
        return comparisons
    
    def format_comparison_table(
        self,
        comparisons: List[ModelComparison],
    ) -> str:
        """
        Formate les comparaisons en tableau Markdown.
        
        Args:
            comparisons: Liste de ModelComparison
            
        Returns:
            Tableau Markdown
        """
        lines = []
        lines.append("| Model A | Model B | Metric | Diff (%) | p-value | Sig. | Effect |")
        lines.append("|---------|---------|--------|----------|---------|------|--------|")
        
        for c in comparisons:
            sig = "✓" if c.test_result.significant else ""
            p_val = c.test_result.corrected_p_value or c.test_result.p_value
            effect = f"{c.test_result.effect_size:.2f}" if c.test_result.effect_size else ""
            
            lines.append(
                f"| {c.model_a} | {c.model_b} | {c.metric} | "
                f"{c.diff_percent:+.1f}% | {p_val:.4f} | {sig} | {effect} |"
            )
        
        return "\n".join(lines)

