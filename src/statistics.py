"""
Statistical Analysis Module
============================

Rigorous statistical analyses for benchmarks:
- Confidence intervals (95% CI)
- Bootstrap for robust estimations
- Significance tests (paired t-test, Wilcoxon)
- Multiple comparisons with correction
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Confidence interval."""
    
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
    """Result of a significance test."""
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_size_name: str = ""
    
    # For multiple comparisons
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
    """Result of comparison between two models."""
    
    model_a: str
    model_b: str
    metric: str
    
    # Descriptive statistics
    mean_a: float
    mean_b: float
    diff_mean: float
    diff_percent: float
    
    # Confidence intervals
    ci_a: ConfidenceInterval = None
    ci_b: ConfidenceInterval = None
    ci_diff: ConfidenceInterval = None
    
    # Significance test
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
    Statistical analyzer for SLM benchmarks.
    
    Provides:
    - Confidence intervals (t-distribution and bootstrap)
    - Parametric and non-parametric significance tests
    - Correction for multiple comparisons
    - Effect sizes
    """
    
    def __init__(self, confidence_level: float = 0.95, seed: int = 42):
        """
        Args:
            confidence_level: Confidence level (default: 0.95 = 95%)
            seed: Seed for bootstrap
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.seed = seed
        np.random.seed(seed)
    
    # =========================================================================
    # CONFIDENCE INTERVALS
    # =========================================================================
    
    def confidence_interval_t(
        self,
        data: np.ndarray,
        confidence_level: Optional[float] = None,
    ) -> ConfidenceInterval:
        """
        Computes confidence interval via t-distribution.
        
        Appropriate for moderate sample sizes (n >= 20).
        
        Args:
            data: Data (1D array)
            confidence_level: Confidence level (optional)
            
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
        
        # t-value for the confidence level
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
        Computes confidence interval via bootstrap.
        
        More robust, does not assume normal distribution.
        Recommended for small samples or non-normal distributions.
        
        Args:
            data: Data (1D array)
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap iterations
            method: 'percentile' or 'bca' (bias-corrected accelerated)
            
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
            # BCa method (more sophisticated)
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
    # SIGNIFICANCE TESTS
    # =========================================================================
    
    def paired_ttest(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        alpha: Optional[float] = None,
    ) -> SignificanceTest:
        """
        Paired t-test.
        
        Used when the same prompts are used for both models.
        Assumes normal distribution of differences.
        
        Args:
            data_a: Model A measurements
            data_b: Model B measurements
            alpha: Significance threshold
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        if len(data_a) != len(data_b):
            raise ValueError("Both samples must have the same size for a paired test")
        
        statistic, p_value = stats.ttest_rel(data_a, data_b)
        
        # Effect size (Cohen's d for paired samples)
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
        Wilcoxon signed-rank test (non-parametric).
        
        Robust alternative to paired t-test.
        Does not assume normal distribution.
        Recommended for small samples or non-normal data.
        
        Args:
            data_a: Model A measurements
            data_b: Model B measurements
            alpha: Significance threshold
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        if len(data_a) != len(data_b):
            raise ValueError("Both samples must have the same size")
        
        # Wilcoxon signed-rank test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = stats.wilcoxon(data_a, data_b, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
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
        Mann-Whitney U test (non-parametric, independent samples).
        
        Used when prompts are different between models.
        
        Args:
            data_a: Model A measurements
            data_b: Model B measurements
            alpha: Significance threshold
            
        Returns:
            SignificanceTest
        """
        alpha = alpha or self.alpha
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        statistic, p_value = stats.mannwhitneyu(
            data_a, data_b, alternative='two-sided'
        )
        
        # Effect size (rank-biserial correlation)
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
    # MULTIPLE COMPARISONS
    # =========================================================================
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Bonferroni correction for multiple comparisons.
        
        Controls FWER (Family-Wise Error Rate).
        Conservative but simple.
        
        Args:
            p_values: List of p-values
            alpha: Significance threshold
            
        Returns:
            List of (corrected_p_value, significant)
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
        Holm-Bonferroni correction (step-down).
        
        Less conservative than Bonferroni, more powerful.
        
        Args:
            p_values: List of p-values
            alpha: Significance threshold
            
        Returns:
            List of (corrected_p_value, significant)
        """
        alpha = alpha or self.alpha
        n = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # Step-down correction
        corrected_p = np.zeros(n)
        for i, p in enumerate(sorted_p):
            corrected_p[i] = p * (n - i)
        
        # Ensure monotonicity
        for i in range(1, n):
            corrected_p[i] = max(corrected_p[i], corrected_p[i-1])
        
        corrected_p = np.minimum(corrected_p, 1.0)
        
        # Restore original order
        result = [(0.0, False)] * n
        for i, orig_idx in enumerate(sorted_indices):
            result[orig_idx] = (corrected_p[i], corrected_p[i] < alpha)
        
        return result
    
    # =========================================================================
    # MODEL COMPARISON
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
        Compare two models on a metric.
        
        Args:
            model_a_name: Model A name
            model_b_name: Model B name
            data_a: Model A measurements
            data_b: Model B measurements
            metric_name: Metric name
            paired: If True, uses paired tests
            use_bootstrap: If True, uses bootstrap for CIs
            
        Returns:
            Complete ModelComparison
        """
        data_a = np.asarray(data_a)
        data_b = np.asarray(data_b)
        
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        diff_mean = mean_a - mean_b
        diff_percent = (diff_mean / mean_b) * 100 if mean_b != 0 else 0
        
        # Confidence intervals
        if use_bootstrap:
            ci_a = self.confidence_interval_bootstrap(data_a)
            ci_b = self.confidence_interval_bootstrap(data_b)
        else:
            ci_a = self.confidence_interval_t(data_a)
            ci_b = self.confidence_interval_t(data_b)
        
        # CI on the difference (for paired data)
        ci_diff = None
        if paired and len(data_a) == len(data_b):
            diff = data_a - data_b
            if use_bootstrap:
                ci_diff = self.confidence_interval_bootstrap(diff)
            else:
                ci_diff = self.confidence_interval_t(diff)
        
        # Significance test
        if paired:
            # Check normality of differences
            diff = data_a - data_b
            _, normality_p = stats.shapiro(diff) if len(diff) >= 3 else (0, 0)
            
            if normality_p > 0.05 and len(data_a) >= 20:
                # Normal distribution: t-test
                test_result = self.paired_ttest(data_a, data_b)
            else:
                # Non-normal or small sample: Wilcoxon
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
        Compare all models against each other.
        
        Args:
            models_data: Dict {model_name: data_array}
            metric_name: Metric name
            paired: If True, uses paired tests
            
        Returns:
            List of ModelComparison for each pair
        """
        model_names = list(models_data.keys())
        comparisons = []
        p_values = []
        
        # Compare all pairs
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
        
        # Correction for multiple comparisons (Holm)
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
        Formats comparisons as a Markdown table.
        
        Args:
            comparisons: List of ModelComparison
            
        Returns:
            Markdown table
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

