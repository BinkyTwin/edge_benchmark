#!/usr/bin/env python3
"""
Generate Benchmark Report
=========================

Génère un rapport consolidé des résultats de benchmark.

Usage:
    python scripts/generate_report.py --input results/ --output report/
    python scripts/generate_report.py --format html
    python scripts/generate_report.py --format markdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


class ReportGenerator:
    """
    Générateur de rapports pour les benchmarks SLM.
    
    Consolide les résultats de performance, capacités et conformité
    en un rapport lisible et exportable.
    """
    
    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Données chargées
        self.performance_results = []
        self.capability_results = []
        self.compliance_results = {}
    
    def load_results(self):
        """Charge tous les résultats depuis le dossier results."""
        print("[Loading] Scanning results directory...")
        
        # Performance results (JSONL)
        for file in self.results_dir.glob("perf_*.jsonl"):
            with open(file) as f:
                for line in f:
                    if line.strip():
                        self.performance_results.append(json.loads(line))
        
        # Capability results (JSON)
        for file in self.results_dir.glob("eval_*.json"):
            with open(file) as f:
                self.capability_results.append(json.load(f))
        
        # Realistic scenarios
        for file in self.results_dir.glob("realistic_*.json"):
            with open(file) as f:
                data = json.load(f)
                self.capability_results.append({
                    "type": "realistic_scenarios",
                    "data": data,
                })
        
        # Coding results
        for file in self.results_dir.glob("coding_*.json"):
            with open(file) as f:
                self.capability_results.append(json.load(f))
        
        # Compliance results
        compliance_dir = self.results_dir / "compliance"
        if compliance_dir.exists():
            for file in compliance_dir.glob("*.json"):
                with open(file) as f:
                    self.compliance_results[file.stem] = json.load(f)
        
        print(f"[Loaded] {len(self.performance_results)} performance results")
        print(f"[Loaded] {len(self.capability_results)} capability results")
        print(f"[Loaded] {len(self.compliance_results)} compliance reports")
    
    def generate_performance_table(self) -> pd.DataFrame:
        """Génère le tableau de résultats de performance."""
        if not self.performance_results:
            return pd.DataFrame()
        
        rows = []
        for result in self.performance_results:
            metrics = result.get("metrics", {})
            row = {
                "Model": result.get("model_id", "Unknown"),
                "Scenario": result.get("scenario_name", "Unknown"),
                "TTFT (ms)": metrics.get("ttft", {}).get("mean_ms", 0),
                "TTFT p95 (ms)": metrics.get("ttft", {}).get("p95_ms", 0),
                "Tokens/s": metrics.get("output_tokens_per_sec", {}).get("mean", 0),
                "Peak RAM (MB)": metrics.get("memory", {}).get("peak_ram_mb", 0),
                "Success Rate": metrics.get("runs", {}).get("success_rate", 0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_capability_table(self) -> pd.DataFrame:
        """Génère le tableau de résultats de capacités."""
        if not self.capability_results:
            return pd.DataFrame()
        
        rows = []
        for result in self.capability_results:
            if result.get("type") == "realistic_scenarios":
                continue  # Traité séparément
            
            row = {
                "Model": result.get("model_id", "Unknown"),
                "Dataset": result.get("dataset_name", result.get("task", "Unknown")),
                "Accuracy": result.get("accuracy", 0),
                "Macro F1": result.get("macro_f1", 0),
                "Samples": result.get("num_samples", 0),
                "Avg Latency (ms)": result.get("avg_latency_ms", 0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_markdown_report(self) -> str:
        """Génère un rapport au format Markdown."""
        lines = []
        
        # Header
        lines.append("# Edge SLM Benchmark Report")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("This report presents the benchmark results for Small Language Models (SLMs) ")
        lines.append("evaluated on Apple Silicon hardware in a simulated banking context.")
        lines.append("")
        
        # Performance Results
        lines.append("## Performance Results")
        lines.append("")
        
        perf_df = self.generate_performance_table()
        if not perf_df.empty:
            lines.append(perf_df.to_markdown(index=False))
        else:
            lines.append("*No performance results available.*")
        lines.append("")
        
        # Capability Results
        lines.append("## Capability Results")
        lines.append("")
        
        cap_df = self.generate_capability_table()
        if not cap_df.empty:
            lines.append(cap_df.to_markdown(index=False))
        else:
            lines.append("*No capability results available.*")
        lines.append("")
        
        # Compliance Summary
        if self.compliance_results:
            lines.append("## Compliance Analysis")
            lines.append("")
            
            if "risk_analysis" in str(self.compliance_results):
                for key, data in self.compliance_results.items():
                    if "risk" in key.lower():
                        summary = data.get("summary", {})
                        lines.append("### Risk Summary")
                        lines.append("")
                        lines.append(f"- **Total Risks**: {summary.get('total_risks', 'N/A')}")
                        lines.append(f"- **Total Controls**: {summary.get('total_controls', 'N/A')}")
                        lines.append("")
                        
                        if "key_findings" in summary:
                            lines.append("### Key Findings")
                            for finding in summary["key_findings"]:
                                lines.append(f"- {finding}")
                        lines.append("")
            
            if "license_audit" in str(self.compliance_results):
                for key, data in self.compliance_results.items():
                    if "license" in key.lower():
                        lines.append("### License Audit")
                        lines.append("")
                        models = data.get("models", {})
                        for model_key, model_data in models.items():
                            license_info = model_data.get("license_info", {})
                            lines.append(f"- **{license_info.get('model_name', model_key)}**: ")
                            lines.append(f"  {license_info.get('license_type', 'Unknown')}")
                        lines.append("")
        
        # Methodology
        lines.append("## Methodology")
        lines.append("")
        lines.append("### Hardware")
        lines.append("- Consumer-grade Apple Silicon laptop (16 GB unified memory class)")
        lines.append("- On-device inference via LM Studio")
        lines.append("")
        lines.append("### Configuration")
        lines.append("- Temperature: 0 (deterministic)")
        lines.append("- 20 runs per scenario with 3 warm-up runs")
        lines.append("- Machine plugged in, power saving disabled")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Report generated by Edge SLM Benchmark Framework*")
        
        return "\n".join(lines)
    
    def generate_html_report(self) -> str:
        """Génère un rapport au format HTML."""
        # Convertir le markdown en HTML simple
        md_content = self.generate_markdown_report()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge SLM Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a4a8a; padding-bottom: 0.5rem; }}
        h2 {{ color: #4a4a8a; margin-top: 2rem; }}
        h3 {{ color: #666; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4a4a8a;
            color: white;
        }}
        tr:hover {{ background: #f0f0f5; }}
        code {{
            background: #e8e8e8;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
        }}
        .summary-box {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
        ul {{ list-style-type: disc; padding-left: 2rem; }}
        li {{ margin: 0.5rem 0; }}
    </style>
</head>
<body>
    <article>
"""
        
        # Conversion simple MD -> HTML
        import re
        
        content = md_content
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        content = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>', content)
        content = content.replace('\n\n', '</p><p>')
        content = f'<p>{content}</p>'
        
        html += content
        html += """
    </article>
</body>
</html>
"""
        return html
    
    def generate_csv_exports(self):
        """Exporte les tableaux en CSV."""
        # Performance
        perf_df = self.generate_performance_table()
        if not perf_df.empty:
            perf_path = self.output_dir / "performance_results.csv"
            perf_df.to_csv(perf_path, index=False)
            print(f"[Saved] {perf_path}")
        
        # Capability
        cap_df = self.generate_capability_table()
        if not cap_df.empty:
            cap_path = self.output_dir / "capability_results.csv"
            cap_df.to_csv(cap_path, index=False)
            print(f"[Saved] {cap_path}")
    
    def save_report(self, format: str = "markdown") -> Path:
        """
        Sauvegarde le rapport.
        
        Args:
            format: 'markdown', 'html', ou 'all'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["markdown", "all"]:
            md_content = self.generate_markdown_report()
            md_path = self.output_dir / f"benchmark_report_{timestamp}.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"[Saved] Markdown report: {md_path}")
        
        if format in ["html", "all"]:
            html_content = self.generate_html_report()
            html_path = self.output_dir / f"benchmark_report_{timestamp}.html"
            with open(html_path, "w") as f:
                f.write(html_content)
            print(f"[Saved] HTML report: {html_path}")
        
        # Toujours exporter les CSV
        self.generate_csv_exports()
        
        return self.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report",
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Input results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "report",
        help="Output report directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "html", "all"],
        default="all",
        help="Report format",
    )
    parser.add_argument(
        "--compliance",
        action="store_true",
        help="Also generate compliance reports",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BENCHMARK REPORT GENERATOR")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print("=" * 60)
    
    # Générer le rapport
    generator = ReportGenerator(
        results_dir=args.input,
        output_dir=args.output,
    )
    
    generator.load_results()
    output_path = generator.save_report(format=args.format)
    
    # Générer les rapports de conformité si demandé
    if args.compliance:
        print("\n[Compliance] Generating compliance reports...")
        
        from src.compliance.risk_analysis import RiskAnalyzer
        from src.compliance.license_audit import LicenseAuditor
        
        compliance_dir = args.output / "compliance"
        compliance_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk Analysis
        risk_analyzer = RiskAnalyzer(output_dir=compliance_dir)
        risk_analyzer.save_report()
        risk_analyzer.print_summary()
        
        # License Audit
        license_auditor = LicenseAuditor(output_dir=compliance_dir)
        license_auditor.save_report()
        license_auditor.print_summary()
    
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Reports saved to: {output_path}")


if __name__ == "__main__":
    main()

