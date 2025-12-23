#!/usr/bin/env python3
"""
Generate Benchmark Report
=========================

Generate a consolidated benchmark results report.

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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


class ReportGenerator:
    """
    Report generator for SLM benchmarks.
    
    Consolidates performance, capability, and compliance results
    into a readable and exportable report.
    """
    
    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loaded data
        self.performance_results = []
        self.capability_results = []
        self.compliance_results = {}
    
    def load_results(self):
        """Load all results from the results directory."""
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
        """Generate the performance results table."""
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
        """Generate the capability results table."""
        if not self.capability_results:
            return pd.DataFrame()
        
        rows = []
        for result in self.capability_results:
            if result.get("type") == "realistic_scenarios":
                continue  # Handled separately
            
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
        """Generate a report in Markdown format."""
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
        """Generate a report in HTML format with real tables."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate HTML tables
        perf_df = self.generate_performance_table()
        cap_df = self.generate_capability_table()
        
        perf_table_html = perf_df.to_html(
            index=False, 
            classes='data-table',
            border=0,
            float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
        ) if not perf_df.empty else '<p class="no-data">No performance results available.</p>'
        
        cap_table_html = cap_df.to_html(
            index=False,
            classes='data-table',
            border=0,
            float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x
        ) if not cap_df.empty else '<p class="no-data">No capability results available.</p>'
        
        # Generate compliance section if available
        compliance_html = ""
        if self.compliance_results:
            compliance_html = '<section class="section"><h2>Compliance Analysis</h2>'
            
            for key, data in self.compliance_results.items():
                if "risk" in key.lower():
                    summary = data.get("summary", {})
                    compliance_html += '''
                    <div class="summary-box">
                        <h3>Risk Summary</h3>
                        <ul>
                            <li><strong>Total Risks:</strong> {}</li>
                            <li><strong>Total Controls:</strong> {}</li>
                        </ul>
                    </div>
                    '''.format(
                        summary.get('total_risks', 'N/A'),
                        summary.get('total_controls', 'N/A')
                    )
                    
                    if "key_findings" in summary:
                        compliance_html += '<div class="summary-box"><h3>Key Findings</h3><ul>'
                        for finding in summary["key_findings"]:
                            compliance_html += f'<li>{finding}</li>'
                        compliance_html += '</ul></div>'
                
                elif "license" in key.lower():
                    compliance_html += '<div class="summary-box"><h3>License Audit</h3><ul>'
                    models = data.get("models", {})
                    for model_key, model_data in models.items():
                        license_info = model_data.get("license_info", {})
                        compliance_html += '<li><strong>{}:</strong> {}</li>'.format(
                            license_info.get('model_name', model_key),
                            license_info.get('license_type', 'Unknown')
                        )
                    compliance_html += '</ul></div>'
            
            compliance_html += '</section>'
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge SLM Benchmark Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Source Serif 4', 'Times New Roman', Times, serif;
            background: #ffffff;
            color: #000000;
            line-height: 1.5;
            font-size: 11pt;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 3rem;
        }}
        
        /* Header */
        header {{
            text-align: center;
            padding: 2rem 0 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #000;
        }}
        
        h1 {{
            font-size: 18pt;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #000;
        }}
        
        .timestamp {{
            color: #444;
            font-size: 10pt;
            font-style: italic;
        }}
        
        /* Sections */
        .section {{
            margin-bottom: 2rem;
        }}
        
        h2 {{
            font-size: 14pt;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid #000;
            color: #000;
        }}
        
        h3 {{
            font-size: 12pt;
            font-weight: 600;
            margin: 1.25rem 0 0.75rem;
            color: #000;
        }}
        
        p {{
            color: #000;
            margin-bottom: 0.75rem;
            text-align: justify;
        }}
        
        /* Tables */
        .table-wrapper {{
            overflow-x: auto;
            margin: 1rem 0;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10pt;
            margin: 1rem 0;
        }}
        
        .data-table th {{
            background: #fff;
            color: #000;
            font-weight: 700;
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-top: 2px solid #000;
            border-bottom: 1px solid #000;
        }}
        
        .data-table td {{
            padding: 0.4rem 0.75rem;
            border-bottom: 1px solid #ccc;
            color: #000;
            font-size: 9pt;
        }}
        
        .data-table tr:last-child td {{
            border-bottom: 2px solid #000;
        }}
        
        /* Summary Box */
        .summary-box {{
            margin: 1rem 0;
            padding-left: 1rem;
        }}
        
        .summary-box h3 {{
            margin-top: 0;
            font-size: 11pt;
        }}
        
        /* Lists */
        ul {{
            list-style: disc;
            padding-left: 1.5rem;
            margin: 0.5rem 0;
        }}
        
        li {{
            padding: 0.15rem 0;
            color: #000;
        }}
        
        li strong {{
            font-weight: 600;
        }}
        
        /* No data message */
        .no-data {{
            text-align: center;
            padding: 1rem;
            color: #666;
            font-style: italic;
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            padding: 1.5rem;
            color: #666;
            font-size: 9pt;
            border-top: 1px solid #000;
            margin-top: 2rem;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Edge SLM Benchmark Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>
    </header>
    
    <main class="container">
        <section class="section">
            <h2>Executive Summary</h2>
            <p>
                This report presents the benchmark results for Small Language Models (SLMs) 
                evaluated on Apple Silicon hardware in a simulated banking context.
            </p>
        </section>
        
        <section class="section">
            <h2>Performance Results</h2>
            <div class="table-wrapper">
                {perf_table_html}
            </div>
        </section>
        
        <section class="section">
            <h2>Capability Results</h2>
            <div class="table-wrapper">
                {cap_table_html}
            </div>
        </section>
        
        {compliance_html}
        
        <section class="section">
            <h2>Methodology</h2>
            
            <h3>Hardware</h3>
            <ul>
                <li>Consumer-grade Apple Silicon laptop (16 GB unified memory class)</li>
                <li>On-device inference via LM Studio</li>
            </ul>
            
            <h3>Configuration</h3>
            <ul>
                <li><strong>Temperature:</strong> 0 (deterministic)</li>
                <li><strong>Runs:</strong> 20 per scenario with 3 warm-up runs</li>
                <li><strong>Power:</strong> Machine plugged in, power saving disabled</li>
            </ul>
        </section>
    </main>
    
    <footer>
        <p>Report generated by Edge SLM Benchmark Framework</p>
    </footer>
</body>
</html>'''
        return html
    
    def generate_csv_exports(self):
        """Export tables to CSV."""
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
        Save the report.
        
        Args:
            format: 'markdown', 'html', or 'all'
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
        
        # Always export CSV files
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
    
    # Generate report
    generator = ReportGenerator(
        results_dir=args.input,
        output_dir=args.output,
    )
    
    generator.load_results()
    output_path = generator.save_report(format=args.format)
    
    # Generate compliance reports if requested
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


