"""Report export utilities for forensic autopsy dossiers (JSON and Markdown)."""

from pathlib import Path
from typing import Union
from engine.models import AutopsyReport
from engine.utils import format_bytes


class ReportExporter:
    """Exports structured forensic autopsy findings to standardized formats."""

    @staticmethod
    def to_json(report: AutopsyReport, indent: int = 2) -> str:
        """Export report to formatted JSON string."""
        return report.to_json(indent=indent)

    @staticmethod
    def to_markdown(report: AutopsyReport) -> str:
        """Render report as a forensic investigation markdown dossier."""
        meta = report.metadata
        prof = report.dataset_profile

        lines = [
            f"# DATA AUTOPSY: Forensic Examination Dossier",
            f"",
            f"**File Examined:** `{meta.filename}`  ",
            f"**SHA256 Digest:** `{meta.sha256_hash}`  ",
            f"**File Size:** {format_bytes(meta.file_size_bytes)}  ",
            f"**Encoding:** {meta.encoding} | **Delimiter:** `{meta.delimiter}`  ",
            f"**Dimensions:** {meta.row_count:,} rows × {meta.column_count:,} columns  ",
            f"**Investigation Duration:** {report.execution_duration_ms:.2f} ms  ",
            f"**Timestamp:** {report.completed_at}  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Findings",
            f"",
        ]

        if not report.findings:
            lines.append("No critical forensic anomalies detected in baseline examination.\n")
        else:
            lines.append(f"Total findings identified: **{len(report.findings)}** (Ranked deterministically by severity and empirical support)\n")
            lines.append("| ID | Severity | Rank Score | Empirical Conf. | Category | Finding |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for f in report.findings:
                lines.append(
                    f"| `{f.id}` | **{f.severity.value}** | `{f.rank_score:.1f}` | `{f.confidence_score:.2f}` | {f.category.value} | {f.title} |"
                )
            lines.append("")

            lines.append("### Detailed Evidence Logs\n")
            for f in report.findings:
                lines.append(f"#### [{f.severity.value}] {f.title} (`{f.rule_id}`)")
                if f.check_name:
                    lines.append(f"- **Check Category:** `{f.check_name}`")
                lines.append(f"- **Deterministic Rank Score:** `{f.rank_score:.1f}` | **Empirical Support:** `{f.confidence_score:.2f}` (sample-calibrated support, not uncalibrated probability)")
                lines.append(f"- **Summary:** {f.summary}")
                lines.append(f"- **Evidence:** {f.evidence.description}")
                if f.evidence.column_names:
                    lines.append(f"- **Affected Columns:** {', '.join(f.evidence.column_names)}")
                if f.evidence.row_indices:
                    lines.append(f"- **Flagged Row Indices:** {f.evidence.row_indices[:20]}")
                if f.evidence.sample_values:
                    lines.append(f"- **Evidence Samples (Observed Data):**")
                    for s in f.evidence.sample_values[:5]:
                        lines.append(f"  - Row `{s.get('row_index')}`: `{s}`")
                if f.evidence.metrics:
                    lines.append(f"- **Measured Values:** `{f.evidence.metrics}`")
                if f.evidence.thresholds:
                    lines.append(f"- **Detection Thresholds:** `{f.evidence.thresholds}`")
                if f.why_it_matters:
                    lines.append(f"- **Why It Matters:** {f.why_it_matters}")
                if f.recommendation:
                    lines.append(f"- **Remediation:** {f.recommendation}")
                if f.limitations:
                    lines.append(f"- **Limitations / False Positives:** {f.limitations}")
                if f.ai_explanation:
                    lines.append(f"- **AI Forensic Note:** {f.ai_explanation}")
                lines.append("")

        if report.baseline_comparison and report.baseline_comparison.get("status") == "success":
            base = report.baseline_comparison
            lines.extend([
                f"---",
                f"",
                f"## 1.1. Empirical Pipeline Baseline Comparison (Target Leakage Impact)",
                f"",
                f"**Task Type:** `{base.get('task_type')}` | **Evaluated Metric:** **{base.get('primary_metric')}**  ",
                f"- **Score WITH suspicious features ({', '.join(base.get('suspicious_features', []))}):** `{base.get('score_with_suspicious')}`",
                f"- **Score WITHOUT suspicious features:** `{base.get('score_without_suspicious')}`",
                f"- **Performance Shift (Delta):** `{base.get('delta'):+.4f}`",
                f"- **Train / Test Sample Counts:** {base.get('train_rows_count')} train rows / {base.get('test_rows_count')} test rows",
                f"- **Empirical Interpretation:** {base.get('interpretation')}",
                f"- **Alternative Explanations / False Positive Note:** {base.get('alternative_explanations')}",
                f"",
            ])

        if report.limitations:
            lines.extend([
                f"---",
                f"",
                f"## 1.2. Forensic Analysis Limitations & Scope Boundaries",
                f"",
                f"Every statistical detector operates under bounded assumptions. The following catalog defines what this investigation establishes and what requires domain verification:",
                f"",
            ])
            for check_key, check_lim in report.limitations.items():
                title = check_lim.get("title", check_key)
                lines.append(f"### {title}")
                lines.append(f"- **Can Establish:** {check_lim.get('can_establish', 'N/A')}")
                lines.append(f"- **Cannot Establish:** {check_lim.get('cannot_establish', 'N/A')}")
                lines.append(f"- **False Positive Risks:** {check_lim.get('false_positive_risks', 'N/A')}")
                lines.append("")

        lines.extend([
            f"---",
            f"",
            f"## 2. Dataset Profiling Summary",
            f"",
            f"- **Total Rows:** {prof.metadata.row_count:,}",
            f"- **Total Columns:** {prof.metadata.column_count:,}",
            f"- **Total Cells:** {prof.total_cells:,}",
            f"- **Missing Cells:** {prof.total_null_cells:,} ({prof.null_cells_percentage:.2f}%)",
            f"- **Duplicate Rows:** {prof.duplicate_rows_count:,} ({prof.duplicate_rows_percentage:.2f}%)",
            f"- **Memory Estimate:** {format_bytes(prof.total_memory_bytes)}",
            f"",
            f"### Column Profiles",
            f"",
            f"| Column | Original Type | Inferred Type | Nulls (%) | Unique Count | Memory |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for col_name, c in prof.columns.items():
            lines.append(
                f"| `{col_name}` | {c.original_dtype} | {c.inferred_type} | "
                f"{c.null_count} ({c.null_percentage}%) | {c.unique_count:,} | {format_bytes(c.memory_bytes)} |"
            )

        if report.analysis_metadata:
            meta_info = report.analysis_metadata
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 3. Analysis Metadata & Reproducibility Context",
                f"",
                f"- **Data Autopsy Version:** `{meta_info.app_version}`",
                f"- **Target Column:** `{meta_info.selected_target or 'None (Unsupervised Mode)'}`",
                f"- **Excluded Columns (Non-mutating):** `{', '.join(meta_info.excluded_columns) if meta_info.excluded_columns else 'None'}`",
                f"- **Checks Executed:** `{', '.join(meta_info.checks_executed)}`",
                f"- **Reproducibility Settings:** `{meta_info.reproducibility_settings}`",
                f"- **Execution Window:** `{meta_info.started_at}` to `{meta_info.completed_at}` ({meta_info.duration_ms:.2f} ms)",
            ])

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def save_to_file(cls, report: AutopsyReport, target_path: Union[str, Path], format_type: str = "json") -> Path:
        """Write autopsy report to target file path."""
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "json":
            content = cls.to_json(report)
        elif format_type.lower() in ["md", "markdown"]:
            content = cls.to_markdown(report)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        p.write_text(content, encoding="utf-8")
        return p
