"""Data Autopsy: Tabular Forensic Analysis Workstation (Streamlit UI).

A serious open-source forensic investigation tool for tabular datasets.
Reuses the existing tested engine for profiling, anomaly detection, leakage checks,
and evidence verification without duplicating detection logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai.explainer import ForensicExplainer
from config import config
from engine.loader import (
    EmptyFileError,
    EncodingDetectionError,
    FileTooLargeError,
    MalformedCSVError,
    load_dataset,
)
from engine.models import AutopsyReport, Finding, Severity
from engine.profiler import generate_autopsy_report
from engine.utils import format_bytes
from engine.validator import validate_finding_evidence, validate_report_evidence
from reports.exporter import ReportExporter

# Configure page layout and forensic workstation theme
st.set_page_config(
    page_title="DATA AUTOPSY // Forensic Dataset Workstation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Forensic workstation aesthetic styling (restrained near-black, warm-white, subtle amber/red accents)
FORENSIC_THEME_CSS = """
<style>
    /* Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e5e5;
    }

    .stApp {
        background-color: #0c0d0e;
    }

    /* Monospace headers and metadata */
    .mono-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        color: #a3a3a3;
    }

    .forensic-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #f5f5f5;
        border-bottom: 1px solid #262626;
        padding-bottom: 0.6rem;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
    }

    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: #f5f5f5;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #f59e0b;
        padding-left: 0.6rem;
    }

    /* Severity badges */
    .badge-critical {
        background-color: #3e1414;
        color: #ff6b6b;
        border: 1px solid #7a2222;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 2px;
        font-weight: 600;
    }

    .badge-high {
        background-color: #38240f;
        color: #f59e0b;
        border: 1px solid #784814;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 2px;
        font-weight: 600;
    }

    .badge-medium {
        background-color: #2b2512;
        color: #fbbf24;
        border: 1px solid #61501b;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 2px;
        font-weight: 500;
    }

    .badge-low {
        background-color: #1a221b;
        color: #6ee7b7;
        border: 1px solid #234832;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 2px;
    }

    .badge-info {
        background-color: #161e2e;
        color: #93c5fd;
        border: 1px solid #233554;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 2px;
    }

    /* Metric card surfaces */
    div[data-testid="stMetric"] {
        background-color: #121315;
        border: 1px solid #262626;
        padding: 0.8rem 1rem;
        border-radius: 3px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem !important;
        color: #f5f5f5 !important;
    }

    /* Forensic Dossier Box */
    .dossier-box {
        background-color: #121315;
        border: 1px solid #262626;
        border-left: 3px solid #f59e0b;
        padding: 1rem;
        margin-bottom: 0.9rem;
        border-radius: 2px;
    }

    .dossier-box-critical {
        background-color: #151112;
        border: 1px solid #3e1b1b;
        border-left: 3px solid #ef4444;
        padding: 1rem;
        margin-bottom: 0.9rem;
        border-radius: 2px;
    }

    .clean-box {
        background-color: #101612;
        border: 1px solid #1a3322;
        border-left: 3px solid #10b981;
        padding: 1rem;
        margin-bottom: 0.9rem;
        border-radius: 2px;
    }
</style>
"""
st.markdown(FORENSIC_THEME_CSS, unsafe_allow_html=True)


def get_badge_html(severity_str: str) -> str:
    """Format severity string as a restrained forensic status badge."""
    sev = severity_str.upper()
    css_class = {
        "CRITICAL": "badge-critical",
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "LOW": "badge-low",
        "INFO": "badge-info",
    }.get(sev, "badge-info")
    return f'<span class="{css_class}">{sev}</span>'


def reset_session() -> None:
    """Clear all cached investigation state and rerun application."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ==============================================================================
# SIDEBAR: Ingestion, Configuration, Target & Session Management
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="forensic-title">DATA AUTOPSY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mono-meta">TABULAR FORENSIC WORKSTATION<br>DETERMINISTIC VERIFICATION SUITE</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("1. Ingestion Source")
    uploaded_file = st.file_uploader(
        "Upload Tabular File (CSV / TSV)",
        type=["csv", "tsv", "txt"],
        help=f"Strict parsing without data modification. Limit: {config.max_file_size_mb} MB.",
    )

    st.markdown("<div class='mono-meta' style='text-align:center;'>— OR TEST FIXTURE —</div>", unsafe_allow_html=True)
    sample_dir = Path(__file__).resolve().parent / "sample_data"
    available_fixtures = sorted([p.name for p in sample_dir.glob("*.csv")]) if sample_dir.exists() else []
    query_fixture = st.query_params.get("fixture", "")
    all_fixtures = ["(Select fixture...)"] + available_fixtures
    fixture_idx = all_fixtures.index(query_fixture) if query_fixture in all_fixtures else 0
    sample_choice = st.selectbox(
        "Select Verification Fixture",
        options=all_fixtures,
        index=fixture_idx,
    )

    st.divider()

    # Local AI Engine Status Card
    explainer = ForensicExplainer(config)
    ai_status = explainer.check_ollama_status()
    st.subheader("2. Local AI Engine")
    if ai_status["status"] == "ready":
        st.markdown(f'<div class="mono-meta" style="color:#6ee7b7;">● Ollama Online ({explainer.model})</div>', unsafe_allow_html=True)
    elif ai_status["status"] == "model_missing":
        st.markdown(f'<div class="mono-meta" style="color:#f59e0b;">▲ Model Missing ({explainer.model})</div>', unsafe_allow_html=True)
        with st.expander("Setup Instructions"):
            st.code(ai_status["setup_instructions"], language="bash")
    else:
        st.markdown('<div class="mono-meta" style="color:#a3a3a3;">○ Ollama Offline (Deterministic Mode)</div>', unsafe_allow_html=True)
        with st.expander("Enable Local AI (Qwen)"):
            st.code(ai_status["setup_instructions"], language="bash")

    st.divider()

    # Session Management
    if st.button("↻ Reset Investigation Session", use_container_width=True, help="Clear memory and reset all findings"):
        reset_session()

    st.markdown(
        f'<div class="mono-meta" style="margin-top:10px;">'
        f'Max Ingest: {config.max_file_size_mb} MB<br>'
        f'Encodings: {len(config.supported_encodings)} types<br>'
        f'Model Guard: Lightweight only</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN WORKSTATION PANEL
# ==============================================================================
st.markdown('<div class="forensic-title">FORENSIC INVESTIGATION WORKSTATION</div>', unsafe_allow_html=True)

# Resolve target dataset input
target_source = None
target_name = None

if uploaded_file is not None:
    target_source = uploaded_file
    target_name = uploaded_file.name
elif sample_choice and sample_choice != "(Select fixture...)":
    sample_file_path = Path(__file__).resolve().parent / "sample_data" / sample_choice
    if sample_file_path.exists():
        target_source = sample_file_path
        target_name = sample_choice

if target_source is None:
    st.info(
        "No dataset currently mounted. Upload an untrusted CSV through the left sidebar "
        "or select a verification test fixture to initiate forensic analysis."
    )
else:
    try:
        # Load and validate dataset through tested engine loader
        df, meta = load_dataset(target_source, filename=target_name)

        # Target Column Selection & Column Exclusion in Sidebar
        DEFAULT_TARGET_MAP = {
            "demo_1_clean_classification.csv": "credit_approved",
            "demo_2_missing_and_duplicates.csv": "churned",
            "demo_3_imbalanced_classification.csv": "is_fraud",
            "demo_4_target_copy_leakage.csv": "readmitted_30d",
            "demo_5_statistical_outliers.csv": "server_failure",
            "demo_6_legitimate_strong_predictors.csv": "monthly_loan_payment",
            "corrupt_clinical_trials.csv": "response_rate",
            "clean_patient_records.csv": "readmitted",
        }

        default_target = DEFAULT_TARGET_MAP.get(target_name, None)
        target_options = ["(None / Unsupervised)"] + list(df.columns)
        query_target = st.query_params.get("target")
        chosen_target = query_target if query_target in target_options else default_target
        default_idx = target_options.index(chosen_target) if chosen_target and chosen_target in target_options else 0

        with st.sidebar:
            st.divider()
            st.subheader("3. Forensic Target")
            target_selection = st.selectbox(
                "Prediction Target Column (Optional)",
                options=target_options,
                index=default_idx,
                help="Specifying a target activates label leakage detection and class imbalance audits.",
            )
            resolved_target = None if target_selection == "(None / Unsupervised)" else target_selection

            st.divider()
            st.subheader("4. Column Exclusion")
            available_to_exclude = [c for c in df.columns if c != resolved_target]
            excluded_selection = st.multiselect(
                "Exclude Columns (Non-mutating)",
                options=available_to_exclude,
                default=[],
                help="Excludes columns from subsequent analysis without modifying the source dataset.",
            )

        # Caching: generate or retrieve autopsy report
        cache_key = f"{meta.sha256_hash}_{resolved_target}_{sorted(excluded_selection)}"
        if st.session_state.get("cached_report_key") != cache_key:
            with st.spinner("Executing forensic engine across 10 deterministic checks..."):
                report = generate_autopsy_report(
                    df,
                    meta,
                    target_column=resolved_target,
                    excluded_columns=excluded_selection,
                )
                st.session_state["cached_report"] = report
                st.session_state["cached_report_key"] = cache_key
        else:
            report = st.session_state["cached_report"]

        # ======================================================================
        # 7 CORE FORENSIC WORKSTATION SECTIONS
        # ======================================================================
        tab_overview, tab_profile, tab_findings, tab_leakage, tab_evidence, tab_ai, tab_export = st.tabs([
            "1. Overview",
            "2. Dataset Profile",
            f"3. Forensic Findings ({len(report.findings)})",
            "4. Leakage Investigation",
            "5. Evidence Inspector",
            "6. AI Investigator",
            "7. Report Export",
        ])

        # ----------------------------------------------------------------------
        # SECTION 1: OVERVIEW
        # ----------------------------------------------------------------------
        with tab_overview:
            st.markdown('<div class="section-header">DATASET IDENTITY & FORENSIC STATUS</div>', unsafe_allow_html=True)

            # Metadata Identity Cards
            id1, id2, id3, id4, id5 = st.columns(5)
            id1.metric("Dimensions", f"{meta.row_count:,} × {meta.column_count:,}")
            id2.metric("Payload Size", format_bytes(meta.file_size_bytes))
            id3.metric("Memory Est.", format_bytes(report.dataset_profile.total_memory_bytes))
            id4.metric("Autopsy Runtime", f"{report.execution_duration_ms:.1f} ms")
            id5.metric("Total Findings", f"{len(report.findings):,}")

            st.markdown(
                f'<div class="mono-meta" style="margin-top: 0.6rem; margin-bottom: 1.2rem; padding: 0.6rem; background-color: #121315; border: 1px solid #262626;">'
                f'<strong>FILE:</strong> <code>{meta.filename}</code> | '
                f'<strong>SHA256:</strong> <code>{meta.sha256_hash}</code> | '
                f'<strong>ENCODING:</strong> {meta.encoding} | <strong>DELIMITER:</strong> <code>{meta.delimiter}</code> | '
                f'<strong>TARGET:</strong> <code>{resolved_target or "None"}</code>'
                + (f' | <strong>EXCLUDED:</strong> <code>{", ".join(excluded_selection)}</code>' if excluded_selection else '')
                + f'</div>',
                unsafe_allow_html=True,
            )

            # Finding Counts by Severity
            sev_counts = {
                Severity.CRITICAL: sum(1 for f in report.findings if f.severity == Severity.CRITICAL),
                Severity.HIGH: sum(1 for f in report.findings if f.severity == Severity.HIGH),
                Severity.MEDIUM: sum(1 for f in report.findings if f.severity == Severity.MEDIUM),
                Severity.LOW: sum(1 for f in report.findings if f.severity == Severity.LOW),
                Severity.INFO: sum(1 for f in report.findings if f.severity == Severity.INFO),
            }

            st.markdown('<div class="section-header">FINDINGS BY SEVERITY TIER</div>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("CRITICAL", f"{sev_counts[Severity.CRITICAL]}", delta=None, help="Fatal integrity breaches or direct target leakage")
            sc2.metric("HIGH", f"{sev_counts[Severity.HIGH]}", delta=None, help="Severe flaws requiring immediate pipeline intervention")
            sc3.metric("MEDIUM", f"{sev_counts[Severity.MEDIUM]}", delta=None, help="Notable anomalies causing model drift or bias")
            sc4.metric("LOW", f"{sev_counts[Severity.LOW]}", delta=None, help="Informative statistical deviations")
            sc5.metric("INFO", f"{sev_counts[Severity.INFO]}", delta=None, help="Structural properties (e.g. potential identifiers)")

            # Evidence-Based Executive Summary
            st.markdown('<div class="section-header">EVIDENCE-BASED FORENSIC SYNOPSIS</div>', unsafe_allow_html=True)
            if not report.findings:
                st.markdown(
                    '<div class="clean-box">'
                    '<strong>CLEAN BASELINE:</strong> No critical anomalies or integrity violations detected under active thresholds. '
                    'All features conform to statistical and structural integrity standards.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                critical_and_high = [f for f in report.findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
                box_cls = "dossier-box-critical" if sev_counts[Severity.CRITICAL] > 0 else "dossier-box"

                synopsis_points = []
                for f in report.findings[:3]:
                    synopsis_points.append(f"• <strong>[{f.severity.value}] {f.title}:</strong> {f.summary}")

                st.markdown(
                    f'<div class="{box_cls}">'
                    f'<strong>AUDIT RESULT:</strong> Forensic analysis identified <strong>{len(report.findings)}</strong> anomalies '
                    f'({sev_counts[Severity.CRITICAL]} Critical, {sev_counts[Severity.HIGH]} High). '
                    f'Calculations were deterministically executed across 10 engine checks in {report.execution_duration_ms:.1f}ms.<br><br>'
                    + "<br>".join(synopsis_points)
                    + f'</div>',
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------------------
        # SECTION 2: DATASET PROFILE
        # ----------------------------------------------------------------------
        with tab_profile:
            prof = report.dataset_profile
            st.markdown('<div class="section-header">STRUCTURAL & STATISTICAL PROFILING</div>', unsafe_allow_html=True)

            pr1, pr2, pr3, pr4 = st.columns(4)
            pr1.metric("Total Cells", f"{prof.total_cells:,}")
            pr2.metric("Missing Cells", f"{prof.total_null_cells:,} ({prof.null_cells_percentage:.2f}%)")
            pr3.metric("Duplicate Rows", f"{prof.duplicate_rows_count:,} ({prof.duplicate_rows_percentage:.2f}%)")
            pr4.metric("Active Columns", f"{len(prof.columns)}")

            # Visualizations Row: Missingness and Inferred Types
            c_chart1, c_chart2 = st.columns([3, 2])

            with c_chart1:
                # Plotly Missingness Horizontal Bar Chart
                missing_series = df.isna().mean() * 100
                missing_cols = missing_series[missing_series > 0].sort_values(ascending=True)

                if not missing_cols.empty:
                    bar_colors = [
                        "#ef4444" if p >= 50 else "#f59e0b" if p >= 20 else "#3b82f6"
                        for p in missing_cols.values
                    ]
                    fig_miss = go.Figure(
                        go.Bar(
                            x=missing_cols.values,
                            y=missing_cols.index,
                            orientation="h",
                            marker=dict(color=bar_colors),
                            text=[f"{p:.1f}%" for p in missing_cols.values],
                            textposition="auto",
                        )
                    )
                    fig_miss.update_layout(
                        title=dict(text="Missingness Rates by Feature (%)", font=dict(family="JetBrains Mono, monospace", size=12, color="#f5f5f5")),
                        paper_bgcolor="#121315",
                        plot_bgcolor="#121315",
                        font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
                        margin=dict(l=20, r=20, t=35, b=25),
                        height=max(220, len(missing_cols) * 26),
                        xaxis=dict(title="Missing %", range=[0, 105], gridcolor="#222428"),
                        yaxis=dict(gridcolor="#222428"),
                    )
                    st.plotly_chart(fig_miss, use_container_width=True)
                else:
                    st.info("Zero missing cells detected across active features.")

            with c_chart2:
                # Plotly Inferred Types Donut Chart
                type_counts = prof.column_type_counts
                if type_counts:
                    fig_types = go.Figure(
                        go.Pie(
                            labels=list(type_counts.keys()),
                            values=list(type_counts.values()),
                            hole=0.55,
                            marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]),
                            textinfo="label+value",
                        )
                    )
                    fig_types.update_layout(
                        title=dict(text="Inferred Column Types", font=dict(family="JetBrains Mono, monospace", size=12, color="#f5f5f5")),
                        paper_bgcolor="#121315",
                        plot_bgcolor="#121315",
                        font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
                        margin=dict(l=20, r=20, t=35, b=25),
                        height=260,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_types, use_container_width=True)

            # Interactive Distribution Explorer
            st.markdown('<div class="section-header">FEATURE DISTRIBUTION EXPLORER</div>', unsafe_allow_html=True)
            selected_feat = st.selectbox("Select Feature to Inspect Distribution", options=list(prof.columns.keys()), index=0)

            if selected_feat:
                c_prof = prof.columns[selected_feat]
                col_data = df[selected_feat].dropna()

                # Bounded sample for large datasets
                if len(col_data) > 50000:
                    col_sample = col_data.sample(50000, random_state=42)
                else:
                    col_sample = col_data

                d1, d2 = st.columns([3, 1])

                with d1:
                    if c_prof.inferred_type in ("numeric", "boolean"):
                        fig_dist = go.Figure(
                            go.Histogram(
                                x=col_sample,
                                nbinsx=35,
                                marker=dict(color="#3b82f6", line=dict(color="#1e3a8a", width=1)),
                            )
                        )
                        fig_dist.update_layout(
                            title=dict(text=f"Numeric Distribution: '{selected_feat}'", font=dict(family="JetBrains Mono, monospace", size=12, color="#f5f5f5")),
                            paper_bgcolor="#121315",
                            plot_bgcolor="#121315",
                            font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
                            margin=dict(l=20, r=20, t=35, b=25),
                            height=280,
                            xaxis=dict(gridcolor="#222428"),
                            yaxis=dict(gridcolor="#222428"),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        top_cats = col_sample.astype(str).value_counts().head(12)
                        fig_cat = go.Figure(
                            go.Bar(
                                x=top_cats.values,
                                y=top_cats.index,
                                orientation="h",
                                marker=dict(color="#10b981"),
                                text=top_cats.values,
                                textposition="auto",
                            )
                        )
                        fig_cat.update_layout(
                            title=dict(text=f"Top Categories: '{selected_feat}'", font=dict(family="JetBrains Mono, monospace", size=12, color="#f5f5f5")),
                            paper_bgcolor="#121315",
                            plot_bgcolor="#121315",
                            font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
                            margin=dict(l=20, r=20, t=35, b=25),
                            height=280,
                            xaxis=dict(gridcolor="#222428"),
                            yaxis=dict(gridcolor="#222428"),
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)

                with d2:
                    st.markdown(
                        f'<div class="dossier-box" style="font-size:0.85rem;">'
                        f'<strong>FEATURE STATS</strong><br>'
                        f'Type: <code>{c_prof.inferred_type}</code><br>'
                        f'Unique Count: <code>{c_prof.unique_count:,}</code><br>'
                        f'Null Count: <code>{c_prof.null_count:,} ({c_prof.null_percentage}%)</code><br>'
                        f'Zeros Count: <code>{c_prof.zeros_count:,} ({c_prof.zeros_percentage}%)</code><br>'
                        + (f'Mean: <code>{c_prof.mean:.4f}</code><br>' if c_prof.mean is not None else '')
                        + (f'Median: <code>{c_prof.median:.4f}</code><br>' if c_prof.median is not None else '')
                        + (f'Std Dev: <code>{c_prof.std_dev:.4f}</code><br>' if c_prof.std_dev is not None else '')
                        + (f'Mode: <code>{c_prof.mode}</code><br>' if c_prof.mode is not None else '')
                        + f'Memory: <code>{format_bytes(c_prof.memory_bytes)}</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Target Class Balance (if discrete target selected)
            if resolved_target and resolved_target in df.columns:
                target_valid = df[resolved_target].dropna()
                if 2 <= target_valid.nunique() <= 15:
                    st.markdown('<div class="section-header">TARGET CLASS BALANCE AUDIT</div>', unsafe_allow_html=True)
                    t_counts = target_valid.astype(str).value_counts()
                    fig_target = go.Figure(
                        go.Bar(
                            x=t_counts.index,
                            y=t_counts.values,
                            marker=dict(color="#f59e0b"),
                            text=[f"{(c / len(target_valid) * 100):.1f}%" for c in t_counts.values],
                            textposition="auto",
                        )
                    )
                    fig_target.update_layout(
                        title=dict(text=f"Target Class Proportions: '{resolved_target}'", font=dict(family="JetBrains Mono, monospace", size=12, color="#f5f5f5")),
                        paper_bgcolor="#121315",
                        plot_bgcolor="#121315",
                        font=dict(family="Inter, sans-serif", color="#a3a3a3", size=11),
                        margin=dict(l=20, r=20, t=35, b=25),
                        height=240,
                        xaxis=dict(gridcolor="#222428"),
                        yaxis=dict(gridcolor="#222428"),
                    )
                    st.plotly_chart(fig_target, use_container_width=True)

            # Detailed Column Profiling Table
            st.markdown('<div class="section-header">DETAILED COLUMN PROFILE METRICS</div>', unsafe_allow_html=True)
            prof_table = []
            for col_name, c in prof.columns.items():
                prof_table.append({
                    "Column": col_name,
                    "Original Dtype": c.original_dtype,
                    "Inferred Type": c.inferred_type,
                    "Nulls (%)": f"{c.null_count} ({c.null_percentage}%)",
                    "Unique Count": c.unique_count,
                    "Zeros (%)": f"{c.zeros_count} ({c.zeros_percentage}%)",
                    "Mode": str(c.mode) if c.mode is not None else "N/A",
                    "Memory": format_bytes(c.memory_bytes),
                    "Sample Values": str(c.sample_values[:3]),
                })
            st.dataframe(pd.DataFrame(prof_table), use_container_width=True)

        # ----------------------------------------------------------------------
        # SECTION 3: FORENSIC FINDINGS
        # ----------------------------------------------------------------------
        with tab_findings:
            st.markdown('<div class="section-header">SEARCH & FILTER FORENSIC FINDINGS</div>', unsafe_allow_html=True)

            # Filter Toolbar
            f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 3])
            with f_col1:
                severity_filter = st.multiselect(
                    "Severity Tiers",
                    options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
                    default=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
                )
            with f_col2:
                category_filter = st.multiselect(
                    "Check Categories",
                    options=list(set(f.category.value for f in report.findings)) if report.findings else [],
                    default=list(set(f.category.value for f in report.findings)) if report.findings else [],
                )
            with f_col3:
                sort_order = st.selectbox(
                    "Rank Order",
                    options=["Deterministic Rank Score", "Severity Hierarchy", "Empirical Confidence", "Rule ID"],
                    index=0,
                )
            with f_col4:
                search_query = st.text_input("Keyword Search", placeholder="Filter by feature, rule ID, or keyword...")

            # Apply Filters
            filtered_findings = []
            for f in report.findings:
                if f.severity.value not in severity_filter:
                    continue
                if category_filter and f.category.value not in category_filter:
                    continue
                if search_query:
                    query = search_query.lower()
                    match_found = (
                        query in f.title.lower()
                        or query in f.summary.lower()
                        or query in f.rule_id.lower()
                        or any(query in c.lower() for c in (f.evidence.column_names or []))
                    )
                    if not match_found:
                        continue
                filtered_findings.append(f)

            # Sort
            if sort_order == "Severity Hierarchy":
                sev_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 0}
                filtered_findings.sort(key=lambda x: -sev_order.get(x.severity, 0))
            elif sort_order == "Empirical Confidence":
                filtered_findings.sort(key=lambda x: -x.confidence_score)
            elif sort_order == "Rule ID":
                filtered_findings.sort(key=lambda x: x.rule_id)
            else:
                filtered_findings.sort(key=lambda x: -x.rank_score)

            st.caption(f"Displaying {len(filtered_findings)} of {len(report.findings)} verified findings")

            if not filtered_findings:
                st.info("No findings match the selected filter criteria.")
            else:
                for f in filtered_findings:
                    badge = get_badge_html(f.severity.value)
                    check_badge = f'<span class="mono-meta" style="color:#d4d4d4; margin-left:8px;">[{f.check_name}]</span>' if f.check_name else ''
                    box_style = "dossier-box-critical" if f.severity == Severity.CRITICAL else "dossier-box"

                    st.markdown(
                        f'<div class="{box_style}">'
                        f'<div>{badge}{check_badge} <strong style="font-size:1.05rem; margin-left:8px;">{f.title}</strong> '
                        f'<span class="mono-meta" style="float:right;">{f.rule_id}</span></div>'
                        f'<div class="mono-meta" style="margin-top:4px; color:#f59e0b;">'
                        f'Deterministic Rank Score: <strong>{f.rank_score:.1f}</strong> | '
                        f'Empirical Support: <strong>{f.confidence_score:.2f}</strong> '
                        f'<span style="color:#9ca3af; font-size:0.75rem;">(sample-calibrated support; not uncalibrated probability)</span>'
                        f'</div>'
                        f'<p style="margin: 8px 0 4px 0; color: #d4d4d4;">{f.summary}</p>'
                        f'<div class="mono-meta" style="color: #a3a3a3; margin-bottom: 4px;"><strong>Evidence:</strong> {f.evidence.description}</div>'
                        + (f'<div class="mono-meta" style="color: #8b949e;"><strong>Measured:</strong> <code>{f.evidence.metrics}</code> | <strong>Threshold:</strong> <code>{f.evidence.thresholds}</code></div>' if f.evidence.metrics else '')
                        + (f'<div style="margin-top:6px; font-size:0.85rem; color:#e2e8f0;"><strong>Why It Matters:</strong> {f.why_it_matters}</div>' if f.why_it_matters else '')
                        + (f'<div style="margin-top:4px; font-size:0.85rem; color:#f59e0b;"><strong>Remediation:</strong> {f.recommendation}</div>' if f.recommendation else '')
                        + (f'<div style="margin-top:4px; font-size:0.82rem; color:#9ca3af;"><em>Limitations / False Positives:</em> {f.limitations}</div>' if f.limitations else '')
                        + f'</div>',
                        unsafe_allow_html=True,
                    )

        # ----------------------------------------------------------------------
        # SECTION 4: LEAKAGE INVESTIGATION
        # ----------------------------------------------------------------------
        with tab_leakage:
            st.markdown('<div class="section-header">TARGET DATA LEAKAGE AUDIT & ABLATION</div>', unsafe_allow_html=True)

            if not resolved_target:
                st.warning(
                    "No prediction target selected. Target leakage checks require an explicit target variable. "
                    "Select a target column in the left sidebar to activate leakage detection and pipeline baseline comparisons."
                )
            else:
                st.markdown(
                    f'<div class="mono-meta" style="padding: 0.6rem; background-color: #121315; border: 1px solid #262626; margin-bottom: 1rem;">'
                    f'Active Target Variable: <code>{resolved_target}</code> | Inferred Dtype: <code>{df[resolved_target].dtype}</code> | '
                    f'Distinct Values: <code>{df[resolved_target].nunique()}</code> | Valid Non-Null Rows: <code>{df[resolved_target].notna().sum():,}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Leakage Checks Status Matrix
                leakage_findings = [f for f in report.findings if f.rule_id.startswith("LEAK-")]
                st.markdown('<div class="section-header">LEAKAGE DETECTOR STATUS MATRIX</div>', unsafe_allow_html=True)

                detector_matrix = [
                    {"Rule ID": "LEAK-001", "Detector Check": "Exact Target Copies / Identity", "Status": "Flagged" if any(f.rule_id == "LEAK-001" for f in leakage_findings) else "Clear"},
                    {"Rule ID": "LEAK-005", "Detector Check": "Inverted Binary Target Proxy", "Status": "Flagged" if any(f.rule_id == "LEAK-005" for f in leakage_findings) else "Clear"},
                    {"Rule ID": "LEAK-006", "Detector Check": "Informative Missingness Mask", "Status": "Flagged" if any(f.rule_id == "LEAK-006" for f in leakage_findings) else "Clear"},
                    {"Rule ID": "LEAK-002", "Detector Check": "Extreme Linear Correlation (|r| >= 0.95)", "Status": "Flagged" if any(f.rule_id == "LEAK-002" for f in leakage_findings) else "Clear"},
                    {"Rule ID": "LEAK-002", "Detector Check": "Near-Perfect ROC-AUC Separation / ANOVA", "Status": "Flagged" if any("MULTI" in f.id or "Separates" in f.title for f in leakage_findings) else "Clear"},
                    {"Rule ID": "LEAK-004", "Detector Check": "Deterministic Target Encoding (100% Purity)", "Status": "Flagged" if any(f.rule_id == "LEAK-004" for f in leakage_findings) else "Clear"},
                ]
                st.dataframe(pd.DataFrame(detector_matrix), use_container_width=True)

                if leakage_findings:
                    st.markdown('<div class="section-header">FLAGGED LEAKAGE CANDIDATES</div>', unsafe_allow_html=True)
                    for lf in leakage_findings:
                        st.markdown(
                            f'<div class="dossier-box-critical">'
                            f'{get_badge_html(lf.severity.value)} <strong style="margin-left:8px;">{lf.title}</strong> '
                            f'<span class="mono-meta" style="float:right;">{lf.rule_id}</span><br>'
                            f'<p style="margin:6px 0; color:#d4d4d4;">{lf.summary}</p>'
                            f'<span style="font-size:0.85rem; color:#f59e0b;">Remediation: {lf.recommendation}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="clean-box">'
                        '<strong>CLEAN LEAKAGE AUDIT:</strong> No active features exhibit exact target duplication, '
                        'inverted binary proxies, or deterministic target encoding.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                # Empirical Pipeline Baseline Comparison
                st.markdown('<div class="section-header">LIGHTWEIGHT PIPELINE BASELINE COMPARISON</div>', unsafe_allow_html=True)
                st.markdown(
                    "<p class='mono-meta'>Measures empirical performance shift when suspected leakage features are ablated. "
                    "Preprocessors strictly fit only on training splits to prevent validation contamination.</p>",
                    unsafe_allow_html=True,
                )

                if report.baseline_comparison and report.baseline_comparison.get("status") == "success":
                    base = report.baseline_comparison
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Task Type", base.get("task_type", "N/A"))
                    bc2.metric(f"{base.get('primary_metric')} (With)", f"{base.get('score_with_suspicious'):.4f}")
                    bc3.metric(f"{base.get('primary_metric')} (Without)", f"{base.get('score_without_suspicious'):.4f}")
                    bc4.metric("Performance Delta", f"{base.get('delta'):+.4f}")

                    st.markdown(
                        f'<div class="dossier-box" style="margin-top:0.8rem;">'
                        f'<strong>Suspicious Feature(s) Ablated:</strong> <code>{", ".join(base.get("suspicious_features", []))}</code><br>'
                        f'<strong>Non-Suspicious Baseline Features:</strong> <code>{", ".join(base.get("non_suspicious_features", []))}</code><br>'
                        f'<strong>Cross-Validation Sample Split:</strong> {base.get("train_rows_count")} train rows / {base.get("test_rows_count")} test rows<br><br>'
                        f'<strong>Empirical Interpretation:</strong> {base.get("interpretation")}<br><br>'
                        f'<span style="color:#f59e0b;"><strong>Alternative Domain Explanations & False-Positive Notice:</strong></span> '
                        f'{base.get("alternative_explanations")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif report.baseline_comparison:
                    st.info(f"Baseline comparison skipped: {report.baseline_comparison.get('reason', 'Requirements not met.')}")

        # ----------------------------------------------------------------------
        # SECTION 5: EVIDENCE INSPECTOR
        # ----------------------------------------------------------------------
        with tab_evidence:
            st.markdown('<div class="section-header">RAW EVIDENCE & CRYPTOGRAPHIC VERIFICATION</div>', unsafe_allow_html=True)

            if not report.findings:
                st.info("No findings flagged to inspect evidence for.")
            else:
                finding_labels = [f"[{f.severity.value}] {f.title} ({f.rule_id})" for f in report.findings]
                selected_idx = st.selectbox("Select Finding to Inspect Evidence", options=range(len(report.findings)), format_func=lambda i: finding_labels[i])
                active_finding = report.findings[selected_idx]

                # Evidence Verification Check
                is_valid, discrepancies = validate_finding_evidence(df, active_finding)
                if is_valid:
                    st.success(f"Evidence Verification Passed: Finding '{active_finding.id}' matches source DataFrame verbatim.")
                else:
                    st.error(f"Evidence Verification Failed: {discrepancies}")

                # Metrics and Thresholds
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    st.markdown("**Measured Statistical Evidence:**")
                    st.json(active_finding.evidence.metrics)
                with ev_col2:
                    st.markdown("**Configured Engine Detection Thresholds:**")
                    st.json(active_finding.evidence.thresholds)

                # Observed Sample Values
                if active_finding.evidence.sample_values:
                    st.markdown("**Observed Evidence Samples (Original Row References & Verbatim Values):**")
                    st.dataframe(pd.DataFrame(active_finding.evidence.sample_values), use_container_width=True)

                # Flagged Row Slices in Raw DataFrame
                if active_finding.evidence.row_indices:
                    st.markdown(f"**Source DataFrame Row Slice (First 20 of {len(active_finding.evidence.row_indices)} Flagged Rows):**")
                    valid_row_indices = [idx for idx in active_finding.evidence.row_indices if idx in df.index][:20]
                    if valid_row_indices:
                        st.dataframe(df.loc[valid_row_indices], use_container_width=True)

            # Global Report Verification
            st.divider()
            st.markdown('<div class="section-header">GLOBAL REPORT EVIDENCE CONSISTENCY AUDIT</div>', unsafe_allow_html=True)
            if st.button("Run Full Evidence Verification Across All Findings"):
                with st.spinner("Validating finding samples against source DataFrame..."):
                    val_res = validate_report_evidence(df, report)
                    if val_res["all_valid"]:
                        st.success(f"100% Verified: All {val_res['total_findings']} findings strictly match source DataFrame records.")
                    else:
                        st.warning(f"{val_res['invalid_count']} finding(s) have discrepancies.")
                        st.json(val_res["discrepancies_by_finding"])

        # ----------------------------------------------------------------------
        # SECTION 6: AI INVESTIGATOR
        # ----------------------------------------------------------------------
        with tab_ai:
            st.markdown('<div class="section-header">LOCAL AI TECHNICAL INVESTIGATOR (OLLAMA)</div>', unsafe_allow_html=True)

            # Model Configuration & Availability
            ai_stat1, ai_stat2, ai_stat3 = st.columns(3)
            ai_stat1.metric("Ollama Host", config.ollama_host)
            ai_stat2.metric("Configured Model", explainer.model)
            ai_stat3.metric("Ollama Daemon Status", ai_status["status"].upper())

            st.markdown(
                f'<div class="mono-meta" style="padding:0.6rem; background-color:#121315; border:1px solid #262626; margin-bottom:1rem;">'
                f'Strict Model Guard: <code>qwen2.5:1.5b</code> (default) | Optional Minimal: <code>qwen2.5:0.5b</code> | '
                f'Heavy 7B/14B+ models disallowed to protect local host memory.<br>'
                f'Zero Cloud Transmissions: Only compact finding metrics are shared locally; raw datasets are never transmitted.'
                f'</div>',
                unsafe_allow_html=True,
            )

            if ai_status["status"] != "ready":
                st.warning(f"Local AI engine not ready: {ai_status['status']}. Showing manual setup instructions below.")
                st.code(ai_status["setup_instructions"], language="bash")

            if not report.findings:
                st.info("No findings flagged to explain.")
            else:
                st.markdown('<div class="section-header">SELECT FINDING FOR LOCAL AI INTERPRETATION</div>', unsafe_allow_html=True)
                ai_finding_labels = [f"[{f.severity.value}] {f.title} ({f.rule_id})" for f in report.findings]
                ai_idx = st.selectbox("Select Finding to Explain", options=range(len(report.findings)), format_func=lambda i: ai_finding_labels[i], key="ai_select_finding")
                target_finding = report.findings[ai_idx]

                ac1, ac2 = st.columns([1, 2])
                with ac1:
                    generate_btn = st.button("Generate Local AI Explanation", key=f"gen_ai_btn_{target_finding.id}", use_container_width=True)
                with ac2:
                    force_fallback = st.checkbox("Force Deterministic Fallback Mode", value=False, help="Inspect deterministic rule explanation without calling local LLM")

                if generate_btn or f"ai_expl_{target_finding.id}" in st.session_state:
                    if generate_btn:
                        with st.spinner("Generating local explanation with Qwen 2.5..."):
                            if force_fallback:
                                exp_res = explainer._fallback_result(target_finding, "manual_fallback")
                            else:
                                exp_res = explainer.explain_finding_structured(target_finding)
                            st.session_state[f"ai_expl_{target_finding.id}"] = exp_res
                    else:
                        exp_res = st.session_state[f"ai_expl_{target_finding.id}"]

                    if exp_res.source == "ai_qwen":
                        st.markdown(
                            f'<div style="background-color:#14171f; border-left:3px solid #3b82f6; padding:1rem; border-radius:2px; margin-top:0.8rem;">'
                            f'<strong style="color:#60a5fa; font-size:1.05rem;">[AI FORENSIC INTERPRETATION ({exp_res.model})]</strong><br>'
                            f'<span style="font-size:0.75rem; color:#9ca3af;">Advisory Notice: AI-generated advisory guidance based on verified findings; not an empirical measurement.</span><br><br>'
                            f'<strong>• Forensic Interpretation:</strong><br>{exp_res.interpretation}<br><br>'
                            f'<strong>• Potential Root Causes:</strong><br>{exp_res.root_causes}<br><br>'
                            f'<strong>• Recommended Investigation Steps:</strong><br>{exp_res.investigation_steps}<br><br>'
                            f'<strong>• Scope Limitations & Alternate Explanations:</strong><br>{exp_res.limitations}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background-color:#1c1d21; border-left:3px solid #f59e0b; padding:1rem; border-radius:2px; margin-top:0.8rem;">'
                            f'<strong style="color:#fbbf24;">[DETERMINISTIC FALLBACK INTERPRETATION]</strong><br>'
                            f'<span style="font-size:0.75rem; color:#9ca3af;">Rule-based fallback active (Status: {exp_res.status}).</span><br><br>'
                            f'{exp_res.formatted_text}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ----------------------------------------------------------------------
        # SECTION 7: REPORT EXPORT
        # ----------------------------------------------------------------------
        with tab_export:
            st.markdown('<div class="section-header">FORENSIC EXAMINATION DOSSIER EXPORT</div>', unsafe_allow_html=True)
            st.markdown(
                "<p class='mono-meta'>Download complete, reproducible forensic audit records. "
                "Export works 100% offline without requiring Ollama.</p>",
                unsafe_allow_html=True,
            )

            # Download Action Buttons
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                json_content = ReportExporter.to_json(report)
                st.download_button(
                    label="⬇ Download Forensic Dossier (JSON)",
                    data=json_content,
                    file_name=f"autopsy_dossier_{meta.filename}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with d_col2:
                md_content = ReportExporter.to_markdown(report)
                st.download_button(
                    label="⬇ Download Forensic Dossier (Markdown)",
                    data=md_content,
                    file_name=f"autopsy_dossier_{meta.filename}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Live Dossier Preview
            st.divider()
            st.markdown('<div class="section-header">LIVE DOSSIER PREVIEW (MARKDOWN)</div>', unsafe_allow_html=True)
            with st.expander("Expand Live Forensic Markdown Dossier", expanded=True):
                st.markdown(md_content)

            # Limitations Catalog Viewer
            st.divider()
            st.markdown('<div class="section-header">FORENSIC ANALYSIS SCOPE & BOUNDARIES CATALOG</div>', unsafe_allow_html=True)
            if report.limitations:
                for check_key, lim in report.limitations.items():
                    with st.expander(f"🔬 {lim.get('title', check_key)}", expanded=False):
                        st.markdown(f"**What It Can Establish:** {lim.get('can_establish', 'N/A')}")
                        st.markdown(f"**What It Cannot Establish:** {lim.get('cannot_establish', 'N/A')}")
                        st.markdown(f"**Operational False Positive Risks:** {lim.get('false_positive_risks', 'N/A')}")

    except EmptyFileError as e:
        st.error(f"Integrity Violation [Empty Payload]: {e}")
    except FileTooLargeError as e:
        st.error(f"Safety Violation [Size Limit]: {e}")
    except MalformedCSVError as e:
        st.error(f"Structural Parse Failure [Malformed CSV]: {e}")
    except EncodingDetectionError as e:
        st.error(f"Encoding Error: {e}")
    except Exception as e:
        st.error(f"Unexpected Forensic Ingestion Failure: {e}")
