"""Local lightweight LLM explanation interface using Ollama with deterministic fallback.

Exclusively supports local lightweight models: default qwen2.5:1.5b, optional qwen2.5:0.5b.
Never defaults to 7B, 14B or larger models.
Strictly isolated: never transmits data to external or third-party APIs.
The model explains verified findings, suggests investigation steps, and notes limitations.
It must never calculate statistics, invent measurements, or declare leakage without evidence.
Never downloads models automatically.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

from config import AppConfig, config as default_config
from engine.models import Finding

logger = logging.getLogger("data_autopsy.ai.explainer")

# Supported lightweight models strictly bounded to prevent workstation memory exhaustion
ALLOWED_LIGHTWEIGHT_MODELS = ("qwen2.5:1.5b", "qwen2.5:0.5b")
DEFAULT_QWEN_MODEL = "qwen2.5:1.5b"


@dataclass
class ExplanationResult:
    """Structured explanation returned by the forensic explainer."""
    source: str  # "ai_qwen" or "deterministic_fallback"
    model: str
    status: str  # "success", "fallback_offline", "fallback_model_missing", "fallback_timeout", "fallback_malformed", "fallback_error"
    interpretation: str
    root_causes: str
    investigation_steps: str
    limitations: str
    formatted_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation result to serializable dictionary."""
        return asdict(self)


class ForensicExplainer:
    """Provides forensic explanations for findings using local Ollama instance.
    
    Strictly isolated: never transmits data to external third-party cloud APIs.
    Operates with rule-based deterministic fallbacks when Ollama is offline or model is missing.
    """

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or default_config
        self.model = self._sanitize_model_name(self.cfg.ollama_model)

    @classmethod
    def _sanitize_model_name(cls, model_name: str) -> str:
        """Enforce strict guard against 7B, 14B or larger models."""
        cleaned = (model_name or "").strip()
        lower = cleaned.lower()
        disallowed_sizes = ["7b", "8b", "14b", "32b", "70b", "72b"]
        if any(sz in lower for sz in disallowed_sizes) or not cleaned:
            logger.warning(
                "Disallowed heavy model '%s' requested. Enforcing lightweight default '%s'.",
                model_name,
                DEFAULT_QWEN_MODEL,
            )
            return DEFAULT_QWEN_MODEL
        return cleaned

    def get_setup_instructions(self) -> str:
        """Provide explicit manual setup instructions for local Ollama and Qwen."""
        return (
            f"Local LLM setup required (Ollama):\n\n"
            f"1. Install Ollama: https://ollama.com\n"
            f"2. Pull the lightweight model manually:\n"
            f"   ollama pull {self.model}\n"
            f"3. Run the model:\n"
            f"   ollama run {self.model}\n\n"
            f"Note: Data Autopsy never downloads models automatically."
        )

    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if local Ollama daemon is active and whether configured model exists.
        
        Returns:
            Dict containing:
                - available (bool): True if daemon is reachable.
                - model_installed (bool): True if configured model is in local tags.
                - installed_models (List[str]): List of models detected.
                - status (str): 'ready', 'ollama_offline', or 'model_missing'.
                - setup_instructions (str): Manual setup instructions if not ready.
        """
        try:
            resp = requests.get(
                f"{self.cfg.ollama_host}/api/tags",
                timeout=2.0,
            )
            if resp.status_code != 200:
                return {
                    "available": False,
                    "model_installed": False,
                    "installed_models": [],
                    "status": "ollama_offline",
                    "setup_instructions": self.get_setup_instructions(),
                }

            data = resp.json()
            raw_models = data.get("models", [])
            installed_names = [m.get("name", "") for m in raw_models]

            target = self.model.lower()
            model_found = False
            for name in installed_names:
                name_clean = name.lower()
                # Match "qwen2.5:1.5b", "qwen2.5:1.5b:latest", or base tag
                if target == name_clean or f"{target}:latest" == name_clean or target == name_clean.split(":")[0]:
                    model_found = True
                    break

            if not model_found:
                return {
                    "available": True,
                    "model_installed": False,
                    "installed_models": installed_names,
                    "status": "model_missing",
                    "setup_instructions": self.get_setup_instructions(),
                }

            return {
                "available": True,
                "model_installed": True,
                "installed_models": installed_names,
                "status": "ready",
                "setup_instructions": "",
            }

        except (requests.RequestException, OSError):
            return {
                "available": False,
                "model_installed": False,
                "installed_models": [],
                "status": "ollama_offline",
                "setup_instructions": self.get_setup_instructions(),
            }

    def is_available(self) -> bool:
        """Check if local Ollama daemon is active and configured model is available locally.
        
        Returns:
            True if Ollama is running and model is installed, False otherwise.
        """
        status = self.check_ollama_status()
        return bool(status.get("available") and status.get("model_installed"))

    def _extract_minimal_evidence(self, finding: Finding) -> Dict[str, Any]:
        """Extract strictly bounded, minimum necessary evidence payload.
        
        Never sends raw data rows, entire tables, or large sample values.
        """
        compact_metrics: Dict[str, Any] = {}
        if finding.evidence and finding.evidence.metrics:
            for k, v in list(finding.evidence.metrics.items())[:6]:
                if isinstance(v, float):
                    compact_metrics[k] = round(v, 4)
                elif isinstance(v, (int, str, bool)):
                    compact_metrics[k] = v
                elif isinstance(v, list) and len(v) <= 3:
                    compact_metrics[k] = v

        return {
            "rule_id": finding.rule_id,
            "category": finding.category.value,
            "severity": finding.severity.value,
            "title": finding.title,
            "summary": finding.summary,
            "affected_columns": finding.evidence.column_names[:3] if finding.evidence else [],
            "metrics": compact_metrics,
            "thresholds": finding.evidence.thresholds if finding.evidence else {},
        }

    def explain_finding_structured(self, finding: Finding) -> ExplanationResult:
        """Generate structured forensic technical explanation with rigorous validation and fallback."""
        status_info = self.check_ollama_status()

        if not status_info["available"]:
            return self._fallback_result(finding, "fallback_offline")

        if not status_info["model_installed"]:
            return self._fallback_result(finding, "fallback_model_missing")

        minimal_evidence = self._extract_minimal_evidence(finding)

        system_prompt = (
            "You are a disciplined forensic tabular data analyst assisting in interpreting verified findings.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Explain ONLY the verified empirical finding provided. Do not invent details.\n"
            "2. Do NOT calculate new statistics, estimate percentages, or invent measurements.\n"
            "3. Do NOT declare target data leakage as 'confirmed' without unambiguous proof; explicitly identify alternate non-malicious explanations.\n"
            "4. Return strict JSON format with exactly these four string keys:\n"
            '   {"interpretation": "...", "root_causes": "...", "investigation_steps": "...", "limitations": "..."}\n'
            "5. Keep descriptions concise, rigorous, and technical."
        )

        user_prompt = (
            f"Verified Finding Data:\n"
            f"{json.dumps(minimal_evidence, indent=2)}\n\n"
            f"Explain this finding in strict JSON format with keys: "
            f"interpretation, root_causes, investigation_steps, limitations."
        )

        try:
            resp = requests.post(
                f"{self.cfg.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 450,
                    },
                },
                timeout=self.cfg.ollama_timeout_seconds,
            )

            if resp.status_code == 200:
                body = resp.json()
                raw_response = body.get("response", "").strip()
                parsed = self._validate_model_json(raw_response)
                if parsed:
                    formatted = self._format_ai_explanation(parsed)
                    return ExplanationResult(
                        source="ai_qwen",
                        model=self.model,
                        status="success",
                        interpretation=parsed["interpretation"],
                        root_causes=parsed["root_causes"],
                        investigation_steps=parsed["investigation_steps"],
                        limitations=parsed["limitations"],
                        formatted_text=formatted,
                    )
                else:
                    logger.warning("Local Qwen output failed schema validation. Falling back to deterministic rules.")
                    return self._fallback_result(finding, "fallback_malformed")

            logger.warning("Local Ollama returned non-200 code: %d", resp.status_code)
            return self._fallback_result(finding, "fallback_error")

        except requests.exceptions.Timeout:
            logger.warning("Local Ollama generation timed out after %ds", self.cfg.ollama_timeout_seconds)
            return self._fallback_result(finding, "fallback_timeout")
        except (requests.exceptions.RequestException, OSError) as e:
            logger.warning("Local Ollama request failed: %s", str(e))
            return self._fallback_result(finding, "fallback_offline")

    def _validate_model_json(self, raw_text: str) -> Optional[Dict[str, str]]:
        """Validate and sanitize structured JSON returned by local model."""
        if not raw_text:
            return None

        # Attempt direct JSON parsing
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                req_keys = ["interpretation", "root_causes", "investigation_steps", "limitations"]
                if all(k in data and isinstance(data[k], str) and len(data[k].strip()) > 0 for k in req_keys):
                    return {k: data[k].strip() for k in req_keys}
        except json.JSONDecodeError:
            pass

        # Attempt regex extraction if model wrapped JSON in markdown code fence
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    req_keys = ["interpretation", "root_causes", "investigation_steps", "limitations"]
                    if all(k in data and isinstance(data[k], str) and len(data[k].strip()) > 0 for k in req_keys):
                        return {k: data[k].strip() for k in req_keys}
            except json.JSONDecodeError:
                pass

        return None

    def _format_ai_explanation(self, parsed: Dict[str, str]) -> str:
        """Format validated AI response with prominent indicator distinguishing AI advisory from empirical data."""
        return (
            f"[AI Forensic Interpretation ({self.model})]\n"
            f"Advisory Notice: AI-generated interpretation based on verified empirical findings; not an empirical measurement.\n\n"
            f"• Forensic Interpretation: {parsed['interpretation']}\n"
            f"• Potential Root Causes: {parsed['root_causes']}\n"
            f"• Recommended Investigation Steps: {parsed['investigation_steps']}\n"
            f"• Scope & Limitations: {parsed['limitations']}"
        )

    def generate_explanation(self, finding: Finding) -> str:
        """Generate formatted forensic technical explanation for a specific finding.
        
        Args:
            finding: The deterministic finding to explain.
            
        Returns:
            Forensic explanation text.
        """
        result = self.explain_finding_structured(finding)
        return result.formatted_text

    def _fallback_result(self, finding: Finding, status_code: str) -> ExplanationResult:
        """Generate deterministic forensic interpretation without calling local LLM."""
        cat = finding.category.value
        rec = finding.recommendation or "Review data pipeline origin."
        why = finding.why_it_matters or "Violates standard data health parameters."
        lim = finding.limitations or "Domain context may legitimately explain observed values."

        formatted = (
            f"[Deterministic Forensic Analysis] "
            f"Category: {cat}. "
            f"The observed condition violates standard data health parameters. "
            f"Recommendation: {rec}"
        )

        cols_str = ", ".join(finding.evidence.column_names) if finding.evidence else "specified features"
        return ExplanationResult(
            source="deterministic_fallback",
            model="deterministic_rules",
            status=status_code,
            interpretation=why,
            root_causes=f"Structural anomaly or pipeline defect in {cols_str}.",
            investigation_steps=rec,
            limitations=lim,
            formatted_text=formatted,
        )
