# Data Autopsy — 90-Second Competition Demo Script

**Total Duration:** 90 Seconds  
**Speaker Tone:** Authoritative, technical, crisp, forensic engineering  
**Target Audience:** Judges, ML engineers, data science leads  

---

### [0:00 – 0:15] The Problem: Silent Tabular Failure
*(Action: Open Data Autopsy in browser. Start on Overview tab.)*

> "Machine learning models rarely fail from bad algorithms. They fail from silent tabular defects: post-event target leakage, duplicate record contamination, and disguised identifiers that inflate validation metrics to 99% accuracy before collapsing in production.
>
> Generic profilers dump uncalibrated charts. **Data Autopsy** performs forensic statistical reverse-engineering on raw tabular datasets."

---

### [0:15 – 0:35] The Investigation: Real Statistical Engines
*(Action: Load `demo_4_target_copy_leakage.csv` from the sample dropdown. Select `readmitted_30d` as target.)*

> "Watch what happens when we feed a clinical dataset with suspected label leakage. Data Autopsy executes ten deterministic detectors in parallel: checking null topologies, duplicate candidates, cardinality, Tukey IQR, and cross-feature mutual information.
>
> It immediately flags `readmit_status_code` as an exact target copy (`LEAK-001`) and `discharge_disposition` as a target-encoded cheat feature (`LEAK-004`). Crucially, clean data does not trigger false alarms, and legitimate physical drivers are separated from post-event data corruption."

---

### [0:35 – 0:55] Verifiable Evidence & Pipeline Ablation
*(Action: Click on 'Leakage Investigation' tab, then scroll down to the 'Baseline Model Performance Comparison' table.)*

> "We never present vague probability scores. Every finding carries an exact mathematical audit trail. Look at the Evidence Inspector: original row indices, observed duplicate tuples, and modified z-scores.
>
> Here in the Leakage tab, Data Autopsy trains two isolated scikit-learn pipelines with zero data snooping: one baseline with all features, and one ablated without suspicious columns. With the leaked feature, accuracy was 100%. Without it, performance drops to real baseline: 58%. The evidence is undeniable."

---

### [0:55 – 1:15] Local AI: Sovereign Forensic Explanation
*(Action: Switch to 'AI Investigator' tab. Show local Ollama `qwen2.5:1.5b` connection badge.)*

> "Notice the AI Investigator tab. We do not transmit sensitive corporate or patient data to external clouds. 
>
> Data Autopsy uses a lightweight local open-weight model—`qwen2.5:1.5b` running entirely inside Ollama. The AI does not invent numbers or run calculations. It receives only bounded evidence payloads, explaining root causes, distinguishing signal from contamination, and recommending concrete feature engineering actions."

---

### [1:15 – 1:30] Limitations & Domain Boundaries
*(Action: Switch to 'Report Export' tab. Show limitations disclaimer and click 'Download Markdown Audit Report'.)*

> "Most importantly, Data Autopsy respects mathematical limits. Outliers are flagged as distributional anomalies, never automatically dropped as errors without domain context. Strong correlation is noted, not conflated with causation.
>
> Every analysis exports to a timestamped Markdown and JSON forensic audit trail. That is Data Autopsy: deterministic, reproducible, local forensic analysis for tabular data."
