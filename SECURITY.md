# Security Policy

## Supported Versions

Security updates and patches are actively maintained for the following versions of Data Autopsy:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Security Architecture & Design Guarantees

Data Autopsy is engineered for sovereign local execution:

* **Local-Only Execution:** All statistical computations, scikit-learn model fitting, and chart rendering execute strictly within the user's local Python process.
* **No Cloud Exfiltration:** User datasets are never transmitted to external APIs, remote telemetry services, or cloud infrastructure.
* **Bounded Local LLM Payloads:** When Ollama is enabled, only compact mathematical summaries (rule IDs, affected columns, observed metrics) are sent over loopback (`localhost:11434`). Raw dataset records are never shared with the model.
* **Safe Ingestion:** CSV and TSV parsing enforces strict maximum file size checks, encoding validation, and bad-line parsing exceptions to guard against buffer and memory exhaustion.

## Reporting a Vulnerability

If you discover a potential security vulnerability in Data Autopsy, please report it responsibly:

1. **Do not disclose publicly:** Do not open a public GitHub issue, pull request, or discussion.
2. **Email Maintainers:** Send an email detailing the vulnerability to `dakshshrivastav56@gmail.com`.
3. **Include Details:**
   - Description of the vulnerability.
   - Steps to reproduce or proof-of-concept script.
   - Affected components, configurations, or environments.
   - Potential impact.

## Response Process

* You will receive an initial response within 48 hours acknowledging receipt of the report.
* The maintainers will investigate the issue and coordinate a timeline for remediation.
* A patch and advisory will be released once the fix is verified.
