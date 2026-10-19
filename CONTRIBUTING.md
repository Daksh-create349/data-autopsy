# Contributing to Data Autopsy

Thank you for your interest in contributing to Data Autopsy. We welcome contributions that improve forensic anomaly detection, documentation, performance, and developer experience.

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Core Engineering Principles

Before opening a pull request, ensure your contribution respects these fundamental design principles:

1. **Determinism:** Core anomaly detection logic must be 100% deterministic and reproducible. Algorithms that involve randomness must accept and honor fixed random seeds (`random_state`).
2. **Traceability:** Every finding must preserve and report exact 0-indexed row numbers and verbatim column headers from the source dataset. Never synthesize or guess row indices.
3. **No Hardcoding:** Never hardcode dataset names, column names, expected finding counts, or fixed outputs into detector logic. All findings must be dynamically computed by real statistical engines.
4. **Local Sovereignty:** Do not introduce dependencies that require cloud credentials, phone-home telemetry, or external API transmission of user data.
5. **Clean Data Baseline:** Clean datasets must never trigger unsupported critical or high severity alerts. Any new detector must be calibrated against clean data.

## Development Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone git@github.com:<your-username>/data-autopsy.git
   cd data-autopsy
   ```
3. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the existing test suite:
   ```bash
   pytest -v
   ```

## Pull Request Guidelines

1. **Branch Naming:** Use clear branch prefixes:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`
   - `test/test-suite-expansion`
2. **Automated Testing:** Every new detector or engine change must include corresponding unit tests under `tests/`.
3. **Run Full Test Suite:** Ensure all 73+ tests pass before opening your pull request:
   ```bash
   pytest -v
   ```
4. **Code Quality:** Ensure code is well-structured, typed with type annotations, and formatted cleanly.
5. **Documentation:** If introducing new detection rules, update the limitations and detector catalog in `engine/limitations.py` and `README.md`.

## Submitting Issues

* **Bug Reports:** Include Python version, OS, minimal reproducible CSV sample, and expected vs actual behavior.
* **Feature Requests:** Detail the proposed forensic check, mathematical rationale, potential false positive risks, and severity mapping.
