# Energy tooling and environment status

Checked: 2026-09-27. This is a point-in-time readiness record, not a service SLA
or evidence that an unrun model evaluation succeeded. No credentials, private
host details, local personal paths or restricted artifacts belong in this file.

| Component | Verified state | Role / next action |
| --- | --- | --- |
| Git | `2.54.0.windows.1`; feature checkout clean before this record | Existing isolated Energy feature branch; no default-branch writes. |
| GitHub CLI | `2.92.0`; authenticated identity query succeeded | Existing PR workflow is usable. Do not reinstall or alter authentication. |
| GitHub plugin | Directory reports installed, enabled | Plugin installation is distinct from a local skill being available. CLI access is independently verified. |
| `yeet` skill | Installed from `openai/skills`, `skills/.curated/yeet`; `SKILL.md` fully read | Use the official installer, not a copied or invented `github:yeet` alias. Follow explicit-file staging and branch isolation required by project rules. |
| LibreOffice | `26.2.5.2`; `soffice.com --version` succeeded | Existing installation is usable via its absolute executable path. Not on PATH is not an installation failure. No document conversion was claimed in this check. |
| OpenSSH | Windows `9.5p2`; existing Iris SSH connection succeeded | Reuse the configured alias; do not copy keys or alter host/authentication settings. |
| Project Python | `3.13.13` in the existing project virtual environment | Local validation uses this environment, not a global dependency upgrade. |
| Validation packages | pytest `9.1.1`, Ruff `0.16.3`, mypy `1.20.2` | Installed package metadata checked; test results remain in the actual run reports. |
| API/schema packages | FastAPI `0.141.1`, Pydantic `2.13.4` | Installed package metadata checked. |
| `uv` | Not found on PATH; optional | Current reproduction and Slurm scripts use `venv` and `pip`; no install needed. This does not assert absence everywhere on disk. |
| Spartan Energy runtime | Dedicated environment and successful CPU preflight already recorded | CPU job `31365198` passed 216 tests, lint, types, all 20 region-day coverage checks and the corrected quarterly-context baseline. Further export-review changes require their own validation. GPU availability must be rechecked at submission, not inferred from this document. |
| Real-model inference | Existing pinned Qwen3-8B / llama.cpp runtime available; new remediation inference not yet run at this checkpoint | GPU contention is a scheduling constraint, not a missing tool. No new model-success or promotion claim. |

Installed `yeet/SKILL.md` SHA-256:
`4829d4909081e1abf91507fb581e6f566ce7064001892414f32191895b08a595`.
The installer returned success and the installed file was read and hashed.
The installed skill is now available and has been used. Existing plugin configuration,
credentials, PATH and other project environments were not modified.

## Installation and registration policy

Check the actual executable and role before installing. Prefer an already usable
tool; record optional dependencies instead of broadening the environment. Record
account authorization, runtime availability, resource contention and application
correctness separately. A successful install or JSON schema check cannot stand in
for a real inference, relevant evidence retrieval or an end-to-end evaluation.

## Remaining application issue, not a missing dependency

Browser review of the deterministic SA1 recording from job `31364557` found that
some retrieved workbook figures concerned gas storage/prices and the following
quarter. The nine numerical/provenance checks establish calculation consistency
and citation presence, not semantic relevance. This preview is not accepted as
the final P1 evidence-grounded demonstration. Period/topic grounding was corrected
and verified in `31365198`. Subsequent review found two exporter gaps: ignoring
runtime verification failures, and screening a longer passage than the displayed
quote. These are application fixes with rejection tests, not missing dependencies.
No real-model remediation success or promotion is asserted by this tools record.

See [remediation review](REMEDIATION_REVIEW_20260927.md) for the P0/P1 execution
record and [demo contract](INTERVIEW_DEMO.md) for the intended deliverable.
