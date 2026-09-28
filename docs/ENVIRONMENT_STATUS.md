# Energy tooling and environment status

Checked: 2026-09-27. This is a point-in-time readiness record, not a service SLA
or evidence that an unrun model evaluation succeeded. No credentials, private
host details, local personal paths or restricted artifacts belong in this file.

| Component | Verified state | Role / next action |
| --- | --- | --- |
| Git | `2.54.0.windows.1`; feature checkout clean before this record | Existing isolated Energy feature branch; no default-branch writes. |
| Git-bundled Bash | `5.3.9(1)-release`; four demo-document Bash blocks passed `bash -n` | Syntax-only check; no Slurm submissions or WSL installation. Use the verified bundled executable instead of assuming a default install path. |
| GitHub CLI | `2.92.0`; authenticated identity query succeeded | Existing PR workflow is usable. Do not reinstall or alter authentication. |
| GitHub plugin | Directory reports installed, enabled | Plugin installation is distinct from a local skill being available. CLI access is independently verified. |
| `yeet` skill | Installed from `openai/skills`, `skills/.curated/yeet`; `SKILL.md` fully read | Use the official installer, not a copied or invented `github:yeet` alias. Follow explicit-file staging and branch isolation required by project rules. |
| LibreOffice | `26.2.5.2`; `soffice.com --version` succeeded | Existing installation is usable via its absolute executable path. Not on PATH is not an installation failure. No document conversion was claimed in this check. |
| OpenSSH | Windows `9.5p2`; existing Iris SSH connection succeeded | Reuse the configured alias; do not copy keys or alter host/authentication settings. |
| Project Python | `3.13.13` in the existing project virtual environment | Local validation uses this environment, not a global dependency upgrade. |
| Validation packages | pytest `9.1.1`, Ruff `0.16.3`, mypy `1.20.2` | Installed package metadata checked; test results remain in the actual run reports. |
| High-confidence scanner | Local ripgrep executes; CI now explicitly installs the `ripgrep` package and checks `command -v rg` / `rg --version` | Required before tests; missing/broken tooling fails closed. Old CI scans were invalid, as recorded below. |
| API/schema packages | FastAPI `0.141.1`, Pydantic `2.13.4` | Installed package metadata checked. |
| `uv` | Not found on PATH; optional | Current reproduction and Slurm scripts use `venv` and `pip`; no install needed. This does not assert absence everywhere on disk. |
| Spartan Energy runtime | Dedicated environment and successful CPU preflight | Latest CPU job `31365441` passed 222 tests, lint, types, all 20 region-day coverage checks and the corrected quarterly-context baseline. GPU availability is checked at submission. |
| Real-model inference | Pinned Qwen3-8B / llama.cpp inference executed; real-model recording passed in `31365444` | Frozen direct holdout `31365531` completed all 504 scored rows. Structured hybrid 54/54 final versus 18/54 initial model paths; no quality advantage over deterministic. Not promoted; GoalSpec remains a separate 0/8 negative pilot. |

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

## CI scanner incident and repair

Runs [36286722679](https://github.com/larry-liyuanfan/australian-energy-market-intelligence-agent/actions/runs/36286722679)
(`1b7f681`) and [36289310946](https://github.com/larry-liyuanfan/australian-energy-market-intelligence-agent/actions/runs/36289310946)
(`9a2cb64`) reported success but their final scan logged `rg: command not found`.
The shell conditional conflated "no match" with execution failure. The prior claim
that those CI scans passed is withdrawn; other completed steps and the independent
local compact-artifact review retain their own evidence, not scan coverage by proxy.

The repaired workflow installs/verifies ripgrep before tests and invokes
`scripts/secret_scan.py`. rg exit 1 means no match (wrapper success); exit 0 means
detected content (wrapper failure); every other exit, launch failure or 120-second
timeout is an error (wrapper failure). Captured scanner output is never echoed.
This remains a bounded high-confidence pattern scan, respects rg ignore rules and
excludes `.git` / `.venv`; it is not a comprehensive secret-history or PII audit.
Eleven regression tests exercise status/error handling and real clean/matching
hidden-file fixtures; missing rg cannot silently skip those tests. Local results:
237 tests, Ruff, strict mypy over 75 files and the actual bounded scan passed.
The repaired commit's CI must independently confirm installation and scan execution;
neither historical green status is sufficient. No model rerun or frozen-result
change is part of this correction.

## Dependency and license audit scope

The P1 export and native GoalSpec transport repair add no third-party Python
dependency. That statement is limited to those patches, **not the entire PR**:
the retained v2 visual experiment adds the optional `visual-benchmark` extra.
The current CI installs `[test,ml,search,redis,workbook]` and runs `pip-audit`
against that installed environment. It does not install or vulnerability-audit
`visual-benchmark`, model weights, or every allowed version in a dependency range.

Declared code licenses checked against publisher sources on 2026-09-27:

| Optional dependency | Declared range / code license | Evidence and status |
| --- | --- | --- |
| `datasets` | `>=3,<5`; Apache-2.0 | [Publisher LICENSE](https://github.com/huggingface/datasets/blob/main/LICENSE); upstream declaration reviewed, not every historical release or transitive dependency. |
| `sentence-transformers` | `>=5,<6`; Apache-2.0 | [Publisher LICENSE](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE); upstream declaration reviewed, not an audit of downloaded checkpoints. |
| `colpali-engine` | `==0.3.1`; MIT | [Exact-release publisher metadata](https://pypi.org/project/colpali-engine/0.3.1/); optional extra is outside current CI vulnerability coverage. |

These entries register dormant experiment dependencies, not deployment approval.
No visual environment was installed or resumed for this review. Model and dataset
licenses remain separate from library code licenses; no weights or third-party
corpora are redistributed. Any future activation requires its own resolved
environment, vulnerability and transitive-license review. P1 and the direct-tool
holdout do not depend on activating this optional experiment.

## Remaining application issue, not a missing dependency

Browser review of the deterministic SA1 recording from job `31364557` found that
some retrieved workbook figures concerned gas storage/prices and the following
quarter. The nine numerical/provenance checks establish calculation consistency
and citation presence, not semantic relevance. This preview is not accepted as
the final P1 evidence-grounded demonstration. Period/topic grounding was corrected
and verified in `31365198`. Subsequent review found two exporter gaps: ignoring
runtime verification failures, and screening a longer passage than the displayed
quote. These are application fixes with rejection tests, not missing dependencies.
Those export gaps and the model's chart/text-route mismatch are now corrected.
The real-model recording passed 13 export checks in `31365444`; it remains
explicitly runtime-assisted. The [completed holdout report](PLANNER_REMEDIATION_V3_EVALUATION.md)
now provides the separate full evaluation and non-promotion decision. Tool
readiness and a completed-case claim are not substitutes for that comparison.

See [remediation review](REMEDIATION_REVIEW_20260927.md) for the P0/P1 execution
record and [demo contract](INTERVIEW_DEMO.md) for the intended deliverable.
