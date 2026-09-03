# Visual retrieval and GoalSpec v2 evaluation status

This report records the 3 September 2026 Spartan execution boundary for the preregistered v2 upgrade. It is an
incomplete evaluation, not a positive model result. Frozen holdout examples were not inspected, full holdout jobs
were not submitted, and no serving-path promotion was made.

## Reproducibility boundary

- Branch: `codex/energy-visual-goal-compiler-v2`
- Clean-room preflight commit: `328c96e84eddf148975a35c4a9d1cdd9207e0469`
- Clean-room preflight job: `30005197`, completed in 57 seconds with 6 CPUs, 16 GB requested memory, and
  1,140,096 KiB batch MaxRSS
- Preflight checks: 185 tests, Ruff, strict mypy, shell syntax, isolated-path contracts, and frozen input hashes
- Private row-level artifacts remain under the task-specific Spartan artifact root and are not copied to GitHub

## GoalSpec development pilot

Job `30003769` completed successfully on the pinned Qwen3-8B/llama.cpp configuration. It used one 20 GB A100 MIG,
6 CPUs, 20 GB requested host memory, 8,110,544 KiB batch MaxRSS, and 11 minutes 47 seconds elapsed time. The private
artifact validator passed.

The four-turn development pilot produced required-field F1 0.4643, valid GoalSpec rate 0.0, compiled task success
0.0, and zero unsafe tool or DSL calls. The direct-tool comparison produced compiled task success 0.25 and raw
complete-path rate 0.5. These development results are below the frozen promotion thresholds, but they were not used
to change the prompt, schema, models, seeds, or gate. They are not holdout estimates.

## ViDoRe infrastructure stop

Three ViDoRe pilot attempts ended before dataset or model scoring:

| Job | Elapsed | Batch MaxRSS | Infrastructure stop |
|---|---:|---:|---|
| `29995830` | 11 s | 232,440 KiB | package required NumPy 2 while pinned ColQwen2 required NumPy 1 |
| `30003770` | 12 s | 229,072 KiB | package required Pillow 11 while pinned ColQwen2 required Pillow below 11 |
| `30005198` | 2 min 23 s | 11,979,144 KiB | Python SSL dependency path was removed by the CUDA module transition |

The first two failures were corrected by making the image-only Qwen and ColQwen runtimes explicit. The final source
fix preserves the Python runtime library path across the CUDA module transition, but it was not executed because the
preregistered limit of two infrastructure retries had been reached. Consequently there are no ViDoRe metrics,
bootstrap comparisons, or 2-of-3 task promotion decision.

## Promotion decision

The evaluation is **not promotable**. GoalSpec holdout thresholds and the ViDoRe 2-of-3 significance gate remain
unevaluated. The SG planner therefore remains deterministic. The new planner is restricted to loopback evaluation
and no visual model is added to the online serving path.
