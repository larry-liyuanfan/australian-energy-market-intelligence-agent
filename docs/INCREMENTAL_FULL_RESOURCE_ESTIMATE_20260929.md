# Frozen full run: resource estimate only

No submission or new evaluation is authorized by this document. The frozen
questions, scoring, strategy, seeds and recovery limits are unchanged.

## Count and time envelope

- Initial turn slots: 24 episodes × 2 turns × 3 paths × 3 seeds = **432**.
  Fail-closed paths may stop before making a request.
- Existing legacy cap: 144 turn slots × (1 initial + 2 replans) = **432 requests**.
- Existing two graph paths: 288 slots × (1 initial + 1 recovery) = **576 requests**.
- Loose whole-run cap: **1,008 provider requests**, including at most 576 extra
  recovery requests. This applies the unchanged per-turn caps everywhere; it is
  not a prediction that every turn offers or needs recovery. Ordinary tool retry
  and model replanning remain different counters.

Confirmation job 31491560 measured 153.639 seconds over 12 path-turns including
graph/checkpoint/verification work (12.803 seconds per turn). Using this as a
rough throughput input gives **92.2 minutes** for 432 initial turns. Scaling the
same average to the loose request cap gives **215.1 minutes**; recovery prompt
sizes and controlled faults were not measured in the normal-turn pilot, so this
is a sizing estimate, not a throughput guarantee or statistical confidence bound.

Recommend a **5-hour single-job walltime cap**, if separately released, using
the same **one 20GB MIG slice, 6 CPU, 16GiB RAM and 10,000MiB node scratch**.
This adds about 39% headroom to the loose empirical request extrapolation.
Observed confirmation MaxRSS was 8,009,000KiB. No measured evidence calls for a
larger model, GPU or memory allocation. Use scratch for the 216 per-episode
SQLite databases and preserve one final archive; project inode availability
must be checked again before submission, without deleting other projects.

The unchanged 45-second provider timeout produces a much larger theoretical
transport-only bound: 1,008 × 45 seconds = **12.6 hours**, excluding local work.
The recommended 5-hour cap therefore **does not guarantee completion under
worst-case repeated timeouts**. If that cap is reached, retain partial/failure
records; do not silently continue in a new job. No additional retry is authorized
here. No AUD infrastructure price is available.

## Frozen identities and release boundary

| Item | Identity |
|---|---|
| Executed source | `b9791d3f134172c2ebba9ac9f5e8c619ec247cbb` |
| Source06 bundle | `b54b2a54289f560c351c1955973c555f4d4548fc01e85e878c96d1203cc59b67` |
| Launcher | `ef3c27ef8e2d909a83a3f1ef3357ae4af8ccd67099ad3e212d344a95b3cc8fab` |
| November archive, `november-inputs-31484156.tar` | `793e781d5758629f91ebb72491ea903683094c00abd5bf4c97c1623e148e3fa6` |
| November market | `76547ea72e398391faccda799ad8aaaac3b9163374ce7d4a946dc7483795c4f2` |
| November snapshots | `ab4563e7d7c1f1e859f0acaa3432759bcb3403e7cca37314849f5b7f63be7fbb` |
| Frozen v2 scenario file | `9de3945eb6cbbf0868f76b9eb204c12ade8b9c5d8f443a5b1ea70dd239ce474f` |
| Confirmation report | `c343b2105f57bef090b9b5ba9afdeb0f8265c87410593e518ec48dcf7b95247f` |
| Confirmation manifest | `616379b5c555fe94bee372297192410249cb9fdab10086a70356073281267cde` |

November hashes come from the retained CPU preparation manifest, not a new
evaluation. Source06 is the bundle executed by confirmation job 31491560.
The confirmation report/manifest are **integration-repair evidence**, not a
model-gain pass. A full quality/release receipt has **not** been created. The
Slurm gate requires a separately reviewed receipt bound to this same source06
SHA; neither the old pilot nor a report hash can be relabelled as that receipt.
Full remains unsubmitted pending explicit coordinator release.

Final delivery commit `95bbd2b5b6a17c463c01ddfda4b09118e9b6c7c8` also passed
[CI 36468040423](https://github.com/larry-liyuanfan/australian-energy-market-intelligence-agent/actions/runs/36468040423):
257 main tests (2 skipped), 49 graph tests, Ruff, strict mypy over 90 sources,
dependency audit and an actually executed ripgrep 14.1.0 secret scan. The added
auditor is post-run code, not a change to source06. Its reproduced report is
byte-identical, and independent read-only mutation checks rejected eight faulty
records. No further test, model or optimization run was used for this estimate.
