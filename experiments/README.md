# Experiments

This directory documents how VisionOps experiments are run. It contains protocols only.
No result is recorded here until it has been measured on this checkout and the exact command,
seed, and artifact paths are written next to it.

## Matched-budget protocol: random vs. uncertainty/diversity selection

The question is whether the selection strategy changes detection quality when the human review
budget is held constant. Both arms must review the same number of frames.

1. Ingest the same source media once and keep the manifest:

   ```bash
   uv run python -m visionops ingest --input <media> --out data/pool --stride 5 --seed 0
   ```

2. Prelabel the pool once with a fixed local model. Both arms consume the same predictions file.

   ```bash
   uv run python -m visionops prelabel --manifest data/pool/manifest.jsonl --model <weights.pt> --out data/pool/predictions.jsonl
   ```

3. Build two review batches with the identical budget `B` and the identical seed:

   ```bash
   uv run python -m visionops select --predictions data/pool/predictions.jsonl --out runs/random/selection.jsonl --budget B --strategy random --seed 0
   uv run python -m visionops select --predictions data/pool/predictions.jsonl --out runs/active/selection.jsonl --budget B --strategy uncertainty --seed 0
   ```

4. Export both batches to Label Studio and review every frame by hand. Never train on a frame
   that has not been corrected and accepted by a reviewer.

5. Split by source group. A group belongs to exactly one of train, val, or test, and the test
   groups are identical for both arms so the comparison is on the same held-out footage.

6. Train both arms from the same starting weights, seed, image size, and epoch count. The only
   difference permitted between the arms is which frames the reviewer labelled.

7. Evaluate both arms on the same test split and record precision, recall, mAP@50, and
   mAP@50-95 from `metrics.json`, together with the seed, the dataset config hash, and the
   model hash.

## Reporting rules

- Report the budget `B`, the number of reviewed frames, and the number of frames per split.
- Report every metric for both arms, including the arm that performed worse.
- Repeat the run with at least two seeds before describing any gap as a difference between
  strategies; a single pair of runs is an observation, not a result.
- State the hardware, the package versions from `training-run.json`, and the exact commands.
