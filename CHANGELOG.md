# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-16

### Added

- Command-line pipeline for person detection in video: `ingest`, `prelabel`,
  `select`, `export-labelstudio`, `train`, `evaluate`, `export-onnx`, `serve`, `benchmark`,
  and `make-showcase`, each a step over a versioned JSONL or JSON contract.
- Deterministic frame ingest that records source path, group, frame index, timestamp, size,
  and the SHA-256 of every extracted image.
- Human review boundary: pre-labels are exported to Label Studio as pre-annotations marked
  `review_required`, and training refuses to start unless the dataset config declares the
  labels reviewed, keeps a test split, and never lets one source group straddle splits.
- Review batch selection under a frame budget, randomly or by confidence ambiguity combined
  with difference-hash diversity.
- ONNX export with a parity check against the source detector, recording preprocessing,
  thresholds, opset, class map, hashes, versions, and the measured deltas in
  `model-manifest.json`.
- ONNX Runtime inference and a FastAPI service exposing `/health` and `/predict`, with
  structured 4xx bodies and no PyTorch fallback at serving time.
- CPU latency and throughput benchmarking over local images.
- Annotated crowd showcase rendering with provenance: source hash, licence, attribution,
  frame rate, frame count, and per-frame `person_count`.
- Slim non-root CPU Docker image with a healthcheck against `/health`.
- `uv`-based development workflow, quality checks, GitHub Actions CI, and tagged GitHub
  Releases containing Python distributions.
- MIT license for the VisionOps source code.

### Notes

- Person class only. The pipeline does not identify people, recognise faces, track
  identities between frames, or estimate crowd density.
- No detector has been trained in this repository, so it publishes no precision, recall, or
  mAP figures. Every number in the README is backed by a generated artifact.
- Raw media, datasets, model weights, and the rendered showcase video are not committed, and
  nothing is downloaded at runtime.

[Unreleased]: https://github.com/ginzburg-dev/visionops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ginzburg-dev/visionops/releases/tag/v0.1.0
