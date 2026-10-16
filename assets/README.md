# Showcase assets

This directory holds the annotated derivative of a crowd clip and its provenance file. Raw
footage is never committed: `assets/raw/` is ignored, and the pipeline never downloads media.

## Status

**The real crowd showcase has not been rendered in this checkout.** Rendering it needs two
inputs that must be placed on disk by a person:

1. A crowd video that passes the gates below, saved under `assets/raw/`.
2. A person-detection model exported to ONNX (`visionops export-onnx`), which currently needs
   the optional `train` extra (`uv sync --extra train`) because the export path goes through
   Ultralytics.

Until both exist, `assets/crowd-showcase.mp4`, `assets/crowd-showcase.png`, and
`assets/crowd-showcase.json` are absent, and no result is claimed anywhere in this repository.

## Source gates

A candidate clip is used only when every gate holds, and the reason for the final choice is
recorded below:

- At least 1920x1080; the renderer refuses anything smaller.
- At least 15 seconds of usable footage so an 8-15 second window can be cut.
- A genuine crowd: many people at once, with overlap and partial occlusion, not two or three
  pedestrians.
- People are large enough to be visibly boxed at 1080p, and the camera is roughly static.
- Current official usage terms of the source site allow redistribution of an annotated
  derivative in a public repository, checked at the time the clip is downloaded.

## Candidate under consideration

- <https://www.pexels.com/video/crowd-of-people-walking-in-urban-area-34901487/>
  Status: not yet evaluated in this checkout. Resolution, crowd density, and the current
  licence text must be confirmed against the gates above before the clip is used, and the
  outcome written here together with the download date.

## Rendering command

```bash
uv run python -m visionops make-showcase \
  --video assets/raw/<clip>.mp4 \
  --model models/person.onnx \
  --out assets \
  --start-s <seconds> \
  --duration-s 10 \
  --source-url <clip page URL> \
  --license "<licence name>" \
  --attribution "<author>"
```

`crowd-showcase.json` records the source SHA-256, the licence and attribution strings passed
above, the model hash and thresholds, the frame rate, the frame count, and the per-frame
`person_count`. Check those values before publishing the clip anywhere.
