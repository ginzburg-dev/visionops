# Showcase assets

This directory holds a still from the annotated derivative of a crowd clip and its
provenance file. Neither raw footage nor rendered video is committed: `assets/raw/` and every
`*.mp4` are ignored, and the pipeline never downloads media.

## Status

The showcase has been rendered over the source described below, using
`models/person.onnx`. `crowd-showcase.png` (the busiest frame) and `crowd-showcase.json` are
committed; the numbers quoted in the root README come from that JSON file.

`crowd-showcase.mp4` is **not** committed. At 43 MB it would sit in the history of every
clone forever, so it is ignored like any other video here. Re-create it with the command
below: the same source, model, and window produce the same clip.

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

## Source in use

- <https://www.pexels.com/video/crowd-of-people-walking-in-urban-area-34901487/>
- Author: Sururi Ballıdağ. Licence: Pexels License. Downloaded by hand on 2026-10-16.

Accepted on 2026-10-16 against every gate above:

| Gate | Observed |
| --- | --- |
| Resolution | 2688x1512 on the source page; the 2560x1440 rendition was downloaded |
| Duration | 0:18, long enough to cut an 8 second window starting at 4s |
| Crowd | A busy Istanbul waterfront walkway: 5 to 20 people detected per frame, with overlap and partial occlusion |
| Camera | Hand-held but fixed vantage point, no pan or cut across the window used |
| Licence | The Pexels License as published on 2026-10-16 allows free use and modification; it forbids unaltered resale, redistribution on other stock platforms, implying endorsement, trade-mark use, and showing identifiable people in a bad light. An annotated derivative published in this repository is within those terms. Attribution is not required but is recorded here and in `crowd-showcase.json` anyway. |

### Why the committed clip is 1920x1080

OpenCV on the rendering machine had no H.264 encoder available, so the writer falls back to
MPEG-4 Part 2, which produced an 82 MB file at the source 2560x1440. The raw clip was
therefore downscaled once, locally, to exactly 1920x1080 before rendering, which still
satisfies the resolution gate and brings the committed clip to 43 MB:

```bash
# one-off, writes into the ignored assets/raw/ directory
python -c "
import cv2
cap = cv2.VideoCapture('assets/raw/14786159_2560_1440_30fps.mp4')
out = cv2.VideoWriter('assets/raw/14786159_1920_1080_30fps-downscaled.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (1920, 1080))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA))
out.release(); cap.release()
"
```

`source_sha256` in `crowd-showcase.json` is the hash of that downscaled file, not of the
original download, because that is the file the renderer actually read.

## Rendering command

```bash
uv run python -m visionops make-showcase \
  --video assets/raw/14786159_1920_1080_30fps-downscaled.mp4 \
  --model models/person.onnx \
  --out assets \
  --start-s 4.0 \
  --duration-s 8.0 \
  --source-url "https://www.pexels.com/video/crowd-of-people-walking-in-urban-area-34901487/" \
  --license "Pexels License" \
  --attribution "Sururi Ballıdağ"
```

`crowd-showcase.json` records the source SHA-256, the licence and attribution strings passed
above, the model hash and thresholds, the frame rate, the frame count, and the per-frame
`person_count`. Check those values before publishing the clip anywhere.
