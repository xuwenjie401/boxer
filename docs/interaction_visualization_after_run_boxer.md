# Interaction and Visualization After `run_boxer.py`

This note explains how to inspect Boxer results after running `run_boxer.py`, and
how to use the interactive prompt viewer for new 2D or text prompts.

The examples assume:

```bash
cd /home/wjxu22/RealityLab/boxer
conda activate jarvis
```

If checkpoints and data live outside the repo, keep the repo-level links in
place so scripts that expect `ckpts/` and `sample_data/` continue to work:

```bash
ln -s /home/wjxu22/huggingface/boxer ckpts
ln -s /home/wjxu22/Datasets/boxer_dataset sample_data
```

## Outputs Created by `run_boxer.py`

A normal run writes results under:

```text
output/<sequence_name>/
```

For example:

```bash
python run_boxer.py --input nym10_gen1 --max_n 90 --track
```

creates:

```text
output/nym10_gen1/owl_2dbbs.csv
output/nym10_gen1/boxer_3dbbs.csv
output/nym10_gen1/boxer_3dbbs_tracked.csv   # only when --track is used
output/nym10_gen1/boxer_viz/                # per-frame visualization images
output/nym10_gen1/boxer_viz_current.jpg
output/nym10_gen1/boxer_viz_final.mp4       # only when visualization is enabled
```

If you used a custom output directory or prefix, pass the same values to the
viewer scripts:

```bash
--output_dir /path/to/output
--write_name my_prefix
```

## Recommended Workflow

First run a short smoke test without video:

```bash
python run_boxer.py --input nym10_gen1 --max_n 5 --track --skip_viz
```

Then run a fuller sequence:

```bash
python run_boxer.py --input nym10_gen1 --max_n 90 --track
```

After that, use one of the viewers below.

## Tracker Viewer

Use `view_tracker.py` to inspect per-frame detections, tracked 3D boxes, camera
trajectory, RGB overlays, and semi-dense points.

```bash
python view_tracker.py --input nym10_gen1 --autoplay
```

It reads:

```text
output/nym10_gen1/boxer_3dbbs_tracked.csv   # preferred if present
output/nym10_gen1/boxer_3dbbs.csv           # fallback
output/nym10_gen1/owl_2dbbs.csv
sample_data/nym10_gen1/
```

Useful options:

```bash
# Load fewer frames into the viewer
python view_tracker.py --input nym10_gen1 --skip_n 2 --max_n 200

# Start in follow-camera mode and show observed points
python view_tracker.py --input nym10_gen1 --autoplay --init_follow --init_show_obs

# Match a custom run_boxer.py output location or prefix
python view_tracker.py --input nym10_gen1 --output_dir output --write_name boxer
```

Common controls:

```text
Space          play / pause
Left / Right   step frames
Esc            pause and leave follow view
Right drag     orbit camera
Middle drag    pan camera
Scroll         zoom
```

Useful UI sections:

```text
Playback       play, step, frame slider, playback FPS, record
Tracker        IoU threshold, min hits, confidence thresholds, max missed
Visualization  raw/tracked boxes, labels, RGB panel, trajectory, frustum, points
Camera         focus on scene and follow-view settings
```

The viewer has a `Record` button. When stopped, it writes an mp4 to `~/Desktop`
using `ffmpeg`.

## Fusion Viewer

Use `view_fusion.py` to inspect static 3D OBBs and run offline fusion from the
viewer UI.

```bash
python view_fusion.py --input nym10_gen1
```

It reads:

```text
output/nym10_gen1/boxer_3dbbs_fused.csv   # preferred if present
output/nym10_gen1/boxer_3dbbs.csv         # fallback
```

If you want to create the fused CSV before opening the viewer:

```bash
python run_boxer.py --input nym10_gen1 --cache3d --fuse
python view_fusion.py --input nym10_gen1
```

Inside the fusion viewer:

```text
RUN FUSION              compute fused boxes with current parameters
3DBB Conf Thresh        filter low-confidence per-frame detections
Semantic Merge Thresh   control semantic similarity for merging
IOU Threshold           control geometric merging
Min Detections          require repeated support before keeping a fused box
Color Mode              PCA, probability, or random colors
Save View               save camera_view.pt under output/<sequence_name>/
Screenshot              save screenshot_###.jpg under output/<sequence_name>/
```

Reopen the same camera view later with:

```bash
python view_fusion.py --input nym10_gen1 --load_view
```

## Prompt Viewer

Use `view_prompt.py` when you want to interactively create new 2D bounding-box
prompts or run OWLv2 text detection on frames. This viewer does not require
`run_boxer.py` CSV output, but it does need the sequence data and checkpoints.

```bash
python view_prompt.py --input nym10_gen1
```

Two prompt modes are available:

```text
Option A: Drag a 2DBB
  Draw a box on the RGB panel. BoxerNet lifts that 2D prompt to a 3D OBB.

Option B: Detect Text (OWL)
  Type a text prompt such as "chair" or "table" and click Detect.
  OWLv2 finds 2D boxes, then BoxerNet lifts them to 3D.
```

Useful controls:

```text
Clear All / Undo Last       manage prompted boxes
2DBB Conf Threshold         OWLv2 detection threshold
3DBB Conf Threshold         BoxerNet 3D confidence threshold
Show RGB Panel              toggle the image panel
Show SDP Depth Overlay      overlay semi-dense depth on the RGB image
Show SDP Patch Overlay      overlay BoxerNet depth patches
Show Points                 toggle semi-dense 3D points
Focus on Scene              refocus camera on the active frame
```

Keyboard and mouse controls:

```text
Space          play / pause
Left / Right   step frames
Left click      draw 2D box in the RGB panel
Right drag      orbit camera
Middle drag     pan camera
Scroll          zoom
Escape          quit
```

## Headless vs GUI

Use `run_boxer.py --skip_viz` for headless checks or when no display is
available:

```bash
python run_boxer.py --input nym10_gen1 --max_n 90 --track --skip_viz
```

The interactive viewers require an OpenGL-capable display and the optional GUI
dependencies:

```bash
python -m pip install -r requirements-viewer.txt
```

If `ffmpeg` is installed outside `jarvis`, expose it through the shell before
commands that create mp4 files:

```bash
export PATH=/home/wjxu22/anaconda3/bin:$PATH
```

## Quick Command Reference

```bash
# Run Boxer and tracker outputs
python run_boxer.py --input nym10_gen1 --max_n 90 --track

# Inspect tracked results
python view_tracker.py --input nym10_gen1 --autoplay

# Generate fused CSV from cached 3D boxes, then inspect fusion
python run_boxer.py --input nym10_gen1 --cache3d --fuse
python view_fusion.py --input nym10_gen1

# Create new interactive 2D/text prompts
python view_prompt.py --input nym10_gen1
```
