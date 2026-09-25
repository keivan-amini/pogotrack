# Pogotrack

Pogotrack is a video-processing and postprocessing toolkit for reproducible analysis of Pogobot swarm experiments. It detects robot pose from video, links detections through time, decodes RGB-based robot IDs, and combines trajectories with controller data recorded during the experiment.

Pogotrack is intended primarily for experiments with [Pogobots](https://pogobot.github.io/), an open-source platform for swarm robotics and programmable active matter. For the complete, step-by-step workflow, see the [Pogotrack tutorial](doc/tutorial_pogotrack.pdf) or its [LaTeX source](doc/tutorial_pogotrack.tex).

<p align="center">
  <img src="img/logo.png" alt="Pogotrack logo" width="300">
</p>

## What the pipeline does

```text
dynamics video + background
        │
        ▼
pose detection and TrackPy linking ──► dynamics CSV

RGB-ID video + background
        │
        ▼
LED measurement and ID decoding ─────► RGB_ID.csv

dynamics CSV + RGB_ID.csv + probe data
        │
        ▼
merged dataset ──────────────────────► summary PNG/PDF
```

The main workflow supports:

- circular and rectangular arena masks;
- standard tracking and phototaxis tracking with separate dark/light processing;
- ROI-based phototaxis detection with Hough-circle fallback;
- TrackPy linking, unit conversion, and trajectory export;
- RGB-ID measurement and ternary-ID decoding;
- automatic merging with controller data stored in Feather files;
- parallel processing of one long dynamics video;
- concurrent processing of several independent experiments.

The `particle` column in the dynamics CSV is a temporary TrackPy identity. The physical robot ID is assigned later, during RGB-ID decoding and merging.

## Installation

Pogotrack supports CPython 3.11--3.13. Python 3.13 is recommended on Apple Silicon macOS. Install the project from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the active interpreter is the virtual environment:

```bash
python --version
which python
```

The first command should report Python 3.11 or newer, and the second should point to `.venv/bin/python`.

On some Homebrew installations, creating the environment requires Homebrew's Expat library to be visible:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix expat)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Do not use an older system interpreter such as `/usr/bin/python3`; the current entry point rejects Python versions older than 3.11.

## Input data layout

The config-driven workflow expects all files for one experiment in the same date directory. For example:

```text
data/
└── 18-09-26/
    ├── alpha10-4_T100_v2.mp4          # dynamics video
    ├── id_alpha10-4_T100_v2.mp4       # RGB-ID video
    ├── bkg.bmp                        # arena background
    └── alpha10-4_T100_v2.feather      # controller/probe data
```

By default, the RGB-ID video is expected to have `id_` added before the dynamics-video name, the background is named `bkg.bmp`, and the Feather file is named after the experiment. These conventions can be overridden in `postprocessing/pipeline.yaml`.

## Track one video

Run standard tracking directly with:

```bash
source .venv/bin/activate

python -m main \
  --video data/18-09-26/alpha10-4_T100_v2.mp4 \
  --background data/18-09-26/bkg.bmp \
  --output results/18-09-26/alpha10-4_T100_v2/alpha10-4_T100_v2.csv \
  --config config/default.yaml
```

The resulting CSV contains:

```text
time, x, y, theta, particle
```

The most important detector settings are in `config/default.yaml`, especially `N_POGO`, `FPS`, `CENTER`, `RADIUS`, the phototaxis thresholds, and the pixel-to-centimetre calibration.

## Parallel processing of one video

One long dynamics video can be divided into contiguous frame ranges and processed by several worker processes:

```bash
python -m main \
  --video data/18-09-26/alpha10-4_T100_v2.mp4 \
  --background data/18-09-26/bkg.bmp \
  --output results/18-09-26/alpha10-4_T100_v2/alpha10-4_T100_v2.csv \
  --config config/default.yaml \
  --workers 2 \
  --warmup-frames 50
```

Each worker receives context frames before its assigned chunk. The parent process restores frame order and performs one global TrackPy linking pass, so trajectories can continue across chunk boundaries.

`--workers 1` is the sequential mode. Start with two workers and benchmark against the sequential run before increasing the value. More workers do not necessarily give proportional speedups because video decoding, memory bandwidth, and circle detection may become bottlenecks.

This mode cannot be combined with RGB-ID analysis or `--visualize`. The `--warmup-frames` value can be increased if results differ near a chunk boundary. Never run two jobs with the same output path.

## RGB-ID tracking

RGB-ID processing is intentionally single-worker because the flashing sequence and the common robot pose must remain synchronised. The normal workflow creates a temporary configuration automatically. For a direct run, copy `config/default.yaml`, set:

```yaml
RGB_ID_ANALYSIS: true
N_POGO: 64
```

Then run:

```bash
python -m main \
  --video data/18-09-26/id_alpha10-4_T100_v2.mp4 \
  --background data/18-09-26/bkg.bmp \
  --output results/18-09-26/alpha10-4_T100_v2/RGB_ID.csv \
  --config config/rgb_id_experiment.yaml
```

This writes `RGB_ID.csv` with columns for frame, time, pose, LED position, and measured `R`, `G`, and `B` values.

## Run the complete workflow

For routine analysis, configure the experiment in `postprocessing/pipeline.yaml`:

```yaml
python: .venv/bin/python

paths:
  data: data
  results: results
  tracking_config: config/default.yaml

batch:
  enabled: true
  max_workers: 1

experiments:
  - id: alpha10-4_T100_v2
    date: 18-09-26
    video: alpha10-4_T100_v2.mp4
    n_robots: 64
    workers: 2
    probe_file: alpha10-4_T100_v2.feather
    merge:
      seconds_difference: auto
      start_rgb_frame: auto
    slides:
      enabled: true
      controller: tanh
```

Then inspect the generated commands:

```bash
python postprocessing/run_pipeline.py --dry-run
```

If the paths are correct, run all stages:

```bash
python postprocessing/run_pipeline.py
```

The stages run in this order:

1. standard dynamics tracking;
2. RGB-ID tracking;
3. RGB-ID decoding and merging with the Feather data;
4. summary PNG/PDF generation.

### Two kinds of parallelism

The two worker settings control different things:

- `experiments[].workers` controls how many processes analyse one experiment's dynamics video. Set it to `1` for sequential processing or `2` to split one video into two chunks.
- `batch.max_workers` controls how many experiments are launched concurrently by `run_pipeline.py`.

Their product is a useful upper bound on the number of dynamics-video workers. For example, `batch.max_workers: 2` and `workers: 2` may use up to four workers at once. Start conservatively on a laptop and increase the values only after benchmarking.

To run only one stage:

```bash
python postprocessing/run_pipeline.py --stage batch   # standard + RGB-ID tracking
python postprocessing/run_pipeline.py --stage rgb     # RGB-ID tracking only
python postprocessing/run_pipeline.py --stage merge   # merge existing outputs
python postprocessing/run_pipeline.py --stage slides  # generate summary figures
```

To restrict a stage to one configured experiment:

```bash
python postprocessing/run_pipeline.py \
  --stage merge \
  --experiment alpha10-4_T100_v2
```

## Outputs

Results are stored under `results/<date>/<experiment>/`:

```text
results/18-09-26/alpha10-4_T100_v2/
├── alpha10-4_T100_v2.csv       # dynamics tracking output
├── RGB_ID.csv                  # per-frame LED measurements
├── score_calib_pose_id.csv     # RGB-ID decoding intermediate
├── merged_alpha10-4_T100_v2.csv
├── alpha10-4_T100_v2.png       # summary figure
├── alpha10-4_T100_v2.pdf       # summary figure
├── tracking.log
├── rgb_tracking.log
├── merge.log
└── slides.log
```

The merged CSV contains the principal analysis columns:

```text
t, x, y, theta, q, w, id
```

Here `t` is the time after the applied recording offset, `x` and `y` are robot coordinates, `theta` is orientation, `q` and `w` are controller variables, and `id` is the decoded physical robot ID.

## Repository layout

```text
pogotrack/
├── main.py                         # CLI for standard, parallel, and RGB-ID tracking
├── src/
│   ├── video_processing.py         # frame processing and VideoProcessor
│   ├── utils.py                    # image processing and TrackPy utilities
│   └── plot_helpers.py             # visualisation and debug helpers
├── config/                         # tracking YAML configurations
├── data/                           # experimental input data
├── results/                        # generated CSV files, logs, and figures
├── postprocessing/
│   ├── pipeline.yaml               # experiment list and workflow settings
│   ├── run_pipeline.py             # config-driven workflow launcher
│   ├── process_id_trits.py         # RGB-ID decoding and merging
│   └── experiment_slides_generator.py
├── doc/tutorial_pogotrack.tex      # detailed user tutorial
├── requirements.txt                # pinned Python dependencies
└── README.md
```

## Troubleshooting

- Verify that the active Python interpreter is the project virtual environment and is at least Python 3.11.
- Check that the dynamics video, RGB-ID video, background, and Feather file belong to the same experiment.
- Check `N_POGO` in the tracking configuration and `n_robots` in `pipeline.yaml`.
- Inspect `tracking.log` and `rgb_tracking.log` before changing detector code.
- Run `--dry-run` to verify generated paths and commands.
- If automatic merge timing is unsuitable, replace `seconds_difference: auto` or `start_rgb_frame: auto` with a measured value.
- If a parallel result differs near a chunk boundary, increase `--warmup-frames` and compare again with a sequential run.
- If the computer becomes unresponsive, reduce either `batch.max_workers` or the per-experiment `workers` value.

Keep raw input files unchanged and write generated data under `results/`. For the full configuration reference and step-by-step manual commands, consult the [tutorial](doc/tutorial_pogotrack.pdf).

## Maintainers

- [Keivan Amini](https://keivan-amini.github.io/), Sorbonne Université / ESPCI
- [Jérémy Fersula](https://www.isir.upmc.fr/personnel/fersula/), Sorbonne Université
