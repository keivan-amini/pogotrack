#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# Parallel batch analysis for PogoTrack using tmux
#
# This script creates four tmux panes.
#
# Each pane runs one worker. Each worker processes its assigned
# experiments sequentially:
#
#   1. Normal tracking
#   2. RGB-ID tracking
#
# Every experiment receives an independent YAML configuration:
#
#   config/multiprocess/config_<date>_<experiment>.yaml
#
# This makes the analyses safe to run in parallel.
# ============================================================

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

DATA_DIR="data"
RESULTS_DIR="results"
BASE_CONFIG_FILE="config/default.yaml"
MULTIPROCESS_CONFIG_DIR="config/multiprocess"

# Maximum number of simultaneous analysis pipelines.
# This script always opens four tmux panes.
MAX_WORKERS=4

# Name of the tmux session that will be created.
TMUX_SESSION="pogotrack_batch"

# ------------------------------------------------------------
# Videos and corresponding N_POGO values
#
# Each video entry has this format:
#
#   "<date-folder>/<video-file>.mp4"
#
# The associated RGB-ID video must be:
#
#   data/<date-folder>/id_<video-file>.mp4
#
# Each date folder must contain:
#
#   data/<date-folder>/bkg.bmp
# ------------------------------------------------------------

VIDEOS=(
  "18-09-26/alpha10-4_T100_v2.mp4"
  #"17-09-26/alpha10-4_T10_v4.mp4"
)

ROBOTS=(
  64
  #63
)

# ============================================================
# Helper functions
# ============================================================

check_file_exists() {
  local file_path="$1"

  if [[ ! -f "$file_path" ]]; then
    echo "ERROR: Required file not found:"
    echo "  $file_path"
    exit 1
  fi
}

set_yaml_value() {
  local config_file="$1"
  local key="$2"
  local value="$3"

  sed -i.bak -E \
    "s|^([[:space:]]*${key}:[[:space:]]*).*|\1${value}|" \
    "$config_file"

  rm -f "${config_file}.bak"
}

run_experiment() {
  local relative_video="$1"
  local n_robots="$2"
  local worker_id="$3"

  local date_folder
  local video_filename
  local experiment_name

  local normal_video
  local rgb_video
  local background

  local output_dir
  local normal_output
  local rgb_output

  local config_name
  local config_file

  date_folder="$(dirname "$relative_video")"
  video_filename="$(basename "$relative_video")"
  experiment_name="${video_filename%.mp4}"

  # ----------------------------------------------------------
  # Input paths
  # ----------------------------------------------------------

  normal_video="${DATA_DIR}/${relative_video}"
  rgb_video="${DATA_DIR}/${date_folder}/id_${video_filename}"
  background="${DATA_DIR}/${date_folder}/bkg.bmp"

  # ----------------------------------------------------------
  # Output paths
  # ----------------------------------------------------------

  output_dir="${RESULTS_DIR}/${date_folder}/${experiment_name}"

  normal_output="${output_dir}/${experiment_name}.csv"
  rgb_output="${output_dir}/rgb_${experiment_name}.csv"

  # Make the config filename unique even if two date directories
  # contain videos with the same filename.
  #
  # Example:
  # config/multiprocess/config_04-09-26_alpha10-2_T10_v2.yaml

  config_name="config_${date_folder//\//_}_${experiment_name}.yaml"
  config_file="${MULTIPROCESS_CONFIG_DIR}/${config_name}"

  # ----------------------------------------------------------
  # Check required input files
  # ----------------------------------------------------------

  check_file_exists "$normal_video"
  check_file_exists "$rgb_video"
  check_file_exists "$background"

  mkdir -p "$output_dir"
  mkdir -p "$MULTIPROCESS_CONFIG_DIR"

  # Each experiment starts from an independent copy of default.yaml.
  cp "$BASE_CONFIG_FILE" "$config_file"

  echo
  echo "============================================================"
  echo "Worker: ${worker_id}"
  echo "Experiment: ${experiment_name}"
  echo "Date folder: ${date_folder}"
  echo "N_POGO: ${n_robots}"
  echo "Config file: ${config_file}"
  echo "============================================================"

  # ----------------------------------------------------------
  # 1. Standard trajectory tracking
  # ----------------------------------------------------------

  echo
  echo "[1/2] Standard tracking"
  echo "Setting N_POGO: ${n_robots}"
  echo "Setting RGB_ID_ANALYSIS: False"

  set_yaml_value "$config_file" "N_POGO" "$n_robots"
  set_yaml_value "$config_file" "RGB_ID_ANALYSIS" "False"

  "$PYTHON" -m main \
    --video "$normal_video" \
    --background "$background" \
    --output "$normal_output" \
    --config "$config_file"

  # ----------------------------------------------------------
  # 2. RGB-ID tracking
  # ----------------------------------------------------------

  echo
  echo "[2/2] RGB-ID tracking"
  echo "Setting N_POGO: ${n_robots}"
  echo "Setting RGB_ID_ANALYSIS: True"

  set_yaml_value "$config_file" "N_POGO" "$n_robots"
  set_yaml_value "$config_file" "RGB_ID_ANALYSIS" "True"

  "$PYTHON" -m main \
    --video "$rgb_video" \
    --background "$background" \
    --output "$rgb_output" \
    --config "$config_file"

  echo
  echo "============================================================"
  echo "Worker ${worker_id} completed:"
  echo "${date_folder}/${experiment_name}"
  echo "============================================================"
}

run_worker() {
  local worker_id="$1"
  local i
  local found_work=0

  echo
  echo "============================================================"
  echo "PogoTrack worker ${worker_id} started"
  echo "============================================================"

  # Each worker handles experiments whose index satisfies:
  #
  # experiment_index modulo MAX_WORKERS = worker_id
  #
  # With three videos:
  #
  # Worker 0 → first video
  # Worker 1 → second video
  # Worker 2 → third video
  # Worker 3 → idle
  #
  # With more than four videos, workers automatically receive
  # further experiments after finishing their first assignment.

  for i in "${!VIDEOS[@]}"; do
    if (( i % MAX_WORKERS == worker_id )); then
      found_work=1

      run_experiment \
        "${VIDEOS[$i]}" \
        "${ROBOTS[$i]}" \
        "$worker_id"
    fi
  done

  if [[ "$found_work" -eq 0 ]]; then
    echo
    echo "Worker ${worker_id} has no assigned experiment."
    echo "This pane is idle."
  fi

  echo
  echo "============================================================"
  echo "Worker ${worker_id} finished."
  echo "============================================================"
}

# ============================================================
# Preliminary validation
# ============================================================

if [[ "${#VIDEOS[@]}" -ne "${#ROBOTS[@]}" ]]; then
  echo "ERROR: VIDEOS and ROBOTS must contain the same number of entries."
  echo "VIDEOS: ${#VIDEOS[@]}"
  echo "ROBOTS: ${#ROBOTS[@]}"
  exit 1
fi

if [[ ! -f "$BASE_CONFIG_FILE" ]]; then
  echo "ERROR: Base configuration file not found:"
  echo "  $BASE_CONFIG_FILE"
  exit 1
fi

if ! "$PYTHON" -c "import tqdm" >/dev/null 2>&1; then
  echo "ERROR: tqdm is not installed for:"
  echo "  $PYTHON"
  echo
  echo "Install it with:"
  echo "  ${PYTHON} -m pip install tqdm"
  exit 1
fi

# ============================================================
# Worker mode
#
# Internal mode used by tmux panes:
#
# ./batch_analysis_tmux.sh --worker 0
# ============================================================

if [[ "${1:-}" == "--worker" ]]; then
  worker_id="${2:-}"

  if [[ ! "$worker_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Worker ID must be an integer."
    exit 1
  fi

  run_worker "$worker_id"
  exit 0
fi

# ============================================================
# Controller mode
#
# Creates the tmux session and launches four workers.
# ============================================================

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is not installed."
  echo
  echo "Install it on macOS with:"
  echo "  brew install tmux"
  exit 1
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: A tmux session named '${TMUX_SESSION}' already exists."
  echo
  echo "Attach to it with:"
  echo "  tmux attach-session -t ${TMUX_SESSION}"
  echo
  echo "Or terminate it with:"
  echo "  tmux kill-session -t ${TMUX_SESSION}"
  exit 1
fi

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

echo
echo "============================================================"
echo "PogoTrack parallel batch launcher"
echo "============================================================"
echo "Experiments: ${#VIDEOS[@]}"
echo "Maximum parallel workers: ${MAX_WORKERS}"
echo "tmux session: ${TMUX_SESSION}"
echo "============================================================"
echo

# Create a detached tmux session with one empty pane.
tmux new-session -d \
  -s "$TMUX_SESSION" \
  -n "tracking"

# Split the window into four panes total.
tmux split-window -t "${TMUX_SESSION}:0" -h
tmux split-window -t "${TMUX_SESSION}:0" -v
tmux split-window -t "${TMUX_SESSION}:0" -v

# Arrange the four panes as a tiled 2 × 2 layout.
tmux select-layout -t "${TMUX_SESSION}:0" tiled

# Start one worker in each pane.
for worker_id in 0 1 2 3; do
  worker_command="bash $(printf '%q' "$SCRIPT_PATH") --worker ${worker_id}; worker_status=\$?; echo; echo \"Worker ${worker_id} exited with status \${worker_status}\"; exec bash"

  tmux send-keys \
    -t "${TMUX_SESSION}:0.${worker_id}" \
    "$worker_command" \
    C-m
done

echo "Launching tmux session..."
echo
echo "Useful tmux shortcuts:"
echo "  Ctrl+b then arrow key    Move between panes"
echo "  Ctrl+b then z            Zoom/unzoom current pane"
echo "  Ctrl+b then d            Detach and leave jobs running"
echo
echo "To reattach later:"
echo "  tmux attach-session -t ${TMUX_SESSION}"
echo

# Attach to the newly created session.
tmux attach-session -t "$TMUX_SESSION"
