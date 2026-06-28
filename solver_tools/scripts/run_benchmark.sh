#!/usr/bin/env bash

set -euo pipefail

# ---- defaults ----
SOLVER="./solve_sa"
MODE="binary"
SCHEDULE="geometric"
MOVE_TYPE="flip"
T_START="10.0"
T_END="0.01"
NUM_RUNS="100"
STEPS_MULTIPLIER="100"
NUM_STEPS=""
SEED="42"
JOBS="32"
GROUND_TRUTH=""
TOLERANCE="1e-6"
VERBOSE=""
INPUT_DIR=""
OUTPUT_CSV=""
CFN_DIR=""

usage() {
    cat <<'USAGE'
Usage: run_benchmark.sh [options]

Required:
  --input-dir DIR       Directory containing .cfn or .json files
  --output-csv FILE     Path for output CSV

Optional:
  --solver PATH         Path to solve_sa binary         (default: ./solve_sa)
  --mode MODE           cfn | binary                    (default: binary)
  --schedule TYPE       geometric | linear              (default: geometric)
  --move-type TYPE      flip | shift | both (CFN only)  (default: flip)
  --T-start FLOAT       Starting temperature            (default: 10.0)
  --T-end FLOAT         Final temperature               (default: 0.01)
  --num-runs N          Runs per problem                (default: 100)
  --steps-multiplier M  steps = M * N * D               (default: 100)
  --num-steps N         Override step count (ignores multiplier)
  --seed N              Base RNG seed                   (default: 42)
  --jobs N              Parallel jobs                   (default: 32)
  --ground-truth E      Known optimum for success count
  --tolerance FLOAT     Tolerance for ground truth      (default: 1e-6)
  --cfn-dir DIR         Source .cfn dir (binary mode): decode best state and
                        record best_cfn_energy / num_feasible / num_best_cfn
  --verbose             Print progress
USAGE
    exit 1
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)           SOLVER="$2";           shift 2;;
        --mode)             MODE="$2";             shift 2;;
        --input-dir)        INPUT_DIR="$2";        shift 2;;
        --output-csv)       OUTPUT_CSV="$2";       shift 2;;
        --schedule)         SCHEDULE="$2";         shift 2;;
        --move-type)        MOVE_TYPE="$2";        shift 2;;
        --T-start)          T_START="$2";          shift 2;;
        --T-end)            T_END="$2";            shift 2;;
        --num-runs)         NUM_RUNS="$2";         shift 2;;
        --steps-multiplier) STEPS_MULTIPLIER="$2"; shift 2;;
        --num-steps)        NUM_STEPS="$2";        shift 2;;
        --seed)             SEED="$2";             shift 2;;
        --jobs)             JOBS="$2";             shift 2;;
        --ground-truth)     GROUND_TRUTH="$2";     shift 2;;
        --tolerance)        TOLERANCE="$2";        shift 2;;
        --cfn-dir)          CFN_DIR="$2";          shift 2;;
        --verbose)          VERBOSE="--verbose";   shift 1;;
        --help|-h)          usage;;
        *)                  echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_CSV" ]]; then
    echo "Error: --input-dir and --output-csv are required."
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# ---- determine file extension ----
if [[ "$MODE" == "cfn" ]]; then
    EXT="cfn"
else
    EXT="json"
fi

# ---- collect input files ----
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.$EXT" -type f | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: no .$EXT files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} .$EXT files in $INPUT_DIR"
echo "Running $NUM_RUNS SA runs per file, $JOBS parallel jobs"

# ---- write CSV header ----
"$SOLVER" --header > "$OUTPUT_CSV"

# ---- build solver command template ----
SOLVER_ARGS=(
    --mode "$MODE"
    --schedule "$SCHEDULE"
    --move-type "$MOVE_TYPE"
    --T-start "$T_START"
    --T-end "$T_END"
    --num-runs "$NUM_RUNS"
    --steps-multiplier "$STEPS_MULTIPLIER"
    --seed "$SEED"
    --tolerance "$TOLERANCE"
)

if [[ -n "$NUM_STEPS" ]]; then
    SOLVER_ARGS+=(--num-steps "$NUM_STEPS")
fi

if [[ -n "$GROUND_TRUTH" ]]; then
    SOLVER_ARGS+=(--ground-truth "$GROUND_TRUTH")
fi

if [[ -n "$CFN_DIR" ]]; then
    SOLVER_ARGS+=(--cfn-dir "$CFN_DIR")
fi

if [[ -n "$VERBOSE" ]]; then
    SOLVER_ARGS+=($VERBOSE)
fi

# ---- run in parallel ----
# GNU parallel distributes files across $JOBS workers.
# Each worker runs solve_sa on one file; output (one CSV row) goes to stdout.

printf '%s\n' "${FILES[@]}" | \
    parallel -j "$JOBS" --bar \
        "$SOLVER" "${SOLVER_ARGS[@]}" --input {} \
    >> "$OUTPUT_CSV"

echo "Done. Results written to $OUTPUT_CSV"
