#!/usr/bin/env bash
# ============================================================================
# run_benchmark_qa.sh  --  D-Wave quantum annealing benchmark
#
# Runs solve_qa on every encoded JSON file in an input directory.
# Each invocation processes one file, submits to D-Wave, and emits
# one aggregate CSV row.  The script collects them into an output CSV.
#
# Note: D-Wave QPU is a shared resource.  Unlike SA, running many
# parallel jobs doesn't improve throughput — jobs are queued on the
# QPU.  Default --jobs is 1 (sequential).  Increase to overlap
# embedding computation with QPU wait times.
#
# Dependencies: GNU parallel, solve_qa (in PATH or via --solver),
#               Python 3 with dwave-ocean-sdk
# ============================================================================

set -euo pipefail

# ---- defaults ----
SOLVER="./solve_qa"
SOLVER_NAME=""
CFN_DIR=""
ANNEALING_TIME="20"
NUM_READS="1000"
DELTA_MAX="0.1"
NO_INHOMOGENEOUS=""
JOBS="1"
GROUND_TRUTH=""
TOLERANCE="1e-6"
VERBOSE=""
INPUT_DIR=""
OUTPUT_CSV=""
PYTHON="python3"

usage() {
    cat <<'USAGE'
Usage: run_benchmark_qa.sh [options]

Required:
  --input-dir DIR       Directory containing encoded .json files
  --output-csv FILE     Path for output CSV
  --cfn-dir DIR         Directory containing source .cfn files
  --solver-name NAME    D-Wave solver name (e.g. Advantage_system6.4)

Optional:
  --solver PATH         Path to solve_qa binary          (default: ./solve_qa)
  --annealing-time F    Annealing time in microseconds    (default: 20)
  --num-reads N         Number of reads (shots)           (default: 1000)
  --delta-max FLOAT     Max anneal offset magnitude       (default: 0.1)
  --no-inhomogeneous    Disable inhomogeneous driving
  --jobs N              Parallel jobs                     (default: 1)
  --ground-truth E      Known optimum for success count
  --tolerance FLOAT     Tolerance for ground truth        (default: 1e-6)
  --python CMD          Python interpreter                (default: python3)
  --verbose             Print progress
USAGE
    exit 1
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)           SOLVER="$2";           shift 2;;
        --solver-name)      SOLVER_NAME="$2";      shift 2;;
        --cfn-dir)          CFN_DIR="$2";          shift 2;;
        --input-dir)        INPUT_DIR="$2";        shift 2;;
        --output-csv)       OUTPUT_CSV="$2";       shift 2;;
        --annealing-time)   ANNEALING_TIME="$2";   shift 2;;
        --num-reads)        NUM_READS="$2";        shift 2;;
        --delta-max)        DELTA_MAX="$2";        shift 2;;
        --no-inhomogeneous) NO_INHOMOGENEOUS="--no-inhomogeneous"; shift 1;;
        --jobs)             JOBS="$2";             shift 2;;
        --ground-truth)     GROUND_TRUTH="$2";     shift 2;;
        --tolerance)        TOLERANCE="$2";        shift 2;;
        --python)           PYTHON="$2";           shift 2;;
        --verbose)          VERBOSE="--verbose";   shift 1;;
        --help|-h)          usage;;
        *)                  echo "Unknown option: $1"; usage;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_CSV" || -z "$CFN_DIR" || -z "$SOLVER_NAME" ]]; then
    echo "Error: --input-dir, --output-csv, --cfn-dir, and --solver-name are required."
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory '$INPUT_DIR' does not exist."
    exit 1
fi

if [[ ! -d "$CFN_DIR" ]]; then
    echo "Error: CFN directory '$CFN_DIR' does not exist."
    exit 1
fi

# ---- collect input files ----
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: no .json files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} .json files in $INPUT_DIR"
echo "D-Wave solver: $SOLVER_NAME"
echo "Annealing time: ${ANNEALING_TIME} us, reads: ${NUM_READS}"
echo "Running with $JOBS parallel job(s)"

# ---- write CSV header ----
"$SOLVER" --header > "$OUTPUT_CSV"

# ---- build solver command template ----
SOLVER_ARGS=(
    --cfn-dir "$CFN_DIR"
    --solver "$SOLVER_NAME"
    --annealing-time "$ANNEALING_TIME"
    --num-reads "$NUM_READS"
    --delta-max "$DELTA_MAX"
    --tolerance "$TOLERANCE"
    --python "$PYTHON"
)

if [[ -n "$NO_INHOMOGENEOUS" ]]; then
    SOLVER_ARGS+=($NO_INHOMOGENEOUS)
fi

if [[ -n "$GROUND_TRUTH" ]]; then
    SOLVER_ARGS+=(--ground-truth "$GROUND_TRUTH")
fi

if [[ -n "$VERBOSE" ]]; then
    SOLVER_ARGS+=($VERBOSE)
fi

# ---- run in parallel ----
printf '%s\n' "${FILES[@]}" | \
    parallel -j "$JOBS" --bar \
        "$SOLVER" "${SOLVER_ARGS[@]}" --input {} \
    >> "$OUTPUT_CSV"

echo "Done. Results written to $OUTPUT_CSV"
