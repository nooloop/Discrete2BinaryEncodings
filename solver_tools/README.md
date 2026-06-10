# solver_tools

High-performance C++17 solvers for Cost Function Networks (CFNs) and their binary-encoded counterparts (QUBO/HUBO/Ising models produced by `encoding_tools`). Includes simulated annealing (`solve_sa`) and D-Wave quantum annealing (`solve_qa`) with inhomogeneous transverse-field driving. Outputs one CSV row per problem instance with aggregate statistics, best solution, and per-run energies. Designed for large-scale benchmarking via GNU parallel and disBatch on multi-core Slurm clusters.

## Table of Contents

- [Contents](#contents)
- [Usage](#usage)
  - [Building](#building)
  - [Simulated annealing (solve_sa)](#simulated-annealing-solve_sa)
  - [D-Wave quantum annealing (solve_qa)](#d-wave-quantum-annealing-solve_qa)
  - [Batch benchmarking](#batch-benchmarking)
  - [Input formats](#input-formats)
  - [Output formats](#output-formats)
  - [Tests](#tests)
- [Dependencies](#dependencies)

## Contents

```
solver_tools/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp                      # SA CLI entry point (solve_sa)
│   ├── main_qa.cpp                   # QA CLI entry point (solve_qa)
│   ├── baseline/
│   │   ├── sa_types.hpp              # SATimer, temperature schedules, SAParams, RunResult, AggregateResult
│   │   ├── qa_types.hpp              # QAParams, DWaveSample, DWaveTiming, DWaveResults, QAResult
│   │   ├── output.hpp                # SA CSV header and row formatting
│   │   └── output_qa.hpp             # QA CSV header and row formatting
│   ├── solvers/
│   │   ├── sa_cfn.hpp                # CFN simulated annealing (flip/shift/both moves)
│   │   ├── sa_binary.hpp             # Binary SA (single-bit flip, delta evaluation)
│   │   └── qa_binary.hpp             # D-Wave QA (effective fields, offsets, submission, decoding)
│   └── utilities/
│       ├── parse_cfn.hpp             # CFN parser (Toulbar2 JSON), CFNModel, delta_energy
│       ├── parse_model.hpp           # Binary model parser, BinaryModel, delta_energy, decode_to_cfn
│       └── dwave_template.hpp        # Embedded Python D-Wave submission script template
├── scripts/
│   ├── run_benchmark.sh              # GNU parallel SA benchmark driver
│   ├── run_benchmark_qa.sh           # GNU parallel QA benchmark driver
│   ├── slurm_qa_benchmark.slurm     # Slurm + disBatch QA benchmark (any encoding)
│   ├── run_qa_one.sh                 # disBatch per-problem wrapper (scratch isolation)
│   └── dwave_submit_template.py      # Reference copy of the D-Wave submission template
└── tests/
    ├── tests.cpp                     # SA unit tests (all solver modes and encodings)
    ├── tests_qa.cpp                  # QA unit + D-Wave integration tests (all encodings)
    ├── test_2x4.cfn                  # Source CFN for test model generation
    └── test_models/                  # Pre-generated encoded models for tests
```

### Solver modes

| Mode | Binary | Input | Variables | Move type |
|---|---|---|---|---|
| **CFN** | `solve_sa` | Toulbar2 `.cfn` JSON | Integer (domain d_i) | flip, shift, both |
| **Binary SA** | `solve_sa` | `encoding_tools` `.json` | BINARY {0,1} or SPIN {-1,+1} | single-bit flip |
| **D-Wave QA** | `solve_qa` | `encoding_tools` `.json` (QUBO/Ising) | BINARY {0,1} or SPIN {-1,+1} | quantum anneal |

### Solution decoding

Binary solutions are decoded to CFN variable assignments using the model's `qubit_map`:

| Encoding | Decoding rule |
|---|---|
| **one_hot** | Choice = index of the single 1-bit in the register |
| **domain_wall** | Choice = number of leading 1-bits |
| **\*_binary** (exact, approximate, truncated) | Choice = register value as integer |

For SPIN models, bits are converted via `b = (1 - s) / 2` before reading the register.

### Inhomogeneous transverse-field driving (D-Wave)

`solve_qa` computes per-qubit effective fields using the O(N_i) recurrence from Adame et al. (2020), normalizes them to anneal offsets, and maps them through the minor embedding to physical qubits. Strongly-coupled qubits are delayed; weakly-coupled qubits are advanced.

## Usage

### Building

Requires CMake 3.14+ and a C++17 compiler. The only C++ dependency (nlohmann/json 3.11.3) is fetched automatically.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Produces `solve_sa`, `solve_qa`, `tests`, and `tests_qa`.

`solve_qa` additionally requires Python 3 with `dwave-ocean-sdk` at runtime:
```bash
pip install dwave-ocean-sdk
```

### Simulated annealing (solve_sa)

```
solve_sa [options]

Required:
  --mode MODE             cfn | binary
  --input FILE            Path to .cfn or encoded .json file

Annealing schedule:
  --schedule TYPE         geometric | linear           (default: geometric)
  --T-start FLOAT         Starting temperature         (default: 10.0)
  --T-end FLOAT           Final temperature            (default: 0.01)

Move type (CFN mode only):
  --move-type TYPE        flip | shift | both          (default: flip)

Run configuration:
  --num-runs N            Number of independent runs   (default: 100)
  --num-steps N           Total SA steps per run       (overrides multiplier)
  --steps-multiplier M    steps = M * N * D            (default: 100)
  --seed N                Base RNG seed                (default: 42)

Optional:
  --ground-truth E        Known optimum for success counting
  --tolerance FLOAT       Tolerance for ground truth   (default: 1e-6)
  --header                Print CSV header and exit
  --verbose               Print progress to stderr
```

Each invocation processes one file, runs all SA trajectories, and emits a single CSV row to stdout.

#### SA examples

```bash
# Native CFN with shift moves
solve_sa --mode cfn --input problem.cfn --move-type shift --num-runs 100

# One-hot encoded QUBO
solve_sa --mode binary --input problem_one_hot.json --num-runs 50

# Print CSV header only
solve_sa --header
```

### D-Wave quantum annealing (solve_qa)

```
solve_qa [options]

Required:
  --input FILE            Path to encoded .json model (QUBO/Ising only)
  --cfn-dir DIR           Directory containing source .cfn files
  --solver NAME           D-Wave solver name

D-Wave parameters:
  --annealing-time FLOAT  Annealing time in microseconds     (default: 20)
  --num-reads N           Number of reads (shots)            (default: 1000)

Inhomogeneous driving:
  --delta-max FLOAT       Max anneal offset magnitude        (default: 0.1)
  --no-inhomogeneous      Disable inhomogeneous driving

Optional:
  --ground-truth E        Known optimum for success counting
  --tolerance FLOAT       Tolerance for ground truth         (default: 1e-6)
  --python CMD            Python interpreter                 (default: python3)
  --header                Print CSV header and exit
  --verbose               Print progress to stderr
```

Each invocation computes anneal offsets (C++), submits to D-Wave (Python), decodes bitstrings to CFN choices, evaluates under the original CFN, and emits a CSV row. The source CFN is located via the model's `source_cfn` field in `--cfn-dir`.

#### QA examples

```bash
# Submit one-hot QUBO to D-Wave
solve_qa --input problem_one_hot.json --cfn-dir cfns/ \
    --solver Advantage2_system1 --num-reads 1000

# Truncated-binary without inhomogeneous driving
solve_qa --input problem_truncated_binary.json --cfn-dir cfns/ \
    --solver Advantage2_system1 --no-inhomogeneous
```

### Batch benchmarking

**SA (GNU parallel):**
```bash
./scripts/run_benchmark.sh \
    --input-dir encoded/one_hot/ --output-csv results.csv \
    --mode binary --num-runs 100 --jobs 32
```

**QA (Slurm + disBatch):** Accepts an input directory of `.jsonl` files (one encoded model per line) and an output directory. Splits JSONL on node-local scratch, runs 32 parallel D-Wave workers, and copies the final CSV to the output directory in a single write. Restartable.

```bash
sbatch scripts/slurm_qa_benchmark.slurm <INPUT_DIR> <OUTPUT_DIR>
```

**QA (GNU parallel):** Simpler alternative for environments without disBatch.

```bash
./scripts/run_benchmark_qa.sh \
    --input-dir encoded/oh/ --output-csv results.csv \
    --cfn-dir cfns/ --solver-name Advantage2_system1
```

### Input formats

**CFN files** (Toulbar2 JSON, for `solve_sa --mode cfn`):
```json
{
  "problem": {"name": "example"},
  "variables": {"x0": 3, "x1": 4},
  "functions": {
    "f0": {"scope": ["x0"], "costs": [1.0, 2.0, 3.0]},
    "f01": {"scope": ["x0", "x1"], "costs": [0.1, 0.2, ..., 1.2]}
  }
}
```

**Encoded model files** (produced by `encoding_tools`, for `solve_sa --mode binary` and `solve_qa`):
```json
{
  "encoding": "one_hot",
  "variable_type": "BINARY",
  "num_logical_qubits": 16,
  "num_auxiliary_qubits": 0,
  "offset": 42.5,
  "source_cfn": "example",
  "qubit_map": {"0": {"variable": 0, "variable_name": "x0", "bit": 0}},
  "terms": {"0": -1.5, "0,1": 2.3}
}
```

### Output formats

**SA CSV** (26 columns): instance metadata, SA configuration, energy statistics (best/mean/std/median), timing, decoded best solution, per-run energies.

**QA CSV** (44 columns): the same 26 SA-compatible columns (with `solver_mode=dwave`) plus D-Wave configuration (`solver_name`, `annealing_time_us`, `delta_max`), D-Wave timing (`inhomog_setup_time_s`, `embedding_time_s`, `qpu_access_time_us`, `qpu_sampling_time_us`, `qpu_programming_time_us`), CFN evaluation (`best_cfn_energy`, `num_feasible`, `num_best_cfn`), and embedding statistics (`emb_num_physical_qubits`, `emb_chain_length_avg/median/var`, `emb_chain_breaks_avg/median/var`).

### Tests

```bash
# SA tests (no external dependencies)
./build/tests

# QA unit tests (no D-Wave needed)
./build/tests_qa --test-dir ./tests

# QA D-Wave integration tests (all 10 encoding variants)
./build/tests_qa --test-dir ./tests --dwave --solver Advantage2_system1 --python ~/dwave-env/bin/python
```

SA tests verify ground-state recovery for all encodings on a 2-variable, 4-choice CFN. QA tests cover the full pipeline (effective fields, anneal offsets, script generation, mock D-Wave results, decode, CFN evaluation, embedding statistics) for one-hot, domain-wall, exact-binary (quadratized), approximate-binary, truncated-binary k=2, and truncated-binary k=3 (quadratized), each in both natural/unsorted and gray/boltzmann orderings where applicable.

## Dependencies

- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14+**
- **[nlohmann/json](https://github.com/nlohmann/json) 3.11.3** (fetched automatically)
- **Python 3** with **[dwave-ocean-sdk](https://docs.ocean.dwavesys.com/)** (for `solve_qa` only)
- **GNU parallel** or **disBatch** (for batch benchmarking scripts only)
