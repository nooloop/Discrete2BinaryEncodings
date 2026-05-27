# solver_tools

High-performance C++17 simulated annealing solver for Cost Function Networks (CFNs) and their binary-encoded counterparts (QUBO/HUBO/Ising models produced by `encoding_tools`). Outputs one CSV row per problem instance with aggregate statistics, best solution, and per-run energies. Designed for large-scale benchmarking via GNU parallel on multi-core nodes.

## Solver modes

| Mode | Input | Variables | Move type | Delta cost |
|---|---|---|---|---|
| **CFN** | Toulbar2 `.cfn` JSON | Integer (domain d_i) | flip, shift, both | O(degree of variable) |
| **Binary** | `encoding_tools` `.json` | BINARY {0,1} or SPIN {-1,+1} | single-bit flip | O(degree of qubit) |

### CFN mode

Operates directly on the native Cost Function Network with integer-valued variables. Each SA step selects a random variable and proposes a new choice according to the move type:

- **flip**: uniform random choice different from the current one
- **shift**: move to an adjacent choice (x +/- 1), with boundary reflection
- **both**: 50/50 random selection between flip and shift at each step

Delta evaluation computes the energy change in O(degree) by updating only the unary cost and adjacent pairwise tables:

```
dE = unary[i][new] - unary[i][old]
   + sum_{(i,j) in edges} [ pw(new, x_j) - pw(old, x_j) ]
```

### Binary mode

Operates on encoded binary models (QUBO, HUBO, or Ising) produced by `encoding_tools`. Each SA step flips a single qubit and evaluates the energy change in O(degree) using a precomputed adjacency structure:

```
BINARY {0,1}:   dE = (1 - 2*b_q) * sum_{S containing q} C_S * prod_{q' in S\{q}} b_{q'}
SPIN {-1,+1}:   dE = -2 * s_q    * sum_{S containing q} C_S * prod_{q' in S\{q}} s_{q'}
```

The best state encountered during each run is tracked and decoded back to CFN choices for the output.

### Solution decoding

Binary solutions are decoded to CFN variable assignments using the qubit-to-variable mapping stored in the model's `qubit_map`:

| Encoding | Decoding rule |
|---|---|
| **one_hot** | Choice = index of the single 1-bit in the register |
| **domain_wall** | Choice = number of leading 1-bits |
| **\*_binary** (exact, approximate, truncated) | Choice = register value as integer |

For SPIN models, bits are converted via `b = (1 - s) / 2` before reading the register.

### Temperature schedules

Two cooling schedules are available:

- **geometric** (default): `T(t) = T_start * (T_end / T_start)^(t / (steps - 1))`
- **linear**: `T(t) = T_start + (T_end - T_start) * t / (steps - 1)`

### Step count

The total number of SA steps per run is determined by:

```
steps = steps_multiplier * N * D
```

where `N` is the number of source CFN variables and `D` is the maximum cardinality. In binary mode, `N` and `D` are extracted from the model's `qubit_map` using encoding-specific logic:

| Encoding | Cardinality from bits |
|---|---|
| one_hot | d = max bits per variable |
| domain_wall | d = max bits per variable + 1 |
| \*_binary | d = 2^(max bits per variable) |

The step count can also be set explicitly with `--num-steps`, which overrides the multiplier.

## Building

Requires CMake 3.14+ and a C++17 compiler. The only dependency (nlohmann/json 3.11.3) is fetched automatically via CMake FetchContent.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On Windows with Visual Studio Build Tools:
```powershell
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

The resulting binaries are `build/Release/solve_sa` and `build/Release/tests` (or `build/solve_sa` and `build/tests` on single-config generators).

## Usage

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

Each invocation processes one input file, runs all SA runs sequentially, and emits a single CSV row to stdout. This design enables efficient parallelization with GNU parallel.

### Examples

Run SA on a native CFN with shift moves:
```bash
solve_sa --mode cfn --input problems/CFN_N10_D4_rho0.3_uniform_1.cfn \
  --move-type shift --num-runs 100 --steps-multiplier 200 --verbose
```

Run SA on a one-hot encoded QUBO:
```bash
solve_sa --mode binary --input encoded/one_hot/CFN_N10_D4_rho0.3_uniform_1_one_hot.json \
  --T-start 5.0 --T-end 0.001 --num-runs 50 --seed 123
```

Run SA on a truncated-binary Ising model with known ground truth:
```bash
solve_sa --mode binary --input encoded/tb3/CFN_N10_D4_rho0.3_uniform_1_truncated_binary.json \
  --ground-truth 42.5 --num-runs 100
```

Print only the CSV header:
```bash
solve_sa --header
```

## Benchmark script

`scripts/run_benchmark.sh` automates parallel execution across all files in a directory using GNU parallel. Designed for 64-core Intel Ice Lake nodes with 32 parallel tasks (leaving 32 cores dormant to avoid thermal throttling).

```bash
./scripts/run_benchmark.sh \
  --input-dir encoded/one_hot/ \
  --output-csv results/oh_results.csv \
  --mode binary \
  --num-runs 100 \
  --steps-multiplier 200 \
  --jobs 32 \
  --verbose
```

```
run_benchmark.sh [options]

Required:
  --input-dir DIR       Directory containing .cfn or .json files
  --output-csv FILE     Path for output CSV

Optional:
  --solver PATH         Path to solve_sa binary        (default: ./solve_sa)
  --mode MODE           cfn | binary                   (default: binary)
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
  --verbose             Print progress
```

The script writes the CSV header first, then dispatches one `solve_sa` invocation per file. Each worker appends its single-line CSV output to the result file.

## Output format

A single CSV file with one row per problem instance. Columns:

- **Instance metadata**: `problem_name`, `source_cfn`, `solver_mode`, `encoding`, `variable_type`, `num_qubits`, `num_variables`, `max_cardinality`, `edge_density`, `distribution`
- **SA configuration**: `schedule`, `move_type`, `T_start`, `T_end`, `num_steps`, `num_runs`, `seed`
- **Energy statistics**: `best_energy`, `mean_energy`, `std_energy`, `median_energy`, `num_optimal`
- **Timing**: `total_runtime_s`, `mean_time_per_run_s`
- **Solution**: `best_solution` (CFN choices as `[X0,X1,...,XN]`)
- **Raw data**: `per_run_energies` (all best energies as `[E0,E1,...,ER]`)

In CFN mode, `best_solution` contains integer variable assignments directly. In binary mode, the best binary state is decoded to CFN choices using the encoding's decoding rule. The `num_optimal` column counts how many runs found the known ground truth (requires `--ground-truth`); it is `NA` if no ground truth is provided.

## Input formats

### CFN files (CFN mode)

Toulbar2 JSON format:

```json
{
  "problem": {"name": "example"},
  "variables": {"x0": 3, "x1": 4, "x2": 3},
  "functions": {
    "f0": {"scope": ["x0"], "costs": [1.0, 2.0, 3.0]},
    "f1": {"scope": ["x1"], "costs": [0.5, 1.5, 2.5, 3.5]},
    "f01": {"scope": ["x0", "x1"], "costs": [0.1, 0.2, ..., 1.2]}
  }
}
```

Costs are flattened row-major with the rightmost scope variable varying fastest. Instance metadata (N, D, rho, distribution) is extracted from filenames matching the pattern `CFN_N<N>_D<D>_rho<rho>_<dist>_<num>.cfn`.

### Encoded model files (binary mode)

JSON files produced by `encoding_tools`:

```json
{
  "encoding": "one_hot",
  "variable_type": "BINARY",
  "num_logical_qubits": 16,
  "num_auxiliary_qubits": 0,
  "offset": 42.5,
  "source_cfn": "example",
  "qubit_map": {
    "0": {"variable": 0, "variable_name": "x0", "bit": 0},
    "1": {"variable": 0, "variable_name": "x0", "bit": 1}
  },
  "terms": {
    "0": -1.5,
    "0,1": 2.3,
    "2,5": -0.7
  }
}
```

Term keys are comma-separated qubit indices. The `qubit_map` maps each logical qubit to its source variable and bit position, enabling solution decoding and step-count calibration.

## Tests

Unit tests verify all solver modes and encoding types on a small 2-variable, 4-choice CFN with known ground state E=1.0 at [3,1]. Build and run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target tests
./build/tests    # or build\Release\tests.exe on Windows
```

Test coverage:

| Test | Encoding | Solver | Expectation |
|---|---|---|---|
| CFN flip/shift/both | native | CFN | Exact ground state E=1.0 at [3,1] |
| One-hot | one_hot | Binary | Exact ground state E=1.0 at [3,1] |
| Domain-wall | domain_wall | Binary | Exact ground state E=1.0 at [3,1] |
| Exact-binary | exact_binary | Binary | Exact ground state E=1.0 at [3,1] |
| Exact-binary (quad) | exact_binary (quadratized) | Binary | Exact ground state E=1.0 at [3,1] |
| Approximate-binary | approximate_binary | Binary | Best decoded CFN energy (cost-approximate) |
| Truncated k=2 | truncated_binary | Binary | Best decoded CFN energy (cost-approximate) |
| Truncated k=3 | truncated_binary | Binary | Best decoded CFN energy (cost-approximate) |
| Truncated k=3 (quad) | truncated_binary (quadratized) | Binary | Best decoded CFN energy (cost-approximate) |

Cost-preserving encodings (one-hot, domain-wall, exact-binary) must find the exact CFN ground state. Cost-approximate encodings (approximate-binary, truncated-binary) decode the best binary solution to CFN choices and verify both the encoded energy and the native CFN energy.

## Project structure

```
solver_tools/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp                      # CLI entry point, argument parsing, run orchestration
│   ├── baseline/
│   │   ├── sa_types.hpp              # SATimer, temperature schedules, SAParams, RunResult, AggregateResult
│   │   └── output.hpp                # CSV header and row formatting
│   ├── solvers/
│   │   ├── sa_cfn.hpp                # CFN simulated annealing (flip/shift/both moves)
│   │   └── sa_binary.hpp             # Binary SA (single-bit flip, delta evaluation)
│   └── utilities/
│       ├── parse_cfn.hpp             # CFN parser (Toulbar2 JSON), CFNModel, delta_energy
│       └── parse_model.hpp           # Binary model parser, BinaryModel, delta_energy, decode_to_cfn
├── scripts/
│   └── run_benchmark.sh              # GNU parallel benchmark driver
└── tests/
    ├── tests.cpp                     # Unit tests (all solver modes and encodings)
    ├── test_2x4.cfn                  # Source CFN for test model generation
    └── test_models/                  # Pre-generated encoded models for tests
        ├── test_2x4_one_hot.json
        ├── test_2x4_domain_wall.json
        ├── test_2x4_exact_binary.json
        ├── test_2x4_exact_binary_quad.json
        ├── test_2x4_approximate_binary.json
        ├── test_2x4_truncated_binary_k2.json
        ├── test_2x4_truncated_binary_k3.json
        └── test_2x4_truncated_binary_k3_quad.json
```

## Dependencies

- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14+**
- **[nlohmann/json](https://github.com/nlohmann/json) 3.11.3** (fetched automatically)
- **GNU parallel** (for benchmark script only)
