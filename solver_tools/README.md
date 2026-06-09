# solver_tools

High-performance C++17 solvers for Cost Function Networks (CFNs) and their binary-encoded counterparts (QUBO/HUBO/Ising models produced by `encoding_tools`). Includes simulated annealing (`solve_sa`) and D-Wave quantum annealing (`solve_qa`) with inhomogeneous transverse-field driving. Outputs one CSV row per problem instance with aggregate statistics, best solution, and per-run energies. Designed for large-scale benchmarking via GNU parallel on multi-core nodes.

## Solver modes

| Mode | Binary | Input | Variables | Move type | Delta cost |
|---|---|---|---|---|---|
| **CFN** | `solve_sa` | Toulbar2 `.cfn` JSON | Integer (domain d_i) | flip, shift, both | O(degree of variable) |
| **Binary SA** | `solve_sa` | `encoding_tools` `.json` | BINARY {0,1} or SPIN {-1,+1} | single-bit flip | O(degree of qubit) |
| **D-Wave QA** | `solve_qa` | `encoding_tools` `.json` (QUBO/Ising only) | BINARY {0,1} or SPIN {-1,+1} | quantum anneal | N/A |

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

### D-Wave QA mode

Submits encoded QUBO/Ising models to a D-Wave quantum annealer via the Ocean SDK. Only degree-2 models are supported (use `--quadratize` in `encoding_tools` for HUBOs). The workflow is:

1. **Effective field computation** (C++, O(N^2)): For each logical qubit, the mean effective field is computed using the O(N_i) recurrence from Adame et al. (2020):

   For SPIN {-1,+1}:
   ```
   f = |h_i|
   for each coupling J: f = (|f + J| + |f - J|) / 2
   eff_field = f
   ```
   For BINARY {0,1}:
   ```
   T = sum of couplings to qubit i
   f = |2*h_i + T|
   for each coupling J: f = (|f + J| + |f - J|) / 2
   eff_field = f / 2
   ```

2. **Anneal offset computation**: Effective fields are normalized to [0,1] and converted to per-qubit anneal offsets:
   ```
   r_i = |F_i| / max_k |F_k|
   delta_i = |delta_max| * (1 - 2 * r_i)
   ```
   Strongly-coupled qubits (large r) are delayed; weakly-coupled qubits are advanced.

3. **D-Wave submission** (Python): A generated Python script reads the encoded model JSON, finds a minor embedding, maps logical offsets to physical qubits, and submits with `uniform_torque_compensation` chain strength.

4. **Result decoding** (C++): D-Wave bitstrings are decoded to CFN choices and evaluated under the original CFN. Reports both the best encoded energy and the best native CFN energy.

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

The resulting binaries are `build/Release/solve_sa`, `build/Release/solve_qa`, and `build/Release/tests` (or `build/solve_sa`, `build/solve_qa`, and `build/tests` on single-config generators).

`solve_qa` additionally requires Python 3 with `dwave-ocean-sdk` at runtime:
```bash
pip install dwave-ocean-sdk
```

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

Each invocation loads one encoded model, computes inhomogeneous anneal offsets (C++), generates and executes a Python D-Wave submission script, then decodes D-Wave bitstrings back to CFN solutions. The source CFN is located automatically via the model's `source_cfn` field in `--cfn-dir`.

#### QA examples

Submit a one-hot QUBO to D-Wave:
```bash
solve_qa --input encoded/oh/example_one_hot.json \
  --cfn-dir cfns/ --solver Advantage_system6.4 \
  --num-reads 1000 --annealing-time 20 --verbose
```

Submit a truncated-binary Ising model without inhomogeneous driving:
```bash
solve_qa --input encoded/tb2/example_truncated_binary.json \
  --cfn-dir cfns/ --solver Advantage_system6.4 \
  --no-inhomogeneous --num-reads 500
```

## Benchmark scripts

### SA benchmark (run_benchmark.sh)

`scripts/run_benchmark.sh` automates parallel SA execution across all files in a directory using GNU parallel. Designed for 64-core Intel Ice Lake nodes with 32 parallel tasks.

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

### QA benchmark (run_benchmark_qa.sh)

`scripts/run_benchmark_qa.sh` automates D-Wave QA execution across all encoded JSON files. Default `--jobs 1` since the D-Wave QPU is a shared resource (parallel jobs are queued). Increase `--jobs` to overlap embedding computation with QPU wait times.

```bash
./scripts/run_benchmark_qa.sh \
  --input-dir encoded/one_hot/ \
  --output-csv results/oh_qa_results.csv \
  --cfn-dir cfns/ \
  --solver-name Advantage_system6.4 \
  --num-reads 1000 \
  --annealing-time 20 \
  --verbose
```

## Output format

### SA CSV (solve_sa)

A single CSV file with one row per problem instance. Columns:

- **Instance metadata**: `problem_name`, `source_cfn`, `solver_mode`, `encoding`, `variable_type`, `num_qubits`, `num_variables`, `max_cardinality`, `edge_density`, `distribution`
- **SA configuration**: `schedule`, `move_type`, `T_start`, `T_end`, `num_steps`, `num_runs`, `seed`
- **Energy statistics**: `best_energy`, `mean_energy`, `std_energy`, `median_energy`, `num_optimal`
- **Timing**: `total_runtime_s`, `mean_time_per_run_s`
- **Solution**: `best_solution` (CFN choices as `[X0,X1,...,XN]`)
- **Raw data**: `per_run_energies` (all best energies as `[E0,E1,...,ER]`)

In CFN mode, `best_solution` contains integer variable assignments directly. In binary mode, the best binary state is decoded to CFN choices using the encoding's decoding rule. The `num_optimal` column counts how many runs found the known ground truth (requires `--ground-truth`); it is `NA` if no ground truth is provided.

### QA CSV (solve_qa)

The QA CSV contains all 26 SA-compatible columns (with `solver_mode=dwave`, SA-specific fields set to `NA`/0) plus additional D-Wave columns:

- **D-Wave configuration**: `solver_name`, `annealing_time_us`, `delta_max`
- **D-Wave timing**: `inhomog_setup_time_s` (effective field computation), `embedding_time_s`, `qpu_access_time_us`, `qpu_sampling_time_us`, `qpu_programming_time_us`
- **CFN evaluation**: `best_cfn_energy` (best energy under original CFN among feasible decoded solutions), `num_feasible` (number of D-Wave samples that decode to valid CFN solutions), `num_best_cfn` (number of feasible samples achieving best CFN energy)
- **Embedding statistics**: `emb_num_physical_qubits`, `emb_chain_length_avg`, `emb_chain_length_median`, `emb_chain_length_var`, `emb_chain_breaks_avg`, `emb_chain_breaks_median`, `emb_chain_breaks_var`

Chain length statistics are computed over the set of logical-to-physical chains in the embedding (one chain per logical variable). Chain break statistics are computed over all D-Wave samples (expanded by `num_occurrences`): for each sample, a chain break is counted when not all physical qubits in a chain agree.

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
│       └── dwave_template.hpp        # Embedded Python submission script template
├── scripts/
│   ├── run_benchmark.sh              # GNU parallel SA benchmark driver
│   ├── run_benchmark_qa.sh           # GNU parallel QA benchmark driver
│   └── dwave_submit_template.py      # Reference copy of the D-Wave submission template
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
- **GNU parallel** (for benchmark scripts only)
- **Python 3** with **[dwave-ocean-sdk](https://docs.ocean.dwavesys.com/)** (for `solve_qa` only)
