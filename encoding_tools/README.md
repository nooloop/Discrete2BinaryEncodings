# encoding_tools

High-performance C++17 tool for converting pairwise-decomposable Cost Function Networks (CFNs) in Toulbar2 JSON format into QUBO/HUBO/Ising models under five binary-variable encodings. Outputs D-Wave-compatible JSON model files and a CSV of benchmarking metrics.

## Table of Contents

- [Contents](#contents)
- [Usage](#usage)
  - [Building](#building)
  - [CLI reference](#cli-reference)
  - [Examples](#examples)
  - [Input format](#input-format)
  - [Output format](#output-format)
  - [Python converter](#python-converter)
- [Dependencies](#dependencies)

## Contents

```
encoding_tools/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp                  # CLI entry point, file discovery, encoding dispatch
│   ├── baseline/
│   │   ├── types.hpp             # BinaryPolynomial, EncodingParams, EncodingResult, Timer
│   │   └── cfn.hpp               # CFN parser (Toulbar2 JSON)
│   ├── encodings/
│   │   ├── one_hot.hpp           # One-hot encoding
│   │   ├── domain_wall.hpp       # Domain-wall encoding
│   │   ├── exact_binary.hpp      # Exact-binary encoding (indicator polynomial expansion)
│   │   ├── approximate_binary.hpp # Approximate-binary encoding (least-squares + refinement)
│   │   └── truncated_binary.hpp  # Truncated-binary encoding (Walsh-Hadamard)
│   ├── utilities/
│   │   ├── lagrange.hpp          # MOMC Lagrange multiplier computation
│   │   ├── assignment.hpp        # Choice ordering, bitstring ordering, LI prioritization
│   │   └── rosenberg.hpp         # Rosenberg quadratization (BINARY and SPIN)
│   └── output.hpp                # JSON writer and CSV formatter
├── python/
│   └── dimod_converter.py        # Load JSON models into dimod objects
├── tests/
│   └── tests.cpp                 # Unit tests
├── test_cfns/                    # Sample CFN inputs
└── test_output/                  # Sample encoded outputs + metrics CSVs
```

### Encodings

| Encoding | Bits per variable | Variable type | Output degree |
|---|---|---|---|
| **One-hot (OH)** | d_i | BINARY {0,1} | 2 (QUBO) |
| **Domain-wall (DW)** | d_i - 1 | BINARY {0,1} | 2 (QUBO) |
| **Exact-binary (EB)** | ceil(log2(d_i)) | BINARY {0,1} | up to N*ceil(log2(d_i)) (HUBO) |
| **Approximate-binary (AB)** | ceil(log2(d_i)) | BINARY {0,1} | k_approx (default 2, QUBO) |
| **Truncated-binary (TB)** | ceil(log2(d_i)) | SPIN {-1,+1} | k_trunc (default 2) |

**One-hot.** Each choice `c` of variable `i` is represented by a dedicated binary variable `b_{i,c}`. Feasibility (exactly one active) is enforced by a MOMC Lagrange penalty on each register.

**Domain-wall.** Variable `i` uses `d_i - 1` ordered binary variables. The choice indicator is `x_{i,c} = b_{i,c-1} - b_{i,c}` with boundary conditions `b_{i,-1} = 1` and `b_{i,d_i-1} = 0`. A MOMC Lagrange penalty enforces the monotonicity constraint.

**Exact-binary.** Variable `i` is encoded in `ceil(log2(d_i))` bits. The choice indicator is expanded as a multilinear polynomial via inclusion-exclusion. Exact but generally HUBO. Unused bitstrings are padded with the lowest-energy choice.

**Approximate-binary.** Same bits as exact-binary but constructs a least-squares QUBO approximation. Available heuristics: choice ordering (unsorted, one_variable, boltzmann), bitstring ordering (natural, gray), weighted least squares, LI prioritization, and nonlinear refinement.

**Truncated-binary.** Walsh-Hadamard spectral decomposition in the Ising spin basis, truncated at Walsh degree `k_trunc`. Pairwise tables are centered (marginals absorbed into unary tables) before transformation to reduce spectral leakage.

**Rosenberg quadratization.** Optional post-processing reducing any HUBO to a QUBO by introducing auxiliary variables. Penalty strength: `M = 2 * max_{|S|>2} |c_S|`. Supports both BINARY and SPIN variable types.

**Lagrange multipliers.** One-hot and domain-wall use per-register MOMC penalties: `lambda_i = (1 + epsilon) * Delta_i`, avoiding dynamic-range inflation from a single global constant.

## Usage

### Building

Requires CMake 3.14+ and a C++17 compiler. Dependencies (nlohmann/json 3.11.3 and Eigen 3.4.0) are fetched automatically via CMake FetchContent. OpenMP is detected and used if available.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On Windows with Visual Studio Build Tools:
```powershell
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

The resulting binary is `build/Release/encode_cfn` (or `build/encode_cfn` on single-config generators).

### CLI reference

```
encode_cfn [options]

Required:
  --input-dir DIR       Directory containing .cfn files
  --output-dir DIR      Directory for output JSON files
  --csv FILE            Path for metrics CSV
  --encoding NAME       one_hot | domain_wall | exact_binary | approximate_binary | truncated_binary

Encoding options:
  --choice-ordering     unsorted | one_variable | boltzmann      (default: unsorted)
  --bitstring-ordering  natural | gray                           (default: natural)
  --weighted-ls         Enable weighted least squares             (AB only)
  --li-prioritization   Enable LI prioritization                  (AB only)
  --nonlinear-refinement Enable nonlinear refinement              (AB only)
  --quadratize          Apply Rosenberg quadratization            (EB/TB only)
  --k-approx N          Max AB interaction degree                 (default: 2)
  --k-trunc N           Walsh truncation order                    (default: 2)
  --epsilon FLOAT       Lagrange multiplier margin                (default: 0.1)
  --temperature FLOAT   Boltzmann temperature                     (default: 1.0)
  --nl-temperature FLOAT  NL refinement temperature               (default: 1.0)
  --nl-tether FLOAT     NL tethering weight                       (default: 1.0)
  --nl-max-iter N       NL max iterations                         (default: 100)
  --threads N           Number of OpenMP threads                  (default: 1)
  --verbose             Print progress
```

### Examples

Encode all CFNs with one-hot:
```bash
encode_cfn --input-dir cfns/ --output-dir out/oh/ --csv metrics.csv --encoding one_hot --verbose
```

Approximate-binary with enhanced assignment (Boltzmann + Gray + LI):
```bash
encode_cfn --input-dir cfns/ --output-dir out/ab/ --csv metrics.csv \
  --encoding approximate_binary \
  --choice-ordering boltzmann --bitstring-ordering gray \
  --weighted-ls --li-prioritization --nonlinear-refinement \
  --k-approx 2 --temperature 1.0 --threads 8
```

Exact-binary with Rosenberg quadratization:
```bash
encode_cfn --input-dir cfns/ --output-dir out/eb_quad/ --csv metrics.csv \
  --encoding exact_binary --quadratize
```

Truncated-binary at degree 3 with quadratization:
```bash
encode_cfn --input-dir cfns/ --output-dir out/tb3/ --csv metrics.csv \
  --encoding truncated_binary --k-trunc 3 --quadratize --threads 8
```

### Input format

CFN files must be in Toulbar2 JSON format:

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

### Output format

**JSON model files.** Each CFN produces one JSON file:

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

Term keys are comma-separated qubit indices. This format is directly consumable by D-Wave samplers and `solver_tools`.

**CSV metrics.** One row per encoded CFN with columns covering: instance metadata, encoding configuration, qubit counts, term statistics by degree, coefficient statistics, Lagrange multiplier and Rosenberg penalty information, approximation quality (L2/L-inf error, spectral profile), per-phase timing, and variable type.

### Python converter

`python/dimod_converter.py` loads the JSON output into Python objects for use with D-Wave's dimod library:

```python
from dimod_converter import load_model, to_bqm, to_polynomial, to_dwave_dict

model = load_model("out/oh/example_one_hot.json")
bqm = to_bqm(model)           # degree <= 2: dimod BinaryQuadraticModel
poly = to_polynomial(model)    # any degree: dimod BinaryPolynomial
Q, offset = to_dwave_dict(model)  # plain dict: {(i, j): Q_ij}
```

## Dependencies

- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14+**
- **[nlohmann/json](https://github.com/nlohmann/json) 3.11.3** (fetched automatically)
- **[Eigen](https://eigen.tuxfamily.org/) 3.4.0** (fetched automatically)
- **OpenMP** (optional, for multithreading)
- **Python 3.6+** with `dimod` (optional, for the Python converter)
