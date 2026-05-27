# encoding_tools

High-performance C++17 tool for converting pairwise-decomposable Cost Function Networks (CFNs) in Toulbar2 JSON format into QUBO/HUBO/Ising models under five binary-variable encodings. Outputs D-Wave-compatible JSON model files and a CSV of benchmarking metrics.

## Encodings

| Encoding | Bits per variable | Variable type | Output degree |
|---|---|---|---|
| **One-hot (OH)** | d_i | BINARY {0,1} | 2 (QUBO) |
| **Domain-wall (DW)** | d_i - 1 | BINARY {0,1} | 2 (QUBO) |
| **Exact-binary (EB)** | ceil(log2(d_i)) | BINARY {0,1} | up to N*ceil(log2(d_i)) (HUBO) |
| **Approximate-binary (AB)** | ceil(log2(d_i)) | BINARY {0,1} | k_approx (default 2, QUBO) |
| **Truncated-binary (TB)** | ceil(log2(d_i)) | SPIN {-1,+1} | k_trunc (default 2) |

### One-hot

Each choice `c` of variable `i` is represented by a dedicated binary variable `b_{i,c}`. Feasibility (exactly one active) is enforced by a MOMC Lagrange penalty on each register: `lambda_i * (sum_c b_{i,c} - 1)^2`.

### Domain-wall

Variable `i` uses `d_i - 1` ordered binary variables. The choice indicator is `x_{i,c} = b_{i,c-1} - b_{i,c}` with boundary conditions `b_{i,-1} = 1` and `b_{i,d_i-1} = 0`. Unary costs are encoded via telescoping sums. A MOMC Lagrange penalty enforces the monotonicity constraint.

### Exact-binary

Variable `i` is encoded in `ceil(log2(d_i))` bits. The choice indicator `x_{i,c}` is expanded as a multilinear polynomial over the bit variables using inclusion-exclusion. The encoding is exact but generally produces terms of degree up to `ceil(log2(d_i))` per register. Unused bitstrings (when `d_i` is not a power of 2) are padded with the lowest-energy choice.

### Approximate-binary

Uses the same `ceil(log2(d_i))` bits as exact-binary but constructs a least-squares QUBO approximation. For each cost table, a design matrix `Q` of monomials up to degree `k_approx` is built and solved via SVD pseudoinverse. Reports L2 and L-infinity approximation errors.

Available heuristics:
- **Choice ordering**: `unsorted`, `one_variable` (sort by unary cost), `boltzmann` (sort by Boltzmann-averaged effective energy)
- **Bitstring ordering**: `natural` (canonical binary), `gray` (Gray code)
- **Weighted least squares**: rows weighted by `sqrt(exp(-E/kBT))` to prioritize low-energy states
- **LI prioritization**: rank-revealing QR assigns linearly independent rows of Q to lower-energy choices
- **Nonlinear refinement**: gradient descent on a Boltzmann probability-matching loss with ground-state tethering

### Truncated-binary

Walsh-Hadamard spectral decomposition in the Ising spin basis. Cost tables are embedded into the full hypercube, transformed via the fast Walsh-Hadamard transform, and truncated at Walsh degree `k_trunc`. Pairwise tables are centered (marginals absorbed into unary tables) before transformation to reduce spectral leakage. Reports a spectral energy profile `P_k = sum_{|S|=k} hat{H}_S^2`.

### Rosenberg quadratization

Optional post-processing that reduces any HUBO to a QUBO by introducing auxiliary variables. Supports both variable types:
- **BINARY {0,1}**: penalty `M * (b_p*b_q - 2*b_p*y - 2*b_q*y + 3*y)` per substitution
- **SPIN {-1,+1}**: penalty `M/4 * (3 + s_p + s_q - 2*t + s_p*s_q - 2*s_p*t - 2*s_q*t)`

Penalty strength: `M = 2 * max_{|S|>2} |c_S|`. Auxiliary variables are shared across terms that substitute the same pair.

### Lagrange multipliers

One-hot and domain-wall use the MOMC (Maximum change in Objective function divided by Minimum Constraint function) method from Ayodele 2022. A single global Lagrange multiplier is computed as:

```
lambda = W_c / gamma

W_c   = max_q  sum_{S containing q} |C_S|     (max single-flip objective change)
gamma = min_{b: H^penalty(b) > 0} H^penalty(b) = 1   (for OH, DW, and KH)
```

The numerator `W_c` is computed from the already-constructed objective polynomial H^(obj) (the encoded cost terms without the penalty). For each qubit `q`, the sum of absolute coefficients of all monomials containing `q` bounds the maximum objective change achievable by flipping `q`. The denominator `gamma = 1` is proven algebraically from the structure of the penalty for all three encodings.

## Building

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

## Usage

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
  --temperature FLOAT   Boltzmann temperature                     (default: 1.0)
  --nl-temperature FLOAT  NL refinement temperature               (default: 1.0)
  --nl-tether FLOAT     NL tethering weight                       (default: 1.0)
  --nl-max-iter N       NL max iterations                         (default: 100)
  --threads N           Number of OpenMP threads                  (default: 1)
  --verbose             Print progress
```

### Examples

Encode all CFNs in a directory with one-hot:
```bash
encode_cfn --input-dir cfns/ --output-dir out/oh/ --csv metrics.csv --encoding one_hot --verbose
```

Approximate-binary with Boltzmann ordering, Gray code, weighted LS, LI prioritization, and nonlinear refinement:
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

## Output format

### JSON model files

Each CFN produces one JSON file containing:

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

Term keys are comma-separated qubit indices. This format is directly consumable by D-Wave samplers.

### CSV metrics

A single CSV file accumulates one row per encoded CFN with columns covering:

- **Instance metadata**: filename, encoding, orderings, N, max cardinality, edge density, distribution
- **Encoding configuration**: k_approx, k_trunc, epsilon, temperature, weighted_ls, li_prioritization, nonlinear_refinement, quadratized
- **Qubit counts**: logical, auxiliary, total
- **Term statistics**: max interaction degree, mean weighted degree, counts by degree (linear/quadratic/cubic/higher), total nonzero terms
- **Coefficient statistics**: max |coeff|, min |coeff|, dynamic range, offset
- **Constraint information**: Lagrange multiplier (global MOMC), Rosenberg penalty strength
- **Approximation quality**: L2 error, L-infinity error, spectral profile (TB only)
- **Timing**: total, parse, choice ordering, bitstring assignment, Lagrange, encoding, quadratization, nonlinear refinement, output (all in seconds)
- **Variable type**: BINARY or SPIN

## Python converter

`python/dimod_converter.py` loads the JSON output into Python objects for use with D-Wave's dimod library.

```python
from dimod_converter import load_model, to_bqm, to_polynomial, to_dwave_dict

model = load_model("out/oh/example_one_hot.json")

# For degree <= 2 (QUBO): dimod BinaryQuadraticModel
bqm = to_bqm(model)

# For any degree (HUBO): dimod BinaryPolynomial
poly = to_polynomial(model)

# Plain Python dict in D-Wave format: {(i, j): Q_ij}
Q, offset = to_dwave_dict(model)
```

CLI usage:
```bash
python python/dimod_converter.py model.json              # print summary
python python/dimod_converter.py model.json --to-bqm     # print BQM
python python/dimod_converter.py model.json --to-poly    # print BinaryPolynomial
python python/dimod_converter.py model.json --to-qubo-dict  # print plain QUBO dict
```

## Input format

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

Costs are flattened row-major with the rightmost scope variable varying fastest. Metadata (N, D, rho, distribution) is extracted from filenames matching the pattern `CFN_N<N>_D<D>_rho<rho>_<dist>_<num>.cfn`.

## Project structure

```
encoding_tools/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp                  # CLI entry point, file discovery, encoding dispatch
│   ├── types.hpp                 # BinaryPolynomial, EncodingParams, EncodingResult, Timer
│   ├── cfn.hpp                   # CFN parser (Toulbar2 JSON)
│   ├── lagrange.hpp              # MOMC Lagrange multiplier computation
│   ├── assignment.hpp            # Choice ordering, bitstring ordering, LI prioritization
│   ├── one_hot.hpp               # One-hot encoding
│   ├── domain_wall.hpp           # Domain-wall encoding
│   ├── exact_binary.hpp          # Exact-binary encoding (indicator polynomial expansion)
│   ├── approximate_binary.hpp    # Approximate-binary encoding (least-squares + refinement)
│   ├── truncated_binary.hpp      # Truncated-binary encoding (Walsh-Hadamard)
│   ├── rosenberg.hpp             # Rosenberg quadratization (BINARY and SPIN)
│   └── output.hpp                # JSON writer and CSV formatter
└── python/
    └── dimod_converter.py        # Load JSON models into dimod objects
```

## Dependencies

- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14+**
- **[nlohmann/json](https://github.com/nlohmann/json) 3.11.3** (fetched automatically)
- **[Eigen](https://eigen.tuxfamily.org/) 3.4.0** (fetched automatically)
- **OpenMP** (optional, for multithreading)
- **Python 3.6+** with `dimod` (optional, for the Python converter)
