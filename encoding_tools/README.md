# encoding_tools

C++17 tool that converts pairwise-decomposable Cost Function Networks (CFNs) in Toulbar2 JSON format into QUBO/HUBO/Ising models under five binary-variable encodings: one-hot, domain-wall, exact-binary, approximate-binary, and truncated-binary. It produces JSON model files as output together with a CSV of benchmarking metrics (including timing information, maximum encoding degree, and number of binary variables used).

Given the indicator-form CFN objective:

$$
H_{\text{CFN}}(\vec{x}) = \sum_{\varnothing \neq S \subseteq [N]} \; \sum_{\boldsymbol{c}\in\mathcal{C}_S} C_{S;\boldsymbol{c}} \prod_{i\in S} x_{i,c_i}
$$

each encoding replaces the indicator $x_{i,c_i}\in\{0,1\}$ with a specific function of binary variables, trading off bit count, interaction degree, infeasible-solution count, and approximation error.

## Table of Contents

- [Contents](#contents)
- [Usage](#usage)
  - [Building](#building)
  - [Testing](#testing)
  - [Examples](#examples)
  - [Input format](#input-format)
  - [Output format](#output-format)
- [Dependencies](#dependencies)

## Contents

```
encoding_tools/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── main.cpp                    # CLI entry point
│   ├── baseline/
│   │   ├── types.hpp               # BinaryPolynomial, EncodingParams, EncodingResult, Timer
│   │   └── cfn.hpp                 # CFN parser
│   ├── encodings/
│   │   ├── one_hot.hpp             # One-hot encoding
│   │   ├── domain_wall.hpp         # Domain-wall encoding
│   │   ├── exact_binary.hpp        # Exact-binary encoding
│   │   ├── approximate_binary.hpp  # Approximate-binary encoding
│   │   └── truncated_binary.hpp    # Truncated-binary encoding
│   ├── utilities/
│   │   ├── lagrange.hpp            # MOMC Lagrange multiplier computation
│   │   ├── assignment.hpp          # Choice ordering and bitstring ordering
│   │   └── rosenberg.hpp           # Rosenberg quadratization
│   └── output.hpp                  # JSON and CSV writers
├── python/
│   └── dimod_converter.py          # Load JSON models into dimod objects
├── tests/
│   └── tests.cpp                   # Unit tests
├── test_cfns/                      # Sample CFN inputs
└── test_output/                    # Sample encoded outputs + metrics CSVs
```

### Encodings

| Encoding | Bits per variable | Variable type | Output degree |
|---|---|---|---|
| **One-hot (OH)** | $\lvert d_i\rvert$ | BINARY $\{0,1\}$ | $2$ (QUBO) |
| **Domain-wall (DW)** | $\lvert d_i\rvert - 1$ | BINARY $\{0,1\}$ | $2$ (QUBO) |
| **Exact-binary (EB)** | $\lceil\log_2\lvert d_i\rvert\rceil$ | BINARY $\{0,1\}$ | up to $\sum_j \lceil\log_2\lvert d_{i_j}\rvert\rceil$ (HUBO) |
| **Approximate-binary (AB)** | $\lceil\log_2\lvert d_i\rvert\rceil$ | BINARY $\{0,1\}$ | $k_{\text{approx}}$ (default $2$, QUBO) |
| **Truncated-binary (TB)** | $\lceil\log_2\lvert d_i\rvert\rceil$ | SPIN $\{-1,+1\}$ | $k_{\text{trunc}}$ (default $2$, QUBO) |

- **One-hot.** Each choice $c_i$ of variable $i$ is represented by $x_{i,c_i}=b_{i,c_i}$. Feasibility (exactly one bit set per register) is enforced by a Lagrange penalty $\lambda_{\text{OH}}\sum_i\big(\sum_{c_i} b_{i,c_i} - 1\big)^2$, where $\lambda_{\text{OH}}$ is set by the MOMC method of [Ayodele, 2022](https://doi.org/10.1007/978-3-031-04148-8_11). 

- **Domain-wall.** Each choice $c_i$ of Variable $i$ is represented by $x_{i,c} = b_{i,c_i-1} - b_{i,c_i}$ with boundary conditions $b_{i,-1}=1$ and $b_{i,\lvert d_i\rvert-1}=0$. A MOMC Lagrange penalty enforces a single $\mathtt{1\!\to\!0}$ domain wall per register by way of $\lambda_{DW}\sum_i\sum_{c_i}\left(b_{i,c_i}-b_{i,c_i}b_{i,c_i-1}\right)$, where $\lambda_{\text{DW}}$ is set by the MOMC method of [Ayodele, 2022](https://doi.org/10.1007/978-3-031-04148-8_11). 

- **Exact-binary.** Variable $i$ is encoded in $D_i=\lceil\log_2\lvert d_i\rvert\rceil$ bits, with each choice assigned a unique bitstring $r^{(c_i)}$. The indicator expands by inclusion–exclusion as $x_{i,c}\mapsto \prod_{q=0}^{D_i-1}\big(b_{i,q}r^{(c)}_q + (1-b_{i,q})(1-r^{(c)}_q)\big)$. Unused bitstrings are padded with the lowest-energy choice. This encoding generally produces HUBOs, but can be quadratized to QUBOs (see below).

- **Approximate-binary.** Same bit-count as exact-binary, but coefficients are fit by least squares onto a degree-$\le k_{\text{approx}}$ monomial basis via the Moore–Penrose pseudoinverse. This eliminates infeasible solutions and Lagrange multipliers while bounding interaction degree, at the cost of bounded approximation error. Optional heuristics include: choice ordering, bitstring ordering, weighted least squares, linearly-indenpendent matrix row prioritization, and nonlinear gradient-descent refinement.

- **Truncated-binary.** A Walsh–Hadamard spectral decomposition of the exact-binary encoding in the Ising spin basis, $H_{\text{EB}}(\vec{b})=\sum_{S}\widehat{H}_S\,\chi_S(\vec{b})$ with $\chi_S(\vec{b})=\prod_{q\in S}(1-2b_q)$, truncated at a user-defined Walsh degree $k_{\text{trunc}}$. This encoding uses the same number of bits as both the exact-binary and approximate-binary encodings, but has a user-defined degree, which if higher than QUBO can be quadratized.

**Rosenberg quadratization.** Optional post-processing that reduces any HUBO to a QUBO by introducing auxiliary variables, with penalty strength $M = 2\max_{\lvert S\rvert>2}\lvert c_S\rvert$. Supports both BINARY and SPIN variable types.

**Lagrange multipliers.** One-hot and domain-wall use per-register MOMC penalties $\lambda_i = W_c/\gamma$ (bit-flip objective gain over minimum infeasible penalty), avoiding the dynamic-range inflation typical of a single global $\alpha\cdot\max_{S,\boldsymbol{c}}\lvert C_{S;\boldsymbol{c}}\rvert$ constant.

For EB, AB, and TB, we test two combinations of choice/bitstring order heuristics: the **naive** canonical binary-order assignment, and the **enhanced** assignment (Boltzmann-average-sorted choices with Gray-code bitstrings, plus linearly-independent prioritization for AB), though combinations are possible.

## Usage

### Building

Requires CMake 3.14+ and a C++17 compiler. Dependencies ([nlohmann/json](https://github.com/nlohmann/json) 3.11.3 and [Eigen](https://eigen.tuxfamily.org/) 3.4.0) are fetched automatically via CMake `FetchContent`. OpenMP is detected and used if available.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This produces the `encode_cfn` binary (`build/encode_cfn`).

### Testing

The same build also produces the `tests` binary:

```bash
./build/tests
```

The unit tests verify exact cost preservation and infeasible-energy dominance for one-hot and domain-wall, Rosenberg quadratization correctness, choice/bitstring ordering, and nonlinear refinement for approximate-binary.

### Examples

#### CLI reference

```
encode_cfn [options]

Required:
  --input-dir DIR        Directory containing .cfn files
  --output-dir DIR       Directory for output JSON files
  --csv FILE             Path for metrics CSV
  --encoding NAME        one_hot | domain_wall | exact_binary | approximate_binary | truncated_binary

Encoding options:
  --choice-ordering      unsorted | one_variable | boltzmann       (default: unsorted)
  --bitstring-ordering   natural | gray                            (default: natural)
  --weighted-ls          Enable weighted least squares             (AB only)
  --li-prioritization    Enable LI prioritization                  (AB only)
  --nonlinear-refinement Enable nonlinear refinement               (AB only)
  --quadratize           Apply Rosenberg quadratization            (EB/AB/TB only)
  --k-approx N           Max AB interaction degree                 (default: 2)
  --k-trunc N            Walsh truncation order                    (default: 2)
  --epsilon FLOAT        Lagrange multiplier margin                (default: 0.1)
  --temperature FLOAT    Boltzmann temperature                     (default: 1.0)
  --nl-temperature FLOAT NL refinement temperature                 (default: 1.0)
  --nl-tether FLOAT      NL tethering weight                       (default: 1.0)
  --nl-max-iter N        NL max iterations                         (default: 100)
  --threads N            Number of OpenMP threads                  (default: 1)
  --verbose              Print progress
```

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

Truncated-binary at degree $k_{\text{trunc}}=3$ with quadratization:

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
    "f01": {"scope": ["x0", "x1"], "costs": [0.1, 0.2, "...", 1.2]}
  }
}
```

Costs are flattened row-major with the rightmost scope variable varying fastest. Instance metadata ($N$, $D$, $\rho$, distribution) is extracted from filenames matching `CFN_N<N>_D<D>_rho<rho>_<dist>_<num>.cfn`. Sample inputs are in [`test_cfns/`](test_cfns).

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

Term keys are comma-separated qubit indices, so a term of degree $k$ contributes $c_S \prod_{q\in S} b_q$. This format is directly consumable the solvers and solver interfaces in [`../solver_tools`](../solver_tools). Sample outputs are in [`test_output/`](test_output).

**CSV metrics.** One row per encoded CFN, with columns covering: instance metadata, encoding configuration, qubit counts, term statistics by degree, coefficient statistics, Lagrange-multiplier and Rosenberg-penalty information, approximation quality, spectral profile $P_k=\sum_{\lvert S\rvert=k}\widehat{H}_S^2$, per-encoding-phase timing, and variable type.

**Python converter.** [`python/dimod_converter.py`](python/dimod_converter.py) loads the JSON output into Python objects for use with D-Wave's `dimod` library:

```python
from dimod_converter import load_model, to_bqm, to_polynomial, to_dwave_dict

model = load_model("out/oh/example_one_hot.json")
bqm  = to_bqm(model)              # degree <= 2: dimod BinaryQuadraticModel
poly = to_polynomial(model)       # any degree: dimod BinaryPolynomial
Q, offset = to_dwave_dict(model)  # plain dict: {(i, j): Q_ij}
```

## Dependencies

| Component | Requirement |
|---|---|
| Build | C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+), CMake 3.14+ |
| JSON I/O | [nlohmann/json](https://github.com/nlohmann/json) 3.11.3 (fetched automatically) |
| Linear algebra | [Eigen](https://eigen.tuxfamily.org/) 3.4.0 (fetched automatically) |
| Multithreading | OpenMP (optional) |
| Python converter | Python 3.6+ with `dimod` (optional) |
