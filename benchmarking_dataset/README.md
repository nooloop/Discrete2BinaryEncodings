# benchmarking_dataset

Generator for the common evaluation set of synthetic, pairwise-decomposable Cost Function Networks (CFNs) used across every encoding and solver in the manuscript. Each instance is written in the [Toulbar2](https://github.com/toulbar2/toulbar2) `.cfn` JSON format and is deterministically seeded, so the entire dataset can be regenerated exactly.

A CFN over $N$ discrete variables $\vec{d}=(d_1,\dots,d_N)$ of cardinalities $\lvert d_1\rvert,\dots,\lvert d_N\rvert$ has the indicator-form objective:

$$
H_{\text{CFN}}(\vec{x}) = \sum_{\varnothing \neq S \subseteq [N]} \; \sum_{\boldsymbol{c}\in\mathcal{C}_S} C_{S;\boldsymbol{c}} \prod_{i\in S} x_{i,c_i}
$$

where $x_{i,c_i}\in\{0,1\}$ indicates variable $i$ taking choice $c_i$, $[N]=\{1,\dots,N\}$, and $\boldsymbol{c}=(c_i)_{i\in S}$. This generator emits the unary ($\lvert S\rvert=1$) and pairwise ($\lvert S\rvert=2$) cost tables $C_{S;\boldsymbol{c}}$ that define each instance.

## Table of Contents

- [Contents](#contents)
- [Usage](#usage)
  - [Examples](#examples)
  - [Input format](#input-format)
  - [Output format](#output-format)
- [Dependencies](#dependencies)

## Contents

This folder contains a single generation script. The generated `.cfn` files are not committed to the repository, but can be produced on demand.

```
benchmarking_dataset/
├── README.md
└── generate_cfn_dataset.py    # deterministic CFN generator (Toulbar2 .cfn)
```

## Usage

### Examples

Generate all instances (reproducible from master seed `20260512`):

```bash
python generate_cfn_dataset.py -o cfns
```

Generate a subset (e.g., only $N=4$, $D=2$, uniform):

```bash
python generate_cfn_dataset.py -o cfns --N 4 --D 2 --dist uniform
```

Dry run (print filenames without writing):

```bash
python generate_cfn_dataset.py --dry-run
```

#### Finding ground states with Toulbar2

Reference ground states are obtained with [Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1, an exact CFN solver combining best-first branch-and-bound with parallel variable-neighbourhood search. Toulbar2 is run directly on the `.cfn` files, with no intermediate conversion:

```bash
toulbar2 cfns/CFN_N4_D2_rho0.3_uniform_1.cfn -w
```

The `-w` flag writes the optimal solution to a file. For batch ground-state computation on a Slurm cluster (64-core Intel Ice Lake nodes, 3.7 GHz, 1 TB RAM), 32 Toulbar2 tasks are run in parallel with 1 CPU per task, with the remaining 32 cores held dormant to avoid thermal throttling and resource contention. This was accomplished using `disBatch`.

The resulting ground-state energies and runtimes are the reference against which all other solvers are scored (see [`../solver_tools`](../solver_tools)).

### Input format

The generator reads no input files; its input is the stratification parameter space below. The full benchmark sweeps across four axes:

| Axis | Symbol | Values |
|---|---|---|
| Variable count | $N$ | $2, 4, 6, \dots, 20$ |
| Cardinality | $\lvert d\rvert = 2^{D}$ | $D \in \{1,2,3,4,5,6,7,8\}$ |
| Edge density | $\rho$ | $0.10, 0.15, 0.20, \dots, 0.50$ |
| Coupling distribution | $R$ | uniform, Gaussian, exponential, Laplace |
| Draws per setting | $P$ | $5$ |

The total instance count is:

$$
\lvert\mathcal{N}\rvert \times \lvert\mathcal{D}\rvert \times \lvert\boldsymbol{\rho}\rvert \times \lvert\mathcal{R}\rvert \times P \;=\; 10 \times 8 \times 9 \times 4 \times 5 \;=\; 14{,}400 .
$$

The edge density is $\rho = \lvert E\rvert / \binom{N}{2}$, where $\lvert E\rvert$ is the number of non-trivial pairwise cost tables. A unary cost table is drawn for every variable; a full $\lvert d\rvert \times \lvert d\rvert$ pairwise table is drawn for every selected edge and is identically zero otherwise. Cardinalities are constrained to powers of two ($\lvert d\rvert = 2^{D}$) to avoid codeword-duplication artifacts in the bit-efficient encodings. Coefficient distribution parameters:

| Distribution | Notation | Parameters |
|---|---|---|
| Uniform | $\mathcal{U}(a,b)$ | $a=-10,\ b=10$ |
| Gaussian | $\mathcal{N}(\mu,\sigma^2)$ | $\mu=0,\ \sigma=10$ |
| Exponential | $\mathrm{Exp}(\lambda)$ | $\lambda=1$ (mean $1$) |
| Laplace (decaying-exponential) | $\mathrm{Lap}(\mu,b)$ | $\mu=0,\ b=1$ |

`generate_cfn_dataset.py` can be easily modified to sweep across alternative axes, and alternative coefficient distributions.

CLI options:

```
-o, --out-dir DIR               Output directory (default: cfn_dataset)
-P, --instances-per-setting N   Draws per setting (default: 5)
--gzip                          Write .cfn.gz (Toulbar2 reads both)
--overwrite                     Overwrite existing files
--N N [N ...]                   Restrict variable counts
--D D [D ...]                   Restrict cardinality exponents
--rho R [R ...]                 Restrict edge densities
--dist D [D ...]                Restrict distributions
--dry-run                       Print filenames without writing
```

### Output format

Each instance is named

```
CFN_N<N>_D<D>_rho<rho>_<distribution>_<number>.cfn
```

where $\lvert d\rvert = 2^{D}$ is the cardinality (choices per variable) and `<number>` indexes the independent draw within each $(N, D, \rho, \text{distribution})$ combination.

Each `.cfn` file is a Toulbar2 JSON document:

```json
{
  "problem": {"name": "CFN_N4_D2_rho0.3_uniform_1", "mustbe": "<42.5"},
  "variables": {"x0": 4, "x1": 4, "x2": 4, "x3": 4},
  "functions": {
    "f_x0": {"scope": ["x0"], "costs": [1.0, -2.5, 3.0, 0.5]},
    "f_x0_x1": {"scope": ["x0", "x1"], "costs": [0.1, 0.2, "...", 1.6]}
  }
}
```

- `variables`: maps each variable name to its cardinality $\lvert d_i\rvert$.
- `functions`: unary tables (one per variable) and pairwise tables (one per edge), with costs flattened row-major and the rightmost scope variable varying fastest (the Toulbar2 convention).
- `mustbe`: an upper bound on the optimal cost, used by Toulbar2 for pruning.

## Dependencies

| Component | Requirement |
|---|---|
| Dataset generation | Python 3.10+, [NumPy](https://numpy.org/) (`pip install numpy`) |
| Ground-state references | [Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1 (not required for generation) |
| Batch ground states | [GNU parallel](https://www.gnu.org/software/parallel/) (optional) |
