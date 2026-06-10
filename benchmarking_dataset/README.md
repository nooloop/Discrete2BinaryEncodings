# benchmarking_dataset

Reproducible generator for synthetic, pairwise-decomposable Cost Function Networks (CFNs) used as the common evaluation set across all solvers in the manuscript. Each instance is written in the [Toulbar2](https://github.com/toulbar2/toulbar2) `.cfn` JSON format and deterministically seeded so the dataset can be regenerated exactly.

## Table of Contents

- [Contents](#contents)
- [Usage](#usage)
  - [Generating the dataset](#generating-the-dataset)
  - [Finding ground states with Toulbar2](#finding-ground-states-with-toulbar2)
  - [Instance naming convention](#instance-naming-convention)
  - [CFN file format](#cfn-file-format)
- [Dependencies](#dependencies)

## Contents

This folder contains the generation script. The generated CFN files themselves are stored on the compute cluster (not committed to the repository due to their number and size).

```
benchmarking_dataset/
├── README.md
└── generate_cfn_dataset.py    # deterministic CFN generator (Toulbar2 .cfn)
```

### Dataset parameters

| Axis | Values |
|---|---|
| Variables N | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 |
| Cardinality d = 2^D | D in {1, 2, 3, 4, 5, 6, 7, 8} |
| Edge density rho | 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 |
| Distribution | uniform, Gaussian, exponential, Laplace |
| Draws per setting | 5 |

The edge density is defined as rho = |E| / (N*(N-1)/2), where |E| is the number of pairwise cost tables and N is the number of variables. A unary cost table is drawn for every variable; a full d x d pairwise cost table is drawn for every selected edge and is identically zero otherwise. Cost-table entries are flattened row-major with the rightmost scope variable varying fastest, matching the Toulbar2 `.cfn` convention.

### Distribution parameters

| Distribution | Notation | Parameters |
|---|---|---|
| Uniform | U(a,b) | a = -10, b = 10 |
| Gaussian | N(mu, sigma^2) | mu = 0, sigma = 10 |
| Exponential | Exp(lambda) | lambda = 1 (mean = 1) |
| Laplace | Lap(mu, b) | mu = 0, b = 1 |

## Usage

### Generating the dataset

Generate all instances (reproducible from master seed `20260512`):

```bash
python generate_cfn_dataset.py -o cfns
```

Generate a subset (e.g., only N=4, D=2, uniform):

```bash
python generate_cfn_dataset.py -o cfns --N 4 --D 2 --dist uniform
```

Dry run (print filenames without writing):

```bash
python generate_cfn_dataset.py --dry-run
```

Additional options:

```
-o, --out-dir DIR               Output directory (default: cfn_dataset)
-P, --instances-per-setting N   Draws per setting (default: 5)
--gzip                          Write .cfn.gz (Toulbar2 reads both)
--overwrite                     Overwrite existing files
--N N [N ...]                   Restrict variable counts
--D D [D ...]                   Restrict cardinality exponents
--rho R [R ...]                 Restrict edge densities
--dist D [D ...]                Restrict distributions
```

### Finding ground states with Toulbar2

Ground-state solutions are obtained using [Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1, an exact CFN solver combining best-first branch-and-bound with parallel variable-neighbourhood search. Toulbar2 is run directly on the `.cfn` files without intermediate conversion:

```bash
toulbar2 cfns/CFN_N4_D2_rho0.3_uniform_1.cfn -w
```

For batch ground-state computation on a Slurm cluster with 64-core Intel Ice Lake nodes (3.7 GHz, 1 TB RAM), 32 Toulbar2 tasks are executed in parallel using 1 CPU per task, with the remaining 32 cores held dormant to avoid thermal throttling and resource contention. This can be orchestrated with GNU parallel or disBatch:

```bash
find cfns/ -name "*.cfn" | parallel -j 32 toulbar2 {} -w
```

The `-w` flag instructs Toulbar2 to write the optimal solution to a file. The resulting ground-state energies and timing information serve as the reference against which all other solvers are scored.

### Instance naming convention

Each instance is named:

```
CFN_N<N>_D<D>_rho<rho>_<distribution>_<number>.cfn
```

where `d = 2^D` is the cardinality (number of choices per variable) and `<number>` indexes the independent draw within each `(N, D, rho, distribution)` combination.

### CFN file format

Each `.cfn` file is a Toulbar2 JSON document:

```json
{
  "problem": {"name": "CFN_N4_D2_rho0.3_uniform_1", "mustbe": "<42.5"},
  "variables": {"x0": 4, "x1": 4, "x2": 4, "x3": 4},
  "functions": {
    "f_x0": {"scope": ["x0"], "costs": [1.0, -2.5, 3.0, 0.5]},
    "f_x0_x1": {"scope": ["x0", "x1"], "costs": [0.1, 0.2, ..., 1.6]}
  }
}
```

- `variables`: maps variable names to their cardinality (number of choices).
- `functions`: unary tables (one per variable) and pairwise tables (one per edge), with costs flattened row-major.
- `mustbe`: an upper bound on the optimal cost, used by Toulbar2 for pruning.

## Dependencies

- **Python 3.10+**
- **NumPy** (`pip install numpy`)
- **[Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1** (for ground-state computation; not required for dataset generation)
