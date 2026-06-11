# Discrete2BinaryEncodings

Benchmarking data, encoders, and solvers for the manuscript:

> Tristan Zaborniak, Ulrike Stege, and Vikram Khipple Mulligan, "Binary encodings of discrete variables for quantum and classical combinatorial optimization," 2026.

This repository provides the complete evaluation pipeline of five binary-variable encodings of pairwise-decomposable Cost Function Networks (CFNs) across classical simulated annealing, quantum annealing, and quantum imaginary time evolution. Tools include: CFN dataset generation, CFN encoding, solvers and solver interfaces, and analysis and plotting scripts. All benchmarking problems are generated deterministically and all solver settings are noted explicitly for reproducibility.

If these benchmarking data, solvers, or analyses are copied, reproduced, or used otherwise, please cite the above manuscript (arXiv: *TODO*).

## Table of Contents

- [Contents](#contents)
- [Pipeline Overview](#pipeline_overview)
- [Encodings](#encodings)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Citation](#citation)

## Contents

```
Discrete2BinaryEncodings/
├── README.md
├── benchmarking_dataset/           # CFN instance generator
│   ├── generate_cfn_dataset.py     #   deterministic CFN generator (Toulbar2 .cfn)
│   └── README.md
├── encoding_tools/                 # CFN to QUBO/HUBO/Ising encoder
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── src/                        #   C++17 source (OH, DW, EB, AB, TB encodings)
│   ├── python/                     #   dimod converter for D-Wave
│   ├── tests/                      #   unit tests
│   ├── test_cfns/                  #   sample CFN inputs
│   └── test_output/                #   sample encoded outputs
└── solver_tools/                   # SA + D-Wave QA solvers
    ├── CMakeLists.txt
    ├── README.md
    ├── src/                        #   C++17 source (SA, QA, parsers, output)
    ├── scripts/                    #   benchmarking scripts
    └── tests/                      #   unit tests
```

Each subfolder has its own README with detailed contents explanations, build instructions, CLI references, usage overviews, and input/output format descriptions.

## Pipeline overview

```
benchmarking_dataset/               encoding_tools/               solver_tools/
generate_cfn_dataset.py   --->   encode_cfn              --->   solve_sa  (SA)
        |                             |                          solve_qa  (D-Wave QA)
   .cfn files                    .json models                   results .csv
   (Toulbar2 format)          (QUBO/HUBO/Ising)           (energies, solutions,
                                                           timing, hardware resource stats)
```

1. **Generate** CFN instances with `benchmarking_dataset/generate_cfn_dataset.py`.
2. **Encode** each CFN into one or more binary-variable models with `encoding_tools/encode_cfn`.
3. **Solve** each encoded model with `solver_tools/solve_sa` (simulated annealing) or `solver_tools/solve_qa` (D-Wave quantum annealing), producing per-instance CSV rows with energy statistics, decoded CFN solutions, and platform-specific metrics.

## Encodings

Five encodings of discrete CFN variables into binary variables are provided:

| Encoding | Bits per variable | Variable type | Output degree |
|---|---|---|---|
| **One-hot (OH)** | d_i | BINARY {0,1} | 2 (QUBO) |
| **Domain-wall (DW)** | d_i - 1 | BINARY {0,1} | 2 (QUBO) |
| **Exact-binary (EB)** | ceil(log2(d_i)) | BINARY {0,1} | up to ceil(log2(d_i)) (HUBO) |
| **Approximate-binary (AB)** | ceil(log2(d_i)) | BINARY {0,1} | k_approx (default 2, QUBO) |
| **Truncated-binary (TB)** | ceil(log2(d_i)) | SPIN {-1,+1} | k_trunc (default 2) |

For EB, TB, and AB, two choice-to-bitstring assignment strategies are tested: the **naive** canonical binary-order assignment, and the **enhanced** assignment (Boltzmann-average-sorted choices with Gray-code bitstrings and linearly-independent prioritization). OH and DW are tested in their canonical form only.

Higher-order encodings (EB, TB with k >= 3) can be reduced to QUBOs via Rosenberg quadratization for platforms that require degree-2 interactions (e.g., D-Wave).

See `encoding_tools/README.md` for detailed encoding descriptions, Lagrange multiplier computation, and quadratization.

## Usage

### 1. Generate the benchmarking dataset

```bash
cd benchmarking_dataset
python generate_cfn_dataset.py -o ../cfns
```

See `benchmarking_dataset/README.md` for dataset parameters and ground-state computation with Toulbar2.

### 2. Encode CFNs

```bash
cd encoding_tools
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# One-hot encoding
./build/encode_cfn --input-dir ../cfns --output-dir ../encoded/oh --csv metrics.csv --encoding one_hot

# Truncated-binary (k=2) with enhanced assignment
./build/encode_cfn --input-dir ../cfns --output-dir ../encoded/tb2 --csv metrics.csv \
    --encoding truncated_binary --k-trunc 2 \
    --choice-ordering boltzmann --bitstring-ordering gray --li-prioritization
```

See `encoding_tools/README.md` for all encoding options and output format.

### 3. Solve

**Simulated annealing** (CPU, any encoding):
```bash
cd solver_tools
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

./build/solve_sa --mode binary --input ../encoded/oh/problem_one_hot.json \
    --num-runs 100 --steps-multiplier 200
```

**D-Wave quantum annealing** (QUBO/Ising encodings only):
```bash
./build/solve_qa --input ../encoded/oh/problem_one_hot.json \
    --cfn-dir ../cfns --solver Advantage2_system1 \
    --num-reads 1000 --annealing-time 20
```

**Batch benchmarking** (Slurm + disBatch):
```bash
sbatch scripts/slurm_qa_benchmark.slurm ../encoded/tb2 ../output/tb2
```

See `solver_tools/README.md` for full CLI references, batch scripts, and output CSV format.

## Dependencies

| Component | Requirements |
|---|---|
| **Dataset generation** | Python 3.10+, NumPy |
| **Ground-state references** | [Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1 |
| **Encoding** | C++17 compiler, CMake 3.14+, nlohmann/json (fetched), Eigen (fetched) |
| **SA solving** | C++17 compiler, CMake 3.14+, nlohmann/json (fetched) |
| **D-Wave QA solving** | Above + Python 3 with [dwave-ocean-sdk](https://docs.ocean.dwavesys.com/), D-Wave Leap access |

## License

*TODO*

## Citation

```bibtex
@article{Zaborniak2026BinaryEncodings,
  title   = {Binary encodings of discrete variables for quantum and classical combinatorial optimization},
  author  = {Zaborniak, Tristan and Stege, Ulrike and Mulligan, Vikram Khipple},
  year    = {2026},
  note    = {arXiv: TODO}
}
```
