# Discrete2BinaryEncodings

Benchmarking data, encoders, solvers, solver interfaces, and analyses for the manuscript:

> Tristan Zaborniak, Ulrike Stege, and Vikram Khipple Mulligan, "Binary encodings of discrete variables for quantum and classical combinatorial optimization," 2026.

This repository provides the evaluation pipeline for five binary-variable encodings of pairwise-decomposable Cost Function Networks (CFNs) across classical simulated annealing, quantum annealing, and gate-based quantum imaginary time evolution. All benchmarking instances are generated deterministically and all solver settings are stated explicitly, for full reproducibility.

A CFN over $N$ discrete variables $\vec{d}=(d_1,\dots,d_N)$ of cardinalities $\lvert d_1\rvert,\dots,\lvert d_N\rvert$ has the indicator-form objective:

$$
H_{\text{CFN}}(\vec{x}) = \sum_{\varnothing \neq S \subseteq [N]} \; \sum_{\boldsymbol{c}\in\mathcal{C}_S} C_{S;\boldsymbol{c}} \prod_{i\in S} x_{i,c_i},
\qquad x_{i,c_i}\in\{0,1\},
$$

and every encoding here replaces the discrete indicator $x_{i,c_i}$ with a specific function of binary variables, producing a QUBO, HUBO, or Ising model.

If these benchmarking data, encoders, solvers, or analyses are copied, reproduced, or otherwise used, please cite the manuscript (see [Citations](#citations)).

## Table of Contents

- [Contents](#contents)
- [Pipeline](#pipeline)
  - [Dataset](#dataset)
  - [Encodings](#encodings)
  - [Solvers](#solvers)
  - [Analyses](#analyses)
- [Dependencies](#dependencies)
- [Citations](#citations)

## Contents

```
Discrete2BinaryEncodings/
├── README.md
├── benchmarking_dataset/           # CFN instance generator
│   ├── generate_cfn_dataset.py     #   deterministic CFN generator (outputs to Toulbar2 .cfn format)
│   └── README.md
├── encoding_tools/                 # CFN to QUBO/HUBO/Ising encoders
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── src/                        #   C++17 source (one-hot, domain-wall, exact-binary, approximate-binary, truncated-binary encodings)
│   ├── python/                     #   dimod converter for D-Wave
│   ├── tests/                      #   unit tests
│   ├── test_cfns/                  #   sample CFN inputs
│   └── test_output/                #   sample encoded outputs
└── solver_tools/                   # SA + D-Wave QA + IBM QITE solvers
    ├── CMakeLists.txt
    ├── README.md
    ├── src/                        #   C++17 source (SA, QA, QITE, parsers, output)
    ├── scripts/                    #   benchmarking scripts
    └── tests/                      #   unit tests
```

Each sub-directory has its own README with detailed contents, build/test instructions, CLI references, usage examples, and input/output format descriptions.

## Pipeline

The benchmark proceeds in four stages: generate a common dataset, encode each instance under every encoding, solve each encoded model on each platform, and analyze the results across problem axes.

```
 benchmarking_dataset/        encoding_tools/            solver_tools/                     analyses
 generate_cfn_dataset.py  ->  encode_cfn           ->    solve_sa   (classical SA)    ->   TTS, optimality gap,
        │                          │                     solve_qa   (D-Wave QA)            scaling fits, qubit /
   .cfn instances            .json models                solve_qite (IBM QITE)             chain / gate statistics
   (Toulbar2 format)        (QUBO / HUBO / Ising)        results .csv
```

### Dataset

[`benchmarking_dataset/`](benchmarking_dataset) generates synthetic, pairwise-decomposable CFNs stratified across four axes (variable count $N$, cardinality $\lvert d\rvert=2^{D}$, edge density $\rho$, and coefficient distribution $R$) with $P=5$ independent draws per setting, for:

$$
\lvert\mathcal{N}\rvert \times \lvert\mathcal{D}\rvert \times \lvert\boldsymbol{\rho}\rvert \times \lvert\mathcal{R}\rvert \times P \;=\; 10 \times 8 \times 9 \times 4 \times 5 \;=\; 14{,}400
$$

instances. Cardinalities are constrained to powers of two to avoid codeword-duplication artifacts in the bit-efficient encodings, and the four distributions (uniform, Gaussian, exponential, Laplace) span bounded/unbounded and light-/heavy-tailed regimes. Every coefficient is drawn from a fixed, reproducible seed (`20260512`). [Toulbar2](https://github.com/toulbar2/toulbar2) is run on every instance to produce reference ground states $E_{\text{GS}}$, against which all solvers are scored. See [`benchmarking_dataset/README.md`](benchmarking_dataset/README.md).

### Encodings

[`encoding_tools/`](encoding_tools) implements five binary encodings, split into two families. The *cost-preserving* encodings (one-hot, domain-wall, exact-binary) reproduce feasible-solution energies exactly; the *cost-approximate* encodings (approximate-binary, truncated-binary) accept bounded error in exchange for minimal bit count and degree.

| Encoding | Family | Bits per variable | Variable type | Output degree |
|---|---|---|---|---|
| **One-hot (OH)** | cost-preserving | $\lvert d_i\rvert$ | BINARY $\{0,1\}$ | $2$ (QUBO) |
| **Domain-wall (DW)** | cost-preserving | $\lvert d_i\rvert - 1$ | BINARY $\{0,1\}$ | $2$ (QUBO) |
| **Exact-binary (EB)** | cost-preserving | $\lceil\log_2\lvert d_i\rvert\rceil$ | BINARY $\{0,1\}$ | up to $\sum_j\lceil\log_2\lvert d_{i_j}\rvert\rceil$ (HUBO) |
| **Approximate-binary (AB)** | cost-approximate | $\lceil\log_2\lvert d_i\rvert\rceil$ | BINARY $\{0,1\}$ | $k_{\text{approx}}$ (default $2$, QUBO) |
| **Truncated-binary (TB)** | cost-approximate | $\lceil\log_2\lvert d_i\rvert\rceil$ | SPIN $\{-1,+1\}$ | $k_{\text{trunc}}$ (default $2$, QUBO) |

- **One-hot** and **domain-wall** use one binary variable per choice (resp. per choice boundary) and enforce feasibility with a MOMC Lagrange penalty $\lambda_i = W_c/\gamma$, where $W_c$ bounds the single-flip objective gain and $\gamma$ the minimum infeasible penalty. See [Lucas, 2014](https://doi.org/10.3389/fphy.2014.00005), [Berwald et al., 2023](https://doi.org/10.1098/rsta.2021.0410), and [Ayodele, 2022](https://doi.org/10.1007/978-3-031-04148-8_11).
- **Exact-binary** expands each indicator by inclusion–exclusion into a multilinear polynomial, using a logarithmic number of bits in the number of choices per variable. The encoding is exact, but generally produces terms of higher order. See [Berwald et al., 2023](https://doi.org/10.1098/rsta.2021.0410).
- **Approximate-binary** fits coefficients by least squares onto a degree-$\le k_{\text{approx}}$ monomial basis, using a logarithmic number of bits in the number of choices per variable. The encoding eliminates infeasible solutions and Lagrange multipliers while bounding degree to the user-specified $k_{\text{approx}}$. See [Zaborniak, 2025](https://hdl.handle.net/1828/22736).
- **Truncated-binary** projects the exact-binary cost onto a degree-$\le k_{\text{trunc}}$ Walsh–Hadamard expansion $H_{\text{EB}}(\vec{b})=\sum_S \widehat{H}_S\,\chi_S(\vec{b})$, with $\chi_S(\vec{b})=\prod_{q\in S}(1-2b_q)$. See [Zaborniak, 2026](https://doi.org/10.48550/arXiv.2605.17143).

Both cost-approximate encodings carry an $L^\infty$ error bound $\varepsilon$ that controls optimum preservation: if the exact-binary energy gap satisfies $\Delta E^\star > 2\varepsilon$, every minimizer of the approximate model is also a minimizer of $H_{\text{EB}}$. For EB, AB, and TB, two choice-to-bitstring assignment strategies are tested: the **naive** canonical binary-order assignment and the **enhanced** assignment (Boltzmann-average-sorted choices with Gray-code bitstrings, plus linearly-independent prioritization for AB). Higher-order encodings (EB, and TB with $k_{\text{trunc}}\ge 3$) are reduced to QUBOs via Rosenberg quadratization, with penalty $M = 2\max_{\lvert S\rvert>2}\lvert c_S\rvert$, for platforms that admit only degree-$2$ interactions. See [`encoding_tools/README.md`](encoding_tools/README.md).

### Solvers

The manuscript benchmarks all encodings on three platforms:

1. **Classical simulated annealing**: a CPU baseline that solves the native CFN or any encoded QUBO/HUBO directly through single-bit-flip (or multi-valued) moves. Per problem, 100 independent trajectories are run with a five-rampdown geometric schedule ($T_{\text{high}}=100$, $T_{\text{low}}=0.3$ kcal/mol) and $100\cdot N\cdot\lvert d\rvert$ steps per trajectory under Metropolis acceptance.
2. **D-Wave `Advantage2` quantum annealing**: an analog, finite-precision quantum annealing device admitting only quadratic couplings (Zephyr topology). Encoded QUBOs are minor-embedded with `minorminer` (smallest of 10 attempts), run with $R=1000$ reads at annealing time $T_a=20\,\mu\text{s}$, chain strength set by uniform torque compensation ($\sqrt{2}$ times the RMS coupling), and per-qubit inhomogeneous transverse-field driving (see [Zaborniak, 2025](https://hdl.handle.net/1828/22736)).
3. **Quantum imaginary time evolution (QITE)** on IBM `Heron` gate-based hardware: a digital quantum device admitting higher-order Pauli strings via phase-gadget decomposition, run under the McLachlan variational principle with a hardware-efficient ansatz.

This repository provides the each solver/solver interface in [`solver_tools/`](solver_tools). Each solver invocation emits one CSV row per instance with energy statistics, the decoded best CFN solution, timing, and platform-specific resource metrics. See [`solver_tools/README.md`](solver_tools/README.md).

### Analyses

The per-instance CSV outputs of the solvers are analyzed across the dataset's stratification axes (solution-space size $S=\prod_i\lvert d_i\rvert = 2^{ND}$, cardinality $\lvert d\rvert$, and edge density $\rho$) using four primary performance measures:

- **Time-to-solution (TTS)** — the wall-clock time to reach $E_{\text{GS}}$ with probability $\ge 99\%$: $\text{TTS} = T_S\cdot\frac{\ln(1-0.99)}{\ln(1-p_S)} + T_E + T_P + T_R$, where $p_S$ is the per-trajectory success probability and $T_S, T_E, T_P, T_R$ are the trajectory, encoding, programming, and readout times.

- **Solution quality** — the optimality gap $\Delta = \frac{E_{\text{best}} - E_{\text{GS}}}{E_{\text{worst}} - E_{\text{GS}}}$, reported as a distribution over instances and as a function of $S$, $\lvert d\rvert$, and $\rho$.

- **Scaling fits** — power laws $\text{TTS}=\alpha S^{\beta}$ fit by right-censored maximum-likelihood estimation under a log-normal noise model, supplemented by a multivariate fit $\text{TTS}=\alpha S^{\beta_S}\lvert d\rvert^{\beta_d}(1+\rho)^{\beta_\rho}$. Pairwise differences in scaling exponents are tested by nested-model likelihood-ratio tests with Holm–Bonferroni family-wise error-rate control over the $\binom{5}{2}=10$ encoding comparisons per platform.

- **Resource accounting and failure modes** — per encoding and platform: logical bit count, couplings by interaction degree, maximum and coefficient-weighted mean degree $\langle k\rangle$, and coefficient dynamic range $\lvert c_{\max}\rvert/\lvert c_{\min}\rvert$ for SA; physical-qubit count, chain length, chain-break fraction, and embedding success rate for QA; and Pauli-string count, Pauli weight, two-qubit gate count, and circuit depth for QITE. Failure modes (SA trapping, D-Wave chain breaks, QITE expressibility floors) are reported.

Figures and tables are produced from the solver CSV outputs described above.See [`analysis_tools/README.md`](analysis_tools/README.md).

## Dependencies

| Component | Requirements |
|---|---|
| **Dataset generation** | Python 3.10+, [NumPy](https://numpy.org/) |
| **Ground-state references** | [Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1 |
| **Encoding** | C++17 compiler, CMake 3.14+, [nlohmann/json](https://github.com/nlohmann/json) (fetched), [Eigen](https://eigen.tuxfamily.org/) (fetched), OpenMP (optional) |
| **SA solving** | C++17 compiler, CMake 3.14+, nlohmann/json (fetched) |
| **D-Wave QA solving** | Above + Python 3 with [dwave-ocean-sdk](https://docs.ocean.dwavesys.com/) and D-Wave Leap access |
| **IBM QITE solving** | Above + Python 3 with [qiskit](https://www.ibm.com/quantum/qiskit) Qiskit Runtime access|

## Citations

If you use this repository, please cite the manuscript:

```bibtex
@article{Zaborniak2026BinaryEncodings,
  title   = {Binary encodings of discrete variables for quantum and classical combinatorial optimization},
  author  = {Zaborniak, Tristan and Stege, Ulrike and Mulligan, Vikram Khipple},
  year    = {2026},
  note    = {arXiv: TODO}
}
```

This work additionally builds on third-party tools that should be cited where appropriate: [Toulbar2](https://github.com/toulbar2/toulbar2) (reference ground states), the [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/) (quantum annealing), [nlohmann/json](https://github.com/nlohmann/json), and [Eigen](https://eigen.tuxfamily.org/), etc.
