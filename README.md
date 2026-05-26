# Discrete2BinaryEncodings

This repository contains all of the benchmarking data, solvers and solver interfaces, and statistical analyses (including reproducible seeds) used in the following manuscript:

> Tristan Zaborniak, Ulrike Stege, and Vikram Khipple Mulligan, ``Binary encodings of discrete variables for quantum and classical combinatorial optimization,'' 2026.

Should these benchmarking data, solvers and solver interfaces, and statistical analyses be copied, reproduced, or used otherwise, please cite the above manuscript, using the arXiv reference: *TODO*.

## Benchmarking Dataset

The `benchmarking_dataset` folder contains:

- A python script (with reproducible seeds) to randomly generate pairwise-decomposable cost function networks (CFNs) in the input format specified by [Toulbar2](https://github.com/toulbar2/toulbar2) (v1.2.1);  
- The CFNs generated using this script used in the above manuscript, where:
	- The number of variables $N\in\{2,4,6,8,10,12,14,16,18,20\}$, 
	- The number of choices per variable $d\in\{2,4,8,16,32,64,128,256\}$, 
	- The edge density $\rho\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9\}$,
	- The CFN coefficients are drawn from $\mathcal{U}(-10,10)$, $\mathcal{N}(0,10)$, $\mathrm{Exp}(1)$, and $\mathrm{Lap}(0,1)$, and
  - The number of CFNs per $(N,d,\rho,\text{distribution})$ combination is 5;
- The [Toulbar2](https://github.com/toulbar2/toulbar2) (v1.2.1) input parameters used to find the optimal solutions to the above problems; and
- The optimal solutions to the above problems, and [Toulbar2](https://github.com/toulbar2/toulbar2) (v1.2.1) timing information for each when run on 3.7 GHz Intel Ice Lake nodes (64 CPUs, 1 TB RAM each), where:
  -  32 optimization tasks are executed in parallel using 1 CPU per task, and
  -  The remaining 32 CPUs are held dormant to avoid thermal throttling and resource contention.

The edge density is defined as $\rho = |E| / \big(|V|(|V|-1)\big)$, where $|E|$ is the number of pairwise cost tables and $|V|=N$ is the number of variables. A unary ("bias") cost table is drawn for every variable; a full $d\times d$ pairwise cost table is drawn for every selected edge, and is identically zero otherwise. Cost-table entries are flattened row-major with the rightmost scope variable varying fastest, matching the Toulbar2 `.cfn` convention.

Each instance is named: `CFN_N<N>_D<D>_rho<rho>_<distribution>_<number>.cfn`, where $d = 2^{D}$, $D\in\{1,\dots,8\}$, and `<number>` indexes the independent draw.

*Note:* We write $\mathcal{U}(a,b)$ for the uniform distribution on $[a,b]$; $\mathcal{N}(\mu,\sigma^2)$ for the normal with mean $\mu$ and variance $\sigma^2$; $\mathrm{Exp}(\lambda)$ for the exponential with rate $\lambda$ (mean $1/\lambda$), supported on $[0,\infty)$; and $\mathrm{Lap}(\mu,b)$ for the Laplace distribution with location $\mu$ and scale $b$.

## Solvers and Solver Interfaces

The `solvers` folder contains the encodings, optimizers, and hardware interfaces used to solve the benchmarking dataset across three platforms: CPU-based simulated annealing, D-Wave `Advantage2` quantum annealing, and quantum imaginary time evolution (QITE) on IBM `Heron` gate-based hardware. All encodings are run on the common dataset above under common per-instance trajectory budgets.

### Encodings

Five encodings of discrete variables into binary variables are provided, each mapping a CFN to a QUBO or higher-order unconstrained binary optimization (HUBO) problem:

- **One-hot (OH)** — $\sum_i d_i$ bits; feasibility enforced by a Lagrange penalty.
- **Domain-wall (DW)** — $\sum_i (d_i - 1)$ bits; feasibility enforced by a Lagrange penalty.
- **Exact-binary (EB)** — $\sum_i \lceil\log_2 d_i\rceil$ bits; exact, generally HUBO (higher-order).
- **Truncated-binary (TB)** — $\sum_i \lceil\log_2 d_i\rceil$ bits, truncated at interaction degree $k$. $k=2$ yields a QUBO directly; $k=3$ retains cubic Walsh modes and requires quadratization.
- **Approximate-binary (AB)** — $\sum_i \lceil\log_2 d_i\rceil$ bits; cost-approximate QUBO.

For EB, TB ($k=2$), TB ($k=3$), and AB, two choice-to-bitstring assignment strategies are tested: the **naive** canonical binary-order assignment, and the **Boltzmann-LI** assignment (Boltzmann-average-sorted with linearly-independent prioritization). One-hot and domain-wall are tested in their canonical form only.

The variant of each encoding run on each platform is summarized below. "Native" denotes the encoding run as-is (HUBO permitted); "Quadratized" denotes Rosenberg quadratization applied to reduce the encoding to a QUBO before solving.

| Encoding | Classical SA | D-Wave `Advantage2` | IBM `Heron` (QITE) |
|---|---|---|---|
| One-hot (OH) | Native | Native | Native |
| Domain-wall (DW) | Native | Native | Native |
| Exact-binary (EB) | Native | Quadratized | Native *and* Quadratized |
| Truncated-binary (TB), $k=2$ | Native (QUBO) | Native (QUBO) | Native (QUBO) |
| Truncated-binary (TB), $k=3$ | Native (HUBO) | Quadratized | Native *and* Quadratized |
| Approximate-binary (AB) | Native | Native | Native |

**Lagrange multipliers.** One-hot and domain-wall use a per-register penalty set by the maximum-of-maximum-contribution (MOMC) bound, $\lambda^{(i)} = (1+\epsilon)\,\Delta_i$ with $\epsilon = 0.1$, where $\Delta_i$ upper-bounds the cost change achievable by altering register $i$. This avoids the dynamic-range inflation incurred by a global constant-times-max-coefficient rule.

**Rosenberg quadratization.** Higher-order terms are reduced to a QUBO via Rosenberg's polynomial, with a uniform penalty $M = 2\cdot\max_{|S|>2}|c_S|$ applied across all reduced terms in an instance.

### Classical Optimizers

- **Toulbar2 reference solver** — [Toulbar2](https://github.com/toulbar2/toulbar2) (v1.2.1), an exact CFN solver combining best-first branch-and-bound with parallel variable-neighbourhood search, used to retrieve the ground-state references against which all other solvers are scored.
- **Custom C++ CFN/HUBO/QUBO simulated annealer** — a single-bit-flip simulated annealer accepting CFNs, HUBOs, and QUBOs through a unified interface, allowing exact-binary and truncated-binary ($k=3$) to be evaluated without quadratization so that any classical-SA performance gap is attributable to encoding-induced landscape features.

### Quantum Optimizers

- **D-Wave `Advantage2` (quantum annealing)** — interfaced via the D-Wave Ocean SDK and the Leap cloud service. Encoded QUBOs are minor-embedded onto the Zephyr topology using `minorminer` (10 attempts per problem, smallest embedding retained); $1000$ anneal-readouts per problem at $T_a = 20\,\mu\text{s}$; chain strength set by uniform torque compensation ($\sqrt{2}$ times the RMS of the quadratic couplings) with majority-vote chain repair; qubit-specific annealing offsets (inhomogeneous transverse-field driving); and per-session calibration-drift controls.
- **IBM `Heron` (QITE)** — interfaced via Qiskit Runtime, run on `ibm_fez` (156-qubit `Heron r2`), with an `Aer` noise-model-matched simulator pass on every tractable instance. Variational QITE under the McLachlan principle with a hardware-efficient ansatz ($R_y$ + CZ, depth $L\in\{2,3,4\}$) and, for QUBO encodings, the imaginary-Hamiltonian variational ansatz (iHVA); imaginary-time step $\Delta\tau\in\{0.01,0.05\}$; total imaginary time $\tau_{\max}\in\{5,10,20\}$; Tikhonov regularization $\epsilon = 10^{-3}$; qubit-wise-commuting Pauli grouping with $10^4$ shots per Pauli string; and error mitigation via zero-noise extrapolation (noise factors $\{1, 1.5, 3\}$), Pauli twirling, and dynamical decoupling.

Per-platform resource accounting (logical/physical qubit counts, chain lengths and chain-break fractions, embedding success rates, coefficient dynamic ranges, Pauli-string counts and weights, two-qubit gate counts, and circuit depths) is recorded alongside every run.

## Statistical Analyses

The `analysis` folder contains the scripts (with reproducible seeds) that compute the performance measures and scaling fits reported in the manuscript:

- **Time-to-solution (TTS)** — the wall-clock time to reach the Toulbar2 ground state with probability $\ge 50\%$,
$$\text{TTS} = T_S\cdot\frac{\ln(1-0.5)}{\ln(1-p_S)} + T_E + T_P + T_R,$$
with per-trajectory success probability $p_S$ and trajectory, encoding, programming, and readout times $T_S, T_E, T_P, T_R$.
- **Solution quality (optimality gap)** — $\Delta = (E_{\text{best}} - E_{\text{GS}})/(E_{\text{worst}} - E_{\text{GS}})$, where $E_{\text{GS}}$ is the Toulbar2 ground state and $E_{\text{worst}}$ is the worst-case feasible energy over the native CFN.
- **Scaling fits** — power-law fits $\text{TTS} = \alpha S^{\beta}$ over solution-space size $S = \prod_i d_i = 2^{ND}$ via right-censored maximum-likelihood estimation under a log-normal multiplicative-noise model, supplemented by a three-factor multivariate fit $\text{TTS} = \alpha S^{\beta_S} d^{\beta_d} (1+\rho)^{\beta_\rho}$ to disentangle the contributions of size, cardinality, and density.
- **Statistical inference** — pairwise differences in scaling exponents tested via nested-model likelihood-ratio tests, with family-wise error rate controlled by Holm–Bonferroni adjustment over the $10$ pairwise comparisons per platform.
- **Failure-mode reporting** — local-minimum trapping (SA), embedding failures and chain-break rates (D-Wave), and ansatz-expressibility floors (QITE); no encoding is excluded from a comparison on the grounds of having failed on a problem class.

## Repository Structure
```
Discrete2BinaryEncodings/
├── benchmarking_dataset/
│   ├── generate_cfn_dataset.py      # reproducible CFN generator (Toulbar2 .cfn)
│   ├── cfns/                        # generated benchmark instances
│   ├── toulbar2_params/             # Toulbar2 v1.2.1 input parameters
│   └── solutions/                   # ground states + timing information
├── solvers/
│   ├── encodings/                   # OH, DW, EB, TB (k=2,3), AB; naive & Boltzmann-LI
│   ├── simulated_annealer/          # custom C++ CFN/HUBO/QUBO simulated annealer
│   ├── dwave/                       # D-Wave Ocean / Advantage2 interface
│   └── qite/                        # Qiskit / IBM Heron QITE interface
├── analysis/                        # performance measures, scaling fits, figures
└── README.md
```

## Requirements

- **Python** 3.10+ with `numpy` and `scipy` (dataset generation and statistical analysis).
- **[Toulbar2](https://github.com/toulbar2/toulbar2) v1.2.1** (and/or `pytoulbar2`) for ground-state references.
- A C++17 compiler for the custom simulated annealer.
- **D-Wave Ocean SDK** (`dwave-system`, `dimod`, `minorminer`) and Leap access for the quantum-annealing experiments.
- **Qiskit** (`qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`) and IBM Quantum access for the QITE experiments.

## Usage

Generate the benchmarking dataset (reproducible from fixed seeds):
```bash
python benchmarking_dataset/generate_cfn_dataset.py -o benchmarking_dataset/cfns
```
Retrieve a reference ground state with Toulbar2:
```bash
toulbar2 benchmarking_dataset/cfns/CFN_N4_D2_rho0.5_uniform_1.cfn
```

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
