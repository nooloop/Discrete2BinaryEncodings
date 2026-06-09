#!/usr/bin/env python3

"""
Generate synthetic, pairwise-decomposable Cost Function Networks (CFNs) of the "Benchmarking Dataset" section of "Binary Encodings of Discrete Variables for Quantum and Classical Optimization" and write each instance in the Toulbar2 .cfn format.

Per instance: one unary cost table per variable, and one pairwise table on each selected edge of the interaction graph is generated. Coefficients are drawn independently from the specified distribution. Every instance is seeded deterministically, so the dataset is reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
from itertools import combinations

import numpy as np

# --- benchmarking stratification axes ---

VARIABLE_COUNTS       = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]                    # N
CARDINALITY_EXPONENTS = [1, 2, 3, 4, 5, 6, 7, 8]                                # D, |d| = 2**D
EDGE_DENSITIES        = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # \rho
DISTRIBUTIONS         = ["uniform", "Gaussian", "exponential", "laplace"]       # R
INSTANCES_PER_SETTING = 5                                                       # number of random repeats per combination

# --- coefficient distribution parameters ---

UNIFORM_LOW, UNIFORM_HIGH  = -10.0, 10.0   # U(-10, 10)
GAUSSIAN_STD               = 10.0          # N(0, 10); second argument is standard deviation
EXPONENTIAL_SCALE          = 1.0           # Exp(1)
LAPLACE_LOC, LAPLACE_SCALE = 0.0, 1.0      # Lap(0, 1)

# --- randomness and output ---

MASTER_SEED     = 20260512      # global seed; change to regenerate a fresh dataset
COST_DECIMALS   = 6             # fixed-point precision for written costs
DEFAULT_OUT_DIR = "cfn_dataset"


def sample_coeffs(dist, rng, size):
    
    # draw `size` coefficients from distribution `dist`:
    
    if dist == "uniform":     return rng.uniform(UNIFORM_LOW, UNIFORM_HIGH, size)
    if dist == "Gaussian":    return rng.normal(0.0, GAUSSIAN_STD, size)
    if dist == "exponential": return rng.exponential(EXPONENTIAL_SCALE, size)
    if dist == "laplace":     return rng.laplace(LAPLACE_LOC, LAPLACE_SCALE, size)
    
    raise ValueError(f"unknown distribution: {dist!r}")


def instance_rng(N, D, rho_idx, dist_idx, number):
    
    # independent, reproducible random-number generator for one instance:
    
    return np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, N, D, rho_idx, dist_idx, number]))


def select_edges(N, rho, rng):
    
    # pick exactly round(rho * {N choose 2}) undirected edges at random; sorted i<j:
    
    pairs = list(combinations(range(N), 2))
    max_edges = len(pairs)                       # {N choose 2}
    m = int(math.floor(rho * max_edges + 0.5))   # undirected density convention
    m = max(0, min(m, max_edges))
    
    if m == 0:
        return []
    
    return sorted(pairs[k] for k in rng.choice(max_edges, size=m, replace=False))


def build_instance(N, D, dist, rho, rho_idx, dist_idx, number):
    
    # return (variables, functions, top, n_edges) for one CFN instance:
    
    card = 2 ** D
    rng = instance_rng(N, D, rho_idx, dist_idx, number)
    names = [f"x{i}" for i in range(N)]

    functions, worst_total = [], 0.0
    
    for i in range(N):                                    # unary tables
        c = sample_coeffs(dist, rng, card)
        functions.append((f"c_{names[i]}", [names[i]], c))
        worst_total += float(c.max())
    
    edges = select_edges(N, rho, rng)
    
    for (i, j) in edges:                                  # pairwise tables
        c = sample_coeffs(dist, rng, card * card)
        functions.append((f"c_{names[i]}_{names[j]}", [names[i], names[j]], c))
        worst_total += float(c.max())
    
    top = math.ceil(worst_total) + 1.0
    
    return [(v, card) for v in names], functions, top, len(edges)


def _fmt_array(arr):
    
    # comma-joined fixed-point list:
    
    rounded = np.round(arr, COST_DECIMALS) + 0.0
    return ",".join(np.char.mod(f"%.{COST_DECIMALS}f", rounded).tolist())


def write_cfn(path, name, variables, functions, top, use_gzip=False):
    
    # write one instance as Toulbar2 JSON:
    
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if use_gzip \
        else (lambda p: open(p, "w", encoding="utf-8"))
    
    with opener(path) as f:
        
        f.write("{\n")
        f.write(f'  "problem": {{"name": "{name}", "mustbe": "<{top:.{COST_DECIMALS}f}"}},\n')
        f.write('  "variables": {%s},\n'
                % ", ".join(f'"{v}": {d}' for v, d in variables))
        f.write('  "functions": {\n')
        
        for k, (fname, scope, costs) in enumerate(functions):
            scope_str = ", ".join(f'"{s}"' for s in scope)
            comma = "," if k < len(functions) - 1 else ""
            f.write(f'    "{fname}": {{"scope": [{scope_str}], '
                    f'"costs": [{_fmt_array(costs)}]}}{comma}\n')
        
        f.write("  }\n}\n")


def iter_settings(args):
    
    # iterate over (N, D, dist, dist_idx, rho, rho_idx, number):
    
    Ns    = args.N    or VARIABLE_COUNTS
    Ds    = args.D    or CARDINALITY_EXPONENTS
    dists = args.dist or DISTRIBUTIONS
    
    for N in Ns:
        for D in Ds:
            for dist in dists:
                di = DISTRIBUTIONS.index(dist)
                for ri, rho in enumerate(EDGE_DENSITIES):
                    if args.rho and not any(abs(rho - r) < 1e-9 for r in args.rho):
                        continue
                    for number in range(1, args.instances_per_setting + 1):
                        yield N, D, dist, di, rho, ri, number


def fname_for(N, D, rho, dist, number):
    return f"CFN_N{N}_D{D}_rho{rho:g}_{dist}_{number}.cfn"


def main(argv=None):
    
    ap = argparse.ArgumentParser(
        description="Generate the benchmarking CFN dataset in Toulbar2 .cfn format.")
    ap.add_argument("-o", "--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("-P", "--instances-per-setting", type=int,
                    default=INSTANCES_PER_SETTING)
    ap.add_argument("--gzip", action="store_true",
                    help="write .cfn.gz (toulbar2 reads both)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--N", type=int, nargs="+", help="restrict variable counts")
    ap.add_argument("--D", type=int, nargs="+", help="restrict cardinality exponents")
    ap.add_argument("--rho", type=float, nargs="+", help="restrict densities")
    ap.add_argument("--dist", nargs="+", choices=DISTRIBUTIONS, help="restrict distributions")
    
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    settings = list(iter_settings(args))
    
    print(f"Planned instances: {len(settings)}")
    print(f"Output directory : {os.path.abspath(args.out_dir)}")
    print(f"Master seed      : {MASTER_SEED}")
    
    if args.dry_run:
        for s in settings[:20]:
            print("  ", fname_for(s[0], s[1], s[4], s[2], s[6]))
        if len(settings) > 20:
            print(f"   ... ({len(settings) - 20} more)")
        return 0

    done = 0
    
    for (N, D, dist, di, rho, ri, number) in settings:
        base = fname_for(N, D, rho, dist, number)
        path = os.path.join(args.out_dir, base + (".gz" if args.gzip else ""))
        if args.overwrite or not os.path.exists(path):
            variables, functions, top, n_edges = build_instance(
                N, D, dist, rho, ri, di, number)
            write_cfn(path, base[:-4], variables, functions, top, args.gzip)
        done += 1
        if done % 200 == 0 or done == len(settings):
            print(f"  [{done}/{len(settings)}] {base}")
    print(f"Done: {done} instances in {os.path.abspath(args.out_dir)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())