#!/usr/bin/env python3
"""
dimod_converter.py
==================

Convert JSON-format QUBO/HUBO/Ising model files produced by encode_cfn
into dimod BinaryQuadraticModel (for degree ≤ 2) or BinaryPolynomial
(for degree > 2) objects.

Usage:
    python dimod_converter.py model.json              # print summary
    python dimod_converter.py model.json --to-bqm     # return BQM (degree ≤ 2)
    python dimod_converter.py model.json --to-poly     # return BinaryPolynomial

As a library:
    from dimod_converter import load_model, to_bqm, to_polynomial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_model(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def to_bqm(model: dict):
    """Convert a JSON model to a dimod BinaryQuadraticModel (degree ≤ 2 only)."""
    import dimod

    vartype_str = model.get("variable_type", "BINARY")
    vartype = dimod.SPIN if vartype_str == "SPIN" else dimod.BINARY
    offset = model.get("offset", 0.0)
    terms = model.get("terms", {})

    linear = {}
    quadratic = {}

    for key, coeff in terms.items():
        indices = [int(x) for x in key.split(",")]
        if len(indices) == 1:
            linear[indices[0]] = linear.get(indices[0], 0) + coeff
        elif len(indices) == 2:
            pair = (indices[0], indices[1])
            quadratic[pair] = quadratic.get(pair, 0) + coeff
        else:
            raise ValueError(
                f"Term {key} has degree {len(indices)} > 2; use to_polynomial() instead"
            )

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


def to_polynomial(model: dict):
    """Convert a JSON model to a dimod BinaryPolynomial (any degree)."""
    import dimod

    vartype_str = model.get("variable_type", "BINARY")
    vartype = dimod.SPIN if vartype_str == "SPIN" else dimod.BINARY
    offset = model.get("offset", 0.0)
    terms = model.get("terms", {})

    poly_terms = {}
    for key, coeff in terms.items():
        indices = tuple(int(x) for x in key.split(","))
        poly_terms[indices] = coeff

    if offset != 0:
        poly_terms[()] = poly_terms.get((), 0) + offset

    return dimod.BinaryPolynomial(poly_terms, vartype)


def to_dwave_dict(model: dict) -> dict:
    """
    Convert to a plain Python dict in D-Wave QUBO format:
    {(i, j): Q_ij} where diagonal (i, i) entries are linear biases.
    Only valid for degree ≤ 2 BINARY models.
    """
    terms = model.get("terms", {})
    offset = model.get("offset", 0.0)
    Q = {}

    for key, coeff in terms.items():
        indices = tuple(int(x) for x in key.split(","))
        if len(indices) == 1:
            Q[(indices[0], indices[0])] = coeff
        elif len(indices) == 2:
            Q[indices] = coeff
        else:
            raise ValueError(f"Degree > 2 term found; not a QUBO")

    return Q, offset


def summarize(model: dict):
    """Print a summary of the model."""
    terms = model.get("terms", {})
    offset = model.get("offset", 0.0)

    degree_counts = {}
    max_idx = -1
    for key in terms:
        indices = [int(x) for x in key.split(",")]
        d = len(indices)
        degree_counts[d] = degree_counts.get(d, 0) + 1
        if indices:
            max_idx = max(max_idx, max(indices))

    print(f"Encoding:      {model.get('encoding', 'N/A')}")
    print(f"Variable type: {model.get('variable_type', 'N/A')}")
    print(f"Logical qubits:  {model.get('num_logical_qubits', 'N/A')}")
    print(f"Auxiliary qubits: {model.get('num_auxiliary_qubits', 'N/A')}")
    print(f"Total variables: {max_idx + 1}")
    print(f"Offset:        {offset}")
    print(f"Total terms:   {len(terms)}")
    for d in sorted(degree_counts):
        print(f"  Degree {d}: {degree_counts[d]} terms")


def main():
    ap = argparse.ArgumentParser(description="Convert encode_cfn JSON to dimod objects")
    ap.add_argument("model", help="Path to JSON model file")
    ap.add_argument("--to-bqm", action="store_true", help="Convert to BQM and print")
    ap.add_argument("--to-poly", action="store_true", help="Convert to BinaryPolynomial and print")
    ap.add_argument("--to-qubo-dict", action="store_true", help="Convert to plain QUBO dict")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.to_bqm:
        bqm = to_bqm(model)
        print(bqm)
    elif args.to_poly:
        poly = to_polynomial(model)
        print(poly)
    elif args.to_qubo_dict:
        Q, offset = to_dwave_dict(model)
        print(f"Q = {Q}")
        print(f"offset = {offset}")
    else:
        summarize(model)


if __name__ == "__main__":
    main()
