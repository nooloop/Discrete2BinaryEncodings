#pragma once
#include "baseline/types.hpp"

// MOMC Lagrange multiplier computation, where:
//   W_c   = max_q sum_{S containing q} |C_S|   (max single-flip objective change)
//   gamma = min_{b: H^penalty(b)>0} H^penalty(b) = 1 for OH, DW, and KH

inline double compute_momc_lambda(const BinaryPolynomial& obj_poly) {
    // For each qubit q, accumulate sum_{S containing q} |C_S|.
    //
    // The bound must be on the largest objective change a single flip of q can
    // produce over ALL settings of the other qubits. Each term S containing q
    // contributes at most |C_S| (the other factors in the monomial are chosen
    // adversarially), so the tight bound is the sum of magnitudes. Summing the
    // signed coefficients instead lets terms of opposite sign cancel, which
    // under-estimates W_c whenever the costs are mixed-sign and yields a penalty
    // too weak to keep the encoded ground state feasible.

    std::unordered_map<int, double> per_qubit_abs_sum;

    for (auto& [indices, coeff] : obj_poly.terms) {
        double abs_c = std::abs(coeff);
        if (abs_c < 1e-18) continue;
        for (int q : indices)
            per_qubit_abs_sum[q] += abs_c;
    }

    // W_c = max over all qubits
    double W_c = 0;
    for (auto& [q, sum] : per_qubit_abs_sum)
        W_c = std::max(W_c, sum);

    return W_c;
}
