#pragma once
#include "baseline/types.hpp"

// MOMC Lagrange multiplier computation, where:
//   W_c   = max_q sum_{S containing q} |C_S|   (max single-flip objective change)
//   gamma = min_{b: H^penalty(b)>0} H^penalty(b) = 1 for OH, DW, and KH

// Strict-feasibility margin. lambda = W_c/gamma satisfies the MOMC criterion only
// with a NON-STRICT inequality: W_c bounds the objective gain of a single flip
// over ALL settings of the other qubits, so when an infeasible state one flip
// from the feasible optimum ATTAINS that bound, it gains exactly W_c on the
// objective and pays exactly gamma*lambda = W_c on the penalty -- coming out
// degenerate with the feasible optimum. The encoded ground state is then not
// unique, and a solver may return an infeasible state that decodes to nothing.
//
// The bound is attained only in the worst case, so typical instances clear it
// with room to spare, but nothing in the construction rules the tie out.
// Scaling lambda by (1 + kFeasibilityMargin) makes the inequality strict: every
// infeasible state is now worse than every feasible one by at least
// kFeasibilityMargin * W_c. The margin is relative (so it is invariant to a
// rescaling of the objective) with an absolute floor for the degenerate W_c = 0
// case, and sits far below the coefficient scale, so it does not otherwise
// distort the landscape.
constexpr double kFeasibilityMargin = 1e-6;

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

    return W_c * (1.0 + kFeasibilityMargin) + kFeasibilityMargin;
}
