#pragma once
#include "baseline/types.hpp"

// MOMC Lagrange multiplier computation, where:
//   W_c   = max_q sum_{S containing q} |C_S|   (max single-flip objective change)
//   gamma = min_{b: H^penalty(b)>0} H^penalty(b) = 1 for OH, DW, and KH

inline double compute_momc_lambda(const BinaryPolynomial& obj_poly) {
    // For each qubit q, accumulate sum_{S containing q} |C_S|
    
    std::unordered_map<int, double> per_qubit_sum_positive;
    std::unordered_map<int, double> per_qubit_sum_negative;

    for (auto& [indices, coeff] : obj_poly.terms) {
        double abs_c = std::abs(coeff);
        if (abs_c < 1e-18) continue;
        for (int q : indices) {
            per_qubit_sum_positive[q] += coeff;
            per_qubit_sum_negative[q] -= coeff;
        }
    }

    std::unordered_map<int, double> per_qubit_sum_max;
    
    for (const auto& [q, neg] : per_qubit_sum_negative)
        per_qubit_sum_max[q] = std::max(neg, per_qubit_sum_positive.at(q));

    // W_c = max over all qubits
    double W_c = 0;
    for (auto& [q, sum] : per_qubit_sum_max)
        W_c = std::max(W_c, sum);

    return W_c;
}
