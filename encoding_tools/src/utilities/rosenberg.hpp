#pragma once
#include "baseline/types.hpp"

// Rosenberg quadratization: reduce a HUBO to QUBO by introducing auxiliary variables.
// For each cubic+ monomial c * b_p b_q b_r..., pick a pair (p,q), introduce y_{pq},
// replace b_p b_q with y, add penalty M*(b_p b_q - 2 b_p y - 2 b_q y + 3 y).
//
// For BINARY {0,1}: standard Rosenberg.
// For SPIN {-1,+1}: Ising-basis Rosenberg.
//   Penalty in Ising: M/4 * (3 + s_p + s_q - 2t + s_p*s_q - 2*s_p*t - 2*s_q*t)
//   Replacement: c * s_p s_q s_r... -> c * t * s_r... + penalty

struct QuadratizationResult {
    BinaryPolynomial poly;
    int num_auxiliaries = 0;
    double penalty_strength = 0;
};

inline QuadratizationResult rosenberg_quadratize(const BinaryPolynomial& input) {
    QuadratizationResult qr;
    qr.poly.var_type = input.var_type;
    qr.poly.offset = input.offset;

    bool is_spin = (input.var_type == VarType::SPIN);

    // Compute penalty strength: M = 2 * max_{|S|>2} |c_S|
    double max_high_coeff = 0;
    for (auto& [k, v] : input.terms) {
        if ((int)k.size() > 2)
            max_high_coeff = std::max(max_high_coeff, std::abs(v));
    }
    double M = 2.0 * max_high_coeff;
    qr.penalty_strength = M;

    // Copy degree-1 and degree-2 terms directly
    for (auto& [k, v] : input.terms) {
        if ((int)k.size() <= 2 && std::abs(v) > 1e-18)
            qr.poly.add_term(k, v);
    }

    // Track auxiliary variables: map (p,q) -> aux_index
    int next_aux = input.num_variables();
    std::map<std::pair<int,int>, int> aux_map;

    auto get_aux = [&](int p, int q) -> int {
        auto key = std::make_pair(std::min(p,q), std::max(p,q));
        auto it = aux_map.find(key);
        if (it != aux_map.end()) return it->second;
        int idx = next_aux++;
        aux_map[key] = idx;
        return idx;
    };

    // Process each high-degree term
    for (auto& [indices, coeff] : input.terms) {
        if ((int)indices.size() <= 2) continue;
        if (std::abs(coeff) < 1e-18) continue;

        // Reduce by repeatedly replacing the first pair with an auxiliary
        std::vector<int> remaining = indices;
        double c = coeff;

        while ((int)remaining.size() > 2) {
            int p = remaining[0], q = remaining[1];
            int aux = get_aux(p, q);

            // Check if penalty for this pair was already added
            auto pkey = std::make_pair(std::min(p,q), std::max(p,q));
            bool first_use = (aux_map.count(pkey) > 0 &&
                              aux_map[pkey] == aux &&
                              aux >= next_aux - (int)aux_map.size());

            if (is_spin) {
                // Ising-basis Rosenberg
                // Replace s_p * s_q with t (aux), add penalty:
                // M/4 * (3 + s_p + s_q - 2t + s_p*s_q - 2*s_p*t - 2*s_q*t)
                qr.poly.add_constant(M * 3.0 / 4.0);
                qr.poly.add_term({p}, M / 4.0);
                qr.poly.add_term({q}, M / 4.0);
                qr.poly.add_term({aux}, -M / 2.0);
                qr.poly.add_term({std::min(p,q), std::max(p,q)}, M / 4.0);
                qr.poly.add_term({std::min(p, aux), std::max(p, aux)}, -M / 2.0);
                qr.poly.add_term({std::min(q, aux), std::max(q, aux)}, -M / 2.0);
            } else {
                // Binary-basis Rosenberg
                // Penalty: M * (b_p*b_q - 2*b_p*y - 2*b_q*y + 3*y)
                qr.poly.add_term({std::min(p,q), std::max(p,q)}, M);
                qr.poly.add_term({std::min(p, aux), std::max(p, aux)}, -2.0 * M);
                qr.poly.add_term({std::min(q, aux), std::max(q, aux)}, -2.0 * M);
                qr.poly.add_term({aux}, 3.0 * M);
            }

            // Replace p,q with aux in remaining indices
            remaining.erase(remaining.begin(), remaining.begin() + 2);
            remaining.insert(remaining.begin(), aux);
            std::sort(remaining.begin(), remaining.end());
        }

        // Now remaining has exactly 2 elements (or 1 if original was degree 3)
        qr.poly.add_term(remaining, c);
    }

    qr.num_auxiliaries = next_aux - input.num_variables();
    qr.poly.prune();
    return qr;
}
