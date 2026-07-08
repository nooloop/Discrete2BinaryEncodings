#pragma once
#include "baseline/types.hpp"

// Rosenberg quadratization: reduce a HUBO to QUBO by introducing auxiliary variables.
// For each cubic+ monomial c * b_p b_q b_r..., pick a pair (p,q), introduce y_{pq},
// replace b_p b_q with y, add penalty M*(b_p b_q - 2 b_p y - 2 b_q y + 3 y).
//
// Rosenberg reduction is only correct in the BINARY {0,1} basis. In the SPIN
// {-1,+1} basis there is NO quadratic Ising penalty on {s_p, s_q, t} that
// enforces the product relation t = s_p * s_q: the valid set {s_p s_q t = +1}
// cannot be separated from its complement by a degree-<=2 Ising form (a symmetric
// quadratic penalty forces contradictory sign conditions on the two invalid
// orbits). The previous SPIN-basis implementation instead enforced the bit-AND
// relation (t_bit = b_p AND b_q) yet substituted the auxiliary spin as if it
// equaled s_p*s_q, which is a different function -- so the quadratized model did
// NOT preserve the HUBO energies (verified: most feasible assignments mismatched).
//
// The correct and simple fix is to quadratize in the binary basis: SPIN inputs
// are first converted to BINARY (s = 1 - 2 b), reduced with the proven binary
// Rosenberg gadget, and emitted as a BINARY QUBO (accepted by every QUBO solver
// and by decode_to_cfn, which reads the register bits directly).

struct QuadratizationResult {
    BinaryPolynomial poly;
    int num_auxiliaries = 0;
    double penalty_strength = 0;
};

// Convert a polynomial from the SPIN {-1,+1} basis to the BINARY {0,1} basis
// using s_q = 1 - 2 b_q. Each spin monomial prod_{q in S} s_q expands to
//   prod_{q in S} (1 - 2 b_q) = sum_{T subseteq S} (-2)^{|T|} prod_{q in T} b_q.
inline BinaryPolynomial spin_to_binary(const BinaryPolynomial& in) {
    BinaryPolynomial out;
    out.var_type = VarType::BINARY;
    out.offset = in.offset;

    for (auto& [S, coeff] : in.terms) {
        int k = (int)S.size();
        int subsets = 1 << k;
        for (int mask = 0; mask < subsets; mask++) {
            int pc = popcount(mask);
            double c = coeff;
            for (int t = 0; t < pc; t++) c *= -2.0;   // (-2)^{|T|}
            if (mask == 0) { out.offset += c; continue; }
            std::vector<int> idx;
            idx.reserve(pc);
            for (int j = 0; j < k; j++)
                if ((mask >> j) & 1) idx.push_back(S[j]);   // S sorted -> idx sorted
            out.terms[idx] += c;
        }
    }
    out.prune();
    return out;
}

inline QuadratizationResult rosenberg_quadratize(const BinaryPolynomial& input_raw) {
    // Quadratize in the binary basis (see header note). SPIN inputs are converted
    // first; the result is always a BINARY QUBO.
    BinaryPolynomial input = (input_raw.var_type == VarType::SPIN)
                                 ? spin_to_binary(input_raw)
                                 : input_raw;

    QuadratizationResult qr;
    qr.poly.var_type = VarType::BINARY;
    qr.poly.offset = input.offset;

    // Penalty strength: M = 2 * max_{|S|>2} |c_S|. Because the penalty is added
    // once per (occurrence of a) reduced pair, a pair reused across T high-degree
    // terms accumulates total penalty >= T*M, which dominates the combined
    // magnitude (<= T * max|c_high| = T*M/2) of those terms -- so the substitution
    // y = b_p b_q holds at every minimizer.
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

            // Binary-basis Rosenberg penalty enforcing y = b_p * b_q:
            //   M * (b_p*b_q - 2*b_p*y - 2*b_q*y + 3*y)
            qr.poly.add_term({std::min(p,q), std::max(p,q)}, M);
            qr.poly.add_term({std::min(p, aux), std::max(p, aux)}, -2.0 * M);
            qr.poly.add_term({std::min(q, aux), std::max(q, aux)}, -2.0 * M);
            qr.poly.add_term({aux}, 3.0 * M);

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
