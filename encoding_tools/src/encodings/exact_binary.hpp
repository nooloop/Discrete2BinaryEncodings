#pragma once
#include "baseline/cfn.hpp"
#include "utilities/assignment.hpp"
#include <vector>

// ---------------------------------------------------------------------------
// Exact-binary encoding.
//
// Each variable i is packed into D_i = ceil(log2 d_i) qubits. A choice c is
// placed at the bitstring ba.assignment[i][c]. The exact multilinear (QUBO/HUBO)
// polynomial is the unique interpolant of the cost function on the register
// hypercube {0,1}^D:
//
//   * unary  i : f(b) = C_{i, choice(b)}                       (b used)
//                     = min_c C_{i,c}                          (b unused: padding)
//   * pairwise ij: f(b_i, b_j) = C_{ij; choice(b_i), choice(b_j)}  (both used)
//                              = 0                                 (otherwise)
//
// Historically this was built by expanding every indicator
//   x_{i,c} = prod_q (b_q r_q + (1-b_q)(1-r_q))
// and multiplying the two indicator polynomials for every (c_i, c_j) pair. That
// costs sum_c |x_{i,c}| = 3^{D_i} per register, i.e. 3^{D_i+D_j} monomial
// products per edge -- (d_i d_j)^{log2 3} ~ (d_i d_j)^1.585 work to produce a
// result of only 2^{D_i+D_j} = d_i d_j terms.
//
// The multilinear expansion is uniquely determined by the function's values on
// the hypercube, so we instead tabulate those 2^D values and recover the
// coefficients with a fast Mobius transform in O(D * 2^D) time -- the same exact
// polynomial, ~ (3/2)^{D_i+D_j} times cheaper, and with no large intermediate
// hash maps. For a D=8 edge this is ~1M ops instead of ~43M.
// ---------------------------------------------------------------------------

// Fast (subset) Mobius transform, in place. On entry a[x] holds f evaluated at
// the hypercube point whose local bit q equals bit q of x. On exit a[S] holds
// the multilinear coefficient of prod_{q in S} b_q, i.e.
//   f(b) = sum_S a[S] prod_{q in S} b_q,   a[S] = sum_{T subseteq S} (-1)^{|S\T|} f(T).
inline void eb_mobius_transform(std::vector<double>& a) {
    int n = (int)a.size();
    for (int bit = 1; bit < n; bit <<= 1)
        for (int x = 0; x < n; x++)
            if (x & bit)
                a[x] -= a[x ^ bit];
}

inline EncodingResult encode_exact_binary(const CFN& cfn,
                                           const BitstringAssignment& ba,
                                           const EncodingParams& params) {
    Timer total_timer;
    EncodingResult result;
    int N = cfn.num_variables;

    // Qubit assignment: variable i gets ceil(log2(|d_i|)) qubits
    result.qubit_start.resize(N);
    result.bits_per_var.resize(N);
    int q = 0;
    for (int i = 0; i < N; i++) {
        result.qubit_start[i] = q;
        int Di = ceil_log2(cfn.cardinalities[i]);
        if (Di == 0) Di = 1;
        result.bits_per_var[i] = Di;
        for (int b = 0; b < Di; b++)
            result.qubit_info.push_back({i, b});
        q += Di;
    }
    result.num_logical_qubits = q;
    result.poly.var_type = VarType::BINARY;

    Timer enc_timer;

    // Emit the nonzero multilinear coefficients of a transformed value table.
    // local_map translates a local bit index (0..D-1) to its global qubit index.
    auto emit_coeffs = [&](const std::vector<double>& coeffs,
                           const std::vector<int>& local_map) {
        for (int S = 0; S < (int)coeffs.size(); S++) {
            double c = coeffs[S];
            if (std::abs(c) < 1e-18) continue;
            if (S == 0) { result.poly.add_constant(c); continue; }
            std::vector<int> idx;
            idx.reserve(popcount(S));
            for (int b = 0; (S >> b) != 0; b++)
                if ((S >> b) & 1) idx.push_back(local_map[b]);
            std::sort(idx.begin(), idx.end());
            result.poly.add_term(idx, c);
        }
    };

    // Unary costs: f(b) = C_{i,choice(b)}, unused bitstrings padded with min cost.
    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int Di = result.bits_per_var[i];
        int total_bs = 1 << Di;

        double pad = *std::min_element(cfn.unary_tables[i].costs.begin(),
                                       cfn.unary_tables[i].costs.end());
        std::vector<double> vals(total_bs, pad);
        for (int c = 0; c < di; c++)
            vals[ba.assignment[i][c]] = cfn.unary_tables[i].costs[c];

        eb_mobius_transform(vals);

        std::vector<int> gmap(Di);
        for (int b = 0; b < Di; b++) gmap[b] = result.qubit_start[i] + b;
        emit_coeffs(vals, gmap);
    }

    // Pairwise costs: f(b_i,b_j) = C_{ij;choice(b_i),choice(b_j)} on used*used
    // register points, 0 elsewhere (matches sum_{c_i,c_j} C x_{i,c_i} x_{j,c_j},
    // since each indicator product is the point-indicator of one used*used cell).
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        int di = cfn.cardinalities[vi], dj = cfn.cardinalities[vj];
        int Di = result.bits_per_var[vi], Dj = result.bits_per_var[vj];
        int total_bs = 1 << (Di + Dj);

        std::vector<double> vals(total_bs, 0.0);
        for (int ci = 0; ci < di; ci++) {
            int bs_i = ba.assignment[vi][ci];
            for (int cj = 0; cj < dj; cj++) {
                int bs_j = ba.assignment[vj][cj];
                vals[bs_i | (bs_j << Di)] = pt.costs[ci * dj + cj];
            }
        }

        eb_mobius_transform(vals);

        // Local bits 0..Di-1 -> variable i's qubits; Di..Di+Dj-1 -> variable j's.
        std::vector<int> gmap(Di + Dj);
        for (int b = 0; b < Di; b++) gmap[b] = result.qubit_start[vi] + b;
        for (int b = 0; b < Dj; b++) gmap[Di + b] = result.qubit_start[vj] + b;
        emit_coeffs(vals, gmap);
    }

    result.time_encoding = enc_timer.elapsed();

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
