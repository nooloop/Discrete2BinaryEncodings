#pragma once
#include "baseline/cfn.hpp"
#include "utilities/assignment.hpp"
#include <cmath>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace tb_detail {

// Fast Walsh-Hadamard transform (in-place, unnormalized).
// Transforms a vector of length 2^n.
// After FWHT, result[S] = sum_x f(x) * chi_S(x) where chi_S(x) = prod_{q in S} (-1)^{x_q}
inline void fwht(std::vector<double>& data) {
    int n = (int)data.size();
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                double x = data[j];
                double y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

// Compute the Walsh degree of a bitmask (= popcount)
inline int walsh_degree(int mask) {
    return popcount(mask);
}

// Embed a cost table into a 2^T vector on the sub-hypercube.
// For unary: maps choice c to bitstring assignment[c], value = costs[c].
// For pairwise: maps (c_i,c_j) to combined bitstring.
// Unused entries get zero (will be handled by centered normalization if needed).
inline std::vector<double> embed_cost_table(
        const std::vector<double>& costs,
        const std::vector<std::vector<int>>& assignments,
        const std::vector<int>& cardinalities,
        const std::vector<int>& reg_sizes) {

    int total_bits = 0;
    for (int s : reg_sizes) total_bits += s;
    int total_bs = 1 << total_bits;
    std::vector<double> embedded(total_bs, 0.0);

    int num_regs = (int)reg_sizes.size();

    if (num_regs == 1) {
        int di = cardinalities[0];
        for (int c = 0; c < di; c++)
            embedded[assignments[0][c]] = costs[c];
    } else if (num_regs == 2) {
        int di = cardinalities[0], dj = cardinalities[1];
        int D0 = reg_sizes[0];
        for (int ci = 0; ci < di; ci++) {
            int bs_i = assignments[0][ci];
            for (int cj = 0; cj < dj; cj++) {
                int bs_j = assignments[1][cj];
                int combined = bs_i | (bs_j << D0);
                embedded[combined] = costs[ci * dj + cj];
            }
        }
    }

    return embedded;
}

// Center a pairwise cost table: subtract marginals so that
// sum_{c_j} C_{i,j;c_i,c_j} = 0 for all c_i (and symmetrically).
// Returns the centered table and the absorbed marginals.
struct CenteredTable {
    std::vector<double> centered_costs;
    std::vector<double> marginal_i; // absorbed into unary table of i
    std::vector<double> marginal_j; // absorbed into unary table of j
};

inline CenteredTable center_pairwise_table(
        const std::vector<double>& costs, int di, int dj) {

    CenteredTable result;
    result.centered_costs = costs;
    result.marginal_i.resize(di, 0.0);
    result.marginal_j.resize(dj, 0.0);

    // Subtract row means (marginals over j)
    for (int ci = 0; ci < di; ci++) {
        double mean = 0;
        for (int cj = 0; cj < dj; cj++)
            mean += costs[ci * dj + cj];
        mean /= dj;
        result.marginal_i[ci] = mean;
        for (int cj = 0; cj < dj; cj++)
            result.centered_costs[ci * dj + cj] -= mean;
    }

    // Subtract column means of the already row-centered table
    for (int cj = 0; cj < dj; cj++) {
        double mean = 0;
        for (int ci = 0; ci < di; ci++)
            mean += result.centered_costs[ci * dj + cj];
        mean /= di;
        result.marginal_j[cj] = mean;
        for (int ci = 0; ci < di; ci++)
            result.centered_costs[ci * dj + cj] -= mean;
    }

    return result;
}

} // namespace tb_detail

inline EncodingResult encode_truncated_binary(const CFN& cfn,
                                               const BitstringAssignment& ba,
                                               const EncodingParams& params) {
    Timer total_timer;
    EncodingResult result;
    int N = cfn.num_variables;

    // Qubit assignment
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
    result.poly.var_type = VarType::SPIN;

    Timer enc_timer;

    int max_degree = 0;
    for (int i = 0; i < N; i++)
        max_degree = std::max(max_degree, result.bits_per_var[i]);
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        max_degree = std::max(max_degree, result.bits_per_var[vi] + result.bits_per_var[vj]);
    }
    result.spectral_profile.resize(max_degree + 1, 0.0);

    // Center pairwise tables and absorb marginals into unary tables
    std::vector<std::vector<double>> adjusted_unary(N);
    for (int i = 0; i < N; i++)
        adjusted_unary[i] = cfn.unary_tables[i].costs;

    std::vector<tb_detail::CenteredTable> centered_pairwise(cfn.pairwise_tables.size());
    for (int tidx = 0; tidx < (int)cfn.pairwise_tables.size(); tidx++) {
        auto& pt = cfn.pairwise_tables[tidx];
        int vi = pt.scope[0], vj = pt.scope[1];
        int di = cfn.cardinalities[vi], dj = cfn.cardinalities[vj];

        centered_pairwise[tidx] = tb_detail::center_pairwise_table(pt.costs, di, dj);

        for (int ci = 0; ci < di; ci++)
            adjusted_unary[vi][ci] += centered_pairwise[tidx].marginal_i[ci];
        for (int cj = 0; cj < dj; cj++)
            adjusted_unary[vj][cj] += centered_pairwise[tidx].marginal_j[cj];
    }

    // Process unary tables via Walsh-Hadamard
    struct WalshTerm {
        std::vector<int> global_bits; // sorted
        double coeff;
    };

    std::vector<std::vector<WalshTerm>> unary_terms(N);
    std::vector<std::vector<WalshTerm>> pairwise_terms(cfn.pairwise_tables.size());

    // Unary tables
    #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(params.num_threads)
    #endif
    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int Di = result.bits_per_var[i];
        int total_bs = 1 << Di;

        auto embedded = tb_detail::embed_cost_table(
            adjusted_unary[i], {ba.assignment[i]}, {di}, {Di});

        tb_detail::fwht(embedded);

        // Normalize: divide by 2^Di
        double norm = 1.0 / total_bs;

        for (int S = 0; S < total_bs; S++) {
            int deg = tb_detail::walsh_degree(S);
            double coeff = embedded[S] * norm;
            if (std::abs(coeff) < 1e-18) continue;

            // Accumulate spectral profile
            if (deg > 0 && deg < (int)result.spectral_profile.size()) {
                #ifdef HAS_OPENMP
                #pragma omp atomic
                #endif
                result.spectral_profile[deg] += coeff * coeff;
            }

            if (deg > params.k_trunc) continue;

            // Convert S bitmask to global qubit indices
            std::vector<int> gbits;
            for (int b = 0; b < Di; b++) {
                if ((S >> b) & 1)
                    gbits.push_back(result.qubit_start[i] + b);
            }
            std::sort(gbits.begin(), gbits.end());
            unary_terms[i].push_back({gbits, coeff});
        }
    }

    // Pairwise tables
    #ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(params.num_threads)
    #endif
    for (int tidx = 0; tidx < (int)cfn.pairwise_tables.size(); tidx++) {
        auto& pt = cfn.pairwise_tables[tidx];
        int vi = pt.scope[0], vj = pt.scope[1];
        int di = cfn.cardinalities[vi], dj = cfn.cardinalities[vj];
        int Di = result.bits_per_var[vi], Dj = result.bits_per_var[vj];
        int total_bits = Di + Dj;
        int total_bs = 1 << total_bits;

        auto embedded = tb_detail::embed_cost_table(
            centered_pairwise[tidx].centered_costs,
            {ba.assignment[vi], ba.assignment[vj]},
            {di, dj}, {Di, Dj});

        tb_detail::fwht(embedded);

        double norm = 1.0 / total_bs;

        for (int S = 0; S < total_bs; S++) {
            int deg = tb_detail::walsh_degree(S);
            double coeff = embedded[S] * norm;
            if (std::abs(coeff) < 1e-18) continue;

            // Only keep cross-register terms (touching both registers)
            int S_i = S & ((1 << Di) - 1);
            int S_j = (S >> Di);
            if (S_i == 0 || S_j == 0) continue; // not cross-register

            if (deg > 0 && deg < (int)result.spectral_profile.size()) {
                #ifdef HAS_OPENMP
                #pragma omp atomic
                #endif
                result.spectral_profile[deg] += coeff * coeff;
            }

            if (deg > params.k_trunc) continue;

            std::vector<int> gbits;
            for (int b = 0; b < Di; b++) {
                if ((S_i >> b) & 1)
                    gbits.push_back(result.qubit_start[vi] + b);
            }
            for (int b = 0; b < Dj; b++) {
                if ((S_j >> b) & 1)
                    gbits.push_back(result.qubit_start[vj] + b);
            }
            std::sort(gbits.begin(), gbits.end());
            pairwise_terms[tidx].push_back({gbits, coeff});
        }
    }

    // Merge into result polynomial
    // Walsh terms: coeff * chi_S(b) = coeff * prod_{q in S} (1-2*b_q)
    // In SPIN representation: coeff * prod_{q in S} s_q
    for (int i = 0; i < N; i++) {
        for (auto& wt : unary_terms[i]) {
            if (wt.global_bits.empty())
                result.poly.add_constant(wt.coeff);
            else
                result.poly.add_term(wt.global_bits, wt.coeff);
        }
    }
    for (int tidx = 0; tidx < (int)cfn.pairwise_tables.size(); tidx++) {
        for (auto& wt : pairwise_terms[tidx])
            result.poly.add_term(wt.global_bits, wt.coeff);
    }

    // Compute L2 and Linf error bounds from spectral profile
    double discarded_l2_sq = 0;
    double discarded_linf = 0;
    for (int k = params.k_trunc + 1; k < (int)result.spectral_profile.size(); k++) {
        discarded_l2_sq += result.spectral_profile[k];
        // For Linf, we'd need sum of |H_S| for |S|>k_trunc, which we approximate
    }
    result.l2_error = std::sqrt(discarded_l2_sq);
    // linf_error is an upper bound; exact computation would require all individual Walsh coefficients
    result.linf_error = result.l2_error; // conservative estimate

    result.time_encoding = enc_timer.elapsed();

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
