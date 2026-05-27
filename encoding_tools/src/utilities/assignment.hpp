#pragma once
#include "baseline/cfn.hpp"
#include <Eigen/Dense>

// Choice-to-bitstring assignment: for each variable, maps choice index to bitstring value.
struct BitstringAssignment {
    // assignment[i][c] = integer bitstring for choice c of variable i
    std::vector<std::vector<int>> assignment;
};

// Compute per-variable choice orderings (permutations of 0..|d_i|-1)
inline std::vector<std::vector<int>> compute_choice_ordering(
        const CFN& cfn, const std::string& method, double temperature = 1.0) {

    int N = cfn.num_variables;
    std::vector<std::vector<int>> orderings(N);

    if (method == "unsorted") {
        for (int i = 0; i < N; i++) {
            orderings[i].resize(cfn.cardinalities[i]);
            std::iota(orderings[i].begin(), orderings[i].end(), 0);
        }
        return orderings;
    }

    if (method == "one_variable") {
        for (int i = 0; i < N; i++) {
            int di = cfn.cardinalities[i];
            orderings[i].resize(di);
            std::iota(orderings[i].begin(), orderings[i].end(), 0);
            auto& costs = cfn.unary_tables[i].costs;
            if ((int)costs.size() == di) {
                std::sort(orderings[i].begin(), orderings[i].end(),
                    [&](int a, int b) { return costs[a] < costs[b]; });
            }
        }
        return orderings;
    }

    if (method == "boltzmann") {
        double kBT = temperature;
        for (int i = 0; i < N; i++) {
            int di = cfn.cardinalities[i];
            std::vector<double> eff_energy(di, 0.0);

            // Start with unary energy
            for (int ci = 0; ci < di; ci++) {
                if (!cfn.unary_tables[i].costs.empty())
                    eff_energy[ci] = cfn.unary_tables[i].costs[ci];
            }

            // Add Boltzmann-averaged pairwise contributions (Eq. 15 in manuscript)
            for (auto& pt : cfn.pairwise_tables) {
                int var_i_pos = -1;
                int var_j_idx = -1;
                if (pt.scope[0] == i) { var_i_pos = 0; var_j_idx = pt.scope[1]; }
                else if (pt.scope[1] == i) { var_i_pos = 1; var_j_idx = pt.scope[0]; }
                else continue;

                int dj = cfn.cardinalities[var_j_idx];

                for (int ci = 0; ci < di; ci++) {
                    double Z = 0, weighted_sum = 0;
                    for (int cj = 0; cj < dj; cj++) {
                        double e;
                        if (var_i_pos == 0)
                            e = pt.costs[ci * dj + cj];
                        else
                            e = pt.costs[cj * di + ci];
                        double w = std::exp(-e / kBT);
                        Z += w;
                        weighted_sum += e * w;
                    }
                    if (Z > 0) eff_energy[ci] += weighted_sum / Z;
                }
            }

            orderings[i].resize(di);
            std::iota(orderings[i].begin(), orderings[i].end(), 0);
            std::sort(orderings[i].begin(), orderings[i].end(),
                [&](int a, int b) { return eff_energy[a] < eff_energy[b]; });
        }
        return orderings;
    }

    throw std::runtime_error("Unknown choice ordering: " + method);
}

// Generate bitstring sequence (natural or Gray code)
inline std::vector<int> bitstring_sequence(int num_bits, const std::string& order) {
    int n = 1 << num_bits;
    std::vector<int> seq(n);
    if (order == "gray") {
        for (int i = 0; i < n; i++) seq[i] = to_gray(i);
    } else {
        std::iota(seq.begin(), seq.end(), 0);
    }
    return seq;
}

// Linearly-independent prioritization for approximate-binary encoding.
// Reorders bitstring assignments so that choices with lower energy get
// linearly independent rows of Q (rank-revealing QR).
inline std::vector<int> li_prioritized_bitstrings(
        int num_bits, int cardinality, int k_approx,
        const std::vector<int>& base_bitstrings) {

    int total_bs = 1 << num_bits;

    // Build monomial column indices for Q matrix
    // Enumerate all subsets of {0,...,num_bits-1} with size 1..k_approx, plus constant
    std::vector<std::vector<int>> monomials;
    for (int sz = 1; sz <= std::min(k_approx, num_bits); sz++) {
        std::vector<int> combo(sz, 0);
        // enumerate all size-sz subsets
        std::function<void(int, int)> enumerate = [&](int start, int depth) {
            if (depth == sz) { monomials.push_back(combo); return; }
            for (int v = start; v < num_bits; v++) {
                combo[depth] = v;
                enumerate(v + 1, depth + 1);
            }
        };
        enumerate(0, 0);
    }
    int M = (int)monomials.size() + 1; // +1 for constant

    // Build Q matrix for all bitstrings
    Eigen::MatrixXd Q(total_bs, M);
    for (int r = 0; r < total_bs; r++) {
        int bs = base_bitstrings[r];
        int col = 0;
        for (auto& mono : monomials) {
            int val = 1;
            for (int bit : mono)
                val &= (bs >> bit) & 1;
            Q(r, col++) = val;
        }
        Q(r, col) = 1.0; // constant
    }

    // Rank-revealing QR on the rows corresponding to the first `cardinality` bitstrings
    // We greedily pick linearly independent rows
    std::vector<bool> used(total_bs, false);
    std::vector<int> li_order;
    std::vector<int> dep_order;
    Eigen::MatrixXd basis(0, M);

    auto is_independent = [&](int row_idx) -> bool {
        if (basis.rows() >= M) return false;
        Eigen::MatrixXd test(basis.rows() + 1, M);
        test.topRows(basis.rows()) = basis;
        test.bottomRows(1) = Q.row(row_idx);
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(test);
        return qr.rank() == (int)test.rows();
    };

    // Try to assign LI rows first (in order of the choice ordering)
    for (int i = 0; i < cardinality && i < total_bs; i++) {
        if (is_independent(i)) {
            li_order.push_back(base_bitstrings[i]);
            used[i] = true;
            Eigen::MatrixXd new_basis(basis.rows() + 1, M);
            new_basis.topRows(basis.rows()) = basis;
            new_basis.bottomRows(1) = Q.row(i);
            basis = new_basis;
        } else {
            dep_order.push_back(base_bitstrings[i]);
        }
    }

    // Combine: LI first, then dependent
    std::vector<int> result;
    result.reserve(cardinality);
    result.insert(result.end(), li_order.begin(), li_order.end());
    result.insert(result.end(), dep_order.begin(), dep_order.end());

    // Pad if cardinality < total_bs (unused bitstrings get padding)
    for (int i = cardinality; i < total_bs; i++) {
        if (!used[i]) result.push_back(base_bitstrings[i]);
    }

    return result;
}

// Full assignment computation
inline BitstringAssignment compute_assignment(
        const CFN& cfn, const EncodingParams& params) {

    BitstringAssignment ba;
    int N = cfn.num_variables;
    ba.assignment.resize(N);

    auto orderings = compute_choice_ordering(cfn, params.choice_ordering, params.temperature);

    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int Di = ceil_log2(di);
        if (Di == 0) Di = 1;
        int total_bs = 1 << Di;

        auto bs_seq = bitstring_sequence(Di, params.bitstring_ordering);

        std::vector<int> final_bs;
        if (params.li_prioritization && params.encoding == "approximate_binary") {
            final_bs = li_prioritized_bitstrings(Di, di, params.k_approx, bs_seq);
        } else {
            final_bs = bs_seq;
        }

        // Map: ordered choice index -> bitstring
        // orderings[i] gives the permutation of original choice indices
        // final_bs gives the bitstrings in priority order
        ba.assignment[i].resize(di);
        for (int rank = 0; rank < di; rank++) {
            int orig_choice = orderings[i][rank];
            ba.assignment[i][orig_choice] = final_bs[rank];
        }
    }

    return ba;
}
