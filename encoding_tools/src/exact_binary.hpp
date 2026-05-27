#pragma once
#include "cfn.hpp"
#include "assignment.hpp"

// Expand the indicator polynomial x_{i,c} for exact-binary encoding.
// x_{i,c} = prod_{q=0}^{D-1} (b_q * r_q + (1-b_q)*(1-r_q))
// Returns a LocalPoly in local qubit indices 0..D-1.
inline LocalPoly eb_indicator(int bitstring, int num_bits) {
    // Expand: product of factors, each is b_q or (1-b_q) depending on r_q
    // x_{i,c} = sum_{T ⊆ S_0} (-1)^|T| * prod_{q ∈ S_1 ∪ T} b_q
    // where S_1 = {q : r_q=1}, S_0 = {q : r_q=0}

    std::vector<int> zeros, ones;
    for (int q = 0; q < num_bits; q++) {
        if ((bitstring >> q) & 1)
            ones.push_back(q);
        else
            zeros.push_back(q);
    }

    int n_zeros = (int)zeros.size();
    int n_subsets = 1 << n_zeros;

    LocalPoly result;
    result.reserve(n_subsets);

    for (int mask = 0; mask < n_subsets; mask++) {
        double sign = (popcount(mask) % 2 == 0) ? 1.0 : -1.0;

        std::vector<int> indices = ones;
        for (int k = 0; k < n_zeros; k++) {
            if ((mask >> k) & 1)
                indices.push_back(zeros[k]);
        }
        std::sort(indices.begin(), indices.end());
        result[indices] += sign;
    }

    return result;
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

    // Build global qubit index maps per variable
    auto global_map = [&](int var) -> std::vector<int> {
        int Di = result.bits_per_var[var];
        std::vector<int> gmap(Di);
        for (int b = 0; b < Di; b++)
            gmap[b] = result.qubit_start[var] + b;
        return gmap;
    };

    // Unary costs: sum_c C_{i;c} * x_{i,c}
    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int Di = result.bits_per_var[i];
        auto gmap = global_map(i);

        for (int c = 0; c < di; c++) {
            double cost = cfn.unary_tables[i].costs[c];
            if (std::abs(cost) < 1e-18) continue;

            auto ind = eb_indicator(ba.assignment[i][c], Di);
            auto scaled = scale_local(ind, cost);
            add_local_to_poly(result.poly, scaled, gmap);
        }
    }

    // Handle unused bitstrings: pad with lowest-energy choice
    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int Di = result.bits_per_var[i];
        int total = 1 << Di;
        if (di >= total) continue;

        // Find lowest-energy choice
        int best_c = 0;
        double best_e = cfn.unary_tables[i].costs[0];
        for (int c = 1; c < di; c++) {
            if (cfn.unary_tables[i].costs[c] < best_e) {
                best_e = cfn.unary_tables[i].costs[c];
                best_c = c;
            }
        }

        // Find unused bitstrings
        std::vector<bool> used(total, false);
        for (int c = 0; c < di; c++)
            used[ba.assignment[i][c]] = true;

        auto gmap = global_map(i);
        int best_bs = ba.assignment[i][best_c];
        for (int bs = 0; bs < total; bs++) {
            if (used[bs]) continue;
            auto ind = eb_indicator(bs, Di);
            auto scaled = scale_local(ind, best_e);
            add_local_to_poly(result.poly, scaled, gmap);
        }
    }

    // Pairwise costs: sum_{c_i,c_j} C * x_{i,c_i} * x_{j,c_j}
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        int di = cfn.cardinalities[vi], dj = cfn.cardinalities[vj];
        int Di = result.bits_per_var[vi], Dj = result.bits_per_var[vj];

        // Combined global map for both registers
        std::vector<int> combined_gmap;
        combined_gmap.reserve(Di + Dj);
        for (int b = 0; b < Di; b++)
            combined_gmap.push_back(result.qubit_start[vi] + b);
        for (int b = 0; b < Dj; b++)
            combined_gmap.push_back(result.qubit_start[vj] + b);

        // Precompute indicator polynomials for variable i
        // Shift variable j's local indices by Di
        auto shift_poly = [&](const LocalPoly& p, int offset) -> LocalPoly {
            LocalPoly shifted;
            for (auto& [k, v] : p) {
                std::vector<int> nk;
                nk.reserve(k.size());
                for (int idx : k) nk.push_back(idx + offset);
                shifted[nk] = v;
            }
            return shifted;
        };

        for (int ci = 0; ci < di; ci++) {
            auto ind_i = eb_indicator(ba.assignment[vi][ci], Di);
            for (int cj = 0; cj < dj; cj++) {
                double cost = pt.costs[ci * dj + cj];
                if (std::abs(cost) < 1e-18) continue;

                auto ind_j = shift_poly(eb_indicator(ba.assignment[vj][cj], Dj), Di);
                auto prod = multiply_local(ind_i, ind_j);
                auto scaled = scale_local(prod, cost);
                add_local_to_poly(result.poly, scaled, combined_gmap);
            }
        }
    }

    result.time_encoding = enc_timer.elapsed();

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
