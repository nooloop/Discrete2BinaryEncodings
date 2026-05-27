#pragma once
#include "baseline/cfn.hpp"
#include "utilities/assignment.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <mutex>
#include <cmath>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ab_detail {

// Enumerate all monomials of a given degree range on num_bits bits.
// Returns vectors of bit indices (sorted) for each monomial.

inline std::vector<std::vector<int>> enumerate_monomials(int num_bits, int min_deg, int max_deg) {
    std::vector<std::vector<int>> result;
    for (int deg = min_deg; deg <= std::min(max_deg, num_bits); deg++) {
        std::vector<int> combo(deg);
        std::function<void(int, int)> gen = [&](int start, int depth) {
            if (depth == deg) { result.push_back(combo); return; }
            for (int v = start; v < num_bits; v++) {
                combo[depth] = v;
                gen(v + 1, depth + 1);
            }
        };
        gen(0, 0);
    }
    return result;
}

// Enumerate cross-register monomials: monomials that touch at least one bit from each register in the scope.
// reg_offsets: local bit offset of each register
// reg_sizes: number of bits per register
inline std::vector<std::vector<int>> enumerate_cross_monomials(
        const std::vector<int>& reg_offsets,
        const std::vector<int>& reg_sizes,
        int k_approx) {
    int total_bits = 0;
    for (int s : reg_sizes) total_bits += s;
    int num_regs = (int)reg_offsets.size();

    auto all_monos = enumerate_monomials(total_bits, num_regs, k_approx);

    // Filter: keep only those touching every register
    std::vector<std::vector<int>> result;
    for (auto& mono : all_monos) {
        std::vector<bool> touched(num_regs, false);
        for (int bit : mono) {
            for (int r = 0; r < num_regs; r++) {
                if (bit >= reg_offsets[r] && bit < reg_offsets[r] + reg_sizes[r])
                    touched[r] = true;
            }
        }
        bool all_touched = true;
        for (bool t : touched) if (!t) { all_touched = false; break; }
        if (all_touched) result.push_back(mono);
    }
    return result;
}

// Build the monomial fitting matrix Q for a set of registers.
// For unary (1 register): all monomials of degree 1..k_approx + constant.
// For n-body (n registers): cross-register monomials of degree n..k_approx + constant.
struct QMatrixInfo {
    Eigen::MatrixXd Q;
    std::vector<std::vector<int>> monomials; // which monomial each column represents (excl. constant)
    int num_rows;
    int num_cols; // includes constant column
};

inline QMatrixInfo build_Q_matrix(
        const std::vector<int>& reg_sizes,
        int k_approx) {
    int num_regs = (int)reg_sizes.size();
    int total_bits = 0;
    std::vector<int> reg_offsets(num_regs);
    for (int r = 0; r < num_regs; r++) {
        reg_offsets[r] = total_bits;
        total_bits += reg_sizes[r];
    }
    int num_rows = 1 << total_bits;

    std::vector<std::vector<int>> monos;
    if (num_regs == 1) {
        monos = enumerate_monomials(total_bits, 1, k_approx);
    } else {
        monos = enumerate_cross_monomials(reg_offsets, reg_sizes, k_approx);
    }
    int num_cols = (int)monos.size() + 1; // +1 for constant

    Eigen::MatrixXd Q(num_rows, num_cols);
    for (int row = 0; row < num_rows; row++) {
        int col = 0;
        for (auto& mono : monos) {
            int val = 1;
            for (int bit : mono)
                val &= (row >> bit) & 1;
            Q(row, col++) = val;
        }
        Q(row, col) = 1.0; // constant column
    }

    return {Q, monos, num_rows, num_cols};
}

// Build energy vector for a cost table, mapping choices to bitstrings.
// Pads unused bitstrings with lowest-energy duplicates.
inline Eigen::VectorXd build_energy_vector(
        const std::vector<double>& costs,
        const std::vector<std::vector<int>>& assignments, // one per register
        const std::vector<int>& cardinalities,
        const std::vector<int>& reg_sizes) {
    int num_regs = (int)reg_sizes.size();
    int total_bits = 0;
    for (int s : reg_sizes) total_bits += s;
    int total_bs = 1 << total_bits;

    Eigen::VectorXd E = Eigen::VectorXd::Zero(total_bs);

    // Find minimum energy for padding
    double min_e = *std::min_element(costs.begin(), costs.end());

    // Fill all entries with min energy (padding default)
    E.setConstant(min_e);

    if (num_regs == 1) {
        int di = cardinalities[0];
        for (int c = 0; c < di; c++) {
            int bs = assignments[0][c];
            E(bs) = costs[c];
        }
    } else if (num_regs == 2) {
        int di = cardinalities[0], dj = cardinalities[1];
        int Dj = reg_sizes[1];
        for (int ci = 0; ci < di; ci++) {
            int bs_i = assignments[0][ci];
            for (int cj = 0; cj < dj; cj++) {
                int bs_j = assignments[1][cj];
                int combined = (bs_i << Dj) | bs_j;
                // Register order: reg0 bits are higher, reg1 bits are lower
                // Actually: reg0 starts at offset 0, reg1 at offset reg_sizes[0]
                // So combined bitstring = bs_j at high bits, bs_i at low bits? No.
                // Let's be explicit: bit index q in the combined bitstring:
                // q < reg_sizes[0]: belongs to reg0, bit q
                // q >= reg_sizes[0]: belongs to reg1, bit q-reg_sizes[0]
                // So combined = bs_i | (bs_j << reg_sizes[0])
                combined = bs_i | (bs_j << reg_sizes[0]);
                E(combined) = costs[ci * dj + cj];
            }
        }
    }

    return E;
}

} // namespace ab_detail

inline EncodingResult encode_approximate_binary(const CFN& cfn,
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
    result.poly.var_type = VarType::BINARY;

    Timer enc_timer;

    // Collect all cost-table tasks (unary + pairwise)
    struct LSTask {
        std::vector<int> scope;
        std::vector<double> costs;
        std::vector<int> reg_sizes;
        std::vector<int> cardinalities;
        std::vector<std::vector<int>> assignments;
        std::vector<int> global_qubit_map;
    };

    std::vector<LSTask> tasks;

    // Unary tasks
    for (int i = 0; i < N; i++) {
        LSTask t;
        t.scope = {i};
        t.costs = cfn.unary_tables[i].costs;
        t.reg_sizes = {result.bits_per_var[i]};
        t.cardinalities = {cfn.cardinalities[i]};
        t.assignments = {ba.assignment[i]};
        t.global_qubit_map.resize(result.bits_per_var[i]);
        for (int b = 0; b < result.bits_per_var[i]; b++)
            t.global_qubit_map[b] = result.qubit_start[i] + b;
        tasks.push_back(std::move(t));
    }

    // Pairwise tasks
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        LSTask t;
        t.scope = pt.scope;
        t.costs = pt.costs;
        t.reg_sizes = {result.bits_per_var[vi], result.bits_per_var[vj]};
        t.cardinalities = {cfn.cardinalities[vi], cfn.cardinalities[vj]};
        t.assignments = {ba.assignment[vi], ba.assignment[vj]};

        int Di = result.bits_per_var[vi], Dj = result.bits_per_var[vj];
        t.global_qubit_map.resize(Di + Dj);
        for (int b = 0; b < Di; b++)
            t.global_qubit_map[b] = result.qubit_start[vi] + b;
        for (int b = 0; b < Dj; b++)
            t.global_qubit_map[Di + b] = result.qubit_start[vj] + b;
        tasks.push_back(std::move(t));
    }

    // Solve all tasks (parallelizable)
    int ntasks = (int)tasks.size();
    std::vector<BinaryPolynomial> task_polys(ntasks);
    std::vector<double> task_l2(ntasks, 0), task_linf(ntasks, 0);

    // Cache for pseudoinverse: key = (total_bits, num_regs, k_approx)
    struct CacheKey {
        std::vector<int> reg_sizes;
        bool operator==(const CacheKey& o) const { return reg_sizes == o.reg_sizes; }
    };
    struct CacheHash {
        size_t operator()(const CacheKey& k) const {
            size_t h = k.reg_sizes.size();
            for (int s : k.reg_sizes)
                h ^= std::hash<int>{}(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<CacheKey, ab_detail::QMatrixInfo, CacheHash> q_cache;
    std::mutex cache_mutex;

    auto solve_task = [&](int tidx) {
        auto& t = tasks[tidx];
        BinaryPolynomial local_poly;
        local_poly.var_type = VarType::BINARY;

        CacheKey ck{t.reg_sizes};
        ab_detail::QMatrixInfo qinfo;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = q_cache.find(ck);
            if (it != q_cache.end()) {
                qinfo = it->second;
            } else {
                qinfo = ab_detail::build_Q_matrix(t.reg_sizes, params.k_approx);
                q_cache[ck] = qinfo;
            }
        }

        auto E = ab_detail::build_energy_vector(t.costs, t.assignments, t.cardinalities, t.reg_sizes);

        Eigen::MatrixXd Q = qinfo.Q;
        Eigen::VectorXd Evec = E;

        // Weighted least squares
        if (params.weighted_ls) {
            double kBT = params.temperature;
            double min_e = Evec.minCoeff();
            for (int r = 0; r < Q.rows(); r++) {
                double w = std::sqrt(std::exp(-(Evec(r) - min_e) / kBT));
                Q.row(r) *= w;
                Evec(r) *= w;
            }
        }

        // Solve via pseudoinverse (SVD)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd B = svd.solve(Evec);

        // Extract monomials into polynomial
        for (int col = 0; col < (int)qinfo.monomials.size(); col++) {
            double coeff = B(col);
            if (std::abs(coeff) < 1e-18) continue;
            std::vector<int> global_idx;
            for (int li : qinfo.monomials[col])
                global_idx.push_back(t.global_qubit_map[li]);
            std::sort(global_idx.begin(), global_idx.end());
            local_poly.add_term(global_idx, coeff);
        }
        // Constant term
        local_poly.add_constant(B(B.size() - 1));

        // Compute approximation errors (on unweighted system)
        auto Qorig = qinfo.Q;
        Eigen::VectorXd residual = Qorig * B - E;
        task_l2[tidx] = residual.norm();
        task_linf[tidx] = residual.cwiseAbs().maxCoeff();

        task_polys[tidx] = std::move(local_poly);
    };

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(params.num_threads)
#endif
    for (int tidx = 0; tidx < ntasks; tidx++) {
        solve_task(tidx);
    }

    // Merge all task polynomials
    for (int tidx = 0; tidx < ntasks; tidx++)
        result.poly.merge_from(task_polys[tidx]);

    // Aggregate errors
    double total_l2_sq = 0;
    double max_linf = 0;
    for (int tidx = 0; tidx < ntasks; tidx++) {
        total_l2_sq += task_l2[tidx] * task_l2[tidx];
        max_linf = std::max(max_linf, task_linf[tidx]);
    }
    result.l2_error = std::sqrt(total_l2_sq);
    result.linf_error = max_linf;

    result.time_encoding = enc_timer.elapsed();

    // Nonlinear refinement (per-unary-table gradient descent)
    if (params.nonlinear_refinement) {
        Timer nl_timer;

        for (int i = 0; i < N; i++) {
            int di = cfn.cardinalities[i];
            int Di = result.bits_per_var[i];
            int total_bs = 1 << Di;
            double kBT = params.nl_temperature;
            double tau = params.nl_tether_weight;

            CacheKey ck{{Di}};
            auto& qinfo = q_cache[ck];
            auto E_desired = ab_detail::build_energy_vector(
                cfn.unary_tables[i].costs, {ba.assignment[i]}, {di}, {Di});

            // Current coefficients for this register
            Eigen::VectorXd B(qinfo.num_cols);
            B.setZero();
            // Extract from result.poly
            for (int col = 0; col < (int)qinfo.monomials.size(); col++) {
                std::vector<int> gidx;
                for (int li : qinfo.monomials[col])
                    gidx.push_back(result.qubit_start[i] + li);
                std::sort(gidx.begin(), gidx.end());
                auto it = result.poly.terms.find(gidx);
                if (it != result.poly.terms.end())
                    B(col) = it->second;
            }
            // Constant
            // Note: the constant from multiple tables is summed into offset;
            // for per-register refinement we use the local constant only
            B(B.size()-1) = 0; // Will be set by the LS solution

            // Recompute B from LS for this register only
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(qinfo.Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
            B = svd.solve(E_desired);

            // Desired Boltzmann probabilities
            double e_min_desired = E_desired.minCoeff();
            Eigen::VectorXd exp_desired(total_bs);
            for (int r = 0; r < total_bs; r++)
                exp_desired(r) = std::exp(-(E_desired(r) - e_min_desired) / kBT);
            double Z_desired = exp_desired.sum();
            Eigen::VectorXd p_desired = exp_desired / Z_desired;

            // Gradient descent
            double lr = 0.01;
            for (int iter = 0; iter < params.nl_max_iter; iter++) {
                Eigen::VectorXd E_actual = qinfo.Q * B;
                double e_min_actual = E_actual.minCoeff();

                Eigen::VectorXd exp_actual(total_bs);
                for (int r = 0; r < total_bs; r++)
                    exp_actual(r) = std::exp(-(E_actual(r) - e_min_actual) / kBT);
                double Z_actual = exp_actual.sum();
                Eigen::VectorXd p_actual = exp_actual / Z_actual;

                // Loss = sum (p_actual - p_desired)^2 + tau * (e_min_actual - e_min_desired)^2
                Eigen::VectorXd dp = p_actual - p_desired;
                double loss = dp.squaredNorm() + tau * std::pow(e_min_actual - e_min_desired, 2);

                // Gradient of p_actual w.r.t. E_actual:
                // dp/dE = -1/kBT * (diag(p) - p*p^T) (softmax Jacobian)
                // dL/dB = Q^T * dL/dE

                // dL/dE_r = 2*(p_actual_r - p_desired_r) * dp_actual_r/dE_actual_r (summed)
                // Using chain rule through softmax:
                Eigen::VectorXd dL_dE(total_bs);
                for (int r = 0; r < total_bs; r++) {
                    double sum_dp_p = 0;
                    for (int s = 0; s < total_bs; s++)
                        sum_dp_p += dp(s) * p_actual(s);
                    dL_dE(r) = -2.0 / kBT * p_actual(r) * (dp(r) - sum_dp_p);
                }

                // Tethering gradient
                int min_idx;
                E_actual.minCoeff(&min_idx);
                dL_dE(min_idx) += 2.0 * tau * (e_min_actual - e_min_desired);

                Eigen::VectorXd grad = qinfo.Q.transpose() * dL_dE;

                B -= lr * grad;

                if (grad.norm() < 1e-10) break;
            }

            // Update result.poly with refined coefficients
            // Remove old terms for this register
            for (int col = 0; col < (int)qinfo.monomials.size(); col++) {
                std::vector<int> gidx;
                for (int li : qinfo.monomials[col])
                    gidx.push_back(result.qubit_start[i] + li);
                std::sort(gidx.begin(), gidx.end());
                result.poly.terms.erase(gidx);
            }
            // Add refined terms
            for (int col = 0; col < (int)qinfo.monomials.size(); col++) {
                double coeff = B(col);
                if (std::abs(coeff) < 1e-18) continue;
                std::vector<int> gidx;
                for (int li : qinfo.monomials[col])
                    gidx.push_back(result.qubit_start[i] + li);
                std::sort(gidx.begin(), gidx.end());
                result.poly.add_term(gidx, coeff);
            }
        }

        result.time_nonlinear_refinement = nl_timer.elapsed();
    }

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
