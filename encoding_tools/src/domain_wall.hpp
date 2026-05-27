#pragma once
#include "cfn.hpp"
#include "lagrange.hpp"

// Domain-wall indicator: x_{i,c} = b_{i,c-1} - b_{i,c}
// with b_{i,-1}=1 and b_{i,|d|-1}=0.
// Returns a local polynomial in the DW qubits of variable i.
// Local qubit indices: 0..|d|-2
struct DWIndicator {
    double constant = 0;                // from b_{i,-1}=1 boundary
    std::vector<std::pair<int, double>> linear;  // (local_qubit, coeff)

    static DWIndicator make(int choice, int cardinality) {
        DWIndicator ind;
        // x_{i,c} = b_{i,c-1} - b_{i,c}
        // c-1 boundary: b_{i,-1}=1 when c=0
        // c boundary: b_{i,|d|-1}=0 when c=|d|-1

        if (choice == 0) {
            // b_{i,-1} - b_{i,0} = 1 - b_{i,0}
            ind.constant = 1.0;
            ind.linear.push_back({0, -1.0});
        } else if (choice == cardinality - 1) {
            // b_{i,|d|-2} - b_{i,|d|-1} = b_{i,|d|-2} - 0
            ind.linear.push_back({cardinality - 2, 1.0});
        } else {
            // b_{i,c-1} - b_{i,c}
            ind.linear.push_back({choice - 1, 1.0});
            ind.linear.push_back({choice, -1.0});
        }
        return ind;
    }
};

inline EncodingResult encode_domain_wall(const CFN& cfn, const EncodingParams& params) {
    Timer total_timer;
    EncodingResult result;
    int N = cfn.num_variables;

    // Qubit assignment: variable i gets |d_i|-1 qubits
    result.qubit_start.resize(N);
    result.bits_per_var.resize(N);
    int q = 0;
    for (int i = 0; i < N; i++) {
        result.qubit_start[i] = q;
        int nbits = cfn.cardinalities[i] - 1;
        result.bits_per_var[i] = nbits;
        for (int c = 0; c < nbits; c++)
            result.qubit_info.push_back({i, c});
        q += nbits;
    }
    result.num_logical_qubits = q;
    result.poly.var_type = VarType::BINARY;

    Timer enc_timer;

    // Unary costs using telescoping: sum_c C_{i;c} * x_{i,c}
    // = C_{i;0} + sum_{q=0}^{|d|-2} (C_{i;q+1} - C_{i;q}) * b_{i,q}
    for (int i = 0; i < N; i++) {
        int di = cfn.cardinalities[i];
        int qs = result.qubit_start[i];
        auto& costs = cfn.unary_tables[i].costs;

        result.poly.add_constant(costs[0]);
        for (int q_loc = 0; q_loc < di - 1; q_loc++) {
            double diff = costs[q_loc + 1] - costs[q_loc];
            if (std::abs(diff) > 1e-18)
                result.poly.add_term({qs + q_loc}, diff);
        }
    }

    // Pairwise costs: sum_{c_i,c_j} C_{i,j;c_i,c_j} * x_{i,c_i} * x_{j,c_j}
    // Each product x_{i,c_i} * x_{j,c_j} expands into at most 4 terms
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        int qi = result.qubit_start[vi], qj = result.qubit_start[vj];
        int di = cfn.cardinalities[vi], dj = cfn.cardinalities[vj];

        for (int ci = 0; ci < di; ci++) {
            DWIndicator ind_i = DWIndicator::make(ci, di);
            for (int cj = 0; cj < dj; cj++) {
                double cost = pt.costs[ci * dj + cj];
                if (std::abs(cost) < 1e-18) continue;

                DWIndicator ind_j = DWIndicator::make(cj, dj);

                // Multiply: (k_i + sum a*b_p) * (k_j + sum a'*b_q)
                // Constant * Constant
                result.poly.add_constant(cost * ind_i.constant * ind_j.constant);

                // Constant_i * Linear_j
                for (auto& [lq, lc] : ind_j.linear) {
                    double c = cost * ind_i.constant * lc;
                    if (std::abs(c) > 1e-18)
                        result.poly.add_term({qj + lq}, c);
                }

                // Linear_i * Constant_j
                for (auto& [lq, lc] : ind_i.linear) {
                    double c = cost * lc * ind_j.constant;
                    if (std::abs(c) > 1e-18)
                        result.poly.add_term({qi + lq}, c);
                }

                // Linear_i * Linear_j
                for (auto& [lq1, lc1] : ind_i.linear) {
                    for (auto& [lq2, lc2] : ind_j.linear) {
                        double c = cost * lc1 * lc2;
                        if (std::abs(c) > 1e-18) {
                            std::vector<int> key = {qi + lq1, qj + lq2};
                            std::sort(key.begin(), key.end());
                            result.poly.add_term(key, c);
                        }
                    }
                }
            }
        }
    }

    result.time_encoding = enc_timer.elapsed();

    // Lagrange multipliers
    Timer lag_timer;
    auto lambdas = compute_momc_lambdas(cfn, params.epsilon_lagrange);
    result.lagrange_multipliers = lambdas;

    // Penalty: lambda_i * sum_{c=1}^{|d|-2} (b_{i,c} - b_{i,c}*b_{i,c-1})
    // (c=0 term is zero because b_{i,-1}=1)
    for (int i = 0; i < N; i++) {
        double lam = lambdas[i];
        int qs = result.qubit_start[i];
        int di = cfn.cardinalities[i];

        for (int c = 1; c < di - 1; c++) {
            result.poly.add_term({qs + c}, lam);
            result.poly.add_term({qs + c - 1, qs + c}, -lam);
        }
    }

    result.time_lagrange = lag_timer.elapsed();

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
