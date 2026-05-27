#pragma once
#include "cfn.hpp"
#include "lagrange.hpp"

inline EncodingResult encode_one_hot(const CFN& cfn, const EncodingParams& params) {
    Timer total_timer;
    EncodingResult result;
    int N = cfn.num_variables;

    // Qubit assignment: variable i gets qubits [start, start + |d_i|)
    result.qubit_start.resize(N);
    result.bits_per_var.resize(N);
    int q = 0;
    for (int i = 0; i < N; i++) {
        result.qubit_start[i] = q;
        result.bits_per_var[i] = cfn.cardinalities[i];
        for (int c = 0; c < cfn.cardinalities[i]; c++)
            result.qubit_info.push_back({i, c});
        q += cfn.cardinalities[i];
    }
    result.num_logical_qubits = q;
    result.poly.var_type = VarType::BINARY;

    Timer enc_timer;

    // Unary costs: C_{i;c} * b_{i,c}
    for (int i = 0; i < N; i++) {
        int qs = result.qubit_start[i];
        int di = cfn.cardinalities[i];
        for (int c = 0; c < di; c++) {
            if (std::abs(cfn.unary_tables[i].costs[c]) > 1e-18)
                result.poly.add_term({qs + c}, cfn.unary_tables[i].costs[c]);
        }
    }

    // Pairwise costs: C_{i,j;c_i,c_j} * b_{i,c_i} * b_{j,c_j}
    for (auto& pt : cfn.pairwise_tables) {
        int i = pt.scope[0], j = pt.scope[1];
        int qi = result.qubit_start[i], qj = result.qubit_start[j];
        int di = cfn.cardinalities[i], dj = cfn.cardinalities[j];
        for (int ci = 0; ci < di; ci++) {
            for (int cj = 0; cj < dj; cj++) {
                double c = pt.costs[ci * dj + cj];
                if (std::abs(c) > 1e-18) {
                    std::vector<int> key = {qi + ci, qj + cj};
                    std::sort(key.begin(), key.end());
                    result.poly.add_term(key, c);
                }
            }
        }
    }

    result.time_encoding = enc_timer.elapsed();

    // Lagrange multipliers
    Timer lag_timer;
    auto lambdas = compute_momc_lambdas(cfn, params.epsilon_lagrange);
    result.lagrange_multipliers = lambdas;

    // Add penalty: lambda_i * (sum_c b_{i,c} - 1)^2
    // = lambda_i * (-sum_c b_{i,c} + 2*sum_{c<c'} b_{i,c}*b_{i,c'} + 1)
    for (int i = 0; i < N; i++) {
        double lam = lambdas[i];
        int qs = result.qubit_start[i];
        int di = cfn.cardinalities[i];

        result.poly.add_constant(lam);

        for (int c = 0; c < di; c++)
            result.poly.add_term({qs + c}, -lam);

        for (int c1 = 0; c1 < di; c1++)
            for (int c2 = c1 + 1; c2 < di; c2++)
                result.poly.add_term({qs + c1, qs + c2}, 2.0 * lam);
    }

    result.time_lagrange = lag_timer.elapsed();

    result.poly.prune();
    result.time_total = total_timer.elapsed();
    return result;
}
