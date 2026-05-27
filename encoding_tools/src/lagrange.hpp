#pragma once
#include "cfn.hpp"

// MOMC Lagrange multiplier computation (Ayodele 2022, Eq. 14-15 in manuscript).
// Per-register: lambda_i = (1+epsilon) * Delta_i
// where Delta_i = range(unary_i) + sum_j range(pairwise_{i,j})
inline std::vector<double> compute_momc_lambdas(const CFN& cfn, double epsilon = 0.1) {
    int N = cfn.num_variables;
    std::vector<double> lambdas(N, 0.0);

    for (int i = 0; i < N; i++) {
        double delta = 0.0;

        // Unary contribution: max - min
        if (!cfn.unary_tables[i].costs.empty()) {
            auto& c = cfn.unary_tables[i].costs;
            auto [mn, mx] = std::minmax_element(c.begin(), c.end());
            delta += *mx - *mn;
        }

        // Pairwise contributions
        for (auto& pt : cfn.pairwise_tables) {
            bool involves_i = false;
            for (int s : pt.scope)
                if (s == i) { involves_i = true; break; }
            if (!involves_i) continue;

            auto [mn, mx] = std::minmax_element(pt.costs.begin(), pt.costs.end());
            delta += *mx - *mn;
        }

        lambdas[i] = (1.0 + epsilon) * delta;
    }

    return lambdas;
}
