#pragma once
#include "baseline/sa_types.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>

// ============================================================================
// CFN model for simulated annealing.
// ============================================================================

struct CFNModel {
    std::string name;
    int num_variables = 0;
    std::vector<std::string> var_names;
    std::vector<int> cardinalities;

    std::vector<std::vector<double>> unary;

    struct PairwiseTable {
        int vi, vj;
        int di, dj;
        std::vector<double> costs;
        double cost(int ci, int cj) const { return costs[ci * dj + cj]; }
    };
    std::vector<PairwiseTable> pairwise;

    struct AdjEntry {
        int table_idx;
        int neighbor;
        bool i_is_first;
    };
    std::vector<std::vector<AdjEntry>> adjacency;

    CFNMetadata meta;

    void build_adjacency() {
        adjacency.clear();
        adjacency.resize(num_variables);
        for (int tidx = 0; tidx < static_cast<int>(pairwise.size()); tidx++) {
            const auto& pt = pairwise[tidx];
            adjacency[pt.vi].push_back({tidx, pt.vj, true});
            adjacency[pt.vj].push_back({tidx, pt.vi, false});
        }
    }
};

inline double compute_energy(const CFNModel& model,
                             const std::vector<int>& state) {
    double E = 0;
    for (int i = 0; i < model.num_variables; i++)
        E += model.unary[i][state[i]];
    for (const auto& pt : model.pairwise)
        E += pt.cost(state[pt.vi], state[pt.vj]);
    return E;
}

inline double delta_energy(const CFNModel& model,
                           const std::vector<int>& state,
                           int i, int new_c) {
    int old_c = state[i];
    double delta = model.unary[i][new_c] - model.unary[i][old_c];

    for (const auto& adj : model.adjacency[i]) {
        const auto& pt = model.pairwise[adj.table_idx];
        int xj = state[adj.neighbor];
        if (adj.i_is_first) {
            delta += pt.costs[new_c * pt.dj + xj]
                   - pt.costs[old_c * pt.dj + xj];
        } else {
            delta += pt.costs[xj * pt.dj + new_c]
                   - pt.costs[xj * pt.dj + old_c];
        }
    }
    return delta;
}

inline CFNModel parse_cfn_for_sa(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open CFN file: " + path);

    nlohmann::json j;
    ifs >> j;

    CFNModel model;
    if (j.contains("problem") && j["problem"].contains("name"))
        model.name = j["problem"]["name"].get<std::string>();

    const auto& vars = j["variables"];
    model.num_variables = static_cast<int>(vars.size());
    model.var_names.reserve(model.num_variables);
    model.cardinalities.reserve(model.num_variables);
    model.unary.resize(model.num_variables);

    std::unordered_map<std::string, int> name_to_idx;
    int idx = 0;
    for (auto it = vars.begin(); it != vars.end(); ++it, ++idx) {
        model.var_names.push_back(it.key());
        model.cardinalities.push_back(it.value().get<int>());
        name_to_idx[it.key()] = idx;
    }

    for (int i = 0; i < model.num_variables; i++)
        model.unary[i].assign(model.cardinalities[i], 0.0);

    const auto& funcs = j["functions"];
    for (auto it = funcs.begin(); it != funcs.end(); ++it) {
        const auto& f = it.value();
        const auto& scope_arr = f["scope"];
        std::vector<int> scope;
        for (const auto& s : scope_arr)
            scope.push_back(name_to_idx.at(s.get<std::string>()));

        std::vector<double> costs;
        for (const auto& c : f["costs"])
            costs.push_back(c.get<double>());

        if (scope.size() == 1) {
            model.unary[scope[0]] = costs;
        } else if (scope.size() == 2) {
            CFNModel::PairwiseTable pt;
            pt.vi = scope[0];
            pt.vj = scope[1];
            pt.di = model.cardinalities[pt.vi];
            pt.dj = model.cardinalities[pt.vj];
            pt.costs = std::move(costs);
            model.pairwise.push_back(std::move(pt));
        }
    }

    std::string stem = path_stem(path);
    model.meta = parse_cfn_metadata(stem);

    model.build_adjacency();
    return model;
}
