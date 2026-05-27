#pragma once
#include "types.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <regex>

struct CostTable {
    std::vector<int> scope;     // variable indices
    std::vector<double> costs;  // flattened row-major, rightmost varies fastest
};

struct CFN {
    std::string name;
    std::string filename;
    int num_variables = 0;
    std::vector<std::string> var_names;
    std::vector<int> cardinalities;
    std::vector<CostTable> unary_tables;    // one per variable (index matches variable index)
    std::vector<CostTable> pairwise_tables; // edges

    // Metadata extracted from filename
    int meta_N = 0, meta_D = 0;
    double meta_rho = 0;
    std::string meta_dist;

    double pairwise_cost(int table_idx, int ci, int cj) const {
        auto& t = pairwise_tables[table_idx];
        int dj = cardinalities[t.scope[1]];
        return t.costs[ci * dj + cj];
    }
};

inline CFN parse_cfn(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open CFN file: " + path);

    nlohmann::json j;
    ifs >> j;

    CFN cfn;
    cfn.filename = path;
    if (j.contains("problem") && j["problem"].contains("name"))
        cfn.name = j["problem"]["name"].get<std::string>();

    auto& vars = j["variables"];
    cfn.num_variables = (int)vars.size();
    cfn.var_names.reserve(cfn.num_variables);
    cfn.cardinalities.reserve(cfn.num_variables);
    cfn.unary_tables.resize(cfn.num_variables);

    std::unordered_map<std::string, int> name_to_idx;
    int idx = 0;
    for (auto it = vars.begin(); it != vars.end(); ++it, ++idx) {
        cfn.var_names.push_back(it.key());
        cfn.cardinalities.push_back(it.value().get<int>());
        name_to_idx[it.key()] = idx;
    }

    auto& funcs = j["functions"];
    for (auto it = funcs.begin(); it != funcs.end(); ++it) {
        auto& f = it.value();
        auto& scope_arr = f["scope"];
        std::vector<int> scope;
        for (auto& s : scope_arr)
            scope.push_back(name_to_idx.at(s.get<std::string>()));

        std::vector<double> costs;
        for (auto& c : f["costs"])
            costs.push_back(c.get<double>());

        CostTable ct{scope, costs};

        if (scope.size() == 1) {
            cfn.unary_tables[scope[0]] = ct;
        } else if (scope.size() == 2) {
            cfn.pairwise_tables.push_back(ct);
        }
    }

    // Extract metadata from filename pattern: CFN_N<N>_D<D>_rho<rho>_<dist>_<num>.cfn
    size_t last_sep = path.find_last_of("/\\");
    std::string basename = (last_sep == std::string::npos) ? path : path.substr(last_sep + 1);
    std::regex re(R"(CFN_N(\d+)_D(\d+)_rho([\d.]+)_(\w+)_(\d+)\.cfn)");
    std::smatch m;
    if (std::regex_match(basename, m, re)) {
        cfn.meta_N = std::stoi(m[1]);
        cfn.meta_D = std::stoi(m[2]);
        cfn.meta_rho = std::stod(m[3]);
        cfn.meta_dist = m[4];
    }

    return cfn;
}
