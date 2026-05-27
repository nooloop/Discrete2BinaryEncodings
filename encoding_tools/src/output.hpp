#pragma once
#include "types.hpp"
#include "cfn.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>

// CSV header (common to all encodings)
inline std::string csv_header() {
    return
        "filename,"
        "encoding,"
        "choice_ordering,"
        "bitstring_ordering,"
        "num_variables,"
        "max_cardinality,"
        "edge_density,"
        "distribution,"
        "solution_space_size,"
        "k_approx,"
        "k_trunc,"
        "epsilon_lagrange,"
        "temperature,"
        "weighted_ls,"
        "li_prioritization,"
        "nonlinear_refinement,"
        "quadratized,"
        "num_logical_qubits,"
        "num_auxiliary_qubits,"
        "total_qubits,"
        "max_interaction_degree,"
        "mean_weighted_interaction_degree,"
        "num_linear_terms,"
        "num_quadratic_terms,"
        "num_cubic_terms,"
        "num_higher_order_terms,"
        "total_nonzero_terms,"
        "coefficient_max_abs,"
        "coefficient_min_abs,"
        "coefficient_dynamic_range,"
        "offset,"
        "lagrange_multiplier_max,"
        "lagrange_multiplier_min,"
        "lagrange_multiplier_mean,"
        "rosenberg_penalty,"
        "l2_approximation_error,"
        "linf_approximation_error,"
        "spectral_profile,"
        "time_total_s,"
        "time_parse_s,"
        "time_choice_ordering_s,"
        "time_bitstring_assignment_s,"
        "time_lagrange_s,"
        "time_encoding_s,"
        "time_quadratization_s,"
        "time_nonlinear_refinement_s,"
        "time_output_s,"
        "variable_type";
}

inline std::string csv_row(const CFN& cfn, const EncodingParams& params,
                           const EncodingResult& res) {
    auto stats = res.poly.compute_stats();
    std::ostringstream ss;
    ss << std::setprecision(12);

    // Filename (basename only)
    std::string fname = cfn.filename;
    size_t pos = fname.find_last_of("/\\");
    if (pos != std::string::npos) fname = fname.substr(pos + 1);
    ss << fname << ",";

    ss << params.encoding << ",";
    ss << params.choice_ordering << ",";
    ss << params.bitstring_ordering << ",";
    ss << cfn.num_variables << ",";
    ss << *std::max_element(cfn.cardinalities.begin(), cfn.cardinalities.end()) << ",";

    // Edge density
    int max_edges = cfn.num_variables * (cfn.num_variables - 1) / 2;
    double density = max_edges > 0 ? (double)cfn.pairwise_tables.size() / max_edges : 0;
    ss << density << ",";

    ss << cfn.meta_dist << ",";

    // Solution space size
    double sol_space = 1;
    for (int d : cfn.cardinalities) sol_space *= d;
    ss << sol_space << ",";

    ss << params.k_approx << ",";
    ss << params.k_trunc << ",";
    ss << params.epsilon_lagrange << ",";
    ss << params.temperature << ",";
    ss << (params.weighted_ls ? "true" : "false") << ",";
    ss << (params.li_prioritization ? "true" : "false") << ",";
    ss << (params.nonlinear_refinement ? "true" : "false") << ",";
    ss << (params.quadratize ? "true" : "false") << ",";

    ss << res.num_logical_qubits << ",";
    ss << res.num_auxiliary_qubits << ",";
    ss << (res.num_logical_qubits + res.num_auxiliary_qubits) << ",";

    ss << stats.max_deg << ",";
    ss << stats.mean_weighted_degree << ",";
    ss << stats.num_linear << ",";
    ss << stats.num_quadratic << ",";
    ss << stats.num_cubic << ",";
    ss << stats.num_higher << ",";
    ss << stats.total << ",";
    ss << stats.coeff_max_abs << ",";
    ss << stats.coeff_min_abs << ",";
    ss << stats.dynamic_range << ",";
    ss << res.poly.offset << ",";

    // Lagrange multipliers
    if (!res.lagrange_multipliers.empty()) {
        ss << *std::max_element(res.lagrange_multipliers.begin(), res.lagrange_multipliers.end()) << ",";
        ss << *std::min_element(res.lagrange_multipliers.begin(), res.lagrange_multipliers.end()) << ",";
        double sum = 0;
        for (double l : res.lagrange_multipliers) sum += l;
        ss << sum / res.lagrange_multipliers.size() << ",";
    } else {
        ss << "NA,NA,NA,";
    }

    ss << (res.rosenberg_penalty > 0 ? std::to_string(res.rosenberg_penalty) : "NA") << ",";
    ss << res.l2_error << ",";
    ss << res.linf_error << ",";

    // Spectral profile as JSON array
    if (!res.spectral_profile.empty()) {
        ss << "\"[";
        for (int k = 0; k < (int)res.spectral_profile.size(); k++) {
            if (k > 0) ss << ",";
            ss << res.spectral_profile[k];
        }
        ss << "]\"" << ",";
    } else {
        ss << "NA,";
    }

    ss << res.time_total << ",";
    ss << res.time_parse << ",";
    ss << res.time_choice_ordering << ",";
    ss << res.time_bitstring_assignment << ",";
    ss << res.time_lagrange << ",";
    ss << res.time_encoding << ",";
    ss << res.time_quadratization << ",";
    ss << res.time_nonlinear_refinement << ",";
    ss << res.time_output << ",";
    ss << (res.poly.var_type == VarType::SPIN ? "SPIN" : "BINARY");

    return ss.str();
}

// Write QUBO/HUBO/Ising model as JSON
inline void write_model_json(const std::string& path,
                              const EncodingResult& res,
                              const CFN& cfn,
                              const EncodingParams& params) {
    nlohmann::json j;
    j["encoding"] = params.encoding;
    j["variable_type"] = (res.poly.var_type == VarType::SPIN) ? "SPIN" : "BINARY";
    j["num_logical_qubits"] = res.num_logical_qubits;
    j["num_auxiliary_qubits"] = res.num_auxiliary_qubits;
    j["offset"] = res.poly.offset;
    j["source_cfn"] = cfn.name;

    // Qubit mapping
    nlohmann::json qmap = nlohmann::json::object();
    for (int q = 0; q < (int)res.qubit_info.size(); q++) {
        auto& [var, bit] = res.qubit_info[q];
        qmap[std::to_string(q)] = {{"variable", var},
                                    {"variable_name", cfn.var_names[var]},
                                    {"bit", bit}};
    }
    j["qubit_map"] = qmap;

    // Terms in D-Wave-compatible format
    nlohmann::json terms = nlohmann::json::object();
    auto sorted = res.poly.sorted_terms();
    for (auto& [indices, coeff] : sorted) {
        std::ostringstream key;
        for (int k = 0; k < (int)indices.size(); k++) {
            if (k > 0) key << ",";
            key << indices[k];
        }
        terms[key.str()] = coeff;
    }
    j["terms"] = terms;

    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error("Cannot open output file: " + path);
    ofs << j.dump(2);
}
