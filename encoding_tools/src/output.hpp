#pragma once
#include "baseline/types.hpp"
#include "baseline/cfn.hpp"
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
        "lagrange_multiplier,"
        "rosenberg_penalty,"
        "l2_approximation_error,"
        "linf_approximation_error,"
        "spectral_profile,"
        "time_total_us,"
        "time_parse_us,"
        "time_choice_ordering_us,"
        "time_bitstring_assignment_us,"
        "time_lagrange_us,"
        "time_encoding_us,"
        "time_quadratization_us,"
        "time_nonlinear_refinement_us,"
        "time_output_us,"
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

    // Lagrange multiplier (global MOMC)
    if (res.lagrange_multiplier > 0)
        ss << res.lagrange_multiplier << ",";
    else
        ss << "NA,";

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
inline nlohmann::json build_model_json(const EncodingResult& res,
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

    // Choice -> bitstring assignment (binary encodings only). Without this,
    // the decoder cannot invert Gray / Boltzmann / LI-prioritized layouts and
    // falls back to natural-binary decoding, which yields wrong CFN choices for
    // the enhanced encodings. Element i is the per-choice bitstring list for
    // variable i: choice_to_bitstring[i][c] = integer bitstring of choice c.
    if (!res.choice_to_bitstring.empty()) {
        nlohmann::json cb = nlohmann::json::array();
        for (const auto& var_assign : res.choice_to_bitstring)
            cb.push_back(var_assign);
        j["choice_to_bitstring"] = cb;
    }

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

    return j;
}

// Backward-compatible per-instance writer (pretty JSON); used when --jsonl is off.
inline void write_model_json(const std::string& path,
                             const EncodingResult& res,
                             const CFN& cfn,
                             const EncodingParams& params) {
    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error("Cannot open output file: " + path);
    ofs << build_model_json(res, cfn, params).dump(2);
}
