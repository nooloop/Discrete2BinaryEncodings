#pragma once
#include "qa_types.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

// ============================================================================
// CSV output for QA solver results.
//
// Columns 1-26 match solve_sa for easy comparison.
// Columns 27+ are D-Wave / QA-specific.
// ============================================================================

inline std::string csv_header_qa() {
    return
        // --- SA-compatible columns (1-26) ---
        "problem_name,"
        "source_cfn,"
        "solver_mode,"
        "encoding,"
        "variable_type,"
        "num_qubits,"
        "num_variables,"
        "max_cardinality,"
        "edge_density,"
        "distribution,"
        "schedule,"
        "move_type,"
        "T_start,"
        "T_end,"
        "num_steps,"
        "num_runs,"
        "seed,"
        "best_energy,"
        "mean_energy,"
        "std_energy,"
        "median_energy,"
        "num_optimal,"
        "total_runtime_s,"
        "mean_time_per_run_s,"
        "best_solution,"
        "per_run_energies,"
        // --- QA-specific columns (27+) ---
        "solver_name,"
        "annealing_time_us,"
        "delta_max,"
        "inhomog_setup_time_s,"
        "embedding_time_s,"
        "qpu_access_time_us,"
        "qpu_sampling_time_us,"
        "qpu_programming_time_us,"
        "best_cfn_energy,"
        "num_feasible,"
        "num_best_cfn,"
        // --- Embedding statistics ---
        "emb_num_physical_qubits,"
        "emb_chain_length_avg,"
        "emb_chain_length_median,"
        "emb_chain_length_var,"
        "emb_chain_breaks_avg,"
        "emb_chain_breaks_median,"
        "emb_chain_breaks_var";
}

inline std::string format_solution_qa(const std::vector<int>& sol) {
    if (sol.empty()) return "NA";
    std::ostringstream ss;
    ss << "\"[";
    for (int k = 0; k < static_cast<int>(sol.size()); k++) {
        if (k > 0) ss << ",";
        ss << sol[k];
    }
    ss << "]\"";
    return ss.str();
}

inline std::string csv_row_qa(const QAResult& r) {
    std::ostringstream ss;
    ss << std::setprecision(12);

    // --- SA-compatible columns ---
    ss << r.problem_name << ",";
    ss << r.source_cfn << ",";
    ss << "dwave,";               // solver_mode
    ss << r.encoding << ",";
    ss << r.variable_type << ",";
    ss << r.num_qubits << ",";
    ss << r.num_variables << ",";
    ss << r.max_cardinality << ",";
    ss << r.edge_density << ",";
    ss << r.distribution << ",";
    ss << "NA,";                  // schedule
    ss << "NA,";                  // move_type
    ss << "0,";                   // T_start
    ss << "0,";                   // T_end
    ss << "0,";                   // num_steps
    ss << r.num_reads << ",";     // num_runs = num_reads
    ss << "0,";                   // seed

    ss << r.best_energy << ",";
    ss << r.mean_energy << ",";
    ss << r.std_energy << ",";
    ss << r.median_energy << ",";

    if (r.num_optimal >= 0)
        ss << r.num_optimal << ",";
    else
        ss << "NA,";

    ss << r.total_runtime_s << ",";
    ss << (r.num_reads > 0 ? r.total_runtime_s / r.num_reads : 0) << ",";

    ss << format_solution_qa(r.best_solution) << ",";

    // Per-run energies as quoted JSON array
    ss << "\"[";
    for (int k = 0; k < static_cast<int>(r.per_run_energies.size()); k++) {
        if (k > 0) ss << ",";
        ss << r.per_run_energies[k];
    }
    ss << "]\",";

    // --- QA-specific columns ---
    ss << r.solver_name << ",";
    ss << r.annealing_time_us << ",";
    ss << r.delta_max << ",";
    ss << r.inhomog_setup_time_s << ",";
    ss << r.embedding_time_s << ",";
    ss << r.qpu_access_time_us << ",";
    ss << r.qpu_sampling_time_us << ",";
    ss << r.qpu_programming_time_us << ",";

    if (std::isnan(r.best_cfn_energy))
        ss << "NA,";
    else
        ss << r.best_cfn_energy << ",";

    ss << r.num_feasible << ",";
    ss << r.num_best_cfn << ",";

    // --- Embedding statistics ---
    ss << r.emb_num_physical_qubits << ",";
    ss << r.emb_chain_length_avg << ",";
    ss << r.emb_chain_length_median << ",";
    ss << r.emb_chain_length_var << ",";
    ss << r.emb_chain_breaks_avg << ",";
    ss << r.emb_chain_breaks_median << ",";
    ss << r.emb_chain_breaks_var;

    return ss.str();
}
