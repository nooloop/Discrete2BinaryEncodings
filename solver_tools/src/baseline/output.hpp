#pragma once
#include "sa_types.hpp"
#include <string>
#include <sstream>
#include <iomanip>

// ============================================================================
// CSV output for solver results.
// ============================================================================

inline std::string csv_header() {
    return
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
        "per_run_energies";
}

inline std::string format_solution(const std::vector<int>& sol) {
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

inline std::string csv_row(const AggregateResult& r) {
    std::ostringstream ss;
    ss << std::setprecision(12);

    ss << r.problem_name << ",";
    ss << r.source_cfn << ",";
    ss << r.solver_mode << ",";
    ss << r.encoding << ",";
    ss << r.variable_type << ",";
    ss << r.num_qubits << ",";
    ss << r.num_variables << ",";
    ss << r.max_cardinality << ",";
    ss << r.edge_density << ",";
    ss << r.distribution << ",";
    ss << r.schedule << ",";
    ss << r.move_type << ",";
    ss << r.T_start << ",";
    ss << r.T_end << ",";
    ss << r.num_steps << ",";
    ss << r.num_runs << ",";
    ss << r.seed << ",";
    ss << r.best_energy << ",";
    ss << r.mean_energy << ",";
    ss << r.std_energy << ",";
    ss << r.median_energy << ",";

    if (r.num_optimal >= 0)
        ss << r.num_optimal << ",";
    else
        ss << "NA,";

    ss << r.total_runtime_s << ",";
    ss << r.mean_time_per_run_s << ",";

    ss << format_solution(r.best_solution) << ",";

    // Per-run energies as quoted JSON array
    ss << "\"[";
    for (int k = 0; k < static_cast<int>(r.per_run_energies.size()); k++) {
        if (k > 0) ss << ",";
        ss << r.per_run_energies[k];
    }
    ss << "]\"";

    return ss.str();
}
