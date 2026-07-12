#pragma once
#include "sa_types.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

// ============================================================================
// CSV output for solver results.
//
// Columns 1-31 are the shared core, identical in name, order and meaning to
// solve_qa (see output_qa.hpp), so SA and QA rows can be concatenated and
// analysed with one code path. solve_qa appends its QPU-specific columns after
// the core.
//
// Two solution columns, two energy vectors -- they mean different things:
//   best_encoded_solution / per_run_energies      under the ENCODING
//   best_cfn_solution     / per_run_cfn_energies  under the SOURCE CFN
// For the cost-approximate encodings (approximate_binary, truncated_binary) the
// encoded energy is not the CFN cost, so only the *_cfn_* columns are
// comparable across encodings. The *_cfn_* columns hold the best DECODED cost
// over each run's trajectory, not the cost of the run's lowest-energy state --
// the two differ, and only the former is a fair score (see cfn_tracker.hpp).
//
// per_run_times_us is the measured wall time of each individual trajectory (the
// anneal only, in microseconds), index-aligned with the two energy vectors, with
// mean_run_time_us its mean. mean_time_per_run_s is coarser: total_runtime_s (all
// runs for this model, including the decode) divided by num_runs.
// ============================================================================

inline std::string csv_core_header() {
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
        "mean_run_time_us,"
        "best_encoded_solution,"
        "best_cfn_solution,"
        "per_run_energies,"
        "per_run_cfn_energies,"
        "per_run_times_us,"
        "best_cfn_energy,"
        "num_feasible,"
        "num_best_cfn";
}

inline std::string csv_header() {
    return csv_core_header();
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

// Quoted JSON array of doubles; NaN entries (infeasible decodes) render as null
// so the column parses as JSON and the infeasible runs stay visible.
inline std::string format_energies(const std::vector<double>& v) {
    if (v.empty()) return "NA";
    std::ostringstream ss;
    ss << std::setprecision(12);
    ss << "\"[";
    for (int k = 0; k < static_cast<int>(v.size()); k++) {
        if (k > 0) ss << ",";
        if (std::isnan(v[k])) ss << "null";
        else                  ss << v[k];
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
    ss << r.mean_run_time_us << ",";

    ss << format_solution(r.best_encoded_solution) << ",";
    ss << format_solution(r.best_cfn_solution) << ",";
    ss << format_energies(r.per_run_energies) << ",";
    ss << format_energies(r.per_run_cfn_energies) << ",";
    ss << format_energies(r.per_run_times_us) << ",";

    if (std::isnan(r.best_cfn_energy)) ss << "NA";
    else                               ss << r.best_cfn_energy;
    ss << ",";
    if (r.num_feasible < 0) ss << "NA"; else ss << r.num_feasible;
    ss << ",";
    if (r.num_best_cfn < 0) ss << "NA"; else ss << r.num_best_cfn;

    return ss.str();
}
