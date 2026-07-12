#pragma once
#include "qa_types.hpp"
#include "output.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

// ============================================================================
// CSV output for QA solver results.
//
// Columns 1-31 are the shared core (csv_core_header, defined in output.hpp and
// emitted verbatim by solve_sa), so SA and QA rows share names, order and
// meaning and can be analysed with one code path. Columns 32+ are the
// D-Wave-specific timing and embedding statistics.
//
// As in solve_sa, the encoding-space and CFN-space results are kept in separate
// columns: best_encoded_solution / per_run_energies are under the encoding,
// best_cfn_solution / per_run_cfn_energies under the source CFN. Only the
// *_cfn_* columns are comparable across encodings.
// ============================================================================

inline std::string csv_header_qa() {
    return csv_core_header() + ","
        // --- QA-specific columns ---
        "solver_name,"
        "annealing_time_us,"
        "delta_max,"
        "inhomog_setup_time_s,"
        "embedding_time_s,"
        "qpu_access_time_us,"
        "qpu_sampling_time_us,"
        "qpu_programming_time_us,"
        // --- Embedding statistics ---
        "emb_num_physical_qubits,"
        "emb_chain_length_avg,"
        "emb_chain_length_median,"
        "emb_chain_length_var,"
        "emb_chain_breaks_avg,"
        "emb_chain_breaks_median,"
        "emb_chain_breaks_var";
}

inline std::string csv_row_qa(const QAResult& r) {
    std::ostringstream ss;
    ss << std::setprecision(12);

    // --- Shared core (columns 1-31), mirroring csv_row in output.hpp ---
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
    ss << "NA,";                  // schedule   (SA-only)
    ss << "NA,";                  // move_type  (SA-only)
    ss << "0,";                   // T_start    (SA-only)
    ss << "0,";                   // T_end      (SA-only)
    ss << "0,";                   // num_steps  (SA-only)
    ss << r.num_reads << ",";     // num_runs = num_reads
    ss << "0,";                   // seed       (SA-only)

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
    ss << r.mean_run_time_us << ",";

    ss << format_solution(r.best_encoded_solution) << ",";
    ss << format_solution(r.best_cfn_solution) << ",";
    ss << format_energies(r.per_run_energies) << ",";
    ss << format_energies(r.per_run_cfn_energies) << ",";
    ss << format_energies(r.per_run_times_us) << ",";   // empty -> NA (no per-read timing)

    if (std::isnan(r.best_cfn_energy))
        ss << "NA,";
    else
        ss << r.best_cfn_energy << ",";

    ss << r.num_feasible << ",";
    ss << r.num_best_cfn << ",";

    // --- QA-specific columns ---
    ss << r.solver_name << ",";
    ss << r.annealing_time_us << ",";
    ss << r.delta_max << ",";
    ss << r.inhomog_setup_time_s << ",";
    ss << r.embedding_time_s << ",";
    ss << r.qpu_access_time_us << ",";
    ss << r.qpu_sampling_time_us << ",";
    ss << r.qpu_programming_time_us << ",";

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
