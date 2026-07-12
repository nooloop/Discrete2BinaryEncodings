#pragma once
#include "sa_types.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// ============================================================================
// QA solver parameters (populated from CLI)
// ============================================================================
struct QAParams {
    std::string input_path;
    std::string cfn_dir;
    std::string solver_name;
    std::string python_cmd    = "python3";
    double annealing_time_us  = 20.0;
    int    num_reads          = 1000;
    double delta_max          = 0.1;
    bool   use_inhomogeneous  = true;
    double ground_truth       = std::numeric_limits<double>::quiet_NaN();
    double tolerance          = 1e-6;
    bool   verbose            = false;
};

// ============================================================================
// D-Wave sample (after unembedding to logical qubits)
// ============================================================================
struct DWaveSample {
    std::vector<int> values;      // qubit values: {0,1} or {-1,+1}
    double energy;
    int    num_occurrences;
};

// ============================================================================
// D-Wave timing information (microseconds unless noted)
// ============================================================================
struct DWaveTiming {
    double qpu_access_time_us             = 0;
    double qpu_sampling_time_us           = 0;
    double qpu_programming_time_us        = 0;
    double post_processing_time_us        = 0;
};

// ============================================================================
// Per-sample chain break count (from embedded sampleset)
// ============================================================================
struct ChainBreakEntry {
    int breaks;
    int num_occurrences;
};

// ============================================================================
// Parsed results from D-Wave submission Python script
// ============================================================================
struct DWaveResults {
    std::vector<DWaveSample> samples;
    DWaveTiming timing;
    double embedding_time_s    = 0;
    double chain_strength      = 0;
    int    num_physical_qubits = 0;
    int    max_chain_length    = 0;

    // Embedding statistics
    std::vector<int> chain_lengths;                // one per logical variable
    std::vector<ChainBreakEntry> chain_breaks;     // per unique embedded sample
};

// ============================================================================
// Complete QA result for CSV output
// ============================================================================
struct QAResult {
    // --- Problem metadata ---
    std::string problem_name;
    std::string source_cfn;
    std::string encoding;
    std::string variable_type;
    int    num_qubits       = 0;
    int    num_variables    = 0;
    int    max_cardinality  = 0;
    double edge_density     = 0;
    std::string distribution;

    // --- D-Wave configuration ---
    std::string solver_name;
    double annealing_time_us = 20.0;
    int    num_reads         = 1000;
    double delta_max         = 0.1;

    // --- Energy statistics (under the encoding) ---
    double best_energy    = 0;
    double mean_energy    = 0;
    double std_energy     = 0;
    double median_energy  = 0;
    int    num_optimal    = -1;    // -1 = not computed

    // --- Timing ---
    double total_runtime_s          = 0;
    double inhomog_setup_time_s     = 0;
    double embedding_time_s         = 0;
    double qpu_access_time_us      = 0;
    double qpu_sampling_time_us    = 0;
    double qpu_programming_time_us = 0;

    // Mean time of a single anneal, in microseconds: the QA counterpart of the
    // SA per-trajectory time, so both solvers fill the same core column. The QPU
    // reports sampling time for the batch rather than per read, so this is
    // qpu_sampling_time_us / num_reads. per_run_times_us is left empty (-> NA):
    // unlike SA, there is no measured time for an individual read to record.
    double mean_run_time_us = 0;

    // --- Embedding statistics ---
    int    emb_num_physical_qubits = 0;
    double emb_chain_length_avg    = 0;
    double emb_chain_length_median = 0;
    double emb_chain_length_var    = 0;
    double emb_chain_breaks_avg    = 0;
    double emb_chain_breaks_median = 0;
    double emb_chain_breaks_var    = 0;

    // --- CFN evaluation ---
    double best_cfn_energy = std::numeric_limits<double>::quiet_NaN();
    int    num_feasible    = 0;
    int    num_best_cfn    = 0;

    // --- Solutions ---
    std::vector<int> best_encoded_solution;  // raw sample (qubits) at best_energy
    std::vector<int> best_cfn_solution;      // decoded CFN choices at best_cfn_energy

    // Per-read vectors, expanded by num_occurrences and index-aligned with each
    // other: entry k is the same read under the encoding and under the CFN.
    // per_run_cfn_energies is NaN for a read that decoded infeasibly.
    std::vector<double> per_run_energies;      // D-Wave energies (encoding)
    std::vector<double> per_run_cfn_energies;  // decoded CFN costs

    // Always empty for QA (-> NA in the CSV): the QPU does not report a time for
    // an individual read. Use mean_run_time_us instead. Present so the shared
    // core column exists in both solvers' output.
    std::vector<double> per_run_times_us;
};
