#pragma once
#include "baseline/qa_types.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "utilities/dwave_template.hpp"

#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <limits>

// ============================================================================
// Validate that the model is at most degree 2 (QUBO / Ising).
// ============================================================================
inline void validate_qubo_model(const BinaryModel& model) {
    for (const auto& t : model.terms) {
        if (t.qubits.size() > 2) {
            throw std::runtime_error(
                "Non-quadratic term found (degree "
                + std::to_string(t.qubits.size())
                + "). D-Wave QPU supports only QUBO/Ising models (degree <= 2). "
                  "Use --quadratize when encoding with encoding_tools."
            );
        }
    }
}

// ============================================================================
// Compute per-qubit effective fields using the O(N_i) recurrence.
//
//   SPIN  {-1,+1}:
//     f = |h_i|
//     for each coupling J: f = (|f+J| + |f-J|) / 2
//     eff_field = f
//
//   BINARY {0,1}:
//     T = sum of couplings J_ij to qubit i
//     f = |2*h_i + T|
//     for each coupling J: f = (|f+J| + |f-J|) / 2
//     eff_field = f / 2
//
// Reference: Adame et al. (2020) / Zaborniak (2025), Algorithm 1.
// ============================================================================
inline std::vector<double> compute_effective_fields(const BinaryModel& model) {
    const int Q = model.num_qubits;
    const bool is_spin = (model.var_type == BinaryModel::SPIN);

    // --- Extract per-qubit biases and aggregated couplings ---
    std::vector<double> biases(Q, 0.0);
    std::vector<std::unordered_map<int, double>> coupling_map(Q);

    for (const auto& t : model.terms) {
        if (t.qubits.size() == 1) {
            biases[t.qubits[0]] += t.coeff;
        } else if (t.qubits.size() == 2) {
            int q0 = t.qubits[0], q1 = t.qubits[1];
            coupling_map[q0][q1] += t.coeff;
            coupling_map[q1][q0] += t.coeff;
        }
    }

    // --- Compute effective field per qubit ---
    std::vector<double> fields(Q, 0.0);

    for (int q = 0; q < Q; q++) {
        double h = biases[q];

        // Collect coupling values to distinct neighbors
        std::vector<double> J_list;
        J_list.reserve(coupling_map[q].size());
        for (const auto& [neighbor, J] : coupling_map[q])
            J_list.push_back(J);

        if (J_list.empty()) {
            fields[q] = std::abs(h);
            continue;
        }

        double f;
        if (is_spin) {
            // Spin recurrence: start from |h|
            f = std::abs(h);
        } else {
            // Binary recurrence: start from |2h + T|
            double T = 0.0;
            for (double j : J_list) T += j;
            f = std::abs(2.0 * h + T);
        }

        // Apply recurrence: f = (|f+J| + |f-J|) / 2
        for (double j : J_list) {
            f = 0.5 * (std::abs(f + j) + std::abs(f - j));
        }

        fields[q] = is_spin ? f : 0.5 * f;
    }

    return fields;
}

// ============================================================================
// Convert effective fields to anneal offsets.
//
//   r_i = |F_i| / max_k |F_k|       (normalize to [0,1])
//   delta_i = |delta_max| * (1 - 2*r_i)
//
// Strongly-coupled qubits (large r) get negative offsets (delayed anneal).
// Weakly-coupled qubits (small r) get positive offsets (advanced anneal).
// ============================================================================
inline std::vector<double> compute_anneal_offsets(
    const std::vector<double>& eff_fields,
    double delta_max
) {
    const int Q = static_cast<int>(eff_fields.size());
    std::vector<double> offsets(Q, std::abs(delta_max));

    double max_f = *std::max_element(eff_fields.begin(), eff_fields.end());
    if (max_f <= 1e-15) return offsets;   // all fields zero, uniform offsets

    for (int q = 0; q < Q; q++) {
        double r = eff_fields[q] / max_f;
        offsets[q] = std::abs(delta_max) * (1.0 - 2.0 * r);
    }
    return offsets;
}

// ============================================================================
// Generate the Python submission script from the embedded template.
// ============================================================================
inline std::string generate_dwave_script(
    const std::string& input_path,
    const std::string& output_path,
    const QAParams& params,
    const std::vector<double>& anneal_offsets
) {
    std::string script(DWAVE_SUBMIT_TEMPLATE);

    // Simple placeholder replacement
    auto replace_all = [&](const std::string& placeholder,
                           const std::string& value) {
        size_t pos = 0;
        while ((pos = script.find(placeholder, pos)) != std::string::npos) {
            script.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    };

    replace_all("__INPUT_PATH__",    input_path);
    replace_all("__OUTPUT_PATH__",   output_path);
    replace_all("__SOLVER_NAME__",   params.solver_name);

    std::ostringstream at_ss;
    at_ss << std::setprecision(12) << params.annealing_time_us;
    replace_all("__ANNEALING_TIME__", at_ss.str());

    replace_all("__NUM_READS__",
                std::to_string(params.num_reads));
    replace_all("__USE_INHOMOGENEOUS__",
                params.use_inhomogeneous ? "True" : "False");

    // Format anneal offsets as Python dict {qubit_idx: offset, ...}
    std::ostringstream off_ss;
    off_ss << std::setprecision(12) << "{";
    for (int q = 0; q < static_cast<int>(anneal_offsets.size()); q++) {
        if (q > 0) off_ss << ", ";
        off_ss << q << ": " << anneal_offsets[q];
    }
    off_ss << "}";
    replace_all("__ANNEAL_OFFSETS__", off_ss.str());

    return script;
}

// ============================================================================
// Parse D-Wave results JSON produced by the Python script.
// ============================================================================
inline DWaveResults parse_dwave_results(const std::string& results_path) {
    std::ifstream ifs(results_path);
    if (!ifs.is_open())
        throw std::runtime_error(
            "Cannot open D-Wave results file: " + results_path);

    nlohmann::json j;
    ifs >> j;
    DWaveResults results;

    // Samples
    for (const auto& s : j["samples"]) {
        DWaveSample sample;
        for (const auto& v : s["values"])
            sample.values.push_back(v.get<int>());
        sample.energy          = s["energy"].get<double>();
        sample.num_occurrences = s["num_occurrences"].get<int>();
        results.samples.push_back(std::move(sample));
    }

    // Timing
    if (j.contains("timing")) {
        const auto& t = j["timing"];
        results.timing.qpu_access_time_us
            = t.value("qpu_access_time", 0.0);
        results.timing.qpu_sampling_time_us
            = t.value("qpu_sampling_time", 0.0);
        results.timing.qpu_programming_time_us
            = t.value("qpu_programming_time", 0.0);
        results.timing.post_processing_time_us
            = t.value("post_processing_time", 0.0);
    }

    results.embedding_time_s    = j.value("embedding_time_s", 0.0);
    results.chain_strength      = j.value("chain_strength", 0.0);
    results.num_physical_qubits = j.value("num_physical_qubits", 0);
    results.max_chain_length    = j.value("max_chain_length", 0);

    // Chain lengths
    if (j.contains("chain_lengths")) {
        for (const auto& v : j["chain_lengths"])
            results.chain_lengths.push_back(v.get<int>());
    }

    // Chain breaks per sample
    if (j.contains("chain_breaks")) {
        for (const auto& cb : j["chain_breaks"]) {
            ChainBreakEntry e;
            e.breaks          = cb["breaks"].get<int>();
            e.num_occurrences = cb["num_occurrences"].get<int>();
            results.chain_breaks.push_back(e);
        }
    }

    return results;
}

// ============================================================================
// Main QA orchestration.
//
//   1. Validate model is QUBO/Ising (degree <= 2).
//   2. Compute effective fields and anneal offsets (C++, O(N^2)).
//   3. Generate Python submission script from template.
//   4. Execute Python script (embedding + D-Wave submission).
//   5. Parse results and decode bitstrings to CFN solutions.
//   6. Evaluate decoded solutions under the original CFN.
//   7. Return aggregate QAResult.
// ============================================================================
inline QAResult run_qa_binary(
    const BinaryModel& model,
    const CFNModel&    cfn,
    const QAParams&    params
) {
    SATimer total_timer;

    // ------------------------------------------------------------------
    // 1. Validate QUBO
    // ------------------------------------------------------------------
    validate_qubo_model(model);

    // ------------------------------------------------------------------
    // 2. Compute effective fields and anneal offsets
    // ------------------------------------------------------------------
    SATimer inhomog_timer;
    std::vector<double> offsets(model.num_qubits, 0.0);
    if (params.use_inhomogeneous) {
        std::vector<double> fields = compute_effective_fields(model);
        offsets = compute_anneal_offsets(fields, params.delta_max);

        if (params.verbose) {
            double max_f = *std::max_element(fields.begin(), fields.end());
            std::cerr << "  Effective fields: max=" << max_f
                      << "  offsets range=["
                      << *std::min_element(offsets.begin(), offsets.end())
                      << ", "
                      << *std::max_element(offsets.begin(), offsets.end())
                      << "]\n";
        }
    }
    double inhomog_time = inhomog_timer.elapsed();

    // ------------------------------------------------------------------
    // 3. Generate Python submission script
    // ------------------------------------------------------------------
    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string suffix      = std::to_string(ts);
    std::string script_path = ".dwave_submit_" + suffix + ".py";
    std::string results_path = ".dwave_results_" + suffix + ".json";

    // Resolve input path (pass as-is; Python runs from same directory)
    std::string script = generate_dwave_script(
        params.input_path, results_path, params, offsets);

    {
        std::ofstream ofs(script_path);
        if (!ofs.is_open())
            throw std::runtime_error(
                "Cannot write submission script: " + script_path);
        ofs << script;
    }

    // ------------------------------------------------------------------
    // 4. Execute Python script
    // ------------------------------------------------------------------
    if (params.verbose) {
        std::cerr << "  Submitting to D-Wave: " << params.solver_name
                  << "  reads=" << params.num_reads
                  << "  annealing=" << params.annealing_time_us << " us\n";
    }

    std::string cmd = params.python_cmd + " " + script_path;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::remove(script_path.c_str());
        throw std::runtime_error(
            "D-Wave submission script failed (exit code "
            + std::to_string(ret) + ").\n"
            "Script: " + script_path);
    }

    // ------------------------------------------------------------------
    // 5. Parse D-Wave results
    // ------------------------------------------------------------------
    DWaveResults dwave = parse_dwave_results(results_path);

    // Clean up temp files
    std::remove(script_path.c_str());
    std::remove(results_path.c_str());

    if (dwave.samples.empty())
        throw std::runtime_error("D-Wave returned no samples.");

    // ------------------------------------------------------------------
    // 6. Expand energies and compute statistics
    // ------------------------------------------------------------------
    std::vector<double> all_energies;
    all_energies.reserve(params.num_reads);
    for (const auto& s : dwave.samples)
        for (int k = 0; k < s.num_occurrences; k++)
            all_energies.push_back(s.energy);

    double best_dw = *std::min_element(
        all_energies.begin(), all_energies.end());

    double sum = std::accumulate(
        all_energies.begin(), all_energies.end(), 0.0);
    double mean = sum / all_energies.size();

    double sq_sum = 0;
    for (double e : all_energies)
        sq_sum += (e - mean) * (e - mean);
    double stddev = (all_energies.size() > 1)
        ? std::sqrt(sq_sum / (all_energies.size() - 1))
        : 0.0;

    std::vector<double> sorted_e = all_energies;
    std::sort(sorted_e.begin(), sorted_e.end());
    int n = static_cast<int>(sorted_e.size());
    double median = (n % 2 == 1)
        ? sorted_e[n / 2]
        : 0.5 * (sorted_e[n / 2 - 1] + sorted_e[n / 2]);

    // ------------------------------------------------------------------
    // 7. Decode bitstrings and evaluate under source CFN
    // ------------------------------------------------------------------
    double best_cfn = std::numeric_limits<double>::infinity();
    std::vector<int> best_cfn_sol;
    int num_feasible = 0;
    int num_best_cfn = 0;

    for (const auto& s : dwave.samples) {
        std::vector<int> choices = decode_to_cfn(model, s.values);
        if (choices.empty()) continue;       // infeasible

        double cfn_energy = compute_energy(cfn, choices);
        num_feasible += s.num_occurrences;

        if (cfn_energy < best_cfn - params.tolerance) {
            best_cfn     = cfn_energy;
            best_cfn_sol = choices;
            num_best_cfn = s.num_occurrences;
        } else if (std::abs(cfn_energy - best_cfn) <= params.tolerance) {
            num_best_cfn += s.num_occurrences;
        }
    }

    // ------------------------------------------------------------------
    // 8. Count ground-truth hits (under encoding energy)
    // ------------------------------------------------------------------
    int num_optimal = -1;
    if (!std::isnan(params.ground_truth)) {
        num_optimal = 0;
        for (double e : all_energies) {
            if (std::abs(e - params.ground_truth) <= params.tolerance)
                num_optimal++;
        }
    }

    // ------------------------------------------------------------------
    // 9. Compute embedding statistics
    // ------------------------------------------------------------------

    // --- Chain length: avg, median, variance ---
    double cl_avg = 0, cl_median = 0, cl_var = 0;
    if (!dwave.chain_lengths.empty()) {
        int nc = static_cast<int>(dwave.chain_lengths.size());
        double cl_sum = 0;
        for (int cl : dwave.chain_lengths) cl_sum += cl;
        cl_avg = cl_sum / nc;

        double cl_sq = 0;
        for (int cl : dwave.chain_lengths)
            cl_sq += (cl - cl_avg) * (cl - cl_avg);
        cl_var = (nc > 1) ? cl_sq / (nc - 1) : 0.0;

        std::vector<int> cl_sorted = dwave.chain_lengths;
        std::sort(cl_sorted.begin(), cl_sorted.end());
        cl_median = (nc % 2 == 1)
            ? cl_sorted[nc / 2]
            : 0.5 * (cl_sorted[nc / 2 - 1] + cl_sorted[nc / 2]);
    }

    // --- Chain breaks: avg, median, variance (expanded by num_occurrences) ---
    double cb_avg = 0, cb_median = 0, cb_var = 0;
    if (!dwave.chain_breaks.empty()) {
        // Expand into full list for correct weighted statistics
        std::vector<int> all_breaks;
        all_breaks.reserve(params.num_reads);
        for (const auto& cb : dwave.chain_breaks)
            for (int k = 0; k < cb.num_occurrences; k++)
                all_breaks.push_back(cb.breaks);

        if (!all_breaks.empty()) {
            int nb = static_cast<int>(all_breaks.size());
            double cb_sum = 0;
            for (int b : all_breaks) cb_sum += b;
            cb_avg = cb_sum / nb;

            double cb_sq = 0;
            for (int b : all_breaks)
                cb_sq += (b - cb_avg) * (b - cb_avg);
            cb_var = (nb > 1) ? cb_sq / (nb - 1) : 0.0;

            std::sort(all_breaks.begin(), all_breaks.end());
            cb_median = (nb % 2 == 1)
                ? all_breaks[nb / 2]
                : 0.5 * (all_breaks[nb / 2 - 1] + all_breaks[nb / 2]);
        }
    }

    // ------------------------------------------------------------------
    // 10. Assemble result
    // ------------------------------------------------------------------
    double total_time = total_timer.elapsed();

    QAResult result;
    result.problem_name   = path_basename(params.input_path);
    result.source_cfn     = model.source_cfn;
    result.encoding       = model.encoding;
    result.variable_type  = (model.var_type == BinaryModel::SPIN)
                                ? "SPIN" : "BINARY";
    result.num_qubits     = model.num_qubits;
    result.num_variables  = model.source_num_variables;
    result.max_cardinality = model.source_max_cardinality;
    result.edge_density   = model.meta.valid ? model.meta.rho : 0;
    result.distribution   = model.meta.valid ? model.meta.dist : "NA";

    result.solver_name       = params.solver_name;
    result.annealing_time_us = params.annealing_time_us;
    result.num_reads         = params.num_reads;
    result.delta_max         = params.use_inhomogeneous ? params.delta_max : 0;

    result.best_energy   = best_dw;
    result.mean_energy   = mean;
    result.std_energy    = stddev;
    result.median_energy = median;
    result.num_optimal   = num_optimal;

    result.total_runtime_s          = total_time;
    result.inhomog_setup_time_s     = inhomog_time;
    result.embedding_time_s         = dwave.embedding_time_s;
    result.qpu_access_time_us      = dwave.timing.qpu_access_time_us;
    result.qpu_sampling_time_us    = dwave.timing.qpu_sampling_time_us;
    result.qpu_programming_time_us = dwave.timing.qpu_programming_time_us;

    result.emb_num_physical_qubits = dwave.num_physical_qubits;
    result.emb_chain_length_avg    = cl_avg;
    result.emb_chain_length_median = cl_median;
    result.emb_chain_length_var    = cl_var;
    result.emb_chain_breaks_avg    = cb_avg;
    result.emb_chain_breaks_median = cb_median;
    result.emb_chain_breaks_var    = cb_var;

    result.best_cfn_energy = (num_feasible > 0) ? best_cfn
                             : std::numeric_limits<double>::quiet_NaN();
    result.num_feasible    = num_feasible;
    result.num_best_cfn    = num_best_cfn;
    result.best_solution   = best_cfn_sol;
    result.per_run_energies = std::move(all_energies);

    return result;
}
