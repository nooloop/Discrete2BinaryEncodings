#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <limits>

// ============================================================================
// High-resolution timer
// ============================================================================
class SATimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
public:
    SATimer() : t0(Clock::now()) {}
    void reset() { t0 = Clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

// ============================================================================
// Temperature schedules
// ============================================================================
//   Geometric: T(t) = T_start * (T_end / T_start)^(t / (steps - 1))
//   Linear:    T(t) = T_start + (T_end - T_start) * t / (steps - 1)
// ============================================================================
inline double temperature_geometric(double T_start, double T_end,
                                     int step, int total_steps) {
    if (total_steps <= 1) return T_start;
    double frac = static_cast<double>(step) / (total_steps - 1);
    return T_start * std::pow(T_end / T_start, frac);
}

inline double temperature_linear(double T_start, double T_end,
                                  int step, int total_steps) {
    if (total_steps <= 1) return T_start;
    double frac = static_cast<double>(step) / (total_steps - 1);
    return T_start + (T_end - T_start) * frac;
}

// ============================================================================
// Solver parameters (populated from CLI)
// ============================================================================
struct SAParams {
    std::string mode;                       // "cfn" or "binary"
    std::string input_path;
    std::string cfn_dir;                    // dir with source .cfn files (binary mode:
                                            // enables decode + best_cfn_energy output)
    std::string schedule    = "geometric";  // "geometric" or "linear"
    std::string move_type   = "flip";       // "flip", "shift", "both" (CFN only)

    // Temperatures. These are ENERGIES, and the energy scale is a property of the
    // encoding, not of the problem: on one source CFN the encoded coefficients
    // span |c|_max = 1.06 (truncated_binary) to 72 (Rosenberg-quadratized
    // exact_binary), a factor of ~70. A fixed T window therefore anneals each
    // encoding at a different point in ITS OWN landscape -- above every
    // coefficient for one encoding (a random walk) and below the penalty scale
    // for another (frozen from step 0) -- which confounds exactly the comparison
    // this benchmark makes. D-Wave does not have this problem: it rescales every
    // model onto the QPU's h/J range, so QA is implicitly scale-free.
    //
    // So the schedule is specified in ACCEPTANCE PROBABILITY, which is
    // dimensionless, and T_start/T_end are calibrated per model from its own
    // uphill-|dE| distribution (see solvers/temperature.hpp). The derived values
    // are written to the T_start/T_end CSV columns, so each row records the
    // temperatures actually used.
    //
    // Passing --T-start / --T-end pins that endpoint to an absolute energy and
    // opts it out of calibration; --no-auto-temp restores the old fixed window.
    bool   auto_temp        = true;
    double accept_start     = 0.8;          // P(accept a typical uphill move) at T_start
    double accept_end       = 0.01;         // P(accept a small uphill move) at T_end
    int    temp_probes      = 1000;         // random states sampled to estimate |dE|
    bool   T_start_given    = false;        // --T-start passed explicitly
    bool   T_end_given      = false;        // --T-end passed explicitly

    double T_start          = 10.0;         // fallback / --no-auto-temp value
    double T_end            = 0.01;
    int    num_runs         = 100;
    int    num_steps        = 0;            // If 0, compute from steps_multiplier
    int    steps_multiplier = 100;          // steps = multiplier * N * D
    uint64_t seed           = 42;
    double ground_truth     = std::numeric_limits<double>::quiet_NaN();
    double tolerance        = 1e-6;
    bool   verbose          = false;
};

// ============================================================================
// Per-run result
// ============================================================================
struct RunResult {
    double best_energy;
    double final_energy;
    double runtime_s;
    std::vector<int> best_state;    // best state found during the run (encoding space)

    // Best DECODED state of the run: the lowest source-CFN cost over every state
    // the trajectory visited, not the CFN cost of best_state. The two differ --
    // best_state minimizes the ENCODED energy, which is a different objective:
    // for the cost-approximate encodings (AB, TB) it is a surrogate by
    // construction, and for OH/DW/quadratized models it carries a Lagrange or
    // Rosenberg penalty, so a run's lowest-energy state can even decode
    // infeasibly while the trajectory did visit good feasible states. Tracking
    // the decode along the trajectory is the SA counterpart of solve_qa decoding
    // every read rather than only the lowest-energy sample.
    //
    // NaN / empty when the run never visited a feasible state, or when no source
    // CFN was supplied (--cfn-dir).
    double best_cfn_energy = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> best_cfn_state;   // decoded CFN choices at best_cfn_energy
};

// ============================================================================
// Metadata extracted from CFN filename pattern:
//   CFN_N<int>_D<int>_rho<float>_<dist>_<id>
// ============================================================================
struct CFNMetadata {
    int N = 0, D = 0;
    double rho = 0;
    std::string dist;
    bool valid = false;
};

inline CFNMetadata parse_cfn_metadata(const std::string& name) {
    CFNMetadata m;
    size_t pos_N = name.find("_N");
    if (pos_N == std::string::npos) return m;
    size_t pos_D = name.find("_D", pos_N + 2);
    if (pos_D == std::string::npos) return m;
    size_t pos_rho = name.find("_rho", pos_D + 2);
    if (pos_rho == std::string::npos) return m;

    try {
        m.N = std::stoi(name.substr(pos_N + 2, pos_D - pos_N - 2));
        m.D = std::stoi(name.substr(pos_D + 2, pos_rho - pos_D - 2));

        size_t rho_start = pos_rho + 4;
        size_t next_us = name.find('_', rho_start);
        if (next_us == std::string::npos) return m;
        m.rho = std::stod(name.substr(rho_start, next_us - rho_start));

        size_t dist_start = next_us + 1;
        size_t dist_end = name.find('_', dist_start);
        m.dist = (dist_end == std::string::npos)
            ? name.substr(dist_start)
            : name.substr(dist_start, dist_end - dist_start);

        m.valid = true;
    } catch (...) {}
    return m;
}

// ============================================================================
// Path utilities
// ============================================================================
inline std::string path_basename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

inline std::string path_stem(const std::string& path) {
    std::string base = path_basename(path);
    size_t dot = base.rfind('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// ============================================================================
// Aggregate result (one row per problem)
// ============================================================================
struct AggregateResult {
    std::string problem_name;
    std::string source_cfn;

    std::string solver_mode;
    std::string encoding;
    std::string variable_type;
    int num_qubits       = 0;
    int num_variables    = 0;
    int max_cardinality  = 0;
    double edge_density  = 0;
    std::string distribution;

    std::string schedule;
    std::string move_type;
    double T_start       = 0;
    double T_end         = 0;
    int num_steps        = 0;
    int num_runs         = 0;
    uint64_t seed        = 0;

    double best_energy   = 0;
    double mean_energy   = 0;
    double std_energy    = 0;
    double median_energy = 0;
    int    num_optimal   = -1;
    double total_runtime_s     = 0;   // solving this model: all runs + decode
    double mean_time_per_run_s = 0;   // total_runtime_s / num_runs
    double mean_run_time_us    = 0;   // mean of the measured per-trajectory times

    // Per-run energies under the encoding. For the cost-approximate encodings
    // (approximate_binary, truncated_binary) these are NOT CFN costs.
    std::vector<double> per_run_energies;

    // Per-run BEST DECODED CFN cost -- the lowest source-CFN cost the run's
    // trajectory visited, not the cost of the run's lowest-ENERGY state (see
    // RunResult::best_cfn_energy). Index-aligned with per_run_energies; NaN for a
    // run that never visited a feasible state.
    // Recorded so success counts can be re-derived post-hoc at any tolerance.
    std::vector<double> per_run_cfn_energies;

    // Wall time of each SA trajectory in MICROSECONDS, index-aligned with the
    // vectors above. This is the anneal itself (RNG init, initial energy, and
    // the num_steps Metropolis sweeps) and excludes model parsing, source-CFN
    // loading and decoding -- unlike mean_time_per_run_s, which is the whole
    // process wall clock divided by num_runs. It is the SA counterpart of the
    // QA per-read sampling time, so a time-to-solution built on it is directly
    // comparable across the two solvers.
    std::vector<double> per_run_times_us;

    std::vector<int> best_encoded_solution;   // raw state (qubits) at best_energy

    // --- Decoded-CFN results. best_cfn_energy is the lowest source-CFN cost any
    //     run's trajectory visited; num_best_cfn is how many runs reached it
    //     (so the per-run success probability for TTS is num_best_cfn/num_runs);
    //     num_feasible is how many runs visited at least one feasible state.
    //     Decoding honours natural (naive) and Gray/Boltzmann (enhanced) layouts
    //     through the model's choice_to_bitstring map. best_cfn_energy stays NaN
    //     when no source CFN is available or no run decoded feasibly. ---
    double best_cfn_energy = std::numeric_limits<double>::quiet_NaN();
    int    num_feasible    = -1;   // -1 => not computed (no source CFN)
    int    num_best_cfn    = -1;
    std::vector<int> best_cfn_solution;   // decoded CFN choices at best_cfn_energy
};

inline AggregateResult aggregate_runs(const std::vector<RunResult>& runs,
                                       double total_time) {
    AggregateResult agg;
    int n = static_cast<int>(runs.size());
    if (n == 0) return agg;

    agg.num_runs = n;
    agg.total_runtime_s = total_time;
    agg.mean_time_per_run_s = total_time / n;

    agg.per_run_energies.resize(n);
    agg.per_run_times_us.resize(n);
    for (int i = 0; i < n; i++) {
        agg.per_run_energies[i] = runs[i].best_energy;
        agg.per_run_times_us[i] = runs[i].runtime_s * 1e6;
    }

    agg.mean_run_time_us = std::accumulate(
        agg.per_run_times_us.begin(), agg.per_run_times_us.end(), 0.0) / n;

    agg.best_energy = *std::min_element(
        agg.per_run_energies.begin(), agg.per_run_energies.end());

    // Store best solution
    for (int i = 0; i < n; i++) {
        if (runs[i].best_energy == agg.best_energy) {
            agg.best_encoded_solution = runs[i].best_state;
            break;
        }
    }

    double sum = std::accumulate(
        agg.per_run_energies.begin(), agg.per_run_energies.end(), 0.0);
    agg.mean_energy = sum / n;

    double sq_sum = 0;
    for (double e : agg.per_run_energies)
        sq_sum += (e - agg.mean_energy) * (e - agg.mean_energy);
    agg.std_energy = (n > 1) ? std::sqrt(sq_sum / (n - 1)) : 0.0;

    std::vector<double> sorted = agg.per_run_energies;
    std::sort(sorted.begin(), sorted.end());
    if (n % 2 == 1)
        agg.median_energy = sorted[n / 2];
    else
        agg.median_energy = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

    return agg;
}
