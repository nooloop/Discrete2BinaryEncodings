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
    std::string schedule    = "geometric";  // "geometric" or "linear"
    std::string move_type   = "flip";       // "flip", "shift", "both" (CFN only)
    double T_start          = 10.0;
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
    std::vector<int> best_state;    // best state found during the run
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
    double total_runtime_s     = 0;
    double mean_time_per_run_s = 0;

    std::vector<double> per_run_energies;
    std::vector<int> best_solution;   // best state across all runs
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
    for (int i = 0; i < n; i++)
        agg.per_run_energies[i] = runs[i].best_energy;

    agg.best_energy = *std::min_element(
        agg.per_run_energies.begin(), agg.per_run_energies.end());

    // Store best solution
    for (int i = 0; i < n; i++) {
        if (runs[i].best_energy == agg.best_energy) {
            agg.best_solution = runs[i].best_state;
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
