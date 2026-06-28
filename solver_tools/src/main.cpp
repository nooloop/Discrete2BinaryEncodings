#include "baseline/sa_types.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/sa_binary.hpp"
#include "solvers/sa_cfn.hpp"
#include "baseline/output.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// CLI usage
// ============================================================================
void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nRequired:\n"
        << "  --mode MODE             cfn | binary\n"
        << "  --input FILE            Path to .cfn or encoded .json file\n"
        << "\nDecoding (binary mode):\n"
        << "  --cfn-dir DIR           Dir with source .cfn files; enables decoding the\n"
        << "                          best state to CFN choices and recording\n"
        << "                          best_cfn_energy / num_feasible / num_best_cfn\n"
        << "\nAnnealing schedule:\n"
        << "  --schedule TYPE         geometric | linear           (default: geometric)\n"
        << "  --T-start FLOAT         Starting temperature         (default: 10.0)\n"
        << "  --T-end FLOAT           Final temperature            (default: 0.01)\n"
        << "\nMove type (CFN mode only):\n"
        << "  --move-type TYPE        flip | shift | both          (default: flip)\n"
        << "\nRun configuration:\n"
        << "  --num-runs N            Number of independent runs   (default: 100)\n"
        << "  --num-steps N           Total SA steps per run       (overrides multiplier)\n"
        << "  --steps-multiplier M    steps = M * N * D            (default: 100)\n"
        << "  --seed N                Base RNG seed                (default: 42)\n"
        << "\nOptional:\n"
        << "  --ground-truth E        Known optimum for success counting\n"
        << "  --tolerance FLOAT       Tolerance for ground truth   (default: 1e-6)\n"
        << "  --header                Print CSV header and exit\n"
        << "  --verbose               Print progress to stderr\n";
}

// ============================================================================
// Argument parser
// ============================================================================
SAParams parse_args(int argc, char** argv, bool& header_only) {
    SAParams params;
    header_only = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };

        if      (a == "--mode")             params.mode = next();
        else if (a == "--input")            params.input_path = next();
        else if (a == "--cfn-dir")          params.cfn_dir = next();
        else if (a == "--schedule")         params.schedule = next();
        else if (a == "--T-start")          params.T_start = std::stod(next());
        else if (a == "--T-end")            params.T_end = std::stod(next());
        else if (a == "--move-type")        params.move_type = next();
        else if (a == "--num-runs")         params.num_runs = std::stoi(next());
        else if (a == "--num-steps")        params.num_steps = std::stoi(next());
        else if (a == "--steps-multiplier") params.steps_multiplier = std::stoi(next());
        else if (a == "--seed")             params.seed = std::stoull(next());
        else if (a == "--ground-truth")     params.ground_truth = std::stod(next());
        else if (a == "--tolerance")        params.tolerance = std::stod(next());
        else if (a == "--header")           { header_only = true; }
        else if (a == "--verbose")          params.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); exit(0); }
        else throw std::runtime_error("Unknown argument: " + a);
    }

    if (!header_only) {
        if (params.mode.empty() || params.input_path.empty()) {
            print_usage(argv[0]);
            throw std::runtime_error("--mode and --input are required");
        }
        if (params.mode != "cfn" && params.mode != "binary") {
            throw std::runtime_error("--mode must be 'cfn' or 'binary'");
        }
    }
    return params;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool header_only = false;
    SAParams params;
    try {
        params = parse_args(argc, argv, header_only);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (header_only) {
        std::cout << csv_header() << "\n";
        return 0;
    }

    try {
        SATimer total_timer;
        std::vector<RunResult> runs;
        AggregateResult agg;

        // Decoded-CFN aggregates (filled by both modes below).
        double cfn_best_energy = std::numeric_limits<double>::quiet_NaN();
        int    cfn_num_feasible = -1;   // -1 => not computed
        int    cfn_num_best     = -1;
        std::vector<int> cfn_best_solution;

        std::string filename = path_basename(params.input_path);

        if (params.mode == "binary") {
            BinaryModel model = parse_binary_model(params.input_path);

            if (params.num_steps == 0) {
                if (model.source_num_variables > 0 && model.source_max_cardinality > 0)
                    params.num_steps = params.steps_multiplier
                                     * model.source_num_variables
                                     * model.source_max_cardinality;
                else
                    params.num_steps = params.steps_multiplier * model.num_qubits;
            }

            if (params.verbose) {
                std::cerr << "Binary SA: " << filename
                          << "  qubits=" << model.num_qubits
                          << "  encoding=" << model.encoding
                          << "  type=" << (model.var_type == BinaryModel::SPIN
                                           ? "SPIN" : "BINARY")
                          << "  steps=" << params.num_steps
                          << "  runs=" << params.num_runs << "\n";
            }

            runs.reserve(params.num_runs);
            for (int r = 0; r < params.num_runs; r++) {
                runs.push_back(run_sa_binary(model, params, params.seed + r));
                if (params.verbose && (r + 1) % 10 == 0)
                    std::cerr << "  run " << (r + 1) << "/" << params.num_runs
                              << "  best=" << runs.back().best_energy << "\n";
            }

            agg.problem_name    = filename;
            agg.source_cfn      = model.source_cfn;
            agg.solver_mode     = "binary";
            agg.encoding        = model.encoding;
            agg.variable_type   = (model.var_type == BinaryModel::SPIN)
                                    ? "SPIN" : "BINARY";
            agg.num_qubits      = model.num_qubits;
            agg.num_variables   = model.source_num_variables;
            agg.max_cardinality = model.source_max_cardinality;
            agg.edge_density    = model.meta.valid ? model.meta.rho : 0;
            agg.distribution    = model.meta.valid ? model.meta.dist : "NA";

            // Decode each run's best state to CFN choices and evaluate the
            // source CFN. decode_to_cfn inverts both natural (naive) and
            // Gray/Boltzmann (enhanced) layouts via the model's
            // choice_to_bitstring map, so best_cfn_energy is the true CFN cost
            // of the best decoded solution regardless of bitstring ordering.
            if (!params.cfn_dir.empty()) {
                if (model.source_cfn.empty()) {
                    std::cerr << "Warning: --cfn-dir set but model has no "
                                 "source_cfn; skipping CFN decode.\n";
                } else {
                    std::string cfn_file = model.source_cfn;
                    if (cfn_file.size() < 4 ||
                        cfn_file.substr(cfn_file.size() - 4) != ".cfn")
                        cfn_file += ".cfn";
                    std::string cfn_path = params.cfn_dir + "/" + cfn_file;
                    try {
                        CFNModel src = parse_cfn_for_sa(cfn_path);
                        cfn_num_feasible = 0;
                        cfn_num_best     = 0;
                        double best = std::numeric_limits<double>::infinity();
                        for (const auto& rr : runs) {
                            std::vector<int> ch = decode_to_cfn(model, rr.best_state);
                            if (ch.empty()) continue;   // infeasible decode
                            cfn_num_feasible++;
                            double ce = compute_energy(src, ch);
                            if (ce < best - params.tolerance) {
                                best = ce;
                                cfn_best_solution = ch;
                                cfn_num_best = 1;
                            } else if (std::abs(ce - best) <= params.tolerance) {
                                cfn_num_best++;
                            }
                        }
                        if (cfn_num_feasible > 0) cfn_best_energy = best;
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: could not load source CFN '"
                                  << cfn_path << "': " << e.what()
                                  << " (skipping CFN decode).\n";
                    }
                }
            }
        }
        else {
            CFNModel model = parse_cfn_for_sa(params.input_path);

            int max_card = *std::max_element(
                model.cardinalities.begin(), model.cardinalities.end());

            if (params.num_steps == 0)
                params.num_steps = params.steps_multiplier
                                 * model.num_variables * max_card;

            if (params.verbose) {
                std::cerr << "CFN SA: " << filename
                          << "  vars=" << model.num_variables
                          << "  max_d=" << max_card
                          << "  edges=" << model.pairwise.size()
                          << "  move=" << params.move_type
                          << "  steps=" << params.num_steps
                          << "  runs=" << params.num_runs << "\n";
            }

            runs.reserve(params.num_runs);
            for (int r = 0; r < params.num_runs; r++) {
                runs.push_back(run_sa_cfn(model, params, params.seed + r));
                if (params.verbose && (r + 1) % 10 == 0)
                    std::cerr << "  run " << (r + 1) << "/" << params.num_runs
                              << "  best=" << runs.back().best_energy << "\n";
            }

            int max_edges = model.num_variables * (model.num_variables - 1) / 2;
            double density = (max_edges > 0)
                ? static_cast<double>(model.pairwise.size()) / max_edges : 0;

            agg.problem_name    = filename;
            agg.source_cfn      = model.name;
            agg.solver_mode     = "cfn";
            agg.encoding        = "native";
            agg.variable_type   = "INTEGER";
            agg.num_qubits      = 0;
            agg.num_variables   = model.num_variables;
            agg.max_cardinality = max_card;
            agg.edge_density    = density;
            agg.distribution    = model.meta.valid ? model.meta.dist : "NA";

            // CFN mode: states already are choices and best_energy already is
            // the CFN cost, so best_cfn_energy mirrors the native result.
            cfn_num_feasible = 0;
            cfn_num_best     = 0;
            double best = std::numeric_limits<double>::infinity();
            for (const auto& rr : runs) {
                cfn_num_feasible++;
                double ce = rr.best_energy;
                if (ce < best - params.tolerance) {
                    best = ce;
                    cfn_best_solution = rr.best_state;
                    cfn_num_best = 1;
                } else if (std::abs(ce - best) <= params.tolerance) {
                    cfn_num_best++;
                }
            }
            cfn_best_energy = best;
        }

        double total_time = total_timer.elapsed();

        AggregateResult stats = aggregate_runs(runs, total_time);

        agg.schedule              = params.schedule;
        agg.move_type             = params.move_type;
        agg.T_start               = params.T_start;
        agg.T_end                 = params.T_end;
        agg.num_steps             = params.num_steps;
        agg.num_runs              = stats.num_runs;
        agg.seed                  = params.seed;
        agg.best_energy           = stats.best_energy;
        agg.mean_energy           = stats.mean_energy;
        agg.std_energy            = stats.std_energy;
        agg.median_energy         = stats.median_energy;
        agg.total_runtime_s       = stats.total_runtime_s;
        agg.mean_time_per_run_s   = stats.mean_time_per_run_s;
        agg.per_run_energies      = stats.per_run_energies;
        agg.best_solution         = stats.best_solution;

        agg.best_cfn_energy       = cfn_best_energy;
        agg.num_feasible          = cfn_num_feasible;
        agg.num_best_cfn          = cfn_num_best;
        agg.best_cfn_solution     = cfn_best_solution;

        if (!std::isnan(params.ground_truth)) {
            agg.num_optimal = 0;
            for (double e : agg.per_run_energies) {
                if (std::abs(e - params.ground_truth) <= params.tolerance)
                    agg.num_optimal++;
            }
        }

        std::cout << csv_row(agg) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
