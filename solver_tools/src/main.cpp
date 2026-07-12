#include "baseline/sa_types.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/sa_binary.hpp"
#include "solvers/temperature.hpp"
#include "solvers/sa_cfn.hpp"
#include "baseline/output.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

// ============================================================================
// CLI usage
// ============================================================================
void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nRequired:\n"
        << "  --mode MODE             cfn | binary\n"
        << "  --input FILE            Path to .cfn, encoded .json, or .jsonl file\n"
        << "\nJSONL (binary mode; encode_cfn --jsonl output):\n"
        << "  A .jsonl input holds one encoded model per line and emits one CSV\n"
        << "  row per model, so an arm needs no per-problem .json staging.\n"
        << "  --jsonl                 Force JSONL (else inferred from .jsonl suffix)\n"
        << "  --num-shards K          Split the file across K workers  (default: 1)\n"
        << "  --shard I               This worker's index, 0 <= I < K   (default: 0)\n"
        << "  --line N                Solve only line N (1-based); for retrying one\n"
        << "  --done-list FILE        Skip models whose problem_name is listed in\n"
        << "                          FILE (one per line) -- resume support\n"
        << "\nDecoding (binary mode):\n"
        << "  --cfn-dir DIR           Dir with source .cfn files; enables decoding the\n"
        << "                          trajectory to CFN choices and recording\n"
        << "                          best_cfn_energy / num_feasible / num_best_cfn\n"
        << "\nAnnealing schedule:\n"
        << "  --schedule TYPE         geometric | linear           (default: geometric)\n"
        << "  Temperatures are calibrated PER MODEL from its own uphill-dE\n"
        << "  distribution, so the schedule means the same thing on every encoding\n"
        << "  (a fixed T window does not: the encoded energy scale varies ~70x\n"
        << "  across encodings of one CFN). The derived values go in the CSV.\n"
        << "  --accept-start P        P(accept typical uphill) at T_start (default: 0.8)\n"
        << "  --accept-end P          P(accept small uphill) at T_end     (default: 0.01)\n"
        << "  --temp-probes N         States sampled to estimate dE       (default: 1000)\n"
        << "  --T-start FLOAT         Pin T_start to an absolute energy (skips calibration)\n"
        << "  --T-end FLOAT           Pin T_end to an absolute energy   (skips calibration)\n"
        << "  --no-auto-temp          Disable calibration; use the fixed 10.0 -> 0.01 window\n"
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
struct JsonlOpts {
    bool        jsonl      = false;
    int         num_shards = 1;
    int         shard      = 0;
    int         line       = 0;    // 1-based; 0 => all
    std::string done_list;
};

SAParams parse_args(int argc, char** argv, bool& header_only, JsonlOpts& jl) {
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
        else if (a == "--T-start")          { params.T_start = std::stod(next());
                                              params.T_start_given = true; }
        else if (a == "--T-end")            { params.T_end = std::stod(next());
                                              params.T_end_given = true; }
        else if (a == "--no-auto-temp")     params.auto_temp = false;
        else if (a == "--accept-start")     params.accept_start = std::stod(next());
        else if (a == "--accept-end")       params.accept_end = std::stod(next());
        else if (a == "--temp-probes")      params.temp_probes = std::stoi(next());
        else if (a == "--move-type")        params.move_type = next();
        else if (a == "--num-runs")         params.num_runs = std::stoi(next());
        else if (a == "--num-steps")        params.num_steps = std::stoi(next());
        else if (a == "--steps-multiplier") params.steps_multiplier = std::stoi(next());
        else if (a == "--seed")             params.seed = std::stoull(next());
        else if (a == "--ground-truth")     params.ground_truth = std::stod(next());
        else if (a == "--tolerance")        params.tolerance = std::stod(next());
        else if (a == "--jsonl")            jl.jsonl = true;
        else if (a == "--num-shards")       jl.num_shards = std::stoi(next());
        else if (a == "--shard")            jl.shard = std::stoi(next());
        else if (a == "--line")             jl.line = std::stoi(next());
        else if (a == "--done-list")        jl.done_list = next();
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
        if (params.mode != "cfn" && params.mode != "binary")
            throw std::runtime_error("--mode must be 'cfn' or 'binary'");

        // Acceptance probabilities: exp(-dE/T) is in (0,1), and the schedule must
        // cool, so accept_end < accept_start.
        auto check_prob = [](const char* flag, double p) {
            if (!(p > 0.0 && p < 1.0))
                throw std::runtime_error(std::string(flag) +
                                         " must lie strictly between 0 and 1");
        };
        check_prob("--accept-start", params.accept_start);
        check_prob("--accept-end",   params.accept_end);
        if (params.accept_end >= params.accept_start)
            throw std::runtime_error("--accept-end must be < --accept-start "
                                     "(the schedule must cool)");
        if (params.temp_probes < 1)
            throw std::runtime_error("--temp-probes must be >= 1");

        // Infer JSONL from the suffix, so the encoder's output can be handed
        // over unchanged.
        const std::string& p = params.input_path;
        if (p.size() >= 6 && p.substr(p.size() - 6) == ".jsonl")
            jl.jsonl = true;

        if (jl.jsonl && params.mode != "binary")
            throw std::runtime_error("--jsonl applies to --mode binary only");
        if (jl.num_shards < 1)
            throw std::runtime_error("--num-shards must be >= 1");
        if (jl.shard < 0 || jl.shard >= jl.num_shards)
            throw std::runtime_error("--shard must satisfy 0 <= shard < num-shards");
    }
    return params;
}

// ============================================================================
// CFN-space aggregates.
//
// Both modes now report the best DECODED cost of each run (run_sa_binary decodes
// along the trajectory; in CFN mode the state is already a choice vector), so
// the two share one aggregation:
//
//   best_cfn_energy  min over runs of the run's best decoded CFN cost
//   num_best_cfn     how many runs reached that cost (within tolerance)
//                    -> the per-run success probability for TTS is
//                       num_best_cfn / num_runs
//   num_feasible     runs that visited at least one feasible state
//
// `tracked` is false in binary mode without --cfn-dir: nothing was decoded, so
// the columns are NA rather than 0.
// ============================================================================
void fill_cfn_aggregates(AggregateResult& agg,
                         const std::vector<RunResult>& runs,
                         double tolerance,
                         bool tracked) {
    if (!tracked) {
        agg.best_cfn_energy = std::numeric_limits<double>::quiet_NaN();
        agg.num_feasible    = -1;   // -> NA
        agg.num_best_cfn    = -1;
        return;
    }

    agg.per_run_cfn_energies.reserve(runs.size());

    double best = std::numeric_limits<double>::infinity();
    int    best_run = -1;
    int    num_feasible = 0;

    for (int r = 0; r < static_cast<int>(runs.size()); r++) {
        double e = runs[r].best_cfn_energy;
        agg.per_run_cfn_energies.push_back(e);   // NaN if the run saw no feasible state
        if (std::isnan(e)) continue;
        num_feasible++;
        if (e < best) { best = e; best_run = r; }
    }

    agg.num_feasible = num_feasible;

    if (best_run < 0) {   // no run ever decoded feasibly
        agg.best_cfn_energy = std::numeric_limits<double>::quiet_NaN();
        agg.num_best_cfn    = 0;
        return;
    }

    int num_best = 0;
    for (double e : agg.per_run_cfn_energies)
        if (!std::isnan(e) && std::abs(e - best) <= tolerance) num_best++;

    agg.best_cfn_energy   = best;
    agg.num_best_cfn      = num_best;
    agg.best_cfn_solution = runs[best_run].best_cfn_state;
}

// ============================================================================
// Solve one model (already parsed) and build its CSV row.
//
// `params` is taken BY VALUE: num_steps is derived per model, and a JSONL shard
// solves many models with one SAParams.
// ============================================================================
AggregateResult solve_binary(const BinaryModel& model,
                             const CFNModel* src_cfn,
                             SAParams params,
                             const std::string& problem_name) {
    SATimer total_timer;

    if (params.num_steps == 0) {
        if (model.source_num_variables > 0 && model.source_max_cardinality > 0)
            params.num_steps = params.steps_multiplier
                             * model.source_num_variables
                             * model.source_max_cardinality;
        else
            params.num_steps = params.steps_multiplier * model.num_qubits;
    }

    apply_temperature_calibration(params, model, src_cfn);

    if (params.verbose) {
        std::cerr << "Binary SA: " << problem_name
                  << "  qubits=" << model.num_qubits
                  << "  encoding=" << model.encoding
                  << "  type=" << (model.var_type == BinaryModel::SPIN
                                   ? "SPIN" : "BINARY")
                  << "  steps=" << params.num_steps
                  << "  runs=" << params.num_runs
                  << "  T=" << params.T_start << "->" << params.T_end << "\n";
    }

    std::vector<RunResult> runs;
    runs.reserve(params.num_runs);
    for (int r = 0; r < params.num_runs; r++)
        runs.push_back(run_sa_binary(model, params, params.seed + r, src_cfn));

    AggregateResult agg = aggregate_runs(runs, total_timer.elapsed());

    agg.problem_name    = problem_name;
    agg.source_cfn      = model.source_cfn;
    agg.solver_mode     = "binary";
    agg.encoding        = model.encoding;
    agg.variable_type   = (model.var_type == BinaryModel::SPIN) ? "SPIN" : "BINARY";
    agg.num_steps       = params.num_steps;   // derived per model, above
    agg.T_start         = params.T_start;     // calibrated per model, above
    agg.T_end           = params.T_end;
    agg.num_qubits      = model.num_qubits;
    agg.num_variables   = model.source_num_variables;
    agg.max_cardinality = model.source_max_cardinality;
    agg.edge_density    = model.meta.valid ? model.meta.rho : 0;
    agg.distribution    = model.meta.valid ? model.meta.dist : "NA";

    fill_cfn_aggregates(agg, runs, params.tolerance,
                        /*tracked=*/src_cfn != nullptr);
    return agg;
}

AggregateResult solve_cfn(const CFNModel& model,
                          SAParams params,
                          const std::string& problem_name) {
    SATimer total_timer;

    int max_card = *std::max_element(model.cardinalities.begin(),
                                     model.cardinalities.end());

    if (params.num_steps == 0)
        params.num_steps = params.steps_multiplier * model.num_variables * max_card;

    apply_temperature_calibration(params, model, &model);

    if (params.verbose) {
        std::cerr << "CFN SA: " << problem_name
                  << "  vars=" << model.num_variables
                  << "  max_d=" << max_card
                  << "  edges=" << model.pairwise.size()
                  << "  move=" << params.move_type
                  << "  steps=" << params.num_steps
                  << "  runs=" << params.num_runs
                  << "  T=" << params.T_start << "->" << params.T_end << "\n";
    }

    std::vector<RunResult> runs;
    runs.reserve(params.num_runs);
    for (int r = 0; r < params.num_runs; r++)
        runs.push_back(run_sa_cfn(model, params, params.seed + r));

    AggregateResult agg = aggregate_runs(runs, total_timer.elapsed());

    int max_edges = model.num_variables * (model.num_variables - 1) / 2;
    double density = (max_edges > 0)
        ? static_cast<double>(model.pairwise.size()) / max_edges : 0;

    agg.problem_name    = problem_name;
    agg.source_cfn      = model.name;
    agg.solver_mode     = "cfn";
    agg.encoding        = "native";
    agg.variable_type   = "INTEGER";
    agg.num_steps       = params.num_steps;   // derived per model, above
    agg.T_start         = params.T_start;     // calibrated per model, above
    agg.T_end           = params.T_end;
    agg.num_qubits      = 0;
    agg.num_variables   = model.num_variables;
    agg.max_cardinality = max_card;
    agg.edge_density    = density;
    agg.distribution    = model.meta.valid ? model.meta.dist : "NA";

    fill_cfn_aggregates(agg, runs, params.tolerance, /*tracked=*/true);
    return agg;
}

// Copy the solver-configuration and ground-truth columns onto a finished row.
// NB: T_start / T_end are NOT set here -- they are calibrated per model inside
// solve_binary / solve_cfn, and the row must record the values actually used.
void finalize_row(AggregateResult& agg, const SAParams& params) {
    agg.schedule  = params.schedule;
    agg.move_type = params.move_type;
    agg.seed      = params.seed;

    if (!std::isnan(params.ground_truth)) {
        agg.num_optimal = 0;
        for (double e : agg.per_run_energies)
            if (std::abs(e - params.ground_truth) <= params.tolerance)
                agg.num_optimal++;
    }
}

// ============================================================================
// Source-CFN loading (binary mode). The .cfn name comes from the model itself.
// ============================================================================
std::unique_ptr<CFNModel> load_source_cfn(const BinaryModel& model,
                                          const SAParams& params) {
    if (params.cfn_dir.empty()) return nullptr;

    if (model.source_cfn.empty()) {
        std::cerr << "Warning: --cfn-dir set but model has no source_cfn; "
                     "skipping CFN decode.\n";
        return nullptr;
    }

    std::string cfn_file = model.source_cfn;
    if (cfn_file.size() < 4 || cfn_file.substr(cfn_file.size() - 4) != ".cfn")
        cfn_file += ".cfn";
    std::string cfn_path = params.cfn_dir + "/" + cfn_file;

    try {
        return std::make_unique<CFNModel>(parse_cfn_for_sa(cfn_path));
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not load source CFN '" << cfn_path
                  << "': " << e.what() << " (skipping CFN decode).\n";
        return nullptr;
    }
}

std::set<std::string> read_done_list(const std::string& path) {
    std::set<std::string> done;
    if (path.empty()) return done;

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Warning: cannot open --done-list '" << path
                  << "'; treating every model as pending.\n";
        return done;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (!line.empty()) done.insert(line);
    }
    return done;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool header_only = false;
    SAParams params;
    JsonlOpts jl;
    try {
        params = parse_args(argc, argv, header_only, jl);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (header_only) {
        std::cout << csv_header() << "\n";
        return 0;
    }

    try {
        // ---- JSONL: stream the file, one CSV row per model ----
        if (jl.jsonl) {
            std::set<std::string> done = read_done_list(jl.done_list);

            JsonlModelReader reader(params.input_path, jl.shard, jl.num_shards,
                                    jl.line);
            BinaryModel model;
            std::string error;
            int index = 0;
            int solved = 0, skipped = 0, failed = 0;

            while (reader.next(model, index, error)) {
                if (!error.empty()) {
                    std::cerr << "Error: " << params.input_path << ": " << error
                              << " (skipping model)\n";
                    failed++;
                    continue;
                }

                // The .jsonl carries no filenames, so the model names itself.
                std::string problem_name =
                    model.source_cfn.empty() ? ("line_" + std::to_string(index + 1))
                                             : model.source_cfn;

                if (done.count(problem_name)) { skipped++; continue; }

                std::unique_ptr<CFNModel> src = load_source_cfn(model, params);
                AggregateResult agg = solve_binary(model, src.get(), params,
                                                   problem_name);
                finalize_row(agg, params);

                // Flush per row: a shard killed by a wall-clock limit still
                // leaves every completed row on disk, and --done-list resumes
                // from exactly there.
                std::cout << csv_row(agg) << std::endl;
                solved++;

                if (reader.exhausted()) break;   // --line: stop, do not scan on
            }

            if (params.verbose || failed > 0) {
                std::cerr << "JSONL " << params.input_path
                          << " [shard " << jl.shard << "/" << jl.num_shards << "]: "
                          << solved << " solved, " << skipped << " already done, "
                          << failed << " failed\n";
            }
            return failed > 0 ? 1 : 0;
        }

        // ---- Single model per file ----
        std::string problem_name = path_basename(params.input_path);
        AggregateResult agg;

        if (params.mode == "binary") {
            BinaryModel model = parse_binary_model(params.input_path);
            std::unique_ptr<CFNModel> src = load_source_cfn(model, params);
            agg = solve_binary(model, src.get(), params, problem_name);
        } else {
            CFNModel model = parse_cfn_for_sa(params.input_path);
            agg = solve_cfn(model, params, problem_name);
        }

        finalize_row(agg, params);
        std::cout << csv_row(agg) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
