#include "baseline/types.hpp"
#include "baseline/cfn.hpp"
#include "utilities/lagrange.hpp"
#include "utilities/assignment.hpp"
#include "encodings/one_hot.hpp"
#include "encodings/domain_wall.hpp"
#include "encodings/exact_binary.hpp"
#include "encodings/approximate_binary.hpp"
#include "encodings/truncated_binary.hpp"
#include "utilities/rosenberg.hpp"
#include "output.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <new>       // std::bad_alloc

namespace fs = std::filesystem;

struct CLIArgs {
    std::string input_dir;
    std::string output_dir;
    std::string csv_path;
    EncodingParams params;
    bool verbose = false;
    bool jsonl = false;
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nRequired:\n"
        << "  --input-dir DIR       Directory containing .cfn files\n"
        << "  --output-dir DIR      Directory for output JSON files\n"
        << "  --jsonl               Write ONE combined <encoding>.jsonl (one model per line)\n"
        << "  --csv FILE            Path for metrics CSV\n"
        << "  --encoding NAME       one_hot|domain_wall|exact_binary|approximate_binary|truncated_binary\n"
        << "\nEncoding options:\n"
        << "  --choice-ordering     unsorted|one_variable|boltzmann  (default: unsorted)\n"
        << "  --bitstring-ordering  natural|gray                     (default: natural)\n"
        << "  --weighted-ls         Enable weighted least squares    (AB only)\n"
        << "  --li-prioritization   Enable LI prioritization         (AB only)\n"
        << "  --nonlinear-refinement Enable nonlinear refinement     (AB only)\n"
        << "  --quadratize          Apply Rosenberg quadratization   (EB/TB only)\n"
        << "  --k-approx N          Max AB interaction degree        (default: 2)\n"
        << "  --k-trunc N           Walsh truncation order            (default: 2)\n"
        << "  --temperature FLOAT   Boltzmann temperature            (default: 1.0)\n"
        << "  --nl-temperature FLOAT  NL refinement temperature      (default: 1.0)\n"
        << "  --nl-tether FLOAT     NL tethering weight              (default: 1.0)\n"
        << "  --nl-max-iter N       NL max iterations                (default: 100)\n"
        << "  --threads N           Number of OpenMP threads         (default: 1)\n"
        << "  --verbose             Print progress\n";
}

CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };

        if (a == "--input-dir")          args.input_dir = next();
        else if (a == "--output-dir")    args.output_dir = next();
        else if (a == "--jsonl")         args.jsonl = true;
        else if (a == "--csv")           args.csv_path = next();
        else if (a == "--encoding")      args.params.encoding = next();
        else if (a == "--choice-ordering")   args.params.choice_ordering = next();
        else if (a == "--bitstring-ordering") args.params.bitstring_ordering = next();
        else if (a == "--weighted-ls")   args.params.weighted_ls = true;
        else if (a == "--li-prioritization") args.params.li_prioritization = true;
        else if (a == "--nonlinear-refinement") args.params.nonlinear_refinement = true;
        else if (a == "--quadratize")    args.params.quadratize = true;
        else if (a == "--k-approx")      args.params.k_approx = std::stoi(next());
        else if (a == "--k-trunc")       args.params.k_trunc = std::stoi(next());
        else if (a == "--temperature")   args.params.temperature = std::stod(next());
        else if (a == "--nl-temperature") args.params.nl_temperature = std::stod(next());
        else if (a == "--nl-tether")     args.params.nl_tether_weight = std::stod(next());
        else if (a == "--nl-max-iter")   args.params.nl_max_iter = std::stoi(next());
        else if (a == "--threads")       args.params.num_threads = std::stoi(next());
        else if (a == "--verbose")       args.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); exit(0); }
        else throw std::runtime_error("Unknown argument: " + a);
    }

    if (args.input_dir.empty() || args.output_dir.empty() ||
        args.csv_path.empty() || args.params.encoding.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("Missing required arguments");
    }
    return args;
}

int main(int argc, char** argv) {
    CLIArgs args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Everything below runs under one guard so a failure in setup (missing input
    // directory, unwritable output, etc.) or an uncaught throw from any single
    // file can never escape to std::terminate (which the cluster reports as a
    // silent exit 134 / SIGABRT that kills the whole batch task).
    int processed = 0;
    int skipped = 0;
    try {

    // Discover .cfn files
    std::vector<std::string> cfn_files;
    for (auto& entry : fs::directory_iterator(args.input_dir)) {
        std::string p = entry.path().string();
        if (p.size() >= 4 && p.substr(p.size() - 4) == ".cfn")
            cfn_files.push_back(p);
    }
    std::sort(cfn_files.begin(), cfn_files.end());

    if (cfn_files.empty()) {
        std::cerr << "No .cfn files found in " << args.input_dir << "\n";
        return 1;
    }

    fs::create_directories(args.output_dir);

    // Open CSV (append if exists, write header if new)
    bool csv_exists = fs::exists(args.csv_path);
    std::ofstream csv(args.csv_path, std::ios::app);
    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV: " << args.csv_path << "\n";
        return 1;
    }
    if (!csv_exists)
        csv << csv_header() << "\n";

    // Optional combined output: one JSONL file total, one model object per line.
    // Avoids creating one file per instance (and the archiving/crawl that needs).
    std::ofstream models;
    if (args.jsonl) {
        std::string models_path = (fs::path(args.output_dir) /
            (args.params.encoding + ".jsonl")).string();
        models.open(models_path);
        if (!models.is_open()) {
            std::cerr << "Cannot open output file: " << models_path << "\n";
            return 1;
        }
    }

    if (args.verbose)
        std::cerr << "Processing " << cfn_files.size() << " CFN files with encoding: "
                  << args.params.encoding << "\n";

    for (auto& cfn_path : cfn_files) {
        try {
            Timer file_timer;

            // Parse CFN
            Timer parse_timer;
            CFN cfn = parse_cfn(cfn_path);
            double t_parse = parse_timer.elapsed();

            EncodingResult result;

            if (args.params.encoding == "one_hot") {
                result = encode_one_hot(cfn, args.params);
            }
            else if (args.params.encoding == "domain_wall") {
                result = encode_domain_wall(cfn, args.params);
            }
            else if (args.params.encoding == "exact_binary" ||
                     args.params.encoding == "approximate_binary" ||
                     args.params.encoding == "truncated_binary") {

                // Compute choice-to-bitstring assignment
                Timer assign_timer;
                Timer order_timer;
                // (choice ordering happens inside compute_assignment)
                BitstringAssignment ba = compute_assignment(cfn, args.params);
                double t_assign = assign_timer.elapsed();

                if (args.params.encoding == "exact_binary") {
                    result = encode_exact_binary(cfn, ba, args.params);
                } else if (args.params.encoding == "approximate_binary") {
                    result = encode_approximate_binary(cfn, ba, args.params);
                } else {
                    result = encode_truncated_binary(cfn, ba, args.params);
                }

                result.time_bitstring_assignment = t_assign;

                // Record the choice->bitstring assignment so the decoder can
                // invert Gray / Boltzmann / LI-prioritized layouts.
                result.choice_to_bitstring = ba.assignment;
            }
            else {
                std::cerr << "Unknown encoding: " << args.params.encoding << "\n";
                return 1;
            }

            result.time_parse = t_parse;

            // Optional quadratization
            if (args.params.quadratize && result.poly.max_degree() > 2) {
                Timer quad_timer;
                auto qr = rosenberg_quadratize(result.poly);
                result.poly = qr.poly;
                result.num_auxiliary_qubits = qr.num_auxiliaries;
                result.rosenberg_penalty = qr.penalty_strength;
                result.time_quadratization = quad_timer.elapsed();
            }

            // Recompute total time including all steps
            result.time_total = file_timer.elapsed();

            // Write output: one combined JSONL line (--jsonl) or a per-instance JSON.
            Timer out_timer;
            std::string basename = fs::path(cfn_path).stem().string();
            if (args.jsonl) {
                models << build_model_json(result, cfn, args.params).dump() << "\n";
                models.flush();
            } else {
                std::string out_path = (fs::path(args.output_dir) /
                    (basename + "_" + args.params.encoding + ".json")).string();
                write_model_json(out_path, result, cfn, args.params);
            }
            result.time_output = out_timer.elapsed();

            // Write CSV row
            csv << csv_row(cfn, args.params, result) << "\n";
            csv.flush();

            processed++;
            if (args.verbose && processed % 100 == 0)
                std::cerr << "  [" << processed << "/" << cfn_files.size() << "] "
                          << basename << " (" << result.time_total << "us)\n";

        } catch (const std::bad_alloc&) {
            // Out of memory on this instance (typically a large exact_binary /
            // truncated_binary HUBO or its Rosenberg quadratization at high
            // cardinality). Skip it instead of letting the throw reach
            // std::terminate. Keep the handler allocation-free and swallow any
            // secondary failure so it cannot escape either.
            try { std::cerr << "Error processing " << cfn_path
                            << ": out of memory (skipped)\n"; } catch (...) {}
            skipped++;
        } catch (const std::exception& e) {
            try { std::cerr << "Error processing " << cfn_path
                            << ": " << e.what() << "\n"; } catch (...) {}
            skipped++;
        } catch (...) {
            try { std::cerr << "Error processing " << cfn_path
                            << ": unknown error (skipped)\n"; } catch (...) {}
            skipped++;
        }
    }

    if (args.verbose)
        std::cerr << "Done: " << processed << "/" << cfn_files.size()
                  << " files processed";
    if (skipped > 0)
        std::cerr << " (" << skipped << " skipped)";
    if (args.verbose || skipped > 0)
        std::cerr << ".\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Fatal: unknown error\n";
        return 1;
    }

    return 0;
}
