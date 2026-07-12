#include "baseline/qa_types.hpp"
#include "baseline/output_qa.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/qa_binary.hpp"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>

// ============================================================================
// CLI usage
// ============================================================================
void print_usage_qa(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nRequired:\n"
        << "  --input FILE            Path to encoded .json model\n"
        << "  --cfn-dir DIR           Directory containing source .cfn files\n"
        << "  --solver NAME           D-Wave solver name\n"
        << "\nD-Wave parameters:\n"
        << "  --annealing-time FLOAT  Annealing time in microseconds     (default: 20)\n"
        << "  --num-reads N           Number of reads (shots)            (default: 1000)\n"
        << "\nInhomogeneous driving:\n"
        << "  --delta-max FLOAT       Max anneal offset magnitude        (default: 0.1)\n"
        << "  --no-inhomogeneous      Disable inhomogeneous driving\n"
        << "\nOptional:\n"
        << "  --ground-truth E        Known optimum for success counting\n"
        << "  --tolerance FLOAT       Tolerance for ground truth         (default: 1e-6)\n"
        << "  --python CMD            Python interpreter                 (default: python3)\n"
        << "  --header                Print CSV header and exit\n"
        << "  --verbose               Print progress to stderr\n";
}

// ============================================================================
// Argument parser
// ============================================================================
QAParams parse_qa_args(int argc, char** argv, bool& header_only) {
    QAParams params;
    header_only = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };

        if      (a == "--input")             params.input_path = next();
        else if (a == "--cfn-dir")           params.cfn_dir = next();
        else if (a == "--solver")            params.solver_name = next();
        else if (a == "--annealing-time")    params.annealing_time_us = std::stod(next());
        else if (a == "--num-reads")         params.num_reads = std::stoi(next());
        else if (a == "--delta-max")         params.delta_max = std::stod(next());
        else if (a == "--no-inhomogeneous")  params.use_inhomogeneous = false;
        else if (a == "--ground-truth")      params.ground_truth = std::stod(next());
        else if (a == "--tolerance")         params.tolerance = std::stod(next());
        else if (a == "--python")            params.python_cmd = next();
        else if (a == "--header")            { header_only = true; }
        else if (a == "--verbose")           params.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage_qa(argv[0]); exit(0); }
        else throw std::runtime_error("Unknown argument: " + a);
    }

    if (!header_only) {
        if (params.input_path.empty())
            throw std::runtime_error("--input is required");
        if (params.cfn_dir.empty())
            throw std::runtime_error("--cfn-dir is required");
        if (params.solver_name.empty())
            throw std::runtime_error("--solver is required");
    }
    return params;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool header_only = false;
    QAParams params;
    try {
        params = parse_qa_args(argc, argv, header_only);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (header_only) {
        std::cout << csv_header_qa() << "\n";
        return 0;
    }

    try {
        // --- Load encoded model ---
        BinaryModel model = parse_binary_model(params.input_path);

        std::string filename = path_basename(params.input_path);

        if (params.verbose) {
            std::cerr << "QA: " << filename
                      << "  qubits=" << model.num_qubits
                      << "  encoding=" << model.encoding
                      << "  type=" << (model.var_type == BinaryModel::SPIN
                                       ? "SPIN" : "BINARY")
                      << "\n";
        }

        // --- Find and load source CFN ---
        std::string cfn_name = model.source_cfn;
        if (cfn_name.empty())
            throw std::runtime_error(
                "Model has no source_cfn field; cannot locate original CFN.");

        // Append .cfn if not already present
        std::string cfn_filename = cfn_name;
        if (cfn_filename.size() < 4 ||
            cfn_filename.substr(cfn_filename.size() - 4) != ".cfn")
            cfn_filename += ".cfn";

        std::string cfn_path = params.cfn_dir + "/" + cfn_filename;

        if (params.verbose)
            std::cerr << "  Source CFN: " << cfn_path << "\n";

        CFNModel cfn = parse_cfn_for_sa(cfn_path);

        // Sanity check
        if (model.source_num_variables > 0 &&
            model.source_num_variables != cfn.num_variables) {
            std::cerr << "Warning: model expects "
                      << model.source_num_variables
                      << " CFN variables but source CFN has "
                      << cfn.num_variables << "\n";
        }

        // --- Run QA ---
        QAResult result = run_qa_binary(model, cfn, params);

        // --- Output CSV row ---
        std::cout << csv_row_qa(result) << "\n";

        if (params.verbose) {
            std::cerr << "  Best D-Wave energy:  " << result.best_energy << "\n";
            std::cerr << "  Best CFN energy:     ";
            if (std::isnan(result.best_cfn_energy))
                std::cerr << "NA (no feasible solutions)\n";
            else
                std::cerr << result.best_cfn_energy << "\n";
            std::cerr << "  Feasible samples:    " << result.num_feasible
                      << " / " << result.num_reads << "\n";
            std::cerr << "  Best CFN solution:   ";
            if (result.best_cfn_solution.empty())
                std::cerr << "NA\n";
            else {
                std::cerr << "[";
                for (int k = 0; k < static_cast<int>(result.best_cfn_solution.size()); k++) {
                    if (k > 0) std::cerr << ",";
                    std::cerr << result.best_cfn_solution[k];
                }
                std::cerr << "]\n";
            }
            std::cerr << "  Total time:          "
                      << result.total_runtime_s << " s\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
