// ============================================================================
// Unit and integration tests for the D-Wave QA pipeline.
//
// Unit tests (no D-Wave access needed):
//   ./tests_qa
//
// Integration test (needs D-Wave access):
//   ./tests_qa --dwave --solver <SOLVER_NAME> [--test-dir <DIR>] [--python <CMD>]
//
// To configure D-Wave access, set your API token:
//   export DWAVE_API_TOKEN=<your-token-here>
// Or configure via:
//   dwave config create
//
// Test CFN (same 2x4 problem used by SA tests):
//   unary_x0 = [3, 1, 4, 2],  unary_x1 = [2.5, 0.5, 1.5, 3.5]
//   Ground state: E(3,1) = 1.0
// ============================================================================

#include "baseline/qa_types.hpp"
#include "baseline/output_qa.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/qa_binary.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstdio>

// ============================================================================
// Simple test framework (same macros as tests.cpp)
// ============================================================================
static int g_checks = 0, g_failures = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        std::cerr << "  FAIL [" << __LINE__ << "]: " << #cond << "\n"; \
    } \
} while(0)

#define APPROX_EQ(a, b, tol) do { \
    g_checks++; \
    if (std::abs((a) - (b)) > (tol)) { \
        g_failures++; \
        std::cerr << "  FAIL [" << __LINE__ << "]: " \
                  << #a << " = " << (a) << " != " << #b << " = " << (b) \
                  << " (tol=" << (tol) << ")\n"; \
    } \
} while(0)

// ============================================================================
// Helpers
// ============================================================================

static std::string fmt(const std::vector<int>& v) {
    std::string s = "[";
    for (int i = 0; i < (int)v.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

// Construct the 2x4 test CFN directly (no file I/O)
static CFNModel make_test_cfn() {
    CFNModel cfn;
    cfn.name = "test_2x4";
    cfn.num_variables = 2;
    cfn.var_names = {"x0", "x1"};
    cfn.cardinalities = {4, 4};
    cfn.unary = {
        {3.0, 1.0, 4.0, 2.0},
        {2.5, 0.5, 1.5, 3.5}
    };
    CFNModel::PairwiseTable pt;
    pt.vi = 0; pt.vj = 1; pt.di = 4; pt.dj = 4;
    pt.costs = {
         0.0,  1.0, -2.0,  0.5,
         1.5,  0.0,  0.5, -1.0,
        -0.5,  2.0,  0.0,  1.0,
         0.5, -1.5,  1.0,  0.0
    };
    cfn.pairwise.push_back(pt);
    cfn.build_adjacency();
    return cfn;
}

// Parse BinaryModel from a JSON string (temp file round-trip)
static BinaryModel parse_model_str(const std::string& json_str) {
    const char* tmp = "_test_qa_model_tmp.json";
    { std::ofstream ofs(tmp); ofs << json_str; }
    auto model = parse_binary_model(tmp);
    std::remove(tmp);
    return model;
}

// Write a string to a temp file and return the path
static std::string write_temp(const std::string& content, const char* name) {
    std::ofstream ofs(name);
    ofs << content;
    return name;
}

// ============================================================================
// Embedded test models (same as test_models/ directory)
// ============================================================================

static const char* ONE_HOT_JSON = R"({
  "encoding": "one_hot",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 8,
  "offset": 13.0,
  "qubit_map": {
    "0": {"bit":0,"variable":0,"variable_name":"x0"},
    "1": {"bit":1,"variable":0,"variable_name":"x0"},
    "2": {"bit":2,"variable":0,"variable_name":"x0"},
    "3": {"bit":3,"variable":0,"variable_name":"x0"},
    "4": {"bit":0,"variable":1,"variable_name":"x1"},
    "5": {"bit":1,"variable":1,"variable_name":"x1"},
    "6": {"bit":2,"variable":1,"variable_name":"x1"},
    "7": {"bit":3,"variable":1,"variable_name":"x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0":-3.5,"0,1":13.0,"0,2":13.0,"0,3":13.0,"0,5":1.0,"0,6":-2.0,"0,7":0.5,
    "1":-5.5,"1,2":13.0,"1,3":13.0,"1,4":1.5,"1,6":0.5,"1,7":-1.0,
    "2":-2.5,"2,3":13.0,"2,4":-0.5,"2,5":2.0,"2,7":1.0,
    "3":-4.5,"3,4":0.5,"3,5":-1.5,"3,6":1.0,
    "4":-4.0,"4,5":13.0,"4,6":13.0,"4,7":13.0,
    "5":-6.0,"5,6":13.0,"5,7":13.0,
    "6":-5.0,"6,7":13.0,
    "7":-3.0
  },
  "variable_type": "BINARY"
})";

static const char* TRUNC_K2_JSON = R"({
  "encoding": "truncated_binary",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 4,
  "offset": 4.6875,
  "qubit_map": {
    "0": {"bit":0,"variable":0,"variable_name":"x0"},
    "1": {"bit":1,"variable":0,"variable_name":"x0"},
    "2": {"bit":0,"variable":1,"variable_name":"x1"},
    "3": {"bit":1,"variable":1,"variable_name":"x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0":1.0625,"0,1":-0.25,"0,2":-0.8125,"0,3":0.1875,
    "1":-0.625,"1,3":0.375,
    "2":-0.0625,"2,3":1.0625,
    "3":-0.3125
  },
  "variable_type": "SPIN"
})";

// A small 3-qubit BINARY QUBO for effective-field unit tests
static const char* TINY_BINARY_JSON = R"({
  "encoding": "test",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 3,
  "offset": 0,
  "qubit_map": {
    "0": {"bit":0,"variable":0,"variable_name":"x0"},
    "1": {"bit":1,"variable":0,"variable_name":"x0"},
    "2": {"bit":0,"variable":1,"variable_name":"x1"}
  },
  "source_cfn": "test",
  "terms": {"0":0.5, "1":-1.0, "2":0.0, "0,1":1.0, "1,2":3.0},
  "variable_type": "BINARY"
})";

// Same topology but SPIN
static const char* TINY_SPIN_JSON = R"({
  "encoding": "test",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 3,
  "offset": 0,
  "qubit_map": {
    "0": {"bit":0,"variable":0,"variable_name":"x0"},
    "1": {"bit":1,"variable":0,"variable_name":"x0"},
    "2": {"bit":0,"variable":1,"variable_name":"x1"}
  },
  "source_cfn": "test",
  "terms": {"0":0.5, "1":-1.0, "2":0.0, "0,1":1.0, "1,2":3.0},
  "variable_type": "SPIN"
})";

// A 3-qubit HUBO (should be rejected)
static const char* TINY_HUBO_JSON = R"({
  "encoding": "test",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 3,
  "offset": 0,
  "qubit_map": {
    "0": {"bit":0,"variable":0,"variable_name":"x0"},
    "1": {"bit":1,"variable":0,"variable_name":"x0"},
    "2": {"bit":0,"variable":1,"variable_name":"x1"}
  },
  "source_cfn": "test",
  "terms": {"0":1.0, "0,1,2":2.5},
  "variable_type": "BINARY"
})";

// ============================================================================
// Mock D-Wave results JSON
//
// For one_hot model (8 qubits, 2 vars x 4 choices):
//   Sample 1: [0,0,0,1, 0,1,0,0]  -> [3,1] (ground state), E=1.0,  occ=5
//   Sample 2: [0,1,0,0, 0,1,0,0]  -> [1,1],                E=2.5,  occ=3
//   Sample 3: [1,1,0,0, 0,1,0,0]  -> infeasible (2 bits),  E=10.0, occ=2
// ============================================================================
static const char* MOCK_RESULTS_ONE_HOT = R"({
  "samples": [
    {"values":[0,0,0,1,0,1,0,0], "energy":1.0,  "num_occurrences":5},
    {"values":[0,1,0,0,0,1,0,0], "energy":2.5,  "num_occurrences":3},
    {"values":[1,1,0,0,0,1,0,0], "energy":10.0, "num_occurrences":2}
  ],
  "timing": {
    "qpu_access_time":15000, "qpu_sampling_time":10000,
    "qpu_programming_time":5000, "post_processing_time":100
  },
  "embedding_time_s": 0.5,
  "chain_strength": 5.0,
  "num_physical_qubits": 12,
  "max_chain_length": 2,
  "chain_lengths": [2,1,2,1,2,1,1,2],
  "chain_breaks": [
    {"breaks":0, "num_occurrences":5},
    {"breaks":1, "num_occurrences":3},
    {"breaks":2, "num_occurrences":2}
  ]
})";

// For truncated_binary_k2 (SPIN, 4 qubits):
//   SPIN->BIN: b = (1-s)/2,  +1->0, -1->1
//   Sample 1: [-1,-1,-1,1]  -> bin [1,1,1,0] -> var0=3, var1=1 -> [3,1]
//   Sample 2: [1,-1,1,-1]   -> bin [0,1,0,1] -> var0=2, var1=2 -> [2,2]
//   Sample 3: [1,1,-1,-1]   -> bin [0,0,1,1] -> var0=0, var1=3 -> [0,3]
static const char* MOCK_RESULTS_SPIN = R"({
  "samples": [
    {"values":[-1,-1,-1,1],  "energy":1.3,  "num_occurrences":4},
    {"values":[1,-1,1,-1],   "energy":2.0,  "num_occurrences":3},
    {"values":[1,1,-1,-1],   "energy":3.5,  "num_occurrences":3}
  ],
  "timing": {
    "qpu_access_time":12000, "qpu_sampling_time":8000,
    "qpu_programming_time":4000, "post_processing_time":50
  },
  "embedding_time_s": 0.3,
  "chain_strength": 3.0,
  "num_physical_qubits": 6,
  "max_chain_length": 2,
  "chain_lengths": [1,2,1,2],
  "chain_breaks": [
    {"breaks":0, "num_occurrences":4},
    {"breaks":1, "num_occurrences":6}
  ]
})";


// ============================================================================
// TEST: Effective field computation — BINARY
// ============================================================================
static void test_effective_fields_binary() {
    std::cout << "\n=== Effective Fields (BINARY) ===\n";
    auto model = parse_model_str(TINY_BINARY_JSON);

    // Topology: q0 -- q1 -- q2
    //   h = [0.5, -1.0, 0.0]
    //   J_{01} = 1.0,  J_{12} = 3.0
    //
    // q0: neighbors={1}, couplings={1.0}
    //   T=1.0, f=|2*0.5+1|=2, after J=1: (|3|+|1|)/2=2, result=2/2=1.0
    //
    // q1: neighbors={0,2}, couplings={1.0,3.0}
    //   T=4.0, f=|2*(-1)+4|=2
    //   after J=1: (|3|+|1|)/2=2
    //   after J=3: (|5|+|1|)/2=3, result=3/2=1.5
    //
    // q2: neighbors={1}, couplings={3.0}
    //   T=3.0, f=|2*0+3|=3, after J=3: (|6|+|0|)/2=3, result=3/2=1.5

    auto fields = compute_effective_fields(model);
    CHECK(fields.size() == 3);
    APPROX_EQ(fields[0], 1.0, 1e-10);
    APPROX_EQ(fields[1], 1.5, 1e-10);
    APPROX_EQ(fields[2], 1.5, 1e-10);
    std::cout << "  fields = [" << fields[0] << ", "
              << fields[1] << ", " << fields[2] << "]\n";
}

// ============================================================================
// TEST: Effective field computation — SPIN
// ============================================================================
static void test_effective_fields_spin() {
    std::cout << "\n=== Effective Fields (SPIN) ===\n";
    auto model = parse_model_str(TINY_SPIN_JSON);

    // q0: f=|0.5|=0.5, after J=1: (|1.5|+|0.5|)/2=1.0
    // q1: f=|-1|=1, after J=1: (|2|+|0|)/2=1, after J=3: (|4|+|2|)/2=3.0
    // q2: f=|0|=0, after J=3: (|3|+|3|)/2=3.0

    auto fields = compute_effective_fields(model);
    CHECK(fields.size() == 3);
    APPROX_EQ(fields[0], 1.0, 1e-10);
    APPROX_EQ(fields[1], 3.0, 1e-10);
    APPROX_EQ(fields[2], 3.0, 1e-10);
    std::cout << "  fields = [" << fields[0] << ", "
              << fields[1] << ", " << fields[2] << "]\n";
}

// ============================================================================
// TEST: Anneal offset computation
// ============================================================================
static void test_anneal_offsets() {
    std::cout << "\n=== Anneal Offsets ===\n";
    // fields = [1.0, 3.0, 3.0], delta_max = 0.1
    //   max_field = 3.0
    //   r = [1/3, 1, 1]
    //   delta = 0.1*(1-2r) = [0.1*(1-2/3), 0.1*(1-2), 0.1*(1-2)]
    //                      = [0.1*1/3, -0.1, -0.1]
    //                      = [0.03333, -0.1, -0.1]

    std::vector<double> fields = {1.0, 3.0, 3.0};
    auto offsets = compute_anneal_offsets(fields, 0.1);

    CHECK(offsets.size() == 3);
    APPROX_EQ(offsets[0], 0.1 * (1.0 - 2.0/3.0), 1e-10);
    APPROX_EQ(offsets[1], -0.1, 1e-10);
    APPROX_EQ(offsets[2], -0.1, 1e-10);
    std::cout << "  offsets = [" << offsets[0] << ", "
              << offsets[1] << ", " << offsets[2] << "]\n";

    // All-zero fields: all get +delta_max
    std::vector<double> zero_fields = {0.0, 0.0};
    auto zo = compute_anneal_offsets(zero_fields, 0.2);
    APPROX_EQ(zo[0], 0.2, 1e-10);
    APPROX_EQ(zo[1], 0.2, 1e-10);
    std::cout << "  zero-field offsets = [" << zo[0] << ", " << zo[1] << "]\n";
}

// ============================================================================
// TEST: QUBO validation (accepts degree <= 2, rejects degree > 2)
// ============================================================================
static void test_validate_qubo() {
    std::cout << "\n=== QUBO Validation ===\n";

    auto qubo = parse_model_str(TINY_BINARY_JSON);
    bool accepted = true;
    try { validate_qubo_model(qubo); } catch (...) { accepted = false; }
    CHECK(accepted);
    std::cout << "  QUBO (degree 2): accepted\n";

    auto hubo = parse_model_str(TINY_HUBO_JSON);
    bool rejected = false;
    try { validate_qubo_model(hubo); } catch (...) { rejected = true; }
    CHECK(rejected);
    std::cout << "  HUBO (degree 3): rejected\n";
}

// ============================================================================
// TEST: Script generation fills all placeholders
// ============================================================================
static void test_script_generation() {
    std::cout << "\n=== Script Generation ===\n";

    QAParams params;
    params.input_path = "/path/to/model.json";
    params.solver_name = "TestSolver";
    params.annealing_time_us = 50.0;
    params.num_reads = 500;
    params.use_inhomogeneous = true;

    std::vector<double> offsets = {0.05, -0.1, 0.0};
    std::string script = generate_dwave_script(
        params.input_path, "/tmp/results.json", params, offsets);

    // No placeholders should remain
    CHECK(script.find("__INPUT_PATH__")    == std::string::npos);
    CHECK(script.find("__OUTPUT_PATH__")   == std::string::npos);
    CHECK(script.find("__SOLVER_NAME__")   == std::string::npos);
    CHECK(script.find("__ANNEALING_TIME__")== std::string::npos);
    CHECK(script.find("__NUM_READS__")     == std::string::npos);
    CHECK(script.find("__USE_INHOMOGENEOUS__") == std::string::npos);
    CHECK(script.find("__ANNEAL_OFFSETS__")== std::string::npos);

    // Values should be present
    CHECK(script.find("/path/to/model.json") != std::string::npos);
    CHECK(script.find("TestSolver")          != std::string::npos);
    CHECK(script.find("500")                 != std::string::npos);
    CHECK(script.find("True")               != std::string::npos);
    std::cout << "  All placeholders filled, values present\n";
}

// ============================================================================
// TEST: Parse mock D-Wave results JSON
// ============================================================================
static void test_parse_mock_results() {
    std::cout << "\n=== Parse Mock Results ===\n";

    const char* tmp = "_test_qa_mock_results.json";
    write_temp(MOCK_RESULTS_ONE_HOT, tmp);
    auto results = parse_dwave_results(tmp);
    std::remove(tmp);

    // Samples
    CHECK(results.samples.size() == 3);
    APPROX_EQ(results.samples[0].energy, 1.0, 1e-10);
    CHECK(results.samples[0].num_occurrences == 5);
    CHECK(results.samples[0].values.size() == 8);
    CHECK(results.samples[0].values == std::vector<int>({0,0,0,1,0,1,0,0}));

    // Timing
    APPROX_EQ(results.timing.qpu_access_time_us, 15000.0, 1e-10);
    APPROX_EQ(results.timing.qpu_sampling_time_us, 10000.0, 1e-10);
    APPROX_EQ(results.embedding_time_s, 0.5, 1e-10);

    // Chain data
    CHECK(results.chain_lengths.size() == 8);
    CHECK(results.chain_lengths == std::vector<int>({2,1,2,1,2,1,1,2}));
    CHECK(results.chain_breaks.size() == 3);
    CHECK(results.chain_breaks[0].breaks == 0);
    CHECK(results.chain_breaks[0].num_occurrences == 5);
    CHECK(results.chain_breaks[2].breaks == 2);

    std::cout << "  3 samples, 8 chain lengths, 3 break entries: OK\n";
}

// ============================================================================
// TEST: Full mock pipeline — one_hot (BINARY, cost-preserving)
//
// Verifies: decode → CFN evaluate → energy stats → embedding stats
// ============================================================================
static void test_pipeline_one_hot() {
    std::cout << "\n=== Pipeline: one_hot (BINARY) ===\n";

    auto model = parse_model_str(ONE_HOT_JSON);
    auto cfn   = make_test_cfn();

    // Parse mock results
    const char* tmp = "_test_qa_pipeline_oh.json";
    write_temp(MOCK_RESULTS_ONE_HOT, tmp);
    auto dwave = parse_dwave_results(tmp);
    std::remove(tmp);

    // --- Decode and evaluate each sample ---
    double best_cfn = 1e30;
    std::vector<int> best_sol;
    int num_feasible = 0, num_best = 0;

    for (const auto& s : dwave.samples) {
        auto choices = decode_to_cfn(model, s.values);
        if (choices.empty()) continue;

        double cfn_e = compute_energy(cfn, choices);
        num_feasible += s.num_occurrences;

        if (cfn_e < best_cfn - 1e-9) {
            best_cfn = cfn_e;
            best_sol = choices;
            num_best = s.num_occurrences;
        } else if (std::abs(cfn_e - best_cfn) <= 1e-9) {
            num_best += s.num_occurrences;
        }
    }

    // Sample 1: [3,1] -> E=1.0, occ=5
    // Sample 2: [1,1] -> E=1.5, occ=3
    // Sample 3: infeasible
    CHECK(num_feasible == 8);
    APPROX_EQ(best_cfn, 1.0, 1e-9);
    CHECK(best_sol == std::vector<int>({3, 1}));
    CHECK(num_best == 5);
    std::cout << "  best_cfn=" << best_cfn << "  solution=" << fmt(best_sol)
              << "  feasible=" << num_feasible << "/" << 10
              << "  num_best=" << num_best << "\n";

    // --- Energy statistics (expanded by num_occurrences) ---
    // Energies: 1.0 x5, 2.5 x3, 10.0 x2  => 10 total
    std::vector<double> all_e;
    for (const auto& s : dwave.samples)
        for (int k = 0; k < s.num_occurrences; k++)
            all_e.push_back(s.energy);

    CHECK(all_e.size() == 10);
    double best_dw = *std::min_element(all_e.begin(), all_e.end());
    APPROX_EQ(best_dw, 1.0, 1e-10);

    double mean = std::accumulate(all_e.begin(), all_e.end(), 0.0) / 10;
    APPROX_EQ(mean, 3.25, 1e-10);

    std::sort(all_e.begin(), all_e.end());
    double median = 0.5 * (all_e[4] + all_e[5]);   // 5th=1.0, 6th=2.5
    APPROX_EQ(median, 1.75, 1e-10);
    std::cout << "  energy: best=" << best_dw << " mean=" << mean
              << " median=" << median << "\n";

    // --- Chain length statistics ---
    // chain_lengths = [2,1,2,1,2,1,1,2], n=8
    // avg = 12/8 = 1.5
    // sorted = [1,1,1,1,2,2,2,2], median = (1+2)/2 = 1.5
    // var = (4*0.25 + 4*0.25)/7 = 2/7 ≈ 0.2857
    {
        auto& cl = dwave.chain_lengths;
        int nc = (int)cl.size();
        double cl_sum = 0;
        for (int c : cl) cl_sum += c;
        double cl_avg = cl_sum / nc;
        APPROX_EQ(cl_avg, 1.5, 1e-10);

        double cl_sq = 0;
        for (int c : cl) cl_sq += (c - cl_avg) * (c - cl_avg);
        double cl_var = cl_sq / (nc - 1);
        APPROX_EQ(cl_var, 2.0/7.0, 1e-10);

        auto cl_s = cl;
        std::sort(cl_s.begin(), cl_s.end());
        double cl_med = 0.5 * (cl_s[3] + cl_s[4]);
        APPROX_EQ(cl_med, 1.5, 1e-10);
        std::cout << "  chain_length: avg=" << cl_avg << " median="
                  << cl_med << " var=" << cl_var << "\n";
    }

    // --- Chain break statistics ---
    // Expanded: [0,0,0,0,0, 1,1,1, 2,2], n=10
    // avg = 7/10 = 0.7
    // sorted = [0,0,0,0,0,1,1,1,2,2], median = (0+1)/2 = 0.5
    // var = (5*0.49 + 3*0.09 + 2*1.69)/9 = 6.1/9 ≈ 0.6778
    {
        std::vector<int> all_b;
        for (const auto& cb : dwave.chain_breaks)
            for (int k = 0; k < cb.num_occurrences; k++)
                all_b.push_back(cb.breaks);
        int nb = (int)all_b.size();
        CHECK(nb == 10);

        double cb_sum = 0;
        for (int b : all_b) cb_sum += b;
        double cb_avg = cb_sum / nb;
        APPROX_EQ(cb_avg, 0.7, 1e-10);

        double cb_sq = 0;
        for (int b : all_b) cb_sq += (b - cb_avg) * (b - cb_avg);
        double cb_var = cb_sq / (nb - 1);
        APPROX_EQ(cb_var, 6.1/9.0, 1e-10);

        std::sort(all_b.begin(), all_b.end());
        double cb_med = 0.5 * (all_b[4] + all_b[5]);
        APPROX_EQ(cb_med, 0.5, 1e-10);
        std::cout << "  chain_breaks: avg=" << cb_avg << " median="
                  << cb_med << " var=" << cb_var << "\n";
    }
}

// ============================================================================
// TEST: Full mock pipeline — truncated_binary (SPIN, cost-approximate)
// ============================================================================
static void test_pipeline_spin() {
    std::cout << "\n=== Pipeline: truncated_binary_k2 (SPIN) ===\n";

    auto model = parse_model_str(TRUNC_K2_JSON);
    auto cfn   = make_test_cfn();

    CHECK(model.var_type == BinaryModel::SPIN);
    CHECK(model.encoding == "truncated_binary");

    const char* tmp = "_test_qa_pipeline_spin.json";
    write_temp(MOCK_RESULTS_SPIN, tmp);
    auto dwave = parse_dwave_results(tmp);
    std::remove(tmp);

    // Decode SPIN samples
    // S1: [-1,-1,-1,1] -> b=[1,1,1,0] -> var0= 1|2=3, var1=1 -> [3,1]
    // S2: [1,-1,1,-1]  -> b=[0,1,0,1] -> var0= 0|2=2, var1=0|2=2 -> [2,2]  -- WAIT
    // Actually: reg[b] = (1 - state[q]) / 2
    // S2: q0=1 -> b=0, q1=-1 -> b=1, q2=1 -> b=0, q3=-1 -> b=1
    //     var0: reg=[0,1], val = 0|(1<<1) = 2
    //     var1: reg=[0,1], val = 0|(1<<1) = 2
    //     -> [2,2]
    // S3: [1,1,-1,-1] -> b=[0,0,1,1] -> var0=0, var1=1|(1<<1)=3 -> [0,3]

    auto ch1 = decode_to_cfn(model, dwave.samples[0].values);
    CHECK(ch1 == std::vector<int>({3, 1}));

    auto ch2 = decode_to_cfn(model, dwave.samples[1].values);
    CHECK(ch2 == std::vector<int>({2, 2}));

    auto ch3 = decode_to_cfn(model, dwave.samples[2].values);
    CHECK(ch3 == std::vector<int>({0, 3}));

    // Evaluate under CFN
    double e1 = compute_energy(cfn, ch1);   // E(3,1) = 1.0
    double e2 = compute_energy(cfn, ch2);   // E(2,2) = 4 + 1.5 + 0 = 5.5
    double e3 = compute_energy(cfn, ch3);   // E(0,3) = 3 + 3.5 + 0.5 = 7.0
    APPROX_EQ(e1, 1.0, 1e-9);

    std::cout << "  [3,1] cfn_E=" << e1
              << "  [2,2] cfn_E=" << e2
              << "  [0,3] cfn_E=" << e3 << "\n";

    // All 3 samples are feasible (binary encoding always decodes)
    int total_occ = 0;
    for (const auto& s : dwave.samples) total_occ += s.num_occurrences;
    CHECK(total_occ == 10);

    // Best CFN energy should be 1.0 at [3,1] with 4 occurrences
    double best_cfn = 1e30;
    int best_occ = 0;
    std::vector<int> best_sol;
    for (const auto& s : dwave.samples) {
        auto ch = decode_to_cfn(model, s.values);
        double ce = compute_energy(cfn, ch);
        if (ce < best_cfn - 1e-9) {
            best_cfn = ce; best_sol = ch; best_occ = s.num_occurrences;
        } else if (std::abs(ce - best_cfn) <= 1e-9) {
            best_occ += s.num_occurrences;
        }
    }
    APPROX_EQ(best_cfn, 1.0, 1e-9);
    CHECK(best_sol == std::vector<int>({3, 1}));
    CHECK(best_occ == 4);
    std::cout << "  best_cfn=" << best_cfn << "  solution=" << fmt(best_sol)
              << "  num_best=" << best_occ << "\n";
}

// ============================================================================
// TEST: Infeasible sample handling (one_hot with bad bitstrings)
// ============================================================================
static void test_infeasible_handling() {
    std::cout << "\n=== Infeasible Sample Handling ===\n";

    auto model = parse_model_str(ONE_HOT_JSON);

    // All bits zero -> no choice selected -> infeasible
    std::vector<int> all_zero = {0,0,0,0, 0,0,0,0};
    auto ch0 = decode_to_cfn(model, all_zero);
    CHECK(ch0.empty());

    // Two bits set in var0 -> infeasible
    std::vector<int> two_bits = {1,1,0,0, 0,1,0,0};
    auto ch1 = decode_to_cfn(model, two_bits);
    CHECK(ch1.empty());

    // Valid one-hot: exactly one bit per register
    std::vector<int> valid = {1,0,0,0, 0,0,1,0};  // [0, 2]
    auto ch2 = decode_to_cfn(model, valid);
    CHECK(!ch2.empty());
    CHECK(ch2 == std::vector<int>({0, 2}));

    std::cout << "  all_zero: infeasible (correct)\n";
    std::cout << "  two_bits: infeasible (correct)\n";
    std::cout << "  valid [0,2]: feasible (correct)\n";
}

// ============================================================================
// TEST: CSV output format
// ============================================================================
static void test_csv_format() {
    std::cout << "\n=== CSV Output Format ===\n";

    // Verify header column count matches row column count
    std::string header = csv_header_qa();
    int h_cols = 1;
    for (char c : header) if (c == ',') h_cols++;

    QAResult r;
    r.problem_name = "test"; r.source_cfn = "test"; r.encoding = "one_hot";
    r.variable_type = "BINARY"; r.distribution = "uniform";
    r.solver_name = "Test"; r.num_reads = 10;
    r.per_run_energies = {1.0, 2.0};
    r.best_solution = {3, 1};

    std::string row = csv_row_qa(r);
    int r_cols = 1;
    // Count commas outside quotes
    bool in_quote = false;
    for (char c : row) {
        if (c == '"') in_quote = !in_quote;
        if (c == ',' && !in_quote) r_cols++;
    }

    CHECK(h_cols == r_cols);
    std::cout << "  header columns: " << h_cols
              << "  row columns: " << r_cols << "\n";

    // Verify solver_mode is "dwave"
    CHECK(row.find(",dwave,") != std::string::npos);
}

// ============================================================================
// TEST: Effective fields on a real encoded model (one_hot)
// ============================================================================
static void test_effective_fields_one_hot() {
    std::cout << "\n=== Effective Fields (one_hot model) ===\n";

    auto model = parse_model_str(ONE_HOT_JSON);
    auto fields = compute_effective_fields(model);

    CHECK(fields.size() == 8);
    // All fields should be positive (all qubits are coupled)
    for (int q = 0; q < 8; q++) {
        CHECK(fields[q] > 0);
    }

    double max_f = *std::max_element(fields.begin(), fields.end());
    double min_f = *std::min_element(fields.begin(), fields.end());
    CHECK(max_f > min_f);   // some variation expected
    std::cout << "  8 qubits, fields range [" << min_f << ", " << max_f << "]\n";

    // Offsets should be in [-delta_max, +delta_max]
    auto offsets = compute_anneal_offsets(fields, 0.1);
    double max_off = *std::max_element(offsets.begin(), offsets.end());
    double min_off = *std::min_element(offsets.begin(), offsets.end());
    CHECK(min_off >= -0.1 - 1e-10);
    CHECK(max_off <=  0.1 + 1e-10);
    std::cout << "  offsets range [" << min_off << ", " << max_off << "]\n";
}

// ============================================================================
// INTEGRATION TEST: End-to-end D-Wave submission
//
// Requires: --dwave --solver <NAME> [--test-dir <DIR>] [--python <CMD>]
//
// Uses the one_hot test model (8 qubits, easy to embed).
// Runs only 10 reads for speed.
// ============================================================================
static void test_dwave_end_to_end(
    const std::string& solver_name,
    const std::string& test_dir,
    const std::string& python_cmd
) {
    std::cout << "\n=== D-Wave End-to-End Test ===\n";
    std::cout << "  Solver: " << solver_name << "\n";

    // Locate test files
    std::string model_path = test_dir + "/test_models/test_2x4_one_hot.json";
    std::string cfn_path   = test_dir + "/test_2x4.cfn";

    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  CFN:   " << cfn_path << "\n";

    // Load model and CFN
    BinaryModel model = parse_binary_model(model_path);
    CFNModel cfn = parse_cfn_for_sa(cfn_path);

    CHECK(model.num_qubits == 8);
    CHECK(model.encoding == "one_hot");
    CHECK(cfn.num_variables == 2);

    // Configure QA params
    QAParams params;
    params.input_path       = model_path;
    params.cfn_dir          = test_dir;
    params.solver_name      = solver_name;
    params.python_cmd       = python_cmd;
    params.annealing_time_us = 20.0;
    params.num_reads        = 10;
    params.delta_max        = 0.1;
    params.use_inhomogeneous = true;
    params.verbose          = true;

    // Run full pipeline
    QAResult result = run_qa_binary(model, cfn, params);

    // --- Sanity checks ---
    CHECK(!std::isnan(result.best_energy));
    CHECK(!std::isinf(result.best_energy));
    CHECK(result.per_run_energies.size() > 0);
    CHECK(result.per_run_energies.size() <= 10);
    CHECK(result.total_runtime_s > 0);
    CHECK(result.inhomog_setup_time_s >= 0);
    CHECK(result.embedding_time_s > 0);
    CHECK(result.qpu_access_time_us > 0);

    std::cout << "\n  Results:\n";
    std::cout << "    best_dwave_energy: " << result.best_energy << "\n";
    std::cout << "    mean_energy:       " << result.mean_energy << "\n";
    std::cout << "    total_runtime:     " << result.total_runtime_s << " s\n";
    std::cout << "    embedding_time:    " << result.embedding_time_s << " s\n";
    std::cout << "    inhomog_setup:     " << result.inhomog_setup_time_s << " s\n";
    std::cout << "    qpu_access:        " << result.qpu_access_time_us << " us\n";

    // Embedding stats
    CHECK(result.emb_num_physical_qubits > 0);
    CHECK(result.emb_chain_length_avg > 0);
    std::cout << "    physical_qubits:   " << result.emb_num_physical_qubits << "\n";
    std::cout << "    chain_length_avg:  " << result.emb_chain_length_avg << "\n";
    std::cout << "    chain_length_med:  " << result.emb_chain_length_median << "\n";
    std::cout << "    chain_breaks_avg:  " << result.emb_chain_breaks_avg << "\n";

    // CFN evaluation
    if (!std::isnan(result.best_cfn_energy)) {
        std::cout << "    best_cfn_energy:   " << result.best_cfn_energy << "\n";
        std::cout << "    best_solution:     " << fmt(result.best_solution) << "\n";
        std::cout << "    feasible:          " << result.num_feasible
                  << "/" << result.num_reads << "\n";
        std::cout << "    num_best_cfn:      " << result.num_best_cfn << "\n";
        CHECK(result.num_feasible > 0);
        CHECK(result.best_cfn_energy >= 1.0 - 1e-6);   // cannot beat ground state
    } else {
        std::cout << "    best_cfn_energy:   NA (no feasible solutions)\n";
    }

    // Verify CSV row is well-formed
    std::string csv = csv_row_qa(result);
    CHECK(!csv.empty());
    CHECK(csv.find("dwave") != std::string::npos);
    std::cout << "\n  CSV row generated successfully (" << csv.size()
              << " chars)\n";
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool run_dwave = false;
    std::string solver_name;
    std::string test_dir = "../../tests";    // default: relative to build/Release/
    std::string python_cmd = "python3";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dwave")    { run_dwave = true; }
        else if (a == "--solver"   && i+1 < argc) { solver_name = argv[++i]; }
        else if (a == "--test-dir" && i+1 < argc) { test_dir    = argv[++i]; }
        else if (a == "--python"   && i+1 < argc) { python_cmd  = argv[++i]; }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Unit tests (no D-Wave needed):\n"
                      << "  " << argv[0] << "\n\n"
                      << "Integration test (needs D-Wave):\n"
                      << "  " << argv[0] << " --dwave --solver <NAME>"
                      << " [--test-dir <DIR>] [--python <CMD>]\n\n"
                      << "Options:\n"
                      << "  --dwave            Enable D-Wave integration test\n"
                      << "  --solver NAME      D-Wave solver name (required with --dwave)\n"
                      << "  --test-dir DIR     Path to tests/ directory"
                      << " (default: ../../tests)\n"
                      << "  --python CMD       Python interpreter"
                      << " (default: python3)\n\n"
                      << "D-Wave access: set DWAVE_API_TOKEN or run 'dwave config create'\n";
            return 0;
        }
    }

    // --- Unit tests (always run) ---
    test_effective_fields_binary();
    test_effective_fields_spin();
    test_anneal_offsets();
    test_validate_qubo();
    test_script_generation();
    test_parse_mock_results();
    test_pipeline_one_hot();
    test_pipeline_spin();
    test_infeasible_handling();
    test_csv_format();
    test_effective_fields_one_hot();

    // --- Integration test (opt-in) ---
    if (run_dwave) {
        if (solver_name.empty()) {
            std::cerr << "\nError: --solver is required with --dwave\n";
            return 1;
        }
        test_dwave_end_to_end(solver_name, test_dir, python_cmd);
    } else {
        std::cout << "\n(Skipping D-Wave integration test."
                  << " Run with --dwave --solver <NAME> to enable.)\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Checks: " << g_checks
              << "  Failures: " << g_failures << "\n";
    return (g_failures > 0) ? 1 : 0;
}
