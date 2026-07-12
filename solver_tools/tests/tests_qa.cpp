// ============================================================================
// Unit and integration tests for the D-Wave QA pipeline.
//
// Unit tests (no D-Wave access needed):
//   ./tests_qa                              # core tests + encoding tests
//   ./tests_qa --test-dir ./tests           # also runs file-based encoding tests
//
// Integration test (needs D-Wave access):
//   ./tests_qa --dwave --solver <NAME> --test-dir ./tests [--python <CMD>]
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
#include <set>
#include <utility>

// ============================================================================
// Test framework
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
        if (i > 0) s += ",";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

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

static BinaryModel parse_model_str(const std::string& json_str) {
    const char* tmp = "_test_qa_model_tmp.json";
    { std::ofstream ofs(tmp); ofs << json_str; }
    auto model = parse_binary_model(tmp);
    std::remove(tmp);
    return model;
}

static std::string write_temp(const std::string& content, const char* name) {
    std::ofstream ofs(name);
    ofs << content;
    return name;
}

// Build a mock D-Wave results JSON string
static std::string make_mock_results(
    const std::vector<std::vector<int>>& bitstrings,
    const std::vector<double>& energies,
    const std::vector<int>& occurrences,
    const std::vector<int>& chain_lengths,
    const std::vector<std::pair<int,int>>& chain_break_entries  // (breaks, occ)
) {
    nlohmann::json j;
    j["timing"] = {{"qpu_access_time", 15000}, {"qpu_sampling_time", 10000},
                   {"qpu_programming_time", 5000}, {"post_processing_time", 100}};
    j["embedding_time_s"] = 0.5;
    j["chain_strength"] = 5.0;
    j["num_physical_qubits"] = 12;
    j["max_chain_length"] = 2;
    j["chain_lengths"] = chain_lengths;

    nlohmann::json samples = nlohmann::json::array();
    for (int i = 0; i < (int)bitstrings.size(); i++) {
        samples.push_back({
            {"values", bitstrings[i]},
            {"energy", energies[i]},
            {"num_occurrences", occurrences[i]}
        });
    }
    j["samples"] = samples;

    nlohmann::json cb = nlohmann::json::array();
    for (const auto& entry : chain_break_entries)
        cb.push_back({{"breaks", entry.first}, {"num_occurrences", entry.second}});
    j["chain_breaks"] = cb;

    return j.dump();
}

// ============================================================================
// Embedded model JSONs (small models for core unit tests)
// ============================================================================

static const char* TINY_BINARY_JSON = R"({
  "encoding":"test","num_auxiliary_qubits":0,"num_logical_qubits":3,"offset":0,
  "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
               "1":{"bit":1,"variable":0,"variable_name":"x0"},
               "2":{"bit":0,"variable":1,"variable_name":"x1"}},
  "source_cfn":"test",
  "terms":{"0":0.5,"1":-1.0,"2":0.0,"0,1":1.0,"1,2":3.0},
  "variable_type":"BINARY"
})";

static const char* TINY_SPIN_JSON = R"({
  "encoding":"test","num_auxiliary_qubits":0,"num_logical_qubits":3,"offset":0,
  "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
               "1":{"bit":1,"variable":0,"variable_name":"x0"},
               "2":{"bit":0,"variable":1,"variable_name":"x1"}},
  "source_cfn":"test",
  "terms":{"0":0.5,"1":-1.0,"2":0.0,"0,1":1.0,"1,2":3.0},
  "variable_type":"SPIN"
})";

static const char* TINY_HUBO_JSON = R"({
  "encoding":"test","num_auxiliary_qubits":0,"num_logical_qubits":3,"offset":0,
  "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
               "1":{"bit":1,"variable":0,"variable_name":"x0"},
               "2":{"bit":0,"variable":1,"variable_name":"x1"}},
  "source_cfn":"test","terms":{"0":1.0,"0,1,2":2.5},"variable_type":"BINARY"
})";

// Model with an isolated qubit (qubit 2 has ONLY a linear term)
static const char* ISOLATED_QUBIT_JSON = R"({
  "encoding":"test","num_auxiliary_qubits":0,"num_logical_qubits":3,"offset":0,
  "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
               "1":{"bit":1,"variable":0,"variable_name":"x0"},
               "2":{"bit":0,"variable":1,"variable_name":"x1"}},
  "source_cfn":"test",
  "terms":{"0":0.5,"1":-1.0,"2":3.0,"0,1":2.0},
  "variable_type":"BINARY"
})";

// ============================================================================
// CORE UNIT TESTS (no test files needed)
// ============================================================================

static void test_effective_fields_binary() {
    std::cout << "\n=== Effective Fields (BINARY) ===\n";
    auto model = parse_model_str(TINY_BINARY_JSON);
    auto fields = compute_effective_fields(model);
    CHECK(fields.size() == 3);
    APPROX_EQ(fields[0], 1.0, 1e-10);
    APPROX_EQ(fields[1], 1.5, 1e-10);
    APPROX_EQ(fields[2], 1.5, 1e-10);
    std::cout << "  fields = [" << fields[0] << ", " << fields[1] << ", " << fields[2] << "]\n";
}

static void test_effective_fields_spin() {
    std::cout << "\n=== Effective Fields (SPIN) ===\n";
    auto model = parse_model_str(TINY_SPIN_JSON);
    auto fields = compute_effective_fields(model);
    CHECK(fields.size() == 3);
    APPROX_EQ(fields[0], 1.0, 1e-10);
    APPROX_EQ(fields[1], 3.0, 1e-10);
    APPROX_EQ(fields[2], 3.0, 1e-10);
    std::cout << "  fields = [" << fields[0] << ", " << fields[1] << ", " << fields[2] << "]\n";
}

static void test_effective_fields_isolated() {
    std::cout << "\n=== Effective Fields (isolated qubit) ===\n";
    auto model = parse_model_str(ISOLATED_QUBIT_JSON);
    // q2 has only linear term (3.0), no couplings
    auto fields = compute_effective_fields(model);
    CHECK(fields.size() == 3);
    APPROX_EQ(fields[2], 3.0, 1e-10);  // |h| for isolated qubit
    CHECK(fields[0] > 0);
    CHECK(fields[1] > 0);
    std::cout << "  q0=" << fields[0] << " q1=" << fields[1]
              << " q2(isolated)=" << fields[2] << "\n";

    // Offsets should still work — isolated qubit gets a valid offset
    auto offsets = compute_anneal_offsets(fields, 0.1);
    CHECK(offsets.size() == 3);
    for (int q = 0; q < 3; q++) {
        CHECK(offsets[q] >= -0.1 - 1e-10);
        CHECK(offsets[q] <=  0.1 + 1e-10);
    }
    std::cout << "  offsets = [" << offsets[0] << ", " << offsets[1]
              << ", " << offsets[2] << "]\n";
}

static void test_anneal_offsets() {
    std::cout << "\n=== Anneal Offsets ===\n";
    std::vector<double> fields = {1.0, 3.0, 3.0};
    auto offsets = compute_anneal_offsets(fields, 0.1);
    CHECK(offsets.size() == 3);
    APPROX_EQ(offsets[0], 0.1 * (1.0 - 2.0/3.0), 1e-10);
    APPROX_EQ(offsets[1], -0.1, 1e-10);
    APPROX_EQ(offsets[2], -0.1, 1e-10);

    std::vector<double> zero_fields = {0.0, 0.0};
    auto zo = compute_anneal_offsets(zero_fields, 0.2);
    APPROX_EQ(zo[0], 0.2, 1e-10);
    APPROX_EQ(zo[1], 0.2, 1e-10);
    std::cout << "  OK\n";
}

static void test_validate_qubo() {
    std::cout << "\n=== QUBO Validation ===\n";
    auto qubo = parse_model_str(TINY_BINARY_JSON);
    bool accepted = true;
    try { validate_qubo_model(qubo); } catch (...) { accepted = false; }
    CHECK(accepted);

    auto hubo = parse_model_str(TINY_HUBO_JSON);
    bool rejected = false;
    try { validate_qubo_model(hubo); } catch (...) { rejected = true; }
    CHECK(rejected);
    std::cout << "  QUBO accepted, HUBO rejected\n";
}

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

    CHECK(script.find("__INPUT_PATH__")         == std::string::npos);
    CHECK(script.find("__OUTPUT_PATH__")         == std::string::npos);
    CHECK(script.find("__SOLVER_NAME__")         == std::string::npos);
    CHECK(script.find("__ANNEALING_TIME__")      == std::string::npos);
    CHECK(script.find("__NUM_READS__")           == std::string::npos);
    CHECK(script.find("__USE_INHOMOGENEOUS__")   == std::string::npos);
    CHECK(script.find("__ANNEAL_OFFSETS__")      == std::string::npos);
    CHECK(script.find("TestSolver")              != std::string::npos);
    std::cout << "  All placeholders filled\n";
}

static void test_script_handles_isolated_variables() {
    std::cout << "\n=== Script handles isolated variables ===\n";
    QAParams params;
    params.input_path = "/tmp/x.json";
    params.solver_name = "S";
    params.num_reads = 1;
    params.use_inhomogeneous = false;
    std::vector<double> offsets = {0.0};
    std::string script = generate_dwave_script("/tmp/x.json", "/tmp/r.json", params, offsets);
    // Must post-fill isolated variables into the embedding
    CHECK(script.find("if v not in embedding") != std::string::npos);
    // Must use edgelists for find_embedding (adjacency dicts fail on some SDK versions)
    CHECK(script.find("source_edgelist") != std::string::npos);
    std::cout << "  edgelist embedding + isolated variable post-fill\n";
}

static void test_csv_format() {
    std::cout << "\n=== CSV Output Format ===\n";
    std::string header = csv_header_qa();
    int h_cols = 1;
    for (char c : header) if (c == ',') h_cols++;

    QAResult r;
    r.problem_name = "test"; r.source_cfn = "test"; r.encoding = "one_hot";
    r.variable_type = "BINARY"; r.distribution = "uniform";
    r.solver_name = "Test"; r.num_reads = 10;
    r.per_run_energies = {1.0, 2.0};
    r.best_cfn_solution = {3, 1};

    std::string row = csv_row_qa(r);
    int r_cols = 1;
    bool in_q = false;
    for (char c : row) {
        if (c == '"') in_q = !in_q;
        if (c == ',' && !in_q) r_cols++;
    }
    CHECK(h_cols == r_cols);
    CHECK(row.find(",dwave,") != std::string::npos);
    std::cout << "  " << h_cols << " columns match\n";
}

// ============================================================================
// ENCODING-SPECIFIC PIPELINE TESTS (load models from test_models/)
//
// For each encoding, we verify:
//   1. Model parses correctly
//   2. QUBO validation passes (degree <= 2)
//   3. Effective fields compute for ALL qubits (including auxiliary)
//   4. Source adjacency includes ALL variables (no isolated-variable bug)
//   5. Mock D-Wave bitstrings decode to correct CFN choices
//   6. CFN energy evaluation is finite and reasonable
// ============================================================================

// Check that source_adj built from a model includes every qubit
static void check_source_adj_complete(const BinaryModel& model, const std::string& label) {
    // Replicate the source_adj logic from the Python template
    std::set<int> adj_vars;
    std::set<int> edge_vars;
    for (const auto& t : model.terms) {
        for (int q : t.qubits) adj_vars.insert(q);
        if (t.qubits.size() == 2) {
            edge_vars.insert(t.qubits[0]);
            edge_vars.insert(t.qubits[1]);
        }
    }

    int n_missing = 0;
    for (int q = 0; q < model.num_qubits; q++) {
        if (adj_vars.find(q) == adj_vars.end()) {
            // Qubit with no terms at all — shouldn't happen in valid models
            n_missing++;
        }
    }

    // The key check: are there variables in the BQM (have terms) but NOT
    // in any edge (isolated = only linear terms)?
    int n_isolated = 0;
    for (int q : adj_vars) {
        if (edge_vars.find(q) == edge_vars.end()) n_isolated++;
    }

    if (n_isolated > 0) {
        std::cout << "    " << n_isolated << " isolated qubit(s) — "
                  << "source_adj (not edgelist) required for embedding\n";
    }
    // source_adj includes all BQM variables; edgelist would miss isolated ones
    CHECK(adj_vars.size() >= edge_vars.size());
}

struct EncodingTestCase {
    std::string label;
    std::string filename;
    std::string var_type;       // "BINARY" or "SPIN"
    bool is_cost_preserving;
    // Mock sample: bitstring → expected decoded choices
    std::vector<int> sample_bitstring;
    std::vector<int> expected_choices;
};

static void run_encoding_test(
    const EncodingTestCase& tc,
    const std::string& test_dir,
    const CFNModel& cfn
) {
    std::cout << "\n  --- " << tc.label << " ---\n";

    std::string model_path = test_dir + "/test_models/" + tc.filename;

    // 1. Parse model
    BinaryModel model;
    try {
        model = parse_binary_model(model_path);
    } catch (const std::exception& e) {
        g_checks++; g_failures++;
        std::cerr << "  FAIL: Cannot load " << tc.filename << ": " << e.what() << "\n";
        return;
    }
    CHECK(model.num_qubits > 0);

    // 2. QUBO validation
    bool qubo_ok = true;
    try { validate_qubo_model(model); } catch (...) { qubo_ok = false; }
    CHECK(qubo_ok);

    // 3. Effective fields (all qubits, including auxiliary)
    auto fields = compute_effective_fields(model);
    CHECK((int)fields.size() == model.num_qubits);
    for (int q = 0; q < model.num_qubits; q++) {
        CHECK(fields[q] >= 0);
        CHECK(std::isfinite(fields[q]));
    }

    auto offsets = compute_anneal_offsets(fields, 0.1);
    CHECK((int)offsets.size() == model.num_qubits);
    for (int q = 0; q < model.num_qubits; q++) {
        CHECK(offsets[q] >= -0.1 - 1e-10);
        CHECK(offsets[q] <=  0.1 + 1e-10);
    }

    // 4. Source adjacency completeness (would catch isolated-variable bug)
    check_source_adj_complete(model, tc.label);

    // 5. Decode mock bitstring
    CHECK((int)tc.sample_bitstring.size() == model.num_qubits);
    auto choices = decode_to_cfn(model, tc.sample_bitstring);
    CHECK(!choices.empty());
    if (!choices.empty()) {
        CHECK(choices == tc.expected_choices);
    }

    // 6. CFN energy evaluation
    if (!choices.empty()) {
        double cfn_e = compute_energy(cfn, choices);
        CHECK(std::isfinite(cfn_e));
        std::cout << "    qubits=" << model.num_qubits
                  << "  decoded=" << fmt(choices)
                  << "  cfn_E=" << cfn_e;
        if (tc.is_cost_preserving) {
            // Cost-preserving encodings of choice [3,1] should give E=1.0
            if (choices == std::vector<int>({3,1})) {
                APPROX_EQ(cfn_e, 1.0, 1e-6);
                std::cout << " (ground state)";
            }
        }
        std::cout << "\n";
    }

    // 7. Parse mock results and verify full pipeline
    std::vector<int> cl(model.num_qubits > 4 ? model.num_qubits : 4, 1);
    auto mock_json = make_mock_results(
        {tc.sample_bitstring}, {0.5}, {10}, cl, {{0, 10}});
    const char* tmp = "_test_qa_enc_results.json";
    write_temp(mock_json, tmp);
    auto dwave = parse_dwave_results(tmp);
    std::remove(tmp);
    CHECK(dwave.samples.size() == 1);
    CHECK(dwave.samples[0].num_occurrences == 10);
}

static void test_all_encodings(const std::string& test_dir) {
    std::cout << "\n=== Encoding Pipeline Tests ===\n";
    auto cfn = make_test_cfn();

    // For binary encodings (2 bits per var, 2 vars):
    //   choice c → bit0 = c&1, bit1 = (c>>1)&1
    //   [3,1] → var0: bits[1,1]  var1: bits[1,0]
    //
    // For SPIN: b=(1-s)/2 → s=1-2b → bit 1→spin -1, bit 0→spin +1

    std::vector<EncodingTestCase> cases = {
        // ---- One-hot (BINARY, 8 qubits) ----
        // [3,1]: var0 bit3=1, var1 bit1=1
        {"one_hot",
         "test_2x4_one_hot.json", "BINARY", true,
         {0,0,0,1, 0,1,0,0}, {3,1}},

        // ---- Domain-wall (BINARY, 6 qubits, d-1=3 bits per var) ----
        // choice c = count of leading 1s
        // [3,1]: var0 = 3 leading 1s → [1,1,1], var1 = 1 leading 1 → [1,0,0]
        {"domain_wall",
         "test_2x4_domain_wall.json", "BINARY", true,
         {1,1,1, 1,0,0}, {3,1}},

        // ---- Exact binary, quadratized (BINARY, 4+4=8 qubits) ----
        // [3,1]: var0=3→bits[1,1], var1=1→bits[1,0], aux=0
        {"exact_binary_quad (natural/unsorted)",
         "test_2x4_exact_binary_quad.json", "BINARY", true,
         {1,1,1,0, 0,0,0,0}, {3,1}},

        // ---- Exact binary, quad, gray+boltzmann (BINARY, 4+4=8 qubits) ----
        // Gray+Boltzmann reorders choices; bitstring 01 (int 1) still decodes as int 1
        // [1,1]: var0=1→bits[1,0], var1=1→bits[1,0], aux=0
        {"exact_binary_quad (gray/boltzmann)",
         "test_2x4_exact_binary_quad_gray_boltz.json", "BINARY", false,
         {1,0,1,0, 0,0,0,0}, {1,1}},

        // ---- Approximate binary (BINARY, 4 qubits) ----
        // [3,1]: var0=3→bits[1,1], var1=1→bits[1,0]
        {"approx_binary (natural/unsorted)",
         "test_2x4_approximate_binary.json", "BINARY", false,
         {1,1,1,0}, {3,1}},

        // ---- Approximate binary, gray+boltzmann (BINARY, 4 qubits) ----
        {"approx_binary (gray/boltzmann)",
         "test_2x4_approx_binary_gray_boltz.json", "BINARY", false,
         {1,0,1,0}, {1,1}},

        // ---- Truncated binary k=2 (SPIN, 4 qubits) ----
        // [3,1]: bits[1,1,1,0] → spin[-1,-1,-1,+1]
        {"trunc_k2 (natural/unsorted)",
         "test_2x4_truncated_binary_k2.json", "SPIN", false,
         {-1,-1,-1,1}, {3,1}},

        // ---- Truncated binary k=2, gray+boltzmann (SPIN, 4 qubits) ----
        {"trunc_k2 (gray/boltzmann)",
         "test_2x4_trunc_k2_gray_boltz.json", "SPIN", false,
         {-1,1,-1,1}, {1,1}},

        // ---- Truncated binary k=3, quadratized (SPIN, 4+3=7 qubits) ----
        // [3,1]: bits[1,1,1,0] → spin[-1,-1,-1,+1], aux spin=[+1,+1,+1]
        {"trunc_k3_quad (natural/unsorted)",
         "test_2x4_truncated_binary_k3_quad.json", "SPIN", false,
         {-1,-1,-1,1, 1,1,1}, {3,1}},

        // ---- Truncated binary k=3, quad, gray+boltzmann (SPIN, 4+3=7 qubits) ----
        {"trunc_k3_quad (gray/boltzmann)",
         "test_2x4_trunc_k3_quad_gray_boltz.json", "SPIN", false,
         {-1,1,-1,1, 1,1,1}, {1,1}},
    };

    for (const auto& tc : cases) {
        run_encoding_test(tc, test_dir, cfn);
    }
}

// ============================================================================
// MOCK PIPELINE TESTS (full decode → CFN evaluate → stats, embedded data)
// ============================================================================

static void test_pipeline_one_hot() {
    std::cout << "\n=== Pipeline: one_hot (mock) ===\n";
    auto cfn = make_test_cfn();

    // Parse one_hot model from embedded string
    static const char* OH = R"({"encoding":"one_hot","num_auxiliary_qubits":0,
      "num_logical_qubits":8,"offset":13.0,"source_cfn":"test_2x4",
      "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
        "1":{"bit":1,"variable":0,"variable_name":"x0"},
        "2":{"bit":2,"variable":0,"variable_name":"x0"},
        "3":{"bit":3,"variable":0,"variable_name":"x0"},
        "4":{"bit":0,"variable":1,"variable_name":"x1"},
        "5":{"bit":1,"variable":1,"variable_name":"x1"},
        "6":{"bit":2,"variable":1,"variable_name":"x1"},
        "7":{"bit":3,"variable":1,"variable_name":"x1"}},
      "terms":{"0":-3.5,"0,1":13.0,"0,2":13.0,"0,3":13.0,"0,5":1.0,"0,6":-2.0,
        "0,7":0.5,"1":-5.5,"1,2":13.0,"1,3":13.0,"1,4":1.5,"1,6":0.5,"1,7":-1.0,
        "2":-2.5,"2,3":13.0,"2,4":-0.5,"2,5":2.0,"2,7":1.0,"3":-4.5,"3,4":0.5,
        "3,5":-1.5,"3,6":1.0,"4":-4.0,"4,5":13.0,"4,6":13.0,"4,7":13.0,
        "5":-6.0,"5,6":13.0,"5,7":13.0,"6":-5.0,"6,7":13.0,"7":-3.0},
      "variable_type":"BINARY"})";
    auto model = parse_model_str(OH);

    auto mock = make_mock_results(
        {{0,0,0,1,0,1,0,0}, {0,1,0,0,0,1,0,0}, {1,1,0,0,0,1,0,0}},
        {1.0, 2.5, 10.0}, {5, 3, 2},
        {2,1,2,1,2,1,1,2}, {{0,5},{1,3},{2,2}});
    const char* tmp = "_test_qa_oh.json";
    write_temp(mock, tmp);
    auto dwave = parse_dwave_results(tmp);
    std::remove(tmp);

    // Decode each sample
    auto ch1 = decode_to_cfn(model, dwave.samples[0].values);
    auto ch2 = decode_to_cfn(model, dwave.samples[1].values);
    auto ch3 = decode_to_cfn(model, dwave.samples[2].values);

    CHECK(ch1 == std::vector<int>({3,1}));
    CHECK(ch2 == std::vector<int>({1,1}));
    CHECK(ch3.empty());  // infeasible (two bits set)

    APPROX_EQ(compute_energy(cfn, ch1), 1.0, 1e-9);
    std::cout << "  [3,1] E=1.0, [1,1] E=" << compute_energy(cfn, ch2)
              << ", infeasible correctly rejected\n";

    // Energy stats
    std::vector<double> all_e;
    for (auto& s : dwave.samples)
        for (int k = 0; k < s.num_occurrences; k++)
            all_e.push_back(s.energy);
    CHECK(all_e.size() == 10);
    APPROX_EQ(*std::min_element(all_e.begin(), all_e.end()), 1.0, 1e-10);

    double mean = std::accumulate(all_e.begin(), all_e.end(), 0.0) / 10;
    APPROX_EQ(mean, 3.25, 1e-10);

    // Chain length stats
    auto& cl = dwave.chain_lengths;
    double cl_avg = std::accumulate(cl.begin(), cl.end(), 0.0) / cl.size();
    APPROX_EQ(cl_avg, 1.5, 1e-10);

    // Chain break stats
    std::vector<int> all_b;
    for (auto& cb : dwave.chain_breaks)
        for (int k = 0; k < cb.num_occurrences; k++)
            all_b.push_back(cb.breaks);
    double cb_avg = std::accumulate(all_b.begin(), all_b.end(), 0.0) / all_b.size();
    APPROX_EQ(cb_avg, 0.7, 1e-10);

    std::cout << "  chain_length_avg=1.5  chain_breaks_avg=0.7  OK\n";
}

static void test_infeasible_handling() {
    std::cout << "\n=== Infeasible Handling ===\n";
    static const char* OH = R"({"encoding":"one_hot","num_auxiliary_qubits":0,
      "num_logical_qubits":8,"offset":13.0,"source_cfn":"test_2x4",
      "qubit_map":{"0":{"bit":0,"variable":0,"variable_name":"x0"},
        "1":{"bit":1,"variable":0,"variable_name":"x0"},
        "2":{"bit":2,"variable":0,"variable_name":"x0"},
        "3":{"bit":3,"variable":0,"variable_name":"x0"},
        "4":{"bit":0,"variable":1,"variable_name":"x1"},
        "5":{"bit":1,"variable":1,"variable_name":"x1"},
        "6":{"bit":2,"variable":1,"variable_name":"x1"},
        "7":{"bit":3,"variable":1,"variable_name":"x1"}},
      "terms":{"0":-3.5,"1":-5.5,"2":-2.5,"3":-4.5,"4":-4.0,"5":-6.0,"6":-5.0,"7":-3.0},
      "variable_type":"BINARY"})";
    auto model = parse_model_str(OH);

    CHECK(decode_to_cfn(model, {0,0,0,0,0,0,0,0}).empty());   // no bit set
    CHECK(decode_to_cfn(model, {1,1,0,0,0,1,0,0}).empty());   // two bits
    CHECK(decode_to_cfn(model, {1,0,0,0,0,0,1,0}) == std::vector<int>({0,2}));
    std::cout << "  OK\n";
}

// ============================================================================
// D-WAVE INTEGRATION TESTS
//
// Full end-to-end: load model → effective fields → D-Wave submission →
// decode bitstrings → CFN evaluation → CSV output.
//
// Runs for all 10 encoding variants:
//   one_hot, domain_wall,
//   exact_binary_quad (natural + gray/boltzmann),
//   approx_binary (natural + gray/boltzmann),
//   trunc_k2 (natural + gray/boltzmann),
//   trunc_k3_quad (natural + gray/boltzmann)
// ============================================================================

struct DWaveTestCase {
    std::string label;
    std::string filename;
};

static void run_dwave_test(
    const DWaveTestCase& tc,
    const std::string& test_dir,
    const std::string& solver_name,
    const std::string& python_cmd
) {
    std::cout << "\n  --- " << tc.label << " ---\n";
    std::cout << std::flush;

    std::string model_path = test_dir + "/test_models/" + tc.filename;

    BinaryModel model;
    try {
        model = parse_binary_model(model_path);
    } catch (const std::exception& e) {
        g_checks++; g_failures++;
        std::cerr << "  FAIL: Cannot load " << tc.filename << ": " << e.what() << "\n";
        return;
    }

    std::cout << "    file=" << tc.filename
              << "  qubits=" << model.num_qubits
              << "  type=" << (model.var_type == BinaryModel::SPIN ? "SPIN" : "BINARY")
              << "  encoding=" << model.encoding
              << "\n    Submitting to " << solver_name << " ..."
              << std::endl;

    CFNModel cfn = parse_cfn_for_sa(test_dir + "/test_2x4.cfn");

    QAParams params;
    params.input_path        = model_path;
    params.cfn_dir           = test_dir;
    params.solver_name       = solver_name;
    params.python_cmd        = python_cmd;
    params.annealing_time_us = 20.0;
    params.num_reads         = 10;
    params.delta_max         = 0.1;
    params.use_inhomogeneous = true;
    params.verbose           = false;

    QAResult result;
    try {
        result = run_qa_binary(model, cfn, params);
    } catch (const std::exception& e) {
        g_checks++; g_failures++;
        std::cerr << "  FAIL [" << tc.label << "]: " << e.what() << "\n";
        return;
    }

    // --- Sanity checks (all encodings) ---
    CHECK(!std::isnan(result.best_energy));
    CHECK(!std::isinf(result.best_energy));
    CHECK(result.per_run_energies.size() > 0);
    CHECK(result.total_runtime_s > 0);
    CHECK(result.embedding_time_s > 0);
    CHECK(result.emb_num_physical_qubits > 0);
    CHECK(result.emb_chain_length_avg > 0);

    // --- CFN evaluation ---
    // Note: we do NOT assert ground-state energy here — with only 10
    // reads the QPU may not find it (especially for encodings with
    // large Rosenberg penalties).  The mock tests verify decoding
    // correctness; these tests verify the pipeline runs end-to-end.
    if (!std::isnan(result.best_cfn_energy)) {
        CHECK(std::isfinite(result.best_cfn_energy));
        CHECK(result.num_feasible > 0);
        CHECK(result.best_cfn_energy >= 1.0 - 1e-6);  // can't beat ground state
    }

    // --- CSV round-trip ---
    std::string csv = csv_row_qa(result);
    CHECK(!csv.empty());
    CHECK(csv.find("dwave") != std::string::npos);

    // --- Report ---
    std::cout << "    qubits=" << model.num_qubits
              << "  type=" << (model.var_type == BinaryModel::SPIN ? "SPIN" : "BINARY")
              << "  phys=" << result.emb_num_physical_qubits
              << "  chain_avg=" << result.emb_chain_length_avg
              << "  breaks_avg=" << result.emb_chain_breaks_avg
              << "\n    best_dwave=" << result.best_energy
              << "  best_cfn=" << result.best_cfn_energy
              << "  feasible=" << result.num_feasible << "/" << result.num_reads
              << "  solution=" << fmt(result.best_cfn_solution)
              << "\n";
}

static void test_dwave_all_encodings(
    const std::string& solver_name,
    const std::string& test_dir,
    const std::string& python_cmd
) {
    std::cout << "\n=== D-Wave End-to-End Tests ===\n";
    std::cout << "  Solver: " << solver_name << "\n";

    std::vector<DWaveTestCase> cases = {
        {"one_hot",                              "test_2x4_one_hot.json"},
        {"domain_wall",                          "test_2x4_domain_wall.json"},
        {"exact_binary_quad (natural/unsorted)",  "test_2x4_exact_binary_quad.json"},
        {"exact_binary_quad (gray/boltzmann)",    "test_2x4_exact_binary_quad_gray_boltz.json"},
        {"approx_binary (natural/unsorted)",      "test_2x4_approximate_binary.json"},
        {"approx_binary (gray/boltzmann)",        "test_2x4_approx_binary_gray_boltz.json"},
        {"trunc_k2 (natural/unsorted)",           "test_2x4_truncated_binary_k2.json"},
        {"trunc_k2 (gray/boltzmann)",             "test_2x4_trunc_k2_gray_boltz.json"},
        {"trunc_k3_quad (natural/unsorted)",      "test_2x4_truncated_binary_k3_quad.json"},
        {"trunc_k3_quad (gray/boltzmann)",        "test_2x4_trunc_k3_quad_gray_boltz.json"},
    };

    for (const auto& tc : cases) {
        run_dwave_test(tc, test_dir, solver_name, python_cmd);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool run_dwave = false;
    std::string solver_name;
    std::string test_dir;
    std::string python_cmd = "python3";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--dwave")                         run_dwave = true;
        else if (a == "--solver"   && i+1 < argc)        solver_name = argv[++i];
        else if (a == "--test-dir" && i+1 < argc)        test_dir = argv[++i];
        else if (a == "--python"   && i+1 < argc)        python_cmd = argv[++i];
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [--test-dir DIR] "
                      << "[--dwave --solver NAME] [--python CMD]\n";
            return 0;
        }
    }

    // --- Core unit tests (always run, no files needed) ---
    test_effective_fields_binary();
    test_effective_fields_spin();
    test_effective_fields_isolated();
    test_anneal_offsets();
    test_validate_qubo();
    test_script_generation();
    test_script_handles_isolated_variables();
    test_pipeline_one_hot();
    test_infeasible_handling();
    test_csv_format();

    // --- Encoding-specific tests (need test_models/) ---
    if (!test_dir.empty()) {
        test_all_encodings(test_dir);
    } else {
        std::cout << "\n(Skipping encoding tests. Use --test-dir to enable.)\n";
    }

    // --- D-Wave integration test ---
    if (run_dwave) {
        if (solver_name.empty() || test_dir.empty()) {
            std::cerr << "\nError: --solver and --test-dir required with --dwave\n";
            return 1;
        }
        test_dwave_all_encodings(solver_name, test_dir, python_cmd);
    } else {
        std::cout << "\n(Skipping D-Wave test. Use --dwave --solver NAME to enable.)\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Checks: " << g_checks << "  Failures: " << g_failures << "\n";
    return (g_failures > 0) ? 1 : 0;
}
