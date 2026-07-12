// ============================================================================
// Unit tests for solver_tools: simulated annealing on CFN and encoded models.
//
// Test CFN (2 variables, 4 choices each):
//   unary_x0 = [3, 1, 4, 2],  unary_x1 = [2.5, 0.5, 1.5, 3.5]
//   pw[c0,c1] = row-major 4x4
//
// Ground state (by exhaustive enumeration): E(3,1) = 1.0
//
// Cost-preserving encodings: best SA energy must equal CFN ground state.
// Cost-approximate encodings: decode best binary solution to CFN choices,
//   report both the encoded energy and the native CFN energy.
// ============================================================================

#include "baseline/sa_types.hpp"
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/sa_binary.hpp"
#include "solvers/sa_cfn.hpp"
#include "solvers/temperature.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

// ============================================================================
// Simple test framework
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
// Construct the 2x4 test CFN directly (no file I/O)
// ============================================================================
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

// ============================================================================
// Brute-force ground state of a CFN
// ============================================================================
struct GroundState {
    double energy;
    std::vector<int> assignment;
};

static GroundState brute_force_cfn(const CFNModel& cfn) {
    int N = cfn.num_variables;
    std::vector<int> best_assign(N, 0);
    double best_E = 1e30;

    // Enumerate all assignments
    std::vector<int> state(N, 0);
    while (true) {
        double E = compute_energy(cfn, state);
        if (E < best_E) {
            best_E = E;
            best_assign = state;
        }
        // Increment state (odometer)
        int pos = 0;
        while (pos < N) {
            state[pos]++;
            if (state[pos] < cfn.cardinalities[pos]) break;
            state[pos] = 0;
            pos++;
        }
        if (pos == N) break;
    }
    return {best_E, best_assign};
}

// ============================================================================
// Evaluate a CFN assignment using the test CFN
// ============================================================================
static double eval_cfn(const CFNModel& cfn, const std::vector<int>& choices) {
    return compute_energy(cfn, choices);
}

// ============================================================================
// Parse BinaryModel from a JSON string (writes to temp file, parses, deletes)
// ============================================================================
static BinaryModel parse_model_str(const std::string& json_str) {
    const char* tmp = "_test_model_tmp.json";
    { std::ofstream ofs(tmp); ofs << json_str; }
    auto model = parse_binary_model(tmp);
    std::remove(tmp);
    return model;
}

// ============================================================================
// Format solution as [X0, X1, ...]
// ============================================================================
static std::string fmt(const std::vector<int>& v) {
    std::string s = "[";
    for (int i = 0; i < (int)v.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

// ============================================================================
// SA parameters for testing: high step count for reliable convergence
// ============================================================================
static SAParams test_params(const std::string& sched = "geometric",
                            const std::string& move = "flip") {
    SAParams p;
    p.schedule = sched;
    p.move_type = move;
    p.T_start = 10.0;
    p.T_end = 0.001;
    p.num_runs = 100;
    p.num_steps = 50000;
    p.seed = 42;
    return p;
}

// ============================================================================
// Run SA on a BinaryModel, decode best solution, return results
// ============================================================================
struct BinaryTestResult {
    double best_encoded_energy;      // best energy under the encoding
    std::vector<int> best_state;     // raw binary/spin state
    std::vector<int> cfn_choices;    // decoded CFN choices (empty if infeasible)
    double cfn_energy;               // energy under original CFN (NaN if infeasible)
};

static BinaryTestResult run_binary_test(const BinaryModel& model,
                                         const CFNModel& cfn,
                                         const SAParams& params) {
    BinaryTestResult res;
    res.cfn_energy = std::numeric_limits<double>::quiet_NaN();

    // Run all SA runs
    double best_E = 1e30;
    for (int r = 0; r < params.num_runs; r++) {
        RunResult rr = run_sa_binary(model, params, params.seed + r);
        if (rr.best_energy < best_E) {
            best_E = rr.best_energy;
            res.best_state = rr.best_state;
        }
    }
    res.best_encoded_energy = best_E;

    // Decode to CFN choices
    res.cfn_choices = decode_to_cfn(model, res.best_state);
    if (!res.cfn_choices.empty())
        res.cfn_energy = eval_cfn(cfn, res.cfn_choices);

    return res;
}

// ============================================================================
// JSON string literals for each test model
// (generated by encoding_tools with unsorted choice ordering, natural bitstring
//  ordering on the 2x4 test CFN)
// ============================================================================

static const char* JSON_ONE_HOT = R"({
  "encoding": "one_hot",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 8,
  "offset": 13.0,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 2, "variable": 0, "variable_name": "x0"},
    "3": {"bit": 3, "variable": 0, "variable_name": "x0"},
    "4": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "5": {"bit": 1, "variable": 1, "variable_name": "x1"},
    "6": {"bit": 2, "variable": 1, "variable_name": "x1"},
    "7": {"bit": 3, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": -3.5, "0,1": 13.0, "0,2": 13.0, "0,3": 13.0, "0,5": 1.0, "0,6": -2.0, "0,7": 0.5,
    "1": -5.5, "1,2": 13.0, "1,3": 13.0, "1,4": 1.5, "1,6": 0.5, "1,7": -1.0,
    "2": -2.5, "2,3": 13.0, "2,4": -0.5, "2,5": 2.0, "2,7": 1.0,
    "3": -4.5, "3,4": 0.5, "3,5": -1.5, "3,6": 1.0,
    "4": -4.0, "4,5": 13.0, "4,6": 13.0, "4,7": 13.0,
    "5": -6.0, "5,6": 13.0, "5,7": 13.0,
    "6": -5.0, "6,7": 13.0,
    "7": -3.0
  },
  "variable_type": "BINARY"
})";

static const char* JSON_DOMAIN_WALL = R"({
  "encoding": "domain_wall",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 6,
  "offset": 5.5,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 2, "variable": 0, "variable_name": "x0"},
    "3": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "4": {"bit": 1, "variable": 1, "variable_name": "x1"},
    "5": {"bit": 2, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": -0.5, "0,1": -5.0, "0,3": -2.5, "0,4": 3.5, "0,5": -4.0,
    "1": 6.0, "1,2": -5.0, "1,3": 4.0, "1,4": -2.5, "1,5": 2.5,
    "2": 4.0, "2,3": -4.5, "2,4": 4.5, "2,5": -2.0,
    "3": -1.0, "3,4": -5.0,
    "4": 3.0, "4,5": -5.0,
    "5": 9.5
  },
  "variable_type": "BINARY"
})";

static const char* JSON_EXACT_BINARY = R"({
  "encoding": "exact_binary",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 4,
  "offset": 5.5,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": -0.5, "0,1": -0.5, "0,1,2": -2.0, "0,1,2,3": 4.0, "0,1,3": -1.0,
    "0,2": -2.5, "0,2,3": -1.5, "0,3": 1.0,
    "1": 0.5, "1,2": 1.5, "1,2,3": -3.0, "1,3": 2.5,
    "2": -1.0, "2,3": 5.5,
    "3": -3.0
  },
  "variable_type": "BINARY"
})";

static const char* JSON_EXACT_BINARY_QUAD = R"({
  "encoding": "exact_binary",
  "num_auxiliary_qubits": 4,
  "num_logical_qubits": 4,
  "offset": 5.5,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": -0.5, "0,1": 23.5, "0,2": 5.5, "0,3": 1.0, "0,5": -16.0, "0,6": -48.0,
    "1": 0.5, "1,2": 9.5, "1,3": 2.5, "1,4": -16.0, "1,6": -48.0,
    "2": -1.0, "2,3": 13.5, "2,4": -16.0, "2,5": -16.0, "2,6": -2.0, "2,7": -16.0,
    "3": -3.0, "3,4": -3.0, "3,5": -1.5, "3,6": -1.0, "3,7": -16.0,
    "4": 24.0, "5": 24.0, "6": 72.0, "6,7": 4.0, "7": 24.0
  },
  "variable_type": "BINARY"
})";

static const char* JSON_APPROXIMATE_BINARY = R"({
  "encoding": "approximate_binary",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 4,
  "offset": 5.762499999999998,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": -1.9999999999999987, "0,2": -1.4500000000000013, "0,3": 0.3833333333333336,
    "1": 0.999999999999999, "1,2": 0.7166666666666672, "1,3": 0.050000000000000155,
    "2": -1.9999999999999982, "2,3": 4.0,
    "3": -0.9999999999999991
  },
  "variable_type": "BINARY"
})";

static const char* JSON_TRUNCATED_K2 = R"({
  "encoding": "truncated_binary",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 4,
  "offset": 4.6875,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": 1.0625, "0,1": -0.25, "0,2": -0.8125, "0,3": 0.1875,
    "1": -0.625, "1,3": 0.375,
    "2": -0.0625, "2,3": 1.0625,
    "3": -0.3125
  },
  "variable_type": "SPIN"
})";

static const char* JSON_TRUNCATED_K3 = R"({
  "encoding": "truncated_binary",
  "num_auxiliary_qubits": 0,
  "num_logical_qubits": 4,
  "offset": 4.6875,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": 1.0625, "0,1": -0.25, "0,1,3": -0.125,
    "0,2": -0.8125, "0,2,3": -0.0625, "0,3": 0.1875,
    "1": -0.625, "1,2,3": 0.125, "1,3": 0.375,
    "2": -0.0625, "2,3": 1.0625,
    "3": -0.3125
  },
  "variable_type": "SPIN"
})";

static const char* JSON_TRUNCATED_K3_QUAD = R"({
  "encoding": "truncated_binary",
  "num_auxiliary_qubits": 3,
  "num_logical_qubits": 4,
  "offset": 5.25,
  "qubit_map": {
    "0": {"bit": 0, "variable": 0, "variable_name": "x0"},
    "1": {"bit": 1, "variable": 0, "variable_name": "x0"},
    "2": {"bit": 0, "variable": 1, "variable_name": "x1"},
    "3": {"bit": 1, "variable": 1, "variable_name": "x1"}
  },
  "source_cfn": "test_2x4",
  "terms": {
    "0": 1.1875, "0,1": -0.1875, "0,2": -0.75, "0,3": 0.1875, "0,5": -0.125, "0,6": -0.125,
    "1": -0.5, "1,2": 0.0625, "1,3": 0.375, "1,4": -0.125, "1,5": -0.125,
    "2": 0.0625, "2,3": 1.0625, "2,4": -0.125, "2,6": -0.125,
    "3": -0.3125, "3,4": 0.125, "3,5": -0.125, "3,6": -0.0625,
    "4": -0.125, "5": -0.125, "6": -0.125
  },
  "variable_type": "SPIN"
})";

// ============================================================================
// Test: CFN solver (flip, shift, both)
// ============================================================================
static void test_cfn_solver() {
    std::cout << "=== CFN Solver Tests ===\n";

    CFNModel cfn = make_test_cfn();
    auto gs = brute_force_cfn(cfn);

    std::cout << "  Ground state: E=" << gs.energy
              << " at " << fmt(gs.assignment) << "\n";
    APPROX_EQ(gs.energy, 1.0, 1e-10);
    CHECK(gs.assignment[0] == 3);
    CHECK(gs.assignment[1] == 1);

    // Test flip moves
    {
        SAParams p = test_params("geometric", "flip");
        double best_E = 1e30;
        std::vector<int> best_state;
        for (int r = 0; r < p.num_runs; r++) {
            RunResult rr = run_sa_cfn(cfn, p, p.seed + r);
            if (rr.best_energy < best_E) {
                best_E = rr.best_energy;
                best_state = rr.best_state;
            }
        }
        std::cout << "  flip:  best_E=" << best_E
                  << " solution=" << fmt(best_state) << "\n";
        APPROX_EQ(best_E, gs.energy, 1e-10);
        CHECK(best_state == gs.assignment);
    }

    // Test shift moves
    {
        SAParams p = test_params("geometric", "shift");
        double best_E = 1e30;
        std::vector<int> best_state;
        for (int r = 0; r < p.num_runs; r++) {
            RunResult rr = run_sa_cfn(cfn, p, p.seed + r);
            if (rr.best_energy < best_E) {
                best_E = rr.best_energy;
                best_state = rr.best_state;
            }
        }
        std::cout << "  shift: best_E=" << best_E
                  << " solution=" << fmt(best_state) << "\n";
        APPROX_EQ(best_E, gs.energy, 1e-10);
        CHECK(best_state == gs.assignment);
    }

    // Test both moves
    {
        SAParams p = test_params("geometric", "both");
        double best_E = 1e30;
        std::vector<int> best_state;
        for (int r = 0; r < p.num_runs; r++) {
            RunResult rr = run_sa_cfn(cfn, p, p.seed + r);
            if (rr.best_energy < best_E) {
                best_E = rr.best_energy;
                best_state = rr.best_state;
            }
        }
        std::cout << "  both:  best_E=" << best_E
                  << " solution=" << fmt(best_state) << "\n";
        APPROX_EQ(best_E, gs.energy, 1e-10);
        CHECK(best_state == gs.assignment);
    }
}

// ============================================================================
// Test: Cost-preserving binary encoding
// ============================================================================
static void test_cost_preserving(const char* label, const char* json,
                                  const CFNModel& cfn,
                                  const GroundState& gs) {
    BinaryModel model = parse_model_str(json);
    SAParams p = test_params();
    auto res = run_binary_test(model, cfn, p);

    std::cout << "  " << label << ": encoded_E=" << res.best_encoded_energy;
    if (!res.cfn_choices.empty())
        std::cout << "  cfn_E=" << res.cfn_energy
                  << "  solution=" << fmt(res.cfn_choices);
    else
        std::cout << "  (infeasible)";
    std::cout << "\n";

    APPROX_EQ(res.best_encoded_energy, gs.energy, 1e-6);
    CHECK(!res.cfn_choices.empty());
    if (!res.cfn_choices.empty()) {
        APPROX_EQ(res.cfn_energy, gs.energy, 1e-6);
        CHECK(res.cfn_choices == gs.assignment);
    }
}

// ============================================================================
// Test: Cost-approximate binary encoding
// ============================================================================
static void test_cost_approximate(const char* label, const char* json,
                                   const CFNModel& cfn,
                                   const GroundState& gs) {
    BinaryModel model = parse_model_str(json);
    SAParams p = test_params();
    auto res = run_binary_test(model, cfn, p);

    std::cout << "  " << label << ":"
              << " encoded_E=" << res.best_encoded_energy;
    if (!res.cfn_choices.empty()) {
        std::cout << "  cfn_E=" << res.cfn_energy
                  << "  solution=" << fmt(res.cfn_choices);
        if (std::abs(res.cfn_energy - gs.energy) < 1e-6)
            std::cout << "  ** matches CFN ground state **";
    } else {
        std::cout << "  (infeasible)";
    }
    std::cout << "\n";

    // Basic sanity checks (we don't require exact ground state for approx)
    CHECK(!res.cfn_choices.empty());
    if (!res.cfn_choices.empty()) {
        // Decoded choices must be in valid range
        for (int i = 0; i < (int)res.cfn_choices.size(); i++)
            CHECK(res.cfn_choices[i] >= 0 &&
                  res.cfn_choices[i] < cfn.cardinalities[i]);
        // CFN energy of decoded solution must be finite
        CHECK(std::isfinite(res.cfn_energy));
    }
}

// ============================================================================
// Test: enhanced binary decode (Gray / Boltzmann / LI assignments)
//
// The encoder emits "choice_to_bitstring" so the solver can invert non-identity
// layouts. This builds models with a deliberately permuted assignment and
// verifies that decode_to_cfn:
//   (a) recovers the original choices for every register value (round-trip),
//   (b) reports unused bitstrings as infeasible,
//   (c) needs the field: without it, natural-binary decode mis-decodes.
// Covered for both BINARY (exact/approximate) and SPIN (truncated) models.
// ============================================================================
static void test_enhanced_binary_decode() {
    std::cout << "\n=== Enhanced Binary Decode Round-Trip ===\n";

    // 2 variables, 2 bits each. Non-identity choice->bitstring maps:
    //   var0 (cardinality 4): choices 0..3 -> bitstrings [0,1,3,2]  (Gray order)
    //   var1 (cardinality 3): choices 0..2 -> bitstrings [2,0,1]    (Boltzmann);
    //                         bitstring 3 is unused -> must decode infeasible.
    std::vector<std::vector<int>> c2b = {{0, 1, 3, 2}, {2, 0, 1}};
    std::vector<int> card = {4, 3};
    const int starts[2] = {0, 2};   // var0 -> qubits 0,1 ; var1 -> qubits 2,3

    auto build_json = [&](const std::string& encoding, bool spin,
                          bool include_field) -> std::string {
        nlohmann::json j;
        j["encoding"]             = encoding;
        j["variable_type"]        = spin ? "SPIN" : "BINARY";
        j["num_logical_qubits"]   = 4;
        j["num_auxiliary_qubits"] = 0;
        j["offset"]               = 0.0;
        j["source_cfn"]           = "enh_test";
        j["terms"]                = nlohmann::json::object();
        nlohmann::json qm = nlohmann::json::object();
        qm["0"] = {{"variable", 0}, {"bit", 0}, {"variable_name", "x0"}};
        qm["1"] = {{"variable", 0}, {"bit", 1}, {"variable_name", "x0"}};
        qm["2"] = {{"variable", 1}, {"bit", 0}, {"variable_name", "x1"}};
        qm["3"] = {{"variable", 1}, {"bit", 1}, {"variable_name", "x1"}};
        j["qubit_map"] = qm;
        if (include_field) j["choice_to_bitstring"] = c2b;
        return j.dump();
    };

    // bit value -> raw state element (SPIN: 0->+1, 1->-1 per decode convention)
    auto bit_to_state = [](int bit, bool spin) { return spin ? (1 - 2 * bit) : bit; };

    auto state_for_choices = [&](const std::vector<int>& choices, bool spin) {
        std::vector<int> state(4, bit_to_state(0, spin));
        for (int i = 0; i < 2; i++) {
            int bs = c2b[i][choices[i]];
            for (int b = 0; b < 2; b++)
                state[starts[i] + b] = bit_to_state((bs >> b) & 1, spin);
        }
        return state;
    };

    // Cover all three binary encodings: exact_binary + approximate_binary are
    // BINARY, truncated_binary is SPIN. The decode path is encoding-agnostic
    // (it inverts choice_to_bitstring), so every one must round-trip.
    struct DecodeCase { const char* enc; bool spin; };
    for (DecodeCase dc : { DecodeCase{"exact_binary", false},
                           DecodeCase{"approximate_binary", false},
                           DecodeCase{"truncated_binary", true} }) {
        std::string enc = dc.enc;
        bool spin = dc.spin;
        BinaryModel model = parse_model_str(build_json(enc, spin, true));
        CHECK(!model.bitstring_to_choice.empty());

        // (a) every choice vector round-trips through decode_to_cfn
        bool all_ok = true;
        for (int c0 = 0; c0 < card[0]; c0++)
            for (int c1 = 0; c1 < card[1]; c1++) {
                std::vector<int> choices = {c0, c1};
                auto decoded = decode_to_cfn(model, state_for_choices(choices, spin));
                if (decoded != choices) all_ok = false;
            }
        CHECK(all_ok);

        // (b) unused bitstring (var1 == 11 == 3) decodes as infeasible
        {
            std::vector<int> state(4, bit_to_state(0, spin));
            state[2] = bit_to_state(1, spin);
            state[3] = bit_to_state(1, spin);
            CHECK(decode_to_cfn(model, state).empty());
        }

        // (c) the field is what fixes it: identical bits, no field -> wrong
        {
            BinaryModel naive = parse_model_str(build_json(enc, spin, false));
            CHECK(naive.bitstring_to_choice.empty());
            std::vector<int> choices = {2, 0};  // var0 ch2->bs3, var1 ch0->bs2
            auto state   = state_for_choices(choices, spin);
            auto correct = decode_to_cfn(model, state);
            auto wrong   = decode_to_cfn(naive, state);
            CHECK(correct == choices);
            CHECK(wrong != choices);            // natural-binary decode mis-decodes
        }

        std::cout << "  " << enc << (spin ? " (SPIN)  " : " (BINARY)")
                  << ": round-trip OK\n";
    }
}

// ============================================================================
// Test: SA + decode recovers the CFN optimum for an ENHANCED model
//
// Builds a 2-var cardinality-2 CFN and a cost-preserving exact_binary model
// under a PERMUTED (enhanced-style) assignment, so the binary minimum decodes
// to the CFN optimum only when choice_to_bitstring is inverted. This exercises
// the same SA->decode->CFN-eval path solve_sa now uses for best_cfn_energy.
// Coefficients are derived by the Mobius transform of the decoded-cost table,
// so the model is cost-preserving by construction (no hand-tuned arithmetic).
// ============================================================================
static void test_sa_decode_enhanced_optimum() {
    std::cout << "\n=== Enhanced SA Decode -> CFN Optimum ===\n";

    CFNModel cfn;
    cfn.name = "enh_cp"; cfn.num_variables = 2; cfn.var_names = {"x0", "x1"};
    cfn.cardinalities = {2, 2};
    cfn.unary = {{1.0, -2.0}, {0.5, 0.5}};
    CFNModel::PairwiseTable pt; pt.vi = 0; pt.vj = 1; pt.di = 2; pt.dj = 2;
    pt.costs = {0.0, 3.0, 3.0, 0.0};               // favours c0 == c1
    cfn.pairwise.push_back(pt); cfn.build_adjacency();
    GroundState gs = brute_force_cfn(cfn);          // optimum: [1,1], E = -1.5

    // Enhanced assignment: var0 swapped (choice0->bs1, choice1->bs0); var1 natural.
    std::vector<std::vector<int>> c2b = {{1, 0}, {0, 1}};
    auto choice_of = [&](int var, int bs) {
        for (int c = 0; c < 2; c++) if (c2b[var][c] == bs) return c; return -1;
    };
    // Decoded-cost table f(b0,b1) and its multilinear (Mobius) coefficients.
    auto fval = [&](int b0, int b1) {
        return eval_cfn(cfn, {choice_of(0, b0), choice_of(1, b1)});
    };
    double f00 = fval(0,0), f10 = fval(1,0), f01 = fval(0,1), f11 = fval(1,1);
    double cst = f00, a = f10 - f00, c = f01 - f00, d = f11 - f10 - f01 + f00;

    nlohmann::json j;
    j["encoding"] = "exact_binary"; j["variable_type"] = "BINARY";
    j["num_logical_qubits"] = 2; j["num_auxiliary_qubits"] = 0;
    j["offset"] = cst; j["source_cfn"] = "enh_cp";
    nlohmann::json terms = nlohmann::json::object();
    if (std::abs(a) > 1e-12) terms["0"]   = a;     // qubit0 = var0 bit0
    if (std::abs(c) > 1e-12) terms["1"]   = c;     // qubit1 = var1 bit0
    if (std::abs(d) > 1e-12) terms["0,1"] = d;
    j["terms"] = terms;
    nlohmann::json qm = nlohmann::json::object();
    qm["0"] = {{"variable",0},{"bit",0},{"variable_name","x0"}};
    qm["1"] = {{"variable",1},{"bit",0},{"variable_name","x1"}};
    j["qubit_map"] = qm;
    j["choice_to_bitstring"] = c2b;

    BinaryModel model = parse_model_str(j.dump());
    SAParams p = test_params();
    auto res = run_binary_test(model, cfn, p);

    std::cout << "  enhanced exact_binary: encoded_E=" << res.best_encoded_energy
              << "  cfn_E=" << res.cfn_energy << "  sol=" << fmt(res.cfn_choices)
              << "  (gs=" << gs.energy << " at " << fmt(gs.assignment) << ")\n";

    APPROX_EQ(res.best_encoded_energy, gs.energy, 1e-6);   // cost-preserving
    CHECK(!res.cfn_choices.empty());
    APPROX_EQ(res.cfn_energy, gs.energy, 1e-6);            // decode -> optimum
    CHECK(res.cfn_choices == gs.assignment);

    // A naive decode (field stripped) would mis-decode the same binary minimum.
    nlohmann::json j2 = j; j2.erase("choice_to_bitstring");
    BinaryModel naive = parse_model_str(j2.dump());
    auto res2 = run_binary_test(naive, cfn, p);
    CHECK(res2.cfn_choices != gs.assignment);
    std::cout << "  (naive decode of same model -> " << fmt(res2.cfn_choices)
              << ", correctly flagged as wrong)\n";
}

// ============================================================================
// Test: the incremental CFN tracker against a brute-force oracle.
//
// CFNTracker maintains the decoded CFN cost in O(deg) per accepted flip, across
// states whose registers decode infeasibly. The oracle re-decodes each visited
// state from scratch (decode_to_cfn + compute_energy) and takes the min over the
// feasible ones -- so a random walk must leave both with the same best.
//
// The walk covers every encoding family, because each stresses a different part
// of the update: one-hot breaks and restores feasibility on nearly every flip,
// domain-wall walks the wall, SPIN exercises the s -> b conversion, and the
// quadratized models carry Rosenberg auxiliaries that map to no variable and
// must leave the decode untouched.
// ============================================================================
static void test_cfn_tracker_vs_oracle() {
    std::cout << "\n=== Incremental CFN Tracker vs Brute-Force Oracle ===\n";

    CFNModel cfn = make_test_cfn();

    struct Case { const char* label; const char* json; };
    const Case cases[] = {
        {"one_hot          ", JSON_ONE_HOT},
        {"domain_wall      ", JSON_DOMAIN_WALL},
        {"exact_binary     ", JSON_EXACT_BINARY},
        {"exact_binary_quad", JSON_EXACT_BINARY_QUAD},
        {"approx_binary    ", JSON_APPROXIMATE_BINARY},
        {"trunc_k3_quad    ", JSON_TRUNCATED_K3_QUAD},
    };

    for (const auto& c : cases) {
        BinaryModel model = parse_model_str(c.json);
        const bool is_spin = (model.var_type == BinaryModel::SPIN);

        std::mt19937_64 rng(12345);
        std::uniform_int_distribution<int> qubit_dist(0, model.num_qubits - 1);

        std::vector<int> state(model.num_qubits);
        for (int q = 0; q < model.num_qubits; q++) {
            int b = static_cast<int>(rng() & 1);
            state[q] = is_spin ? (b * 2 - 1) : b;
        }

        CFNTracker tracker(model, cfn, state);

        // Oracle: min CFN cost over every visited state that decodes feasibly.
        double oracle = std::numeric_limits<double>::quiet_NaN();
        auto observe = [&](const std::vector<int>& s) {
            std::vector<int> ch = decode_to_cfn(model, s);
            if (ch.empty()) return;                 // infeasible: not a candidate
            double e = eval_cfn(cfn, ch);
            if (std::isnan(oracle) || e < oracle) oracle = e;
        };
        observe(state);

        for (int step = 0; step < 4000; step++) {
            int q = qubit_dist(rng);
            state[q] = is_spin ? -state[q] : 1 - state[q];
            tracker.on_flip(q, state);              // walk accepts every move
            observe(state);
        }

        double tracked = tracker.best_energy();
        bool agree = (std::isnan(oracle) && std::isnan(tracked)) ||
                     (std::abs(tracked - oracle) < 1e-9);
        CHECK(agree);

        // The reported cost must be the cost of the reported solution.
        if (!std::isnan(tracked)) {
            CHECK(std::abs(eval_cfn(cfn, tracker.best_choices()) - tracked) < 1e-9);
        }

        std::cout << "  " << c.label << ": tracker=" << tracked
                  << "  oracle=" << oracle
                  << (agree ? "  OK" : "  MISMATCH") << "\n";
    }
}

// ============================================================================
// Test: Rosenberg auxiliaries decode to nothing.
//
// qubit_map lists only logical qubits, and BinaryModel::QubitInfo default-
// constructs to {variable 0, bit 0} -- so without an explicit -1 marker an
// auxiliary is indistinguishable from bit 0 of variable 0, and flipping one
// would corrupt variable 0's register. Flipping an auxiliary must leave the
// decoded choices completely unchanged.
// ============================================================================
static void test_auxiliary_qubits_decode_to_nothing() {
    std::cout << "\n=== Rosenberg Auxiliaries Decode To Nothing ===\n";

    CFNModel cfn = make_test_cfn();
    BinaryModel model = parse_model_str(JSON_EXACT_BINARY_QUAD);

    CHECK(model.num_auxiliary > 0);

    int n_aux_marked = 0;
    for (int q = 0; q < model.num_qubits; q++)
        if (model.qubit_to_var[q] < 0) n_aux_marked++;
    CHECK(n_aux_marked == model.num_auxiliary);

    std::vector<int> state(model.num_qubits, 0);
    std::vector<int> before = decode_to_cfn(model, state);
    CHECK(!before.empty());

    CFNTracker tracker(model, cfn, state);
    double best_before = tracker.best_energy();

    for (int q = 0; q < model.num_qubits; q++) {
        if (model.qubit_to_var[q] >= 0) continue;   // logical
        state[q] = 1 - state[q];
        tracker.on_flip(q, state);
    }

    CHECK(decode_to_cfn(model, state) == before);              // choices unchanged
    CHECK(std::abs(tracker.best_energy() - best_before) < 1e-12);

    std::cout << "  " << n_aux_marked << " auxiliaries flipped; decoded choices "
              << fmt(before) << " unchanged\n";
}

// ============================================================================
// Test: JSONL reader (encode_cfn --jsonl output).
//
// Blank lines are skipped, a malformed line is reported and skipped rather than
// killing the shard, and the reported index is the 0-based line index that the
// shard/line selectors in main.cpp partition on.
// ============================================================================
static void test_jsonl_reader() {
    std::cout << "\n=== JSONL Reader ===\n";

    const char* tmp = "_test_models_tmp.jsonl";
    {
        std::ofstream ofs(tmp);
        ofs << nlohmann::json::parse(JSON_ONE_HOT).dump() << "\n"
            << "\n"                                     // blank line: skipped
            << "{ not valid json\n"                     // malformed: reported
            << nlohmann::json::parse(JSON_EXACT_BINARY).dump() << "\n";
    }

    JsonlModelReader reader(tmp);
    BinaryModel model;
    std::string error;
    int index = 0;

    std::vector<std::string> encodings;
    std::vector<int> indices;
    int n_errors = 0;

    while (reader.next(model, index, error)) {
        if (!error.empty()) { n_errors++; continue; }
        encodings.push_back(model.encoding);
        indices.push_back(index);
    }
    std::remove(tmp);

    CHECK(encodings.size() == 2);
    CHECK(n_errors == 1);
    CHECK(encodings[0] == "one_hot");
    CHECK(encodings[1] == "exact_binary");
    CHECK(indices[0] == 0);      // line 1
    CHECK(indices[1] == 3);      // line 4 -- blank and malformed lines still count
    CHECK(model.num_qubits > 0);

    std::cout << "  2 models parsed, 1 malformed line skipped, blank line ignored\n";
}

// ============================================================================
// Test: temperature calibration.
//
// The properties that make the encoding comparison fair:
//
//   (a) SCALE-FREE. Multiplying every coefficient of the model and its source
//       CFN by k multiplies both temperatures by k and changes nothing else. A
//       fixed T window has no such invariance -- it anneals a model at k=1 and
//       the same model at k=70 in completely different regimes, which is exactly
//       the confound, since the encodings of one CFN span ~70x in energy scale.
//
//   (b) SHARED COLD END. Every encoding of a given CFN cools to the SAME T_end,
//       because the cold end is measured on the source CFN -- the one scale the
//       encodings share, and the one the solver is scored on.
//
//   (c) ENCODING-SPECIFIC HOT END. A penalty encoding (one-hot: every flip pays
//       the Lagrange penalty) must start hotter than a penalty-free one, or its
//       chain cannot cross its own barriers.
// ============================================================================
static void test_temperature_calibration() {
    std::cout << "\n=== Temperature Calibration ===\n";

    CFNModel cfn = make_test_cfn();
    SAParams params = test_params();

    BinaryModel oh = parse_model_str(JSON_ONE_HOT);
    BinaryModel eb = parse_model_str(JSON_EXACT_BINARY);

    TemperatureRange t_oh = calibrate_temperatures(oh, &cfn, params, 99);
    TemperatureRange t_eb = calibrate_temperatures(eb, &cfn, params, 99);

    CHECK(t_oh.T_start > 0 && t_oh.T_end > 0);
    CHECK(t_oh.T_end < t_oh.T_start);      // the schedule must cool

    // (b) same source CFN => same cold end, whatever the encoding
    CHECK(std::abs(t_oh.T_end - t_eb.T_end) < 1e-12);

    // (c) the penalty encoding starts hotter
    CHECK(t_oh.T_start > t_eb.T_start);

    // (a) scale the encoded model AND the source CFN by k; T must scale by k
    const double k = 100.0;
    BinaryModel oh_k = oh;
    for (auto& t : oh_k.terms) t.coeff *= k;
    oh_k.offset *= k;
    oh_k.build_adjacency();

    CFNModel cfn_k = cfn;
    for (auto& u : cfn_k.unary)
        for (double& c : u) c *= k;
    for (auto& pt : cfn_k.pairwise)
        for (double& c : pt.costs) c *= k;

    TemperatureRange t_k = calibrate_temperatures(oh_k, &cfn_k, params, 99);

    CHECK(std::abs(t_k.T_start - k * t_oh.T_start) < 1e-6 * k * t_oh.T_start);
    CHECK(std::abs(t_k.T_end   - k * t_oh.T_end)   < 1e-6 * k * t_oh.T_end);

    // The hot end delivers the requested acceptance on a typical uphill move.
    std::vector<double> uphill = sample_uphill(oh, params, 99);
    double mean = uphill_mean(uphill);
    double p_accept = std::exp(-mean / t_oh.T_start);
    CHECK(std::abs(p_accept - params.accept_start) < 1e-9);

    // --T-start / --T-end pin an endpoint and opt it out of calibration.
    SAParams pinned = params;
    pinned.T_start = 3.5;
    pinned.T_start_given = true;
    apply_temperature_calibration(pinned, oh, &cfn);
    CHECK(pinned.T_start == 3.5);                          // untouched
    CHECK(std::abs(pinned.T_end - t_oh.T_end) < 1e-12);    // still calibrated

    SAParams off = params;
    off.auto_temp = false;
    double t0 = off.T_start, t1 = off.T_end;
    apply_temperature_calibration(off, oh, &cfn);
    CHECK(off.T_start == t0 && off.T_end == t1);           // --no-auto-temp

    std::cout << "  one_hot     : T=" << t_oh.T_start << " -> " << t_oh.T_end << "\n"
              << "  exact_binary: T=" << t_eb.T_start << " -> " << t_eb.T_end
              << "   (same cold end, cooler hot end)\n"
              << "  scaled x100 : T=" << t_k.T_start << " -> " << t_k.T_end
              << "   (both scale exactly)\n";
}

// ============================================================================
// Main test driver
// ============================================================================
int main() {
    std::cout << std::setprecision(8);

    CFNModel cfn = make_test_cfn();
    auto gs = brute_force_cfn(cfn);

    // ------- CFN solver -------
    test_cfn_solver();

    // ------- Cost-preserving encodings -------
    std::cout << "\n=== Cost-Preserving Encoding Tests ===\n";
    std::cout << "  (expect: encoded_E = cfn_E = " << gs.energy
              << " at " << fmt(gs.assignment) << ")\n";

    test_cost_preserving("one_hot         ", JSON_ONE_HOT, cfn, gs);
    test_cost_preserving("domain_wall     ", JSON_DOMAIN_WALL, cfn, gs);
    test_cost_preserving("exact_binary    ", JSON_EXACT_BINARY, cfn, gs);
    test_cost_preserving("exact_binary_quad", JSON_EXACT_BINARY_QUAD, cfn, gs);

    // ------- Cost-approximate encodings -------
    std::cout << "\n=== Cost-Approximate Encoding Tests ===\n";
    std::cout << "  (decode best binary solution and evaluate under original CFN)\n";

    test_cost_approximate("approx_binary    ", JSON_APPROXIMATE_BINARY, cfn, gs);
    test_cost_approximate("trunc_binary_k2  ", JSON_TRUNCATED_K2, cfn, gs);
    test_cost_approximate("trunc_binary_k3  ", JSON_TRUNCATED_K3, cfn, gs);
    test_cost_approximate("trunc_binary_k3_q", JSON_TRUNCATED_K3_QUAD, cfn, gs);

    // ------- Enhanced (Gray / Boltzmann / LI) decode -------
    test_enhanced_binary_decode();
    test_sa_decode_enhanced_optimum();

    // ------- Trajectory decoding, auxiliaries, JSONL -------
    test_temperature_calibration();
    test_cfn_tracker_vs_oracle();
    test_auxiliary_qubits_decode_to_nothing();
    test_jsonl_reader();

    // ------- Summary -------
    std::cout << "\n========================================\n"
              << "Checks: " << g_checks
              << "  Failures: " << g_failures << "\n";

    return g_failures > 0 ? 1 : 0;
}
