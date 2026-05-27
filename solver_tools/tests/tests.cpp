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

    // ------- Summary -------
    std::cout << "\n========================================\n"
              << "Checks: " << g_checks
              << "  Failures: " << g_failures << "\n";

    return g_failures > 0 ? 1 : 0;
}
