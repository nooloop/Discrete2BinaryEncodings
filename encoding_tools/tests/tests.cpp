// Unit tests for exact encoding cost preservation and infeasible-energy dominance (OH, DW).
// Unit tests for Rosenberg quadratization correctness.
// Unit tests for choice ordering and bitstring ordering.
// Unit tests for nonlinear refinement for approximate-binary.

#include "baseline/types.hpp"
#include "baseline/cfn.hpp"
#include "utilities/assignment.hpp"
#include "utilities/rosenberg.hpp"
#include "encodings/one_hot.hpp"
#include "encodings/domain_wall.hpp"
#include "encodings/exact_binary.hpp"
#include "encodings/approximate_binary.hpp"
#include "encodings/truncated_binary.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static const double TOL = 1e-9;
static int g_passed = 0;
static int g_failed = 0;

static void check(bool cond, const std::string& msg) {
    if (cond) {
        g_passed++;
    } else {
        g_failed++;
        std::cerr << "  FAIL: " << msg << "\n";
    }
}

// ---------------------------------------------------------------------------
// Polynomial evaluator
// ---------------------------------------------------------------------------
static double evaluate(const BinaryPolynomial& poly,
                       const std::vector<int>& bits) {
    double val = poly.offset;
    for (auto& [indices, coeff] : poly.terms) {
        double prod = 1.0;
        for (int q : indices)
            prod *= bits[q];
        val += coeff * prod;
    }
    return val;
}

// ---------------------------------------------------------------------------
// CFN cost for a discrete choice vector
// ---------------------------------------------------------------------------
static double cfn_cost(const CFN& cfn, const std::vector<int>& choices) {
    double cost = 0;
    for (int i = 0; i < cfn.num_variables; i++)
        cost += cfn.unary_tables[i].costs[choices[i]];
    for (auto& pt : cfn.pairwise_tables) {
        int vi = pt.scope[0], vj = pt.scope[1];
        int dj = cfn.cardinalities[vj];
        cost += pt.costs[choices[vi] * dj + choices[vj]];
    }
    return cost;
}

// ---------------------------------------------------------------------------
// Enumerate all discrete choice combinations
// ---------------------------------------------------------------------------
static void for_each_choice(const CFN& cfn,
                            const std::function<void(const std::vector<int>&)>& fn) {
    int N = cfn.num_variables;
    std::vector<int> choices(N, 0);
    while (true) {
        fn(choices);
        int carry = N - 1;
        while (carry >= 0) {
            choices[carry]++;
            if (choices[carry] < cfn.cardinalities[carry]) break;
            choices[carry] = 0;
            carry--;
        }
        if (carry < 0) break;
    }
}

// ---------------------------------------------------------------------------
// Choice -> binary-config converters
// ---------------------------------------------------------------------------

// One-hot: bit (qubit_start[i] + c_i) is 1, rest of register is 0
static std::vector<int> to_oh_bits(const CFN& cfn,
                                   const EncodingResult& res,
                                   const std::vector<int>& choices) {
    std::vector<int> bits(res.num_logical_qubits, 0);
    for (int i = 0; i < cfn.num_variables; i++)
        bits[res.qubit_start[i] + choices[i]] = 1;
    return bits;
}

// Domain-wall: choice c means bits 0..c-1 are 1, bits c..d-2 are 0
static std::vector<int> to_dw_bits(const CFN& cfn,
                                   const EncodingResult& res,
                                   const std::vector<int>& choices) {
    std::vector<int> bits(res.num_logical_qubits, 0);
    for (int i = 0; i < cfn.num_variables; i++) {
        int qs = res.qubit_start[i];
        for (int q = 0; q < choices[i]; q++)
            bits[qs + q] = 1;
    }
    return bits;
}

// Exact-binary: bits spell out the assigned bitstring for each register
static std::vector<int> to_eb_bits(const CFN& cfn,
                                   const EncodingResult& res,
                                   const BitstringAssignment& ba,
                                   const std::vector<int>& choices) {
    std::vector<int> bits(res.num_logical_qubits, 0);
    for (int i = 0; i < cfn.num_variables; i++) {
        int Di = res.bits_per_var[i];
        int qs = res.qubit_start[i];
        int bs = ba.assignment[i][choices[i]];
        for (int b = 0; b < Di; b++)
            bits[qs + b] = (bs >> b) & 1;
    }
    return bits;
}

// ---------------------------------------------------------------------------
// Feasibility checkers
// ---------------------------------------------------------------------------

// OH feasible: exactly one bit set per register
static bool is_oh_feasible(const CFN& cfn, const EncodingResult& res,
                           const std::vector<int>& bits) {
    for (int i = 0; i < cfn.num_variables; i++) {
        int qs = res.qubit_start[i];
        int di = cfn.cardinalities[i];
        int sum = 0;
        for (int c = 0; c < di; c++)
            sum += bits[qs + c];
        if (sum != 1) return false;
    }
    return true;
}

// DW feasible: monotone non-increasing per register (1...10...0)
static bool is_dw_feasible(const CFN& cfn, const EncodingResult& res,
                           const std::vector<int>& bits) {
    for (int i = 0; i < cfn.num_variables; i++) {
        int qs = res.qubit_start[i];
        int nbits = res.bits_per_var[i];
        bool seen_zero = false;
        for (int q = 0; q < nbits; q++) {
            if (bits[qs + q] == 0) seen_zero = true;
            else if (seen_zero) return false;  // 1 after a 0 -> not monotone
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test CFN factories
// ---------------------------------------------------------------------------

// CFN 1: 2 variables, cardinalities [3, 2], 1 pairwise table
static CFN make_cfn1() {
    CFN cfn;
    cfn.name = "test1";
    cfn.filename = "test1.cfn";
    cfn.num_variables = 2;
    cfn.var_names = {"x0", "x1"};
    cfn.cardinalities = {3, 2};
    cfn.unary_tables.resize(2);
    cfn.unary_tables[0] = {{0}, {1.0, 2.0, 3.0}};
    cfn.unary_tables[1] = {{1}, {0.5, 1.5}};
    cfn.pairwise_tables.push_back({{0, 1}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}});
    return cfn;
}

// CFN 2: 3 variables, cardinalities [2, 2, 2], 2 pairwise tables
static CFN make_cfn2() {
    CFN cfn;
    cfn.name = "test2";
    cfn.filename = "test2.cfn";
    cfn.num_variables = 3;
    cfn.var_names = {"x0", "x1", "x2"};
    cfn.cardinalities = {2, 2, 2};
    cfn.unary_tables.resize(3);
    cfn.unary_tables[0] = {{0}, {1.0, 2.0}};
    cfn.unary_tables[1] = {{1}, {3.0, 4.0}};
    cfn.unary_tables[2] = {{2}, {0.0, 1.0}};
    cfn.pairwise_tables.push_back({{0, 1}, {0.5, 1.0, 1.5, 2.0}});
    cfn.pairwise_tables.push_back({{1, 2}, {0.1, 0.2, 0.3, 0.4}});
    return cfn;
}

// CFN 3: 2 variables, cardinalities [4, 3], 1 pairwise table.
// Non-monotone costs make different orderings produce distinct assignments.
// EB pairwise degree = ceil(log2(4)) + ceil(log2(3)) = 2+2 = 4 -> requires quadratization.
static CFN make_cfn3() {
    CFN cfn;
    cfn.name = "test3";
    cfn.filename = "test3.cfn";
    cfn.num_variables = 2;
    cfn.var_names = {"x0", "x1"};
    cfn.cardinalities = {4, 3};
    cfn.unary_tables.resize(2);
    cfn.unary_tables[0] = {{0}, {3.0, 1.0, 4.0, 2.0}};
    cfn.unary_tables[1] = {{1}, {2.5, 0.5, 1.5}};
    cfn.pairwise_tables.push_back({{0, 1}, {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2
    }});
    return cfn;
}

// CFN for NL refinement: single variable, cardinality 5, large cost spread.
// D_i = 3 bits, so AB at k_approx=2 is approximate (8 rows, 7 columns, 1 residual DOF).
static CFN make_cfn_nl() {
    CFN cfn;
    cfn.name = "test_nl";
    cfn.filename = "test_nl.cfn";
    cfn.num_variables = 1;
    cfn.var_names = {"x0"};
    cfn.cardinalities = {5};
    cfn.unary_tables.resize(1);
    cfn.unary_tables[0] = {{0}, {1.0, 10.0, 2.0, 15.0, 3.0}};
    return cfn;
}

// ===========================================================================
// Cost-preservation tests
// ===========================================================================

static void test_cost_preservation_oh(const CFN& cfn, const std::string& label) {
    std::cerr << "[OH cost preservation] " << label << "\n";
    EncodingParams params;
    params.encoding = "one_hot";
    auto res = encode_one_hot(cfn, params);

    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto bits = to_oh_bits(cfn, res, choices);
        double enc = evaluate(res.poly, bits);
        double orig = cfn_cost(cfn, choices);
        std::ostringstream msg;
        msg << "choices=(";
        for (int i = 0; i < (int)choices.size(); i++) {
            if (i) msg << ",";
            msg << choices[i];
        }
        msg << ") enc=" << enc << " orig=" << orig << " diff=" << (enc - orig);
        check(std::abs(enc - orig) < TOL, msg.str());
    });
}

static void test_cost_preservation_dw(const CFN& cfn, const std::string& label) {
    std::cerr << "[DW cost preservation] " << label << "\n";
    EncodingParams params;
    params.encoding = "domain_wall";
    auto res = encode_domain_wall(cfn, params);

    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto bits = to_dw_bits(cfn, res, choices);
        double enc = evaluate(res.poly, bits);
        double orig = cfn_cost(cfn, choices);
        std::ostringstream msg;
        msg << "choices=(";
        for (int i = 0; i < (int)choices.size(); i++) {
            if (i) msg << ",";
            msg << choices[i];
        }
        msg << ") enc=" << enc << " orig=" << orig << " diff=" << (enc - orig);
        check(std::abs(enc - orig) < TOL, msg.str());
    });
}

static void test_cost_preservation_eb(const CFN& cfn, const std::string& label) {
    std::cerr << "[EB cost preservation] " << label << "\n";
    EncodingParams params;
    params.encoding = "exact_binary";
    params.choice_ordering = "unsorted";
    params.bitstring_ordering = "natural";

    BitstringAssignment ba = compute_assignment(cfn, params);
    auto res = encode_exact_binary(cfn, ba, params);

    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto bits = to_eb_bits(cfn, res, ba, choices);
        double enc = evaluate(res.poly, bits);
        double orig = cfn_cost(cfn, choices);
        std::ostringstream msg;
        msg << "choices=(";
        for (int i = 0; i < (int)choices.size(); i++) {
            if (i) msg << ",";
            msg << choices[i];
        }
        msg << ") enc=" << enc << " orig=" << orig << " diff=" << (enc - orig);
        check(std::abs(enc - orig) < TOL, msg.str());
    });
}

// ===========================================================================
// Infeasible-energy dominance tests
// ===========================================================================

static void test_infeasible_penalty_oh(const CFN& cfn, const std::string& label) {
    std::cerr << "[OH infeasible > min feasible] " << label << "\n";
    EncodingParams params;
    params.encoding = "one_hot";
    auto res = encode_one_hot(cfn, params);

    // Find minimum feasible energy
    double min_feasible = std::numeric_limits<double>::max();
    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto bits = to_oh_bits(cfn, res, choices);
        double e = evaluate(res.poly, bits);
        min_feasible = std::min(min_feasible, e);
    });

    // Enumerate all binary configurations
    int total_qubits = res.num_logical_qubits;
    long long total_configs = 1LL << total_qubits;
    int infeasible_count = 0;

    for (long long mask = 0; mask < total_configs; mask++) {
        std::vector<int> bits(total_qubits);
        for (int q = 0; q < total_qubits; q++)
            bits[q] = (int)((mask >> q) & 1);

        if (is_oh_feasible(cfn, res, bits)) continue;

        infeasible_count++;
        double e = evaluate(res.poly, bits);
        std::ostringstream msg;
        msg << "infeasible mask=" << mask
            << " energy=" << e << " min_feasible=" << min_feasible;
        check(e > min_feasible - TOL, msg.str());
    }

    std::cerr << "  checked " << infeasible_count << " infeasible configurations\n";
}

static void test_infeasible_penalty_dw(const CFN& cfn, const std::string& label) {
    std::cerr << "[DW infeasible > min feasible] " << label << "\n";
    EncodingParams params;
    params.encoding = "domain_wall";
    auto res = encode_domain_wall(cfn, params);

    // Find minimum feasible energy
    double min_feasible = std::numeric_limits<double>::max();
    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto bits = to_dw_bits(cfn, res, choices);
        double e = evaluate(res.poly, bits);
        min_feasible = std::min(min_feasible, e);
    });

    // Enumerate all binary configurations
    int total_qubits = res.num_logical_qubits;
    long long total_configs = 1LL << total_qubits;
    int infeasible_count = 0;

    for (long long mask = 0; mask < total_configs; mask++) {
        std::vector<int> bits(total_qubits);
        for (int q = 0; q < total_qubits; q++)
            bits[q] = (int)((mask >> q) & 1);

        if (is_dw_feasible(cfn, res, bits)) continue;

        infeasible_count++;
        double e = evaluate(res.poly, bits);
        std::ostringstream msg;
        msg << "infeasible mask=" << mask
            << " energy=" << e << " min_feasible=" << min_feasible;
        check(e > min_feasible - TOL, msg.str());
    }

    std::cerr << "  checked " << infeasible_count << " infeasible configurations\n";
}

// ===========================================================================
// Quadratization tests  (Rosenberg reduction, Appendix A.5)
//
// For each feasible discrete assignment, set the logical bits to the EB
// encoding, then search over all auxiliary-bit settings.  The Rosenberg
// penalty (Eq. A.18) vanishes when y_{pq} = b_p * b_q, so the minimum
// QUBO energy over auxiliaries must equal the HUBO energy.
// ===========================================================================

static void test_quadratization_eb(const CFN& cfn, const std::string& label) {
    std::cerr << "[EB quadratization] " << label << "\n";
    EncodingParams params;
    params.encoding = "exact_binary";
    params.choice_ordering = "unsorted";
    params.bitstring_ordering = "natural";

    BitstringAssignment ba = compute_assignment(cfn, params);
    auto hubo_res = encode_exact_binary(cfn, ba, params);

    int hubo_deg = hubo_res.poly.max_degree();
    std::cerr << "  HUBO max degree: " << hubo_deg << "\n";

    auto qr = rosenberg_quadratize(hubo_res.poly);

    // QUBO must have max degree <= 2
    int qubo_deg = qr.poly.max_degree();
    check(qubo_deg <= 2,
          label + " QUBO degree <= 2 (got " + std::to_string(qubo_deg) + ")");

    int num_aux = qr.num_auxiliaries;
    int num_logical = hubo_res.num_logical_qubits;
    int total_qubits = num_logical + num_aux;
    std::cerr << "  Auxiliaries: " << num_aux
              << ", total qubits: " << total_qubits << "\n";

    // Degree-2 HUBOs need no auxiliaries (quadratization is a no-op)
    if (hubo_deg <= 2)
        check(num_aux == 0,
              label + " degree-2 HUBO should need 0 auxiliaries");

    // For each feasible assignment, min_aux QUBO energy == HUBO energy
    for_each_choice(cfn, [&](const std::vector<int>& choices) {
        auto logical_bits = to_eb_bits(cfn, hubo_res, ba, choices);
        double hubo_energy = evaluate(hubo_res.poly, logical_bits);

        long long aux_configs = (num_aux > 0) ? (1LL << num_aux) : 1;
        double min_qubo = std::numeric_limits<double>::max();

        for (long long amask = 0; amask < aux_configs; amask++) {
            std::vector<int> all_bits(total_qubits);
            for (int q = 0; q < num_logical; q++)
                all_bits[q] = logical_bits[q];
            for (int a = 0; a < num_aux; a++)
                all_bits[num_logical + a] = (int)((amask >> a) & 1);
            double e = evaluate(qr.poly, all_bits);
            min_qubo = std::min(min_qubo, e);
        }

        std::ostringstream msg;
        msg << "choices=(";
        for (int i = 0; i < (int)choices.size(); i++) {
            if (i) msg << ",";
            msg << choices[i];
        }
        msg << ") hubo=" << hubo_energy << " qubo_min=" << min_qubo;
        check(std::abs(min_qubo - hubo_energy) < TOL, msg.str());
    });
}

// ===========================================================================
// Choice ordering & bitstring ordering tests  (Section 3.4)
// ===========================================================================

// EB cost preservation must hold under every ordering combination.
static void test_orderings_eb(const CFN& cfn, const std::string& label) {
    std::cerr << "[EB ordering combos] " << label << "\n";

    const std::string co[] = {"unsorted", "one_variable", "boltzmann"};
    const std::string bo[] = {"natural", "gray"};

    for (auto& c : co) {
        for (auto& b : bo) {
            std::cerr << "  " << c << " + " << b << "\n";
            EncodingParams params;
            params.encoding = "exact_binary";
            params.choice_ordering = c;
            params.bitstring_ordering = b;
            params.temperature = 1.0;

            BitstringAssignment ba = compute_assignment(cfn, params);
            auto res = encode_exact_binary(cfn, ba, params);

            for_each_choice(cfn, [&](const std::vector<int>& choices) {
                auto bits = to_eb_bits(cfn, res, ba, choices);
                double enc = evaluate(res.poly, bits);
                double orig = cfn_cost(cfn, choices);
                std::ostringstream msg;
                msg << c << "+" << b << " (";
                for (int i = 0; i < (int)choices.size(); i++) {
                    if (i) msg << ",";
                    msg << choices[i];
                }
                msg << ") enc=" << enc << " orig=" << orig;
                check(std::abs(enc - orig) < TOL, msg.str());
            });
        }
    }
}

// Verify the ordering functions themselves:
//   unsorted     -> identity permutation
//   one_variable -> ascending unary cost  (Eq. 30 in Section 3.4)
//   boltzmann    -> valid permutation sorted by effective energy (Eq. 31)
static void test_choice_ordering_values(const CFN& cfn, const std::string& label) {
    std::cerr << "[Choice ordering values] " << label << "\n";

    // unsorted: identity
    auto ord_un = compute_choice_ordering(cfn, "unsorted");
    for (int i = 0; i < cfn.num_variables; i++)
        for (int c = 0; c < cfn.cardinalities[i]; c++)
            check(ord_un[i][c] == c,
                  "unsorted var " + std::to_string(i) + " identity");

    // one_variable: ascending unary cost
    auto ord_1v = compute_choice_ordering(cfn, "one_variable");
    for (int i = 0; i < cfn.num_variables; i++) {
        auto& costs = cfn.unary_tables[i].costs;
        for (int r = 1; r < cfn.cardinalities[i]; r++)
            check(costs[ord_1v[i][r - 1]] <= costs[ord_1v[i][r]],
                  "one_variable var " + std::to_string(i) +
                  " ascending at rank " + std::to_string(r));
    }

    // boltzmann: valid permutation
    auto ord_bz = compute_choice_ordering(cfn, "boltzmann", 1.0);
    for (int i = 0; i < cfn.num_variables; i++) {
        std::vector<bool> seen(cfn.cardinalities[i], false);
        for (int r = 0; r < cfn.cardinalities[i]; r++) {
            int ch = ord_bz[i][r];
            check(ch >= 0 && ch < cfn.cardinalities[i] && !seen[ch],
                  "boltzmann var " + std::to_string(i) +
                  " valid perm at rank " + std::to_string(r));
            if (ch >= 0 && ch < cfn.cardinalities[i]) seen[ch] = true;
        }
    }
}

// Natural ordering: 0,1,2,...,2^n-1.
// Gray ordering: successive entries differ by exactly 1 bit (Section 3.4).
static void test_bitstring_ordering() {
    std::cerr << "[Bitstring ordering] natural vs gray\n";

    for (int n = 1; n <= 4; n++) {
        int sz = 1 << n;

        // Natural
        auto nat = bitstring_sequence(n, "natural");
        check((int)nat.size() == sz,
              "natural " + std::to_string(n) + "-bit size");
        for (int i = 0; i < sz; i++)
            check(nat[i] == i,
                  "natural " + std::to_string(n) + "-bit [" +
                  std::to_string(i) + "]");

        // Gray: Hamming-1 successive, valid permutation
        auto gray = bitstring_sequence(n, "gray");
        check((int)gray.size() == sz,
              "gray " + std::to_string(n) + "-bit size");

        for (int i = 1; i < sz; i++)
            check(popcount(gray[i] ^ gray[i - 1]) == 1,
                  "gray " + std::to_string(n) + "-bit Hamming-1 at " +
                  std::to_string(i));

        std::vector<bool> seen(sz, false);
        for (int i = 0; i < sz; i++) {
            check(gray[i] >= 0 && gray[i] < sz && !seen[gray[i]],
                  "gray " + std::to_string(n) + "-bit unique at " +
                  std::to_string(i));
            if (gray[i] >= 0 && gray[i] < sz) seen[gray[i]] = true;
        }
    }
}

// Verify that different orderings produce distinct bitstring assignments
// on a CFN with non-monotone unary costs.
static void test_orderings_differ(const CFN& cfn, const std::string& label) {
    std::cerr << "[Orderings differ] " << label << "\n";

    EncodingParams p_un, p_1v, p_gr;
    p_un.encoding = p_1v.encoding = p_gr.encoding = "exact_binary";
    p_un.choice_ordering  = "unsorted";      p_un.bitstring_ordering = "natural";
    p_1v.choice_ordering  = "one_variable";  p_1v.bitstring_ordering = "natural";
    p_gr.choice_ordering  = "unsorted";      p_gr.bitstring_ordering = "gray";

    auto ba_un = compute_assignment(cfn, p_un);
    auto ba_1v = compute_assignment(cfn, p_1v);
    auto ba_gr = compute_assignment(cfn, p_gr);

    // one_variable should differ from unsorted (non-monotone costs)
    bool diff_1v = false;
    for (int i = 0; i < cfn.num_variables && !diff_1v; i++)
        for (int c = 0; c < cfn.cardinalities[i] && !diff_1v; c++)
            if (ba_un.assignment[i][c] != ba_1v.assignment[i][c])
                diff_1v = true;
    check(diff_1v, label + " one_variable differs from unsorted");

    // gray should differ from natural
    bool diff_gr = false;
    for (int i = 0; i < cfn.num_variables && !diff_gr; i++)
        for (int c = 0; c < cfn.cardinalities[i] && !diff_gr; c++)
            if (ba_un.assignment[i][c] != ba_gr.assignment[i][c])
                diff_gr = true;
    check(diff_gr, label + " gray differs from natural");
}

// ===========================================================================
// Approximate-binary nonlinear refinement tests  (Appendix A.4, Eq. 44)
//
// The NL refinement minimises a Boltzmann probability-matching loss:
//   L(B_i) = sum_c (p_actual - p_desired)^2  +  tau*(E_min_actual - E_min_desired)^2
// on the per-register coefficients B_i (starting from the LS solution).
//
// We verify that:
//   (a) the refinement runs and produces finite output,
//   (b) it actually modifies the polynomial,
//   (c) the probability-matching component of the loss does not worsen.
// The test uses a single-variable CFN so that the per-register energies
// can be evaluated unambiguously from the full polynomial.
// ===========================================================================

static void test_nonlinear_refinement_ab(const CFN& cfn, const std::string& label) {
    std::cerr << "[AB NL refinement] " << label << "\n";

    EncodingParams params_base;
    params_base.encoding            = "approximate_binary";
    params_base.choice_ordering     = "unsorted";
    params_base.bitstring_ordering  = "natural";
    params_base.k_approx            = 2;

    EncodingParams params_nl  = params_base;
    params_nl.nonlinear_refinement  = true;
    params_nl.nl_temperature        = 1.0;
    params_nl.nl_tether_weight      = 1.0;
    params_nl.nl_max_iter           = 200;

    BitstringAssignment ba = compute_assignment(cfn, params_base);

    auto res_base = encode_approximate_binary(cfn, ba, params_base);
    auto res_nl   = encode_approximate_binary(cfn, ba, params_nl);

    // (a) NL refinement ran
    check(res_nl.time_nonlinear_refinement > 0,
          label + " NL time > 0");

    // (a) all coefficients finite
    bool all_fin = std::isfinite(res_nl.poly.offset);
    for (auto& [k, v] : res_nl.poly.terms)
        if (!std::isfinite(v)) { all_fin = false; break; }
    check(all_fin, label + " NL poly finite");

    // (b) NL actually changed the evaluated energies
    {
        int Di = res_base.bits_per_var[0];
        bool changed = false;
        for (int bs = 0; bs < (1 << Di) && !changed; bs++) {
            std::vector<int> bits_b(res_base.num_logical_qubits, 0);
            std::vector<int> bits_n(res_nl.num_logical_qubits, 0);
            for (int b = 0; b < Di; b++) {
                bits_b[res_base.qubit_start[0] + b] = (bs >> b) & 1;
                bits_n[res_nl.qubit_start[0] + b]   = (bs >> b) & 1;
            }
            if (std::abs(evaluate(res_base.poly, bits_b) -
                         evaluate(res_nl.poly, bits_n)) > 1e-12)
                changed = true;
        }
        check(changed, label + " NL changed energies");
    }

    // (c) Probability-matching component of the loss (shift-invariant)
    double kBT = params_nl.nl_temperature;

    for (int var = 0; var < cfn.num_variables; var++) {
        int Di = res_base.bits_per_var[var];
        int total_bs = 1 << Di;
        int di = cfn.cardinalities[var];

        // Desired energy vector (padding = min energy)
        double min_cost = *std::min_element(
            cfn.unary_tables[var].costs.begin(),
            cfn.unary_tables[var].costs.end());
        std::vector<double> E_desired(total_bs, min_cost);
        for (int c = 0; c < di; c++)
            E_desired[ba.assignment[var][c]] = cfn.unary_tables[var].costs[c];

        // Evaluate polynomial at each bitstring of this register
        auto eval_reg = [&](const EncodingResult& res) {
            std::vector<double> E(total_bs);
            for (int bs = 0; bs < total_bs; bs++) {
                std::vector<int> bits(res.num_logical_qubits, 0);
                for (int b = 0; b < Di; b++)
                    bits[res.qubit_start[var] + b] = (bs >> b) & 1;
                E[bs] = evaluate(res.poly, bits);
            }
            return E;
        };

        auto E_base = eval_reg(res_base);
        auto E_nl   = eval_reg(res_nl);

        // Boltzmann probabilities
        auto boltzmann = [&](const std::vector<double>& E) {
            double e_min = *std::min_element(E.begin(), E.end());
            std::vector<double> p(E.size());
            double Z = 0;
            for (int r = 0; r < (int)E.size(); r++) {
                p[r] = std::exp(-(E[r] - e_min) / kBT);
                Z += p[r];
            }
            for (auto& pi : p) pi /= Z;
            return p;
        };

        auto p_desired = boltzmann(E_desired);
        auto p_base    = boltzmann(E_base);
        auto p_nl      = boltzmann(E_nl);

        double loss_b = 0, loss_n = 0;
        for (int r = 0; r < total_bs; r++) {
            loss_b += (p_base[r] - p_desired[r]) * (p_base[r] - p_desired[r]);
            loss_n += (p_nl[r]   - p_desired[r]) * (p_nl[r]   - p_desired[r]);
        }

        std::cerr << "  var " << var << ": prob_loss  base=" << loss_b
                  << "  nl=" << loss_n << "\n";

        // NL should not significantly worsen probability matching
        check(loss_n <= loss_b + 0.01,
              label + " var " + std::to_string(var) +
              " NL prob loss not worse (base=" +
              std::to_string(loss_b) + " nl=" + std::to_string(loss_n) + ")");
    }
}

// Smoke test: NL refinement on a multi-variable CFN (checks no crash, finite output).
static void test_nl_refinement_smoke(const CFN& cfn, const std::string& label) {
    std::cerr << "[AB NL smoke] " << label << "\n";
    EncodingParams params;
    params.encoding             = "approximate_binary";
    params.choice_ordering      = "boltzmann";
    params.bitstring_ordering   = "gray";
    params.k_approx             = 2;
    params.nonlinear_refinement = true;
    params.nl_temperature       = 1.0;
    params.nl_tether_weight     = 1.0;
    params.nl_max_iter          = 50;

    BitstringAssignment ba = compute_assignment(cfn, params);
    auto res = encode_approximate_binary(cfn, ba, params);

    check(res.num_logical_qubits > 0, label + " has qubits");
    check(std::isfinite(res.poly.offset), label + " offset finite");
    check(res.time_nonlinear_refinement > 0, label + " NL time > 0");

    bool all_fin = true;
    for (auto& [k, v] : res.poly.terms)
        if (!std::isfinite(v)) { all_fin = false; break; }
    check(all_fin, label + " all coefficients finite");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    auto cfn1   = make_cfn1();
    auto cfn2   = make_cfn2();
    auto cfn3   = make_cfn3();
    auto cfn_nl = make_cfn_nl();

    // ----- existing tests -----
    std::cerr << "=== Cost Preservation Tests ===\n";
    test_cost_preservation_oh(cfn1, "CFN1 [3,2]");
    test_cost_preservation_oh(cfn2, "CFN2 [2,2,2]");
    test_cost_preservation_dw(cfn1, "CFN1 [3,2]");
    test_cost_preservation_dw(cfn2, "CFN2 [2,2,2]");
    test_cost_preservation_eb(cfn1, "CFN1 [3,2]");
    test_cost_preservation_eb(cfn2, "CFN2 [2,2,2]");

    std::cerr << "\n=== Infeasible Energy Dominance Tests ===\n";
    test_infeasible_penalty_oh(cfn1, "CFN1 [3,2]");
    test_infeasible_penalty_oh(cfn2, "CFN2 [2,2,2]");
    test_infeasible_penalty_dw(cfn1, "CFN1 [3,2]");
    test_infeasible_penalty_dw(cfn2, "CFN2 [2,2,2]");

    // ----- new tests -----
    std::cerr << "\n=== Quadratization Tests ===\n";
    test_quadratization_eb(cfn1, "CFN1 [3,2]");    // degree-3 HUBO
    test_quadratization_eb(cfn2, "CFN2 [2,2,2]");  // degree-2 (no-op)
    test_quadratization_eb(cfn3, "CFN3 [4,3]");    // degree-4 HUBO

    std::cerr << "\n=== Choice & Bitstring Ordering Tests ===\n";
    test_bitstring_ordering();
    test_choice_ordering_values(cfn3, "CFN3 [4,3]");
    test_orderings_differ(cfn3, "CFN3 [4,3]");
    test_orderings_eb(cfn1, "CFN1 [3,2]");
    test_orderings_eb(cfn3, "CFN3 [4,3]");

    std::cerr << "\n=== Nonlinear Refinement Tests ===\n";
    test_nonlinear_refinement_ab(cfn_nl, "CFN_NL [5]");
    test_nl_refinement_smoke(cfn1, "CFN1 [3,2]");
    test_nl_refinement_smoke(cfn3, "CFN3 [4,3]");

    // ----- summary -----
    std::cerr << "\n========================================\n";
    std::cerr << g_passed << " passed, " << g_failed << " failed\n";
    if (g_failed == 0)
        std::cerr << "ALL TESTS PASSED\n";
    else
        std::cerr << "SOME TESTS FAILED\n";

    return g_failed > 0 ? 1 : 0;
}
