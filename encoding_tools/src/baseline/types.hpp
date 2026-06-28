#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cassert>
#include <iostream>
#include <sstream>
#include <limits>

// ============================================================================
// Hash for vector<int> keys
// ============================================================================
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = v.size();
        for (auto x : v)
            seed ^= std::hash<int>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// ============================================================================
// Variable type: BINARY {0,1} or SPIN {-1,+1}
// ============================================================================
enum class VarType { BINARY, SPIN };

// ============================================================================
// BinaryPolynomial: sparse representation of a QUBO/HUBO/Ising model
// Keys are sorted vectors of qubit indices; values are coefficients.
// ============================================================================
class BinaryPolynomial {
public:
    using TermMap = std::unordered_map<std::vector<int>, double, VectorHash>;

    VarType var_type = VarType::BINARY;
    double offset = 0.0;
    TermMap terms;

    void add_term(const std::vector<int>& indices, double coeff) {
        if (std::abs(coeff) < 1e-18) return;
        terms[indices] += coeff;
    }

    void add_term(std::vector<int>&& indices, double coeff) {
        if (std::abs(coeff) < 1e-18) return;
        std::sort(indices.begin(), indices.end());
        terms[std::move(indices)] += coeff;
    }

    void add_constant(double c) { offset += c; }

    void merge_from(const BinaryPolynomial& other) {
        offset += other.offset;
        for (auto& [k, v] : other.terms)
            terms[k] += v;
    }

    void prune(double tol = 1e-15) {
        for (auto it = terms.begin(); it != terms.end();) {
            if (std::abs(it->second) < tol)
                it = terms.erase(it);
            else
                ++it;
        }
    }

    int num_variables() const {
        int mx = -1;
        for (auto& [k, v] : terms)
            if (!k.empty()) mx = std::max(mx, k.back());
        return mx + 1;
    }

    int max_degree() const {
        int md = 0;
        for (auto& [k, v] : terms)
            md = std::max(md, (int)k.size());
        return md;
    }

    struct Stats {
        int num_linear = 0, num_quadratic = 0, num_cubic = 0, num_higher = 0;
        int total = 0;
        double coeff_max_abs = 0, coeff_min_abs = std::numeric_limits<double>::max();
        double dynamic_range = 0;
        double mean_weighted_degree = 0;
        int max_deg = 0;
    };

    Stats compute_stats() const {
        Stats s;
        double sum_abs = 0, sum_wdeg = 0;
        for (auto& [k, v] : terms) {
            if (std::abs(v) < 1e-18) continue;
            int d = (int)k.size();
            if (d == 1) s.num_linear++;
            else if (d == 2) s.num_quadratic++;
            else if (d == 3) s.num_cubic++;
            else s.num_higher++;
            s.total++;
            double a = std::abs(v);
            s.coeff_max_abs = std::max(s.coeff_max_abs, a);
            s.coeff_min_abs = std::min(s.coeff_min_abs, a);
            s.max_deg = std::max(s.max_deg, d);
            sum_abs += a;
            sum_wdeg += a * d;
        }
        if (s.total == 0) s.coeff_min_abs = 0;
        s.dynamic_range = (s.coeff_min_abs > 0) ? s.coeff_max_abs / s.coeff_min_abs : 0;
        s.mean_weighted_degree = (sum_abs > 0) ? sum_wdeg / sum_abs : 0;
        return s;
    }

    std::map<std::vector<int>, double> sorted_terms() const {
        std::map<std::vector<int>, double> out;
        for (auto& [k, v] : terms)
            if (std::abs(v) > 1e-18) out[k] = v;
        return out;
    }
};

// ============================================================================
// Local polynomial (polynomial in local bit indices, used during construction)
// ============================================================================
using LocalPoly = std::unordered_map<std::vector<int>, double, VectorHash>;

inline LocalPoly multiply_local(const LocalPoly& a, const LocalPoly& b) {
    LocalPoly result;
    result.reserve(a.size() * b.size());
    for (auto& [ka, va] : a) {
        for (auto& [kb, vb] : b) {
            double c = va * vb;
            if (std::abs(c) < 1e-18) continue;
            std::vector<int> key;
            key.reserve(ka.size() + kb.size());
            std::merge(ka.begin(), ka.end(), kb.begin(), kb.end(), std::back_inserter(key));
            // Remove duplicates (b_q^2 = b_q in binary)
            key.erase(std::unique(key.begin(), key.end()), key.end());
            result[key] += c;
        }
    }
    return result;
}

inline LocalPoly scale_local(const LocalPoly& p, double s) {
    LocalPoly result = p;
    for (auto& [k, v] : result) v *= s;
    return result;
}

inline void add_local_to_poly(BinaryPolynomial& poly,
                              const LocalPoly& local,
                              const std::vector<int>& global_map) {
    for (auto& [local_idx, coeff] : local) {
        if (std::abs(coeff) < 1e-18) continue;
        if (local_idx.empty()) {
            poly.add_constant(coeff);
            continue;
        }
        std::vector<int> global_idx;
        global_idx.reserve(local_idx.size());
        for (int li : local_idx)
            global_idx.push_back(global_map[li]);
        std::sort(global_idx.begin(), global_idx.end());
        poly.add_term(global_idx, coeff);
    }
}

// ============================================================================
// Encoding parameters
// ============================================================================
struct EncodingParams {
    std::string encoding;
    std::string choice_ordering = "unsorted";   // unsorted, one_variable, boltzmann
    std::string bitstring_ordering = "natural";  // natural, gray
    bool weighted_ls = false;
    bool li_prioritization = false;
    bool nonlinear_refinement = false;
    bool quadratize = false;
    int k_approx = 2;
    int k_trunc = 2;
    double temperature = 1.0;
    int num_threads = 1;
    double nl_temperature = 1.0;
    double nl_tether_weight = 1.0;
    int nl_max_iter = 100;
};

// ============================================================================
// Encoding result
// ============================================================================
struct EncodingResult {
    BinaryPolynomial poly;

    int num_logical_qubits = 0;
    int num_auxiliary_qubits = 0;

    // Timing (seconds)
    double time_total = 0;
    double time_parse = 0;
    double time_choice_ordering = 0;
    double time_bitstring_assignment = 0;
    double time_lagrange = 0;
    double time_encoding = 0;
    double time_quadratization = 0;
    double time_nonlinear_refinement = 0;
    double time_output = 0;

    // Encoding-specific
    double lagrange_multiplier = 0;
    double rosenberg_penalty = 0;
    double l2_error = 0;
    double linf_error = 0;
    std::vector<double> spectral_profile;

    // Qubit mapping: qubit_info[q] = (variable_index, local_bit_index)
    std::vector<std::pair<int,int>> qubit_info;
    std::vector<int> qubit_start;  // first qubit index per variable
    std::vector<int> bits_per_var; // bits per variable

    // Choice -> bitstring assignment per variable (binary encodings only;
    // empty for one_hot / domain_wall). choice_to_bitstring[i][c] is the
    // integer bitstring assigned to choice c of variable i. The decoder needs
    // this to invert non-identity assignments produced by Gray ordering,
    // Boltzmann choice ordering, and LI prioritization. Without it, decoding
    // can only assume natural-binary (identity) layout, which is wrong for the
    // enhanced encodings.
    std::vector<std::vector<int>> choice_to_bitstring;
};

// ============================================================================
// High-resolution timer
// ============================================================================
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
public:
    Timer() : t0(Clock::now()) {}
    void reset() { t0 = Clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
};

// ============================================================================
// Gray code utilities
// ============================================================================
inline int to_gray(int n) { return n ^ (n >> 1); }

inline int from_gray(int g) {
    int n = g;
    while (g >>= 1) n ^= g;
    return n;
}

inline int ceil_log2(int n) {
    if (n <= 1) return 0;
    int k = 0;
    int v = 1;
    while (v < n) { v <<= 1; k++; }
    return k;
}

inline int popcount(int n) {
    int c = 0;
    while (n) { c += n & 1; n >>= 1; }
    return c;
}
