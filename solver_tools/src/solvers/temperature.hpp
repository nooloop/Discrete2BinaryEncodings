#pragma once
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================================
// Per-model temperature calibration.
//
// A temperature is only meaningful relative to the energy scale it acts on:
// Metropolis accepts an uphill move with probability exp(-dE/T), so what a given
// T *does* depends entirely on the size of a typical dE. Encodings of the same
// CFN differ in that scale by nearly two orders of magnitude (a Lagrange lambda
// or a Rosenberg penalty M = 2 max|c_S| dwarfs the objective coefficients), so a
// fixed T window silently anneals each arm at a different place in its own
// landscape. Comparing the arms then compares annealing regimes, not encodings.
//
// The fix: specify the schedule by ACCEPTANCE PROBABILITY -- dimensionless, and
// the thing the schedule is actually for -- and solve for the T that delivers it.
// The two ends are calibrated against DIFFERENT scales, and must be:
//
//   T_start  P(accept a typical uphill move) = accept_start (default 0.80),
//            measured on the ENCODED model:
//                T_start = -mean(dE+ | encoded) / ln(accept_start)
//            The hot end must be able to cross whatever barriers this particular
//            encoding erects, and for OH / DW / quadratized models those barriers
//            ARE the penalty. Deriving it per encoding is what lets a one-hot
//            chain leave a feasible basin at all.
//
//   T_end    P(accept a small uphill move) = accept_end (default 0.01), measured
//            on the SOURCE CFN:
//                T_end = -q10(dE+ | source CFN) / ln(accept_end)
//
// Why the cold end must NOT come from the encoded model: in a penalty encoding
// EVERY single-bit flip moves the penalty, so the encoded dE+ distribution has no
// small-move mode at all. On a 6x4 instance the *minimum* uphill dE over 4000
// one-hot probes is 8.06 -- there is no such thing as a cheap flip. Taking a low
// quantile of that lands the cold end on the PENALTY scale (T_end ~ 3.8), where
// an objective-scale difference of ~1 is still accepted with probability
// exp(-1/3.8) = 0.77: the chain would finish the anneal still diffusing over the
// objective, never optimizing it. Calibrating the cold end on the source CFN
// pins it to the structure the solver is actually scored on, and pins it to the
// SAME criterion for every arm -- every encoding here represents the CFN cost in
// CFN units (the penalty is added on top), so this is the one scale they share.
//
// Both endpoints are linear in the energy scale they are measured against, so
// multiplying every coefficient by k multiplies T by k: the anneal is scale-free
// and invariant to a rescaling of the model, which the fixed window was not.
//
// dE is sampled from the SAME move distribution the solver uses (a random single
// bit flip; a flip/shift/both reassignment in CFN mode), at random states, so the
// estimate matches the chain's actual step. Cost is temp_probes single-move delta
// evaluations, negligible next to num_runs * num_steps.
//
// Without --cfn-dir there is no source CFN to calibrate the cold end against, and
// it falls back to the encoded low decile (correct for the penalty-free
// encodings, too hot for the rest -- another reason to always pass --cfn-dir).
// ============================================================================

struct TemperatureRange {
    double T_start;
    double T_end;
};

// Mean of a sorted uphill sample: the "typical" barrier, for the hot end.
inline double uphill_mean(const std::vector<double>& uphill) {
    if (uphill.empty()) return 0.0;
    return std::accumulate(uphill.begin(), uphill.end(), 0.0)
         / static_cast<double>(uphill.size());
}

// Low decile of a sorted uphill sample: the fine structure the chain should
// still resolve at the cold end. Falls back to the smallest strictly-positive
// sample if the decile is degenerate (a heavily-tied landscape).
inline double uphill_low_decile(const std::vector<double>& sorted_uphill) {
    if (sorted_uphill.empty()) return 0.0;
    double q10 = sorted_uphill[sorted_uphill.size() / 10];
    if (q10 > 0.0) return q10;
    auto it = std::upper_bound(sorted_uphill.begin(), sorted_uphill.end(), 0.0);
    return (it != sorted_uphill.end()) ? *it : 0.0;
}

// ---- Binary models: the move is a single random bit flip ----
inline std::vector<double> sample_uphill(const BinaryModel& model,
                                         const SAParams& params,
                                         uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> qubit_dist(0, model.num_qubits - 1);
    std::uniform_int_distribution<int> bit01(0, 1);

    const bool is_spin = (model.var_type == BinaryModel::SPIN);

    std::vector<int> state(model.num_qubits);
    std::vector<double> uphill;
    uphill.reserve(params.temp_probes);

    for (int t = 0; t < params.temp_probes; t++) {
        for (int q = 0; q < model.num_qubits; q++) {
            int b = bit01(rng);
            state[q] = is_spin ? (b * 2 - 1) : b;
        }
        double dE = delta_energy(model, state, qubit_dist(rng));
        if (dE > 0.0) uphill.push_back(dE);
    }
    std::sort(uphill.begin(), uphill.end());
    return uphill;
}

// ---- CFN models: the move is the configured flip / shift / both ----
inline std::vector<double> sample_uphill(const CFNModel& model,
                                         const SAParams& params,
                                         uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> var_dist(0, model.num_variables - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const bool shift_moves = (params.move_type == "shift");
    const bool both_moves  = (params.move_type == "both");

    std::vector<int> state(model.num_variables);
    std::vector<double> uphill;
    uphill.reserve(params.temp_probes);

    for (int t = 0; t < params.temp_probes; t++) {
        for (int i = 0; i < model.num_variables; i++) {
            std::uniform_int_distribution<int> choice(0, model.cardinalities[i] - 1);
            state[i] = choice(rng);
        }

        int i  = var_dist(rng);
        int di = model.cardinalities[i];
        if (di <= 1) continue;
        int old_c = state[i];
        int new_c;

        bool shift = shift_moves || (both_moves && unif(rng) < 0.5);
        if (shift) {
            if (old_c == 0)            new_c = 1;
            else if (old_c == di - 1)  new_c = di - 2;
            else                       new_c = (unif(rng) < 0.5) ? old_c - 1 : old_c + 1;
        } else {
            std::uniform_int_distribution<int> choice(0, di - 2);
            new_c = choice(rng);
            if (new_c >= old_c) new_c++;
        }

        double dE = delta_energy(model, state, i, new_c);
        if (dE > 0.0) uphill.push_back(dE);
    }
    std::sort(uphill.begin(), uphill.end());
    return uphill;
}

// Hot end from the model the chain actually moves on; cold end from
// `objective_ref` -- the source CFN, the scale the solver is scored on. Pass
// nullptr only when no source CFN is available; the cold end then falls back to
// the encoded model and runs too hot on the penalty encodings.
template <typename Model>
inline TemperatureRange calibrate_temperatures(const Model& model,
                                               const CFNModel* objective_ref,
                                               const SAParams& params,
                                               uint64_t seed) {
    TemperatureRange tr{params.T_start, params.T_end};   // fallbacks

    std::vector<double> enc = sample_uphill(model, params, seed);
    if (enc.empty()) return tr;   // no uphill move: no scale to calibrate against

    double mean = uphill_mean(enc);

    // The cold end is measured on the source CFN when we have it.
    double q10 = objective_ref
        ? uphill_low_decile(sample_uphill(*objective_ref, params, seed + 1))
        : uphill_low_decile(enc);
    if (!(q10 > 0.0)) q10 = uphill_low_decile(enc);

    if (!(mean > 0.0) || !(q10 > 0.0)) return tr;

    tr.T_start = -mean / std::log(params.accept_start);
    tr.T_end   = -q10  / std::log(params.accept_end);

    // A monotone decreasing schedule needs T_end < T_start. These are measured
    // against different models, so nothing structurally forbids an inversion --
    // it just means the objective's fine scale is as coarse as the encoded
    // barriers (a penalty-free encoding of a flat CFN). Cool by a fixed ratio
    // rather than emit a rising "cooling" schedule.
    if (!(tr.T_end < tr.T_start))
        tr.T_end = tr.T_start * 1e-3;

    return tr;
}

// Apply calibration to `params`, leaving any endpoint the user pinned explicitly.
// Derived from a stream independent of the run seeds, so it is deterministic and
// identical across every run of a model.
template <typename Model>
inline void apply_temperature_calibration(SAParams& params, const Model& model,
                                          const CFNModel* objective_ref) {
    if (!params.auto_temp) return;
    if (params.T_start_given && params.T_end_given) return;

    TemperatureRange tr = calibrate_temperatures(
        model, objective_ref, params, params.seed ^ 0x9E3779B97F4A7C15ULL);

    if (!params.T_start_given) params.T_start = tr.T_start;
    if (!params.T_end_given)   params.T_end   = tr.T_end;
}
