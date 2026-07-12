#pragma once
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include "solvers/cfn_tracker.hpp"
#include <random>
#include <cmath>
#include <memory>

// ============================================================================
// Simulated annealing for binary (QUBO / HUBO / Ising) models.
//
// Single-bit flip with O(degree) delta evaluation.
//
// Tracks TWO bests, which are not the same state:
//   best_energy / best_state          lowest ENCODED energy visited
//   best_cfn_energy / best_cfn_state  lowest SOURCE-CFN cost visited, decoded
//                                     along the trajectory (see cfn_tracker.hpp)
// The CFN tracking is active only when `src_cfn` is supplied (--cfn-dir); pass
// nullptr and the *_cfn_* fields come back NaN / empty, at zero cost.
// ============================================================================

inline RunResult run_sa_binary(const BinaryModel& model,
                               const SAParams& params,
                               uint64_t run_seed,
                               const CFNModel* src_cfn = nullptr) {
    SATimer timer;
    std::mt19937_64 rng(run_seed);
    std::uniform_int_distribution<int> qubit_dist(0, model.num_qubits - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const int Q = model.num_qubits;
    const int steps = params.num_steps;
    const bool is_spin = (model.var_type == BinaryModel::SPIN);

    // --- Random initial state ---
    std::vector<int> state(Q);
    if (is_spin) {
        std::uniform_int_distribution<int> bit01(0, 1);
        for (int q = 0; q < Q; q++)
            state[q] = bit01(rng) * 2 - 1;
    } else {
        std::uniform_int_distribution<int> bit01(0, 1);
        for (int q = 0; q < Q; q++)
            state[q] = bit01(rng);
    }

    double energy = compute_energy(model, state);
    double best_energy = energy;
    std::vector<int> best_state = state;

    // Decodes the initial state, then follows every accepted flip.
    std::unique_ptr<CFNTracker> cfn_tracker;
    if (src_cfn && !model.var_qubits.empty())
        cfn_tracker = std::make_unique<CFNTracker>(model, *src_cfn, state);

    auto temp_fn = (params.schedule == "linear")
        ? temperature_linear
        : temperature_geometric;

    for (int step = 0; step < steps; step++) {
        int q = qubit_dist(rng);
        double dE = delta_energy(model, state, q);
        double T = temp_fn(params.T_start, params.T_end, step, steps);

        if (dE <= 0.0 || unif(rng) < std::exp(-dE / T)) {
            if (is_spin)
                state[q] = -state[q];
            else
                state[q] = 1 - state[q];
            energy += dE;

            if (cfn_tracker) cfn_tracker->on_flip(q, state);

            if (energy < best_energy) {
                best_energy = energy;
                best_state = state;
            }
        }
    }

    RunResult result;
    result.best_energy  = best_energy;
    result.final_energy = energy;
    result.runtime_s    = timer.elapsed();
    result.best_state   = std::move(best_state);

    if (cfn_tracker) {
        result.best_cfn_energy = cfn_tracker->best_energy();
        result.best_cfn_state  = cfn_tracker->best_choices();
    }
    return result;
}
