#pragma once
#include "utilities/parse_cfn.hpp"
#include <random>
#include <cmath>

// ============================================================================
// Simulated annealing for Cost Function Networks.
//
// Move types: flip, shift, both.
// Tracks the best state encountered during the run.
// ============================================================================

inline RunResult run_sa_cfn(
    const CFNModel& model,
    const SAParams& params,
    uint64_t run_seed
) {
    
    SATimer timer;
    std::mt19937_64 rng(run_seed);
    std::uniform_int_distribution<int> var_dist(0, model.num_variables - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const int N = model.num_variables;
    const int steps = params.num_steps;

    enum MoveType { FLIP, SHIFT, BOTH };
    MoveType move = FLIP;
    if (params.move_type == "shift") move = SHIFT;
    else if (params.move_type == "both") move = BOTH;

    // --- Random initial state ---
    std::vector<int> state(N);
    for (int i = 0; i < N; i++) {
        std::uniform_int_distribution<int> choice(0, model.cardinalities[i] - 1);
        state[i] = choice(rng);
    }

    double energy = compute_energy(model, state);
    double best_energy = energy;
    std::vector<int> best_state = state;

    auto temp_fn = (params.schedule == "linear")
        ? temperature_linear
        : temperature_geometric;

    for (int step = 0; step < steps; step++) {
        int i = var_dist(rng);
        int di = model.cardinalities[i];
        int old_c = state[i];
        int new_c;

        MoveType this_move = move;
        if (move == BOTH)
            this_move = (unif(rng) < 0.5) ? FLIP : SHIFT;

        if (this_move == SHIFT) {
            if (di <= 1) continue;
            if (old_c == 0)
                new_c = 1;
            else if (old_c == di - 1)
                new_c = di - 2;
            else
                new_c = (unif(rng) < 0.5) ? old_c - 1 : old_c + 1;
        } else {
            if (di <= 1) continue;
            std::uniform_int_distribution<int> choice(0, di - 2);
            new_c = choice(rng);
            if (new_c >= old_c) new_c++;
        }

        double dE = delta_energy(model, state, i, new_c);
        double T = temp_fn(params.T_start, params.T_end, step, steps);

        if (dE <= 0.0 || unif(rng) < std::exp(-dE / T)) {
            state[i] = new_c;
            energy += dE;

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
    return result;
}
