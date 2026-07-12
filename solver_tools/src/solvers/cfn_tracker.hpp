#pragma once
#include "utilities/parse_model.hpp"
#include "utilities/parse_cfn.hpp"
#include <limits>

// ============================================================================
// Tracks the SOURCE-CFN cost of a binary SA trajectory, state by state, and
// keeps the best feasible one.
//
// Why not just decode the run's best state at the end? Because the state that
// minimizes the ENCODED energy is not the state that minimizes the CFN cost:
//
//   - AB / TB encode a surrogate objective, so their energy ordering is not the
//     CFN cost ordering;
//   - OH / DW / quadratized models add a Lagrange or Rosenberg penalty, and a
//     run's lowest-energy state can be an infeasible local minimum that decodes
//     to nothing, throwing away a run that did visit good feasible states.
//
// Decoding every visited state is the SA counterpart of solve_qa decoding every
// read rather than only the lowest-energy sample.
//
// COST: O(deg_i) per ACCEPTED flip, the same order as the SA delta itself.
// A flip of qubit q changes exactly one CFN variable's register (or none, for a
// Rosenberg auxiliary), so only that variable's unary term and its incident
// pairwise terms move.
//
// INFEASIBLE REGISTERS: `partial_` sums only the terms all of whose variables
// currently decode feasibly, and `num_infeasible_` counts the registers that do
// not. When num_infeasible_ == 0 every term is included, so partial_ IS the CFN
// cost of the fully-decoded state. Carrying the sum across infeasible states
// this way (rather than rebuilding it whenever the walk re-enters the feasible
// set) is what keeps the update O(deg_i): a one-hot walk breaks and restores
// feasibility on almost every flip, so an O(N + |E|) rebuild per re-entry would
// dominate the anneal.
// ============================================================================
class CFNTracker {
public:
    CFNTracker(const BinaryModel& bm, const CFNModel& cfn,
               const std::vector<int>& state)
        : bm_(bm), cfn_(cfn) {
        reset(state);
    }

    // Decode `state` from scratch and start tracking it. O(N + |E|).
    void reset(const std::vector<int>& state) {
        int N = bm_.source_num_variables;
        choices_.assign(N, -1);
        num_infeasible_ = 0;
        partial_ = 0.0;

        for (int i = 0; i < N; i++) {
            choices_[i] = decode_register(bm_, i, state);
            if (choices_[i] < 0) num_infeasible_++;
        }
        for (int i = 0; i < N; i++)
            if (choices_[i] >= 0) partial_ += cfn_.unary[i][choices_[i]];
        for (const auto& pt : cfn_.pairwise) {
            int ci = choices_[pt.vi], cj = choices_[pt.vj];
            if (ci >= 0 && cj >= 0) partial_ += pt.cost(ci, cj);
        }
        record();
    }

    // Call once per ACCEPTED flip, AFTER `state` has been updated. O(deg_i).
    void on_flip(int q, const std::vector<int>& state) {
        int i = bm_.qubit_to_var[q];
        if (i < 0) return;   // Rosenberg auxiliary: decodes to nothing

        int old_c = choices_[i];
        int new_c = decode_register(bm_, i, state);
        if (new_c == old_c) return;   // e.g. a flip inside an already-broken register

        // Swap variable i's contributions. The neighbours' choices are unchanged,
        // so every term not touching i is untouched.
        if (old_c >= 0) partial_ -= cfn_.unary[i][old_c];
        if (new_c >= 0) partial_ += cfn_.unary[i][new_c];

        for (const auto& adj : cfn_.adjacency[i]) {
            int cj = choices_[adj.neighbor];
            if (cj < 0) continue;   // edge is excluded from partial_ either way
            const auto& pt = cfn_.pairwise[adj.table_idx];
            if (adj.i_is_first) {
                if (old_c >= 0) partial_ -= pt.costs[old_c * pt.dj + cj];
                if (new_c >= 0) partial_ += pt.costs[new_c * pt.dj + cj];
            } else {
                if (old_c >= 0) partial_ -= pt.costs[cj * pt.dj + old_c];
                if (new_c >= 0) partial_ += pt.costs[cj * pt.dj + new_c];
            }
        }

        if (old_c < 0 && new_c >= 0) num_infeasible_--;
        if (old_c >= 0 && new_c < 0) num_infeasible_++;
        choices_[i] = new_c;

        record();
    }

    // NaN / empty if the trajectory never visited a feasible state.
    double best_energy() const { return best_; }
    const std::vector<int>& best_choices() const { return best_choices_; }

private:
    void record() {
        if (num_infeasible_ != 0) return;                      // not decodable
        // NB: every comparison against a NaN best_ is false, so the first
        // feasible state must be taken on the isnan branch, not by comparison.
        if (!std::isnan(best_) && !(partial_ < best_)) return;

        // partial_ is an incremental sum over the whole trajectory, so it carries
        // accumulated rounding. Re-evaluate the CFN exactly on the (rare)
        // improvements so the REPORTED cost is drift-free and matches what
        // compute_energy(cfn, best_choices) gives downstream.
        best_ = compute_energy(cfn_, choices_);
        best_choices_ = choices_;
    }

    const BinaryModel& bm_;
    const CFNModel&    cfn_;

    std::vector<int> choices_;          // -1 = register decodes infeasibly
    int              num_infeasible_ = 0;
    double           partial_ = 0.0;    // CFN cost when num_infeasible_ == 0

    double           best_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> best_choices_;
};
