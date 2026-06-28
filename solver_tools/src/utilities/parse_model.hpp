#pragma once
#include "baseline/sa_types.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>

// ============================================================================
// Binary model (QUBO / HUBO / Ising) loaded from encoding_tools JSON.
// ============================================================================

struct BinaryModel {
    enum VarType { BINARY, SPIN };

    VarType     var_type;
    int         num_qubits;          // logical + auxiliary
    int         num_logical;
    int         num_auxiliary;
    double      offset;
    std::string encoding;
    std::string source_cfn;

    // --- Source-CFN properties (extracted from qubit_map) ---
    int source_num_variables   = 0;
    int source_max_cardinality = 0;

    // --- Qubit-to-variable mapping (for solution decoding) ---
    struct QubitInfo {
        int variable;
        int bit;
    };
    std::vector<QubitInfo> qubit_info;   // indexed by qubit ID (logical only)
    std::vector<int> qubit_start;        // first qubit index per variable
    std::vector<int> bits_per_var;       // bits per variable

    // Inverse choice<->bitstring assignment for binary encodings, built from
    // the encoder's "choice_to_bitstring" field. bitstring_to_choice[i][bs] is
    // the CFN choice that bitstring bs maps to for variable i, or -1 if that
    // bitstring is unused (decodes as infeasible). Empty when the field is
    // absent (older models): decoding then falls back to natural-binary, which
    // is only correct for naive (unsorted + natural, no LI) assignments.
    std::vector<std::vector<int>> bitstring_to_choice;

    // --- Flat term storage ---
    struct Term {
        std::vector<int> qubits;
        double coeff;
    };
    std::vector<Term> terms;

    // --- Adjacency for delta evaluation ---
    struct Contribution {
        double coeff;
        std::vector<int> other_qubits;
    };
    std::vector<std::vector<Contribution>> adjacency;

    // --- Source-CFN metadata (parsed from source_cfn name) ---
    CFNMetadata meta;

    void build_adjacency() {
        adjacency.clear();
        adjacency.resize(num_qubits);
        for (const auto& t : terms) {
            for (int i = 0; i < static_cast<int>(t.qubits.size()); i++) {
                int q = t.qubits[i];
                Contribution c;
                c.coeff = t.coeff;
                c.other_qubits.reserve(t.qubits.size() - 1);
                for (int j = 0; j < static_cast<int>(t.qubits.size()); j++) {
                    if (j != i) c.other_qubits.push_back(t.qubits[j]);
                }
                adjacency[q].push_back(std::move(c));
            }
        }
    }
};

// ============================================================================
// Full-evaluation energy (for initialization / verification)
// ============================================================================
inline double compute_energy(const BinaryModel& model,
                             const std::vector<int>& state) {
    double E = model.offset;
    for (const auto& t : model.terms) {
        double prod = t.coeff;
        for (int q : t.qubits) {
            prod *= state[q];
            if (prod == 0.0) break;
        }
        E += prod;
    }
    return E;
}

// ============================================================================
// Delta energy for flipping qubit q.
//
//   BINARY {0,1}:  DE = (1 - 2*b_q) * sum C_S prod b_{q'}
//   SPIN {-1,+1}:  DE = -2 * s_q    * sum C_S prod s_{q'}
// ============================================================================
inline double delta_energy(const BinaryModel& model,
                           const std::vector<int>& state, int q) {
    double sum = 0.0;
    for (const auto& c : model.adjacency[q]) {
        double prod = c.coeff;
        for (int oq : c.other_qubits) {
            prod *= state[oq];
            if (prod == 0.0) break;
        }
        sum += prod;
    }
    if (model.var_type == BinaryModel::BINARY)
        return (1 - 2 * state[q]) * sum;
    else
        return -2 * state[q] * sum;
}

// ============================================================================
// Decode binary state to CFN choices.
//
//   one_hot:      choice = which bit is 1 in the register
//   domain_wall:  choice = number of leading 1-bits
//   *_binary:     choice = bitstring_to_choice[i][register value] when the
//                 encoder supplied an explicit assignment (handles Gray /
//                 Boltzmann / LI layouts); otherwise the natural register value
//                 (correct only for naive unsorted+natural models).
//
// For SPIN variables, converts s -> b = (1-s)/2 before reading.
// Returns empty vector if decoding fails (infeasible state).
// ============================================================================
inline std::vector<int> decode_to_cfn(const BinaryModel& model,
                                       const std::vector<int>& state) {
    if (model.source_num_variables == 0 || model.qubit_start.empty())
        return {};

    int N = model.source_num_variables;
    std::vector<int> choices(N, -1);

    for (int i = 0; i < N; i++) {
        int start = model.qubit_start[i];
        int nbits = model.bits_per_var[i];

        // Read register bits (convert SPIN to BINARY if needed)
        std::vector<int> reg(nbits);
        for (int b = 0; b < nbits; b++) {
            int q = start + b;
            if (q >= static_cast<int>(state.size())) return {};
            if (model.var_type == BinaryModel::SPIN)
                reg[b] = (1 - state[q]) / 2;   // +1 -> 0, -1 -> 1
            else
                reg[b] = state[q];
        }

        if (model.encoding == "one_hot") {
            // Find the single 1-bit
            int count = 0, choice = -1;
            for (int b = 0; b < nbits; b++) {
                if (reg[b] == 1) { count++; choice = b; }
            }
            if (count != 1) return {};   // infeasible
            choices[i] = choice;
        }
        else if (model.encoding == "domain_wall") {
            // Count leading 1s: valid pattern is 1...10...0
            int ones = 0;
            while (ones < nbits && reg[ones] == 1) ones++;
            // Check remaining bits are 0
            for (int b = ones; b < nbits; b++)
                if (reg[b] != 0) return {};   // infeasible
            choices[i] = ones;
        }
        else {
            // Binary encodings: read the register as a bitstring integer.
            int val = 0;
            for (int b = 0; b < nbits; b++)
                val |= (reg[b] << b);

            if (!model.bitstring_to_choice.empty() &&
                i < static_cast<int>(model.bitstring_to_choice.size()) &&
                !model.bitstring_to_choice[i].empty()) {
                // Invert the encoder's explicit assignment. This is the only
                // path that decodes Gray / Boltzmann / LI-prioritized layouts
                // correctly; the raw integer is NOT the choice index there.
                const std::vector<int>& inv = model.bitstring_to_choice[i];
                if (val >= static_cast<int>(inv.size()) || inv[val] < 0)
                    return {};   // unused bitstring -> infeasible
                choices[i] = inv[val];
            } else {
                // Backward-compatible natural-binary decoding (identity
                // assignment): correct only for naive unsorted+natural models.
                if (val >= model.source_max_cardinality) return {};  // out of range
                choices[i] = val;
            }
        }
    }
    return choices;
}

// ============================================================================
// Parser
// ============================================================================
inline BinaryModel parse_binary_model(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open model file: " + path);

    nlohmann::json j;
    ifs >> j;

    BinaryModel model;
    model.encoding     = j["encoding"].get<std::string>();
    std::string vt     = j["variable_type"].get<std::string>();
    model.var_type     = (vt == "SPIN") ? BinaryModel::SPIN : BinaryModel::BINARY;
    model.num_logical  = j["num_logical_qubits"].get<int>();
    model.num_auxiliary = j["num_auxiliary_qubits"].get<int>();
    model.num_qubits   = model.num_logical + model.num_auxiliary;
    model.offset       = j["offset"].get<double>();
    model.source_cfn   = j.value("source_cfn", "");

    // Parse terms
    const auto& jterms = j["terms"];
    model.terms.reserve(jterms.size());
    for (auto it = jterms.begin(); it != jterms.end(); ++it) {
        BinaryModel::Term t;
        t.coeff = it.value().get<double>();
        const std::string& key = it.key();
        std::istringstream ss(key);
        std::string token;
        while (std::getline(ss, token, ','))
            t.qubits.push_back(std::stoi(token));
        std::sort(t.qubits.begin(), t.qubits.end());

        if (!t.qubits.empty()) {
            int mx = t.qubits.back() + 1;
            if (mx > model.num_qubits)
                model.num_qubits = mx;
        }
        model.terms.push_back(std::move(t));
    }

    // Parse qubit_map for decoding and metadata
    if (j.contains("qubit_map")) {
        std::unordered_map<int, int> max_bit_per_var;
        std::unordered_map<int, int> min_qubit_per_var;

        model.qubit_info.resize(model.num_qubits);
        for (auto it = j["qubit_map"].begin(); it != j["qubit_map"].end(); ++it) {
            int qid = std::stoi(it.key());
            int var = it.value()["variable"].get<int>();
            int bit = it.value()["bit"].get<int>();

            if (qid < model.num_qubits)
                model.qubit_info[qid] = {var, bit};

            auto fmb = max_bit_per_var.find(var);
            if (fmb == max_bit_per_var.end())
                max_bit_per_var[var] = bit;
            else
                fmb->second = std::max(fmb->second, bit);

            auto fmq = min_qubit_per_var.find(var);
            if (fmq == min_qubit_per_var.end())
                min_qubit_per_var[var] = qid;
            else
                fmq->second = std::min(fmq->second, qid);
        }

        model.source_num_variables = static_cast<int>(max_bit_per_var.size());

        // Build qubit_start and bits_per_var arrays
        model.qubit_start.resize(model.source_num_variables);
        model.bits_per_var.resize(model.source_num_variables);
        for (const auto& [var, mb] : max_bit_per_var) {
            model.bits_per_var[var] = mb + 1;
            model.qubit_start[var] = min_qubit_per_var[var];
        }

        // Determine cardinality from encoding type
        int max_bits = 0;
        for (const auto& [var, mb] : max_bit_per_var)
            max_bits = std::max(max_bits, mb + 1);

        if (model.encoding == "one_hot")
            model.source_max_cardinality = max_bits;
        else if (model.encoding == "domain_wall")
            model.source_max_cardinality = max_bits + 1;
        else
            model.source_max_cardinality = 1 << max_bits;
    }

    // Parse choice->bitstring assignment (if present) and precompute the
    // inverse map used by decode_to_cfn. Must run after the qubit_map block so
    // bits_per_var is known.
    if (j.contains("choice_to_bitstring")) {
        const auto& cb = j["choice_to_bitstring"];
        int N = static_cast<int>(cb.size());
        model.bitstring_to_choice.assign(N, {});
        for (int i = 0; i < N; i++) {
            int nbits = (i < static_cast<int>(model.bits_per_var.size()))
                            ? model.bits_per_var[i] : 0;
            int total_bs = 1 << nbits;
            std::vector<int> inv(total_bs, -1);
            const auto& var_assign = cb[i];
            for (int c = 0; c < static_cast<int>(var_assign.size()); c++) {
                int bs = var_assign[c].get<int>();
                if (bs >= 0 && bs < total_bs)
                    inv[bs] = c;
            }
            model.bitstring_to_choice[i] = std::move(inv);
        }
    }

    model.meta = parse_cfn_metadata(model.source_cfn);
    model.build_adjacency();
    return model;
}
