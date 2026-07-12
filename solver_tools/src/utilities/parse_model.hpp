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

    // var_qubits[i][b] = qubit ID holding bit b of variable i, read straight from
    // qubit_map. Decoding through this instead of qubit_start[i] + b drops the
    // assumption that a variable's qubits are contiguous and in bit order.
    std::vector<std::vector<int>> var_qubits;

    // Inverse of the above, indexed by qubit ID. qubit_to_var[q] == -1 marks a
    // qubit that decodes to nothing: a Rosenberg auxiliary. Auxiliaries MUST be
    // distinguishable -- qubit_info default-constructs to {variable 0, bit 0},
    // so an auxiliary is otherwise indistinguishable from bit 0 of variable 0.
    std::vector<int> qubit_to_var;
    std::vector<int> qubit_to_bit;

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
// Decode ONE variable's register. Returns the CFN choice, or -1 if the register
// holds an infeasible pattern (broken one-hot / broken wall / unused bitstring).
// Shared by decode_to_cfn and by the incremental tracker in cfn_tracker.hpp, so
// the trajectory decode and the final decode can never drift apart.
inline int decode_register(const BinaryModel& model, int i,
                           const std::vector<int>& state) {
    int nbits = model.bits_per_var[i];
    const std::vector<int>& qubits = model.var_qubits[i];
    const bool is_spin = (model.var_type == BinaryModel::SPIN);
    const int  n_state = static_cast<int>(state.size());

    // Bit b of the register, as {0,1}. Reads state directly -- the tracker calls
    // this on every accepted flip, so it must not allocate. Returns -1 if the
    // qubit is out of range.
    auto bit = [&](int b) -> int {
        int q = qubits[b];
        if (q < 0 || q >= n_state) return -1;
        return is_spin ? (1 - state[q]) / 2      // +1 -> 0, -1 -> 1
                       : state[q];
    };

    if (model.encoding == "one_hot") {
        // Find the single 1-bit
        int count = 0, choice = -1;
        for (int b = 0; b < nbits; b++) {
            int v = bit(b);
            if (v < 0) return -1;
            if (v == 1) { count++; choice = b; }
        }
        if (count != 1) return -1;   // infeasible
        return choice;
    }

    if (model.encoding == "domain_wall") {
        // Valid pattern is 1...10...0: count leading 1s, then all bits after the
        // wall must be 0.
        int ones = 0;
        bool in_ones = true;
        for (int b = 0; b < nbits; b++) {
            int v = bit(b);
            if (v < 0) return -1;
            if (in_ones && v == 1) { ones++; continue; }
            in_ones = false;
            if (v != 0) return -1;   // infeasible: 1 after the wall
        }
        return ones;
    }

    // Binary encodings: read the register as a bitstring integer.
    int val = 0;
    for (int b = 0; b < nbits; b++) {
        int v = bit(b);
        if (v < 0) return -1;
        val |= (v << b);
    }

    if (!model.bitstring_to_choice.empty() &&
        i < static_cast<int>(model.bitstring_to_choice.size()) &&
        !model.bitstring_to_choice[i].empty()) {
        // Invert the encoder's explicit assignment. This is the only path that
        // decodes Gray / Boltzmann / LI-prioritized layouts correctly; the raw
        // integer is NOT the choice index there.
        const std::vector<int>& inv = model.bitstring_to_choice[i];
        if (val >= static_cast<int>(inv.size()) || inv[val] < 0)
            return -1;   // unused bitstring -> infeasible
        return inv[val];
    }

    // Backward-compatible natural-binary decoding (identity assignment):
    // correct only for naive unsorted+natural models.
    if (val >= model.source_max_cardinality) return -1;   // out of range
    return val;
}

inline std::vector<int> decode_to_cfn(const BinaryModel& model,
                                       const std::vector<int>& state) {
    if (model.source_num_variables == 0 || model.var_qubits.empty())
        return {};

    int N = model.source_num_variables;
    std::vector<int> choices(N, -1);

    for (int i = 0; i < N; i++) {
        int c = decode_register(model, i, state);
        if (c < 0) return {};   // infeasible
        choices[i] = c;
    }
    return choices;
}

// ============================================================================
// Parser
// ============================================================================
inline BinaryModel parse_binary_model_json(const nlohmann::json& j) {
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

        // Build the explicit qubit <-> (variable, bit) maps from qubit_map. -1
        // leaves auxiliaries (Rosenberg y's, absent from qubit_map) marked as
        // decoding to nothing.
        model.qubit_to_var.assign(model.num_qubits, -1);
        model.qubit_to_bit.assign(model.num_qubits, -1);
        model.var_qubits.resize(model.source_num_variables);
        for (int i = 0; i < model.source_num_variables; i++)
            model.var_qubits[i].assign(model.bits_per_var[i], -1);

        for (auto it = j["qubit_map"].begin(); it != j["qubit_map"].end(); ++it) {
            int qid = std::stoi(it.key());
            int var = it.value()["variable"].get<int>();
            int bit = it.value()["bit"].get<int>();
            if (qid < 0 || qid >= model.num_qubits) continue;
            if (var < 0 || var >= model.source_num_variables) continue;
            if (bit < 0 || bit >= model.bits_per_var[var]) continue;

            model.qubit_to_var[qid] = var;
            model.qubit_to_bit[qid] = bit;
            model.var_qubits[var][bit] = qid;
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

// One model per file (encode_cfn's default output).
inline BinaryModel parse_binary_model(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open model file: " + path);

    nlohmann::json j;
    ifs >> j;
    return parse_binary_model_json(j);
}

// ============================================================================
// JSONL: one model per line (encode_cfn --jsonl).
//
// Streamed a line at a time -- never more than one model in memory, so an arm's
// whole .jsonl (thousands of models) is processed in bounded memory.
//
// Sharding is by line index rather than by byte offset because a line is only
// locatable by scanning: seeking to problem k would mean re-reading k lines, so
// one-task-per-problem would re-scan the file 14,400 times. num_shards workers
// each streaming the file once and taking lines where (index % num_shards ==
// shard) reads it num_shards times total instead.
// ============================================================================
class JsonlModelReader {
    std::ifstream ifs_;
    std::string   path_;
    int line_index_  = -1;     // 0-based index of the last line read
    int shard_       = 0;
    int num_shards_  = 1;
    int only_line_   = 0;      // 1-based; 0 => every line in this shard

    // Selection is decided from the line INDEX alone, before the line is parsed.
    // Parsing is by far the dominant per-line cost, so a shard must not pay it
    // for the lines it does not own -- otherwise K shards parse the whole file K
    // times over.
    bool selected(int index) const {
        if (only_line_ > 0) return index == only_line_ - 1;
        return index % num_shards_ == shard_;
    }

public:
    explicit JsonlModelReader(const std::string& path,
                              int shard = 0, int num_shards = 1, int only_line = 0)
        : ifs_(path), path_(path),
          shard_(shard), num_shards_(num_shards), only_line_(only_line) {
        if (!ifs_.is_open())
            throw std::runtime_error("Cannot open JSONL file: " + path);
    }

    // Reads the next non-blank line THIS SHARD OWNS into `model`. Returns false
    // at EOF. A malformed line is reported through `error` and skipped rather
    // than killing the shard: one bad model out of thousands should cost one row.
    bool next(BinaryModel& model, int& index, std::string& error) {
        std::string line;
        while (std::getline(ifs_, line)) {
            line_index_++;

            // Blank lines still consume an index, so a given line has the same
            // index in every shard.
            if (line.find_first_not_of(" \t\r\n") == std::string::npos)
                continue;
            if (!selected(line_index_)) continue;

            index = line_index_;
            try {
                model = parse_binary_model_json(nlohmann::json::parse(line));
                error.clear();
            } catch (const std::exception& e) {
                error = "line " + std::to_string(line_index_ + 1) + ": " + e.what();
            }
            return true;
        }
        return false;
    }

    // True once every line this shard owns has been read.
    bool exhausted() const {
        return only_line_ > 0 && line_index_ >= only_line_ - 1;
    }
};
