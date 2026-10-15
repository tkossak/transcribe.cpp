// Private selection policy shared by serial and batched Whisper detection.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace transcribe::whisper {
inline bool has_language_pair(const std::vector<std::string> & codes,
                              const std::vector<int32_t> & token_ids,
                              int64_t vocab_size,
                              bool supports_language_detect) {
    if (!supports_language_detect || codes.size() < 2 || codes.size() != token_ids.size()) {
        return false;
    }
    bool has_distinct_language = false;
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i].empty() || token_ids[i] < 0 || token_ids[i] >= vocab_size) {
            return false;
        }
        if (codes[i] != codes[0] && token_ids[i] != token_ids[0]) {
            has_distinct_language = true;
        }
    }
    return has_distinct_language;
}


inline int select_language_index(const std::vector<std::string> & codes,
                                 const std::vector<int32_t> & token_ids,
                                 const float * logits,
                                 int64_t vocab_size,
                                 const char * const * candidates,
                                 int32_t n_candidates) {
    int candidate_indices[2] = { -1, -1 };
    if (n_candidates != 0) {
        if (n_candidates != 2 || candidates == nullptr || candidates[0] == nullptr || candidates[1] == nullptr ||
            candidates[0][0] == '\0' || candidates[1][0] == '\0' || std::strcmp(candidates[0], candidates[1]) == 0 ||
            codes.size() != token_ids.size()) {
            return -1;
        }
        for (size_t i = 0; i < codes.size(); ++i) {
            for (int c = 0; c < 2; ++c) {
                if (codes[i] == candidates[c]) {
                    const int32_t id = token_ids[i];
                    if (id < 0 || id >= vocab_size) {
                        return -1;
                    }
                    candidate_indices[c] = static_cast<int>(i);
                }
            }
        }
        if (candidate_indices[0] < 0 || candidate_indices[1] < 0) {
            return -1;
        }
        // Resolve picker order to stable model order before comparing only
        // the two candidate scores.
        if (candidate_indices[0] > candidate_indices[1]) {
            const int first = candidate_indices[0];
            candidate_indices[0] = candidate_indices[1];
            candidate_indices[1] = first;
        }
        float best = -std::numeric_limits<float>::infinity();
        int best_index = -1;
        for (int c = 0; c < 2; ++c) {
            const int index = candidate_indices[c];
            const float score = logits[token_ids[static_cast<size_t>(index)]];
            if (score > best || (best_index < 0 && score == best)) {
                best = score;
                best_index = index;
            }
        }
        return best_index;
    }

    // Iterate model order, never picker order: equal scores keep the first
    // native language token, just as unrestricted detection does.
    float best = -std::numeric_limits<float>::infinity();
    int best_index = -1;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        const int32_t id = token_ids[i];
        if (id >= 0 && id < vocab_size && logits[id] > best) {
            best = logits[id];
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

}  // namespace transcribe::whisper
