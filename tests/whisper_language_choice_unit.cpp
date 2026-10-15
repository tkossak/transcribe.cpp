// Deterministic language decisions at Whisper's private candidate-choice seam.

#include "arch/whisper/language-choice.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
const std::vector<std::string> k_codes = { "en", "pl", "de" };
const std::vector<int32_t> k_token_ids = { 3, 1, 4 };

void check_choice(const char * what,
                  const std::vector<float> & logits,
                  const char * const * candidates,
                  int32_t n_candidates,
                  int expected,
                  const std::vector<int32_t> & token_ids = k_token_ids) {
    const int got = transcribe::whisper::select_language_index(
        k_codes, token_ids, logits.data(), static_cast<int64_t>(logits.size()), candidates, n_candidates);
    if (got != expected) {
        std::fprintf(stderr, "FAIL %s: language index %d, expected %d\n", what, got, expected);
        ++g_failures;
    }
}

}  // namespace

int main(void) {
    const char * pair[] = { "en", "pl" };
    check_choice("higher-scoring candidate wins despite higher outsider", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, pair, 2, 1);

    const char * reversed_pair[] = { "pl", "en" };
    check_choice("reversed pair has the same higher-scoring winner", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f },
                 reversed_pair, 2, 1);
    check_choice("English can win a fresh recording", { 0.0f, -3.0f, 0.0f, -1.0f, 100.0f }, pair, 2, 0);
    check_choice("tie uses model order rather than token ID order", { 0.0f, 4.0f, 0.0f, 4.0f, 100.0f }, pair, 2, 0);
    check_choice("reversed equal-score pair keeps the same winner", { 0.0f, 4.0f, 0.0f, 4.0f, 100.0f },
                 reversed_pair, 2, 0);
    check_choice("Auto still considers the outsider", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, nullptr, 0, 2);
    check_choice("Auto still uses stable model order on ties", { 0.0f, 4.0f, 0.0f, 4.0f, 4.0f }, nullptr, 0, 0);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    check_choice("Auto preserves unresolved all-negative-infinity scores", { 0.0f, neg_inf, 0.0f, neg_inf, neg_inf },
                 nullptr, 0, -1);
    check_choice("pair negative-infinity tie still chooses stable model candidate",
                 { 0.0f, neg_inf, 0.0f, neg_inf, 100.0f }, pair, 2, 0);
    check_choice("reversed negative-infinity tie keeps the same candidate",
                 { 0.0f, neg_inf, 0.0f, neg_inf, 100.0f }, reversed_pair, 2, 0);
    const char * missing_pair[] = { "en", "fr" };
    check_choice("missing model language does not reduce the pair", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f },
                 missing_pair, 2, -1);
    const char * duplicate_pair[] = { "en", "en" };
    check_choice("duplicate candidates cannot become forced language", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f },
                 duplicate_pair, 2, -1);
    check_choice("incomplete candidate request cannot become Auto", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, pair, 1, -1);
    check_choice("missing candidate array cannot become Auto", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, nullptr, 2, -1);
    check_choice("missing candidate token rejects instead of picking the other member",
                 { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, pair, 2, -1, { 3, -1, 4 });
    check_choice("candidate token outside model vocab rejects instead of widening",
                 { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f }, pair, 2, -1, { 3, 5, 4 });
    check_choice("Auto preserves skipping unusable tokens", { 0.0f, 4.0f, 0.0f, 2.0f, 100.0f },
                 nullptr, 0, 2, { 3, -1, 4 });

    if (g_failures > 0) {
        std::fprintf(stderr, "whisper_language_choice_unit: %d failures\n", g_failures);
        return 1;
    }
    std::fprintf(stdout, "whisper_language_choice_unit: ok\n");
    return 0;
}
