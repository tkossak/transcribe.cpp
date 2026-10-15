// Focused public-ABI pair regression for either Whisper GGUF or legacy .bin.
#include "transcribe.h"
#include "transcribe/whisper.h"
#include "wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures; \
        } \
    } while (0)

#include "whisper_pair_checks.h"
}  // namespace

int main() {
    const char * path = std::getenv("TRANSCRIBE_PAIR_MODEL");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "TRANSCRIBE_PAIR_MODEL unset: skipping\n");
        return 77;
    }
    const char * backend_dir = std::getenv("TRANSCRIBE_PAIR_BACKEND_DIR");
    CHECK((backend_dir != nullptr ? transcribe_init_backends(backend_dir) :
                                   transcribe_init_backends_default()) == TRANSCRIBE_OK);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    std::vector<float> english;
    std::vector<float> german;
    std::string error;
    CHECK(transcribe_cli::load_wav_mono_16k(
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav", english, error));
    CHECK(transcribe_cli::load_wav_mono_16k(
        std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/german.wav", german, error));
    if (failures != 0) {
        std::fprintf(stderr, "audio load failed: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    transcribe_model_load_params load;
    transcribe_model_load_params_init(&load);
    load.backend = TRANSCRIBE_BACKEND_CPU;
    transcribe_model * model = nullptr;
    CHECK(transcribe_model_load_file(path, &load, &model) == TRANSCRIBE_OK);
    if (model == nullptr) {
        return EXIT_FAILURE;
    }
    CHECK(std::strcmp(transcribe_model_arch_string(model), "whisper") == 0);
    transcribe_capabilities capabilities;
    transcribe_capabilities_init(&capabilities);
    CHECK(transcribe_model_get_capabilities(model, &capabilities) == TRANSCRIBE_OK);
    CHECK(capabilities.supports_language_candidates);
    transcribe_session_params options;
    transcribe_session_params_init(&options);
    transcribe_session * session = nullptr;
    CHECK(transcribe_session_init(model, &options, &session) == TRANSCRIBE_OK);
    if (session != nullptr) {
        check_whisper_pair_results(session, english, german);
        transcribe_session_free(session);
    }
    transcribe_model_free(model);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
