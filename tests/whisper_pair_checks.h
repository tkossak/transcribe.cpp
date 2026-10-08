// Public run/result regressions shared by GGUF and legacy binary smoke tests.
#pragma once

#include "transcribe.h"
#include "transcribe/whisper.h"

#include <cstring>
#include <string>
#include <vector>

static void check_whisper_pair_results(transcribe_session * session,
                                       const std::vector<float> & english,
                                       const std::vector<float> & german) {
    const char * pair[] = { "en", "de" };
    transcribe_run_params p;
    transcribe_run_params_init(&p);
    p.language_candidates = pair;
    p.n_language_candidates = 2;
    p.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    CHECK(transcribe_run(session, english.data(), static_cast<int>(english.size()), &p) == TRANSCRIBE_OK);
    CHECK(std::strcmp(transcribe_detected_language(session), "en") == 0);
    CHECK(transcribe_run(session, german.data(), static_cast<int>(german.size()), &p) == TRANSCRIBE_OK);
    CHECK(std::strcmp(transcribe_detected_language(session), "de") == 0);
    const std::string detected = transcribe_detected_language(session);
    pair[0] = "de";
    pair[1] = "en";
    CHECK(transcribe_run(session, german.data(), static_cast<int>(german.size()), &p) == TRANSCRIBE_OK);
    CHECK(detected == transcribe_detected_language(session));

    const float * inputs[] = { english.data(), german.data() };
    const int lengths[] = { static_cast<int>(english.size()), static_cast<int>(german.size()) };
    CHECK(transcribe_run_batch(session, inputs, lengths, 2, &p) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_n_results(session) == 2);
    CHECK(transcribe_batch_status(session, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(session, 1) == TRANSCRIBE_OK);
    CHECK(std::strcmp(transcribe_batch_detected_language(session, 0), "en") == 0);
    CHECK(std::strcmp(transcribe_batch_detected_language(session, 1), "de") == 0);

    // A one-utterance batch always takes Whisper's serial fallback, on CPU
    // and GPU. Use a valid long-form prompt policy, then verify each slot
    // succeeded before inspecting its independently detected source.
    transcribe_whisper_run_ext w;
    transcribe_whisper_run_ext_init(&w);
    w.prompt_condition = TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS;
    w.condition_on_prev_tokens = true;
    p.family = &w.ext;
    for (int i = 0; i < 2; ++i) {
        CHECK(transcribe_run_batch(session, inputs + i, lengths + i, 1, &p) == TRANSCRIBE_OK);
        CHECK(transcribe_batch_n_results(session) == 1);
        CHECK(transcribe_batch_status(session, 0) == TRANSCRIBE_OK);
        CHECK(std::strcmp(transcribe_batch_detected_language(session, 0), i == 0 ? "en" : "de") == 0);
    }
    p.family = nullptr;

    // More than one encoder window: the later German speech cannot change
    // the initial English recognition decision. This is a timing invariant,
    // not a claim of mixed-language transcription quality.
    std::vector<float> long_audio;
    while (long_audio.size() < 480000) {
        long_audio.insert(long_audio.end(), english.begin(), english.end());
    }
    long_audio.resize(480000);
    long_audio.insert(long_audio.end(), german.begin(), german.end());
    CHECK(transcribe_run(session, long_audio.data(), static_cast<int>(long_audio.size()), &p) == TRANSCRIBE_OK);
    CHECK(std::strcmp(transcribe_detected_language(session), "en") == 0);

    // The source decision stays within the pair even when the input language
    // is outside it. Do not impose an accuracy or confidence threshold.
    pair[0] = "pl";
    pair[1] = "fr";
    CHECK(transcribe_run(session, english.data(), static_cast<int>(english.size()), &p) == TRANSCRIBE_OK);
    const std::string outsider_choice = transcribe_detected_language(session);
    CHECK(outsider_choice == "pl" || outsider_choice == "fr");

    pair[0] = "de";
    pair[1] = "pl";
    p.task = TRANSCRIBE_TASK_TRANSLATE;
    p.target_language = "en";
    CHECK(transcribe_run(session, german.data(), static_cast<int>(german.size()), &p) == TRANSCRIBE_OK);
    CHECK(std::strcmp(transcribe_detected_language(session), "de") == 0);
    const std::string translated = transcribe_full_text(session);
    CHECK(translated.find("beach") != std::string::npos || translated.find("sun") != std::string::npos);
}
