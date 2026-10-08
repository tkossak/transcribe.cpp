//! Real-model gated pair detection through the public safe run/result API.
//! Set TRANSCRIBE_PAIR_MODEL (multilingual GGUF) and/or TRANSCRIBE_PAIR_BIN_MODEL
//! plus TRANSCRIBE_PAIR_AUDIO_PL (16 kHz mono PCM16 human Polish speech).

mod common;

use transcribe_cpp::{Error, Model, RunOptions, Session, StreamOptions, Task};

fn pair(codes: [&str; 2]) -> RunOptions {
    RunOptions {
        language_candidates: codes.map(String::from).to_vec(),
        ..Default::default()
    }
}

fn assert_pair_capability(model: &Model) {
    let caps = model.capabilities();
    assert_eq!(model.arch(), "whisper");
    assert!(caps.supports_language_detect, "{caps:?}");
    assert!(caps.supports_language_candidates, "{caps:?}");
    assert!(caps.languages.iter().any(|l| l == "pl"), "{caps:?}");
    assert!(caps.languages.iter().any(|l| l == "en"), "{caps:?}");
}

#[test]
fn pair_detects_each_recording_independently_of_candidate_order() {
    let (Some(polish), Some(english)) = (common::pair_audio("pl"), common::pair_audio("en"))
    else {
        eprintln!("skip pair detection: Polish/English speech fixtures unavailable");
        return;
    };
    for path in common::pair_models("pair_detects_each_recording_independently_of_candidate_order") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let mut session = model.session().unwrap();
        for candidates in [["pl", "en"], ["en", "pl"]] {
            let options = pair(candidates);
            for (audio, expected) in [(&polish, "pl"), (&english, "en"), (&polish, "pl")] {
                let result = session.run(audio, &options).unwrap();
                assert_eq!(result.language.as_deref(), Some(expected), "{path:?}: {result:?}");
            }
        }
    }
}

#[test]
fn pair_batch_keeps_each_utterances_detected_language() {
    let (Some(polish), Some(english)) = (common::pair_audio("pl"), common::pair_audio("en"))
    else {
        eprintln!("skip pair batch: Polish/English speech fixtures unavailable");
        return;
    };
    for path in common::pair_models("pair_batch_keeps_each_utterances_detected_language") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let mut session = model.session().unwrap();
        for candidates in [["pl", "en"], ["en", "pl"]] {
            let results = session
                .run_batch(&[&polish, &english, &polish], &pair(candidates))
                .unwrap();
            assert_eq!(results.len(), 3);
            for (result, expected) in results.into_iter().zip(["pl", "en", "pl"]) {
                let result = result.unwrap();
                assert_eq!(result.language.as_deref(), Some(expected), "{path:?}: {result:?}");
            }
        }
    }
}

#[test]
fn third_language_never_widens_the_pair() {
    let Some(german) = common::pair_audio("de") else {
        eprintln!("skip third-language pair detection: German speech fixture unavailable");
        return;
    };
    for path in common::pair_models("third_language_never_widens_the_pair") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let mut session = model.session().unwrap();
        let first = session.run(&german, &pair(["pl", "en"])).unwrap();
        assert!(
            matches!(first.language.as_deref(), Some("pl") | Some("en")),
            "{path:?}: {first:?}"
        );
        let reversed = session.run(&german, &pair(["en", "pl"])).unwrap();
        assert_eq!(reversed.language, first.language, "{path:?}");
    }
}

#[test]
fn malformed_pairs_and_forced_language_conflicts_are_rejected() {
    let audio = [0.0; 1600];
    for path in common::pair_models("malformed_pairs_and_forced_language_conflicts_are_rejected") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let mut session = model.session().unwrap();
        for codes in [vec!["en"], vec!["pl", "en", "de"], vec!["en", "en"], vec!["", "en"]] {
            let options = RunOptions {
                language_candidates: codes.into_iter().map(String::from).collect(),
                ..Default::default()
            };
            assert!(
                matches!(session.run(&audio, &options), Err(Error::InvalidArgument(_))),
                "{path:?}: {options:?}"
            );
            assert!(
                matches!(session.run_batch(&[&audio], &options), Err(Error::InvalidArgument(_))),
                "{path:?}: {options:?}"
            );
        }
        let conflict = RunOptions {
            language: Some("en".into()),
            ..pair(["pl", "en"])
        };
        assert!(matches!(session.run(&audio, &conflict), Err(Error::InvalidArgument(_))));
        assert!(matches!(session.run_batch(&[&audio], &conflict), Err(Error::InvalidArgument(_))));
    }
}

#[test]
fn unadvertised_candidates_are_rejected_instead_of_falling_back() {
    let audio = [0.0; 1600];
    for path in common::pair_models("unadvertised_candidates_are_rejected_instead_of_falling_back") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let mut session = model.session().unwrap();
        for candidates in [["pl", "not-a-language"], ["not-a-language", "also-not-a-language"], ["pl", "EN"]] {
            let options = pair(candidates);
            assert!(
                matches!(session.run(&audio, &options), Err(Error::Unsupported(_))),
                "{path:?}: {options:?}"
            );
            assert!(
                matches!(session.run_batch(&[&audio], &options), Err(Error::Unsupported(_))),
                "{path:?}: {options:?}"
            );
        }
    }
}

fn assert_incompatible_pair_is_rejected(session: &mut Session) {
    let audio = [0.0; 1600];
    let options = pair(["pl", "en"]);
    assert!(matches!(session.run(&audio, &options), Err(Error::Unsupported(_))));
    assert!(matches!(session.run_batch(&[&audio], &options), Err(Error::Unsupported(_))));
    assert!(matches!(
        session.stream(&options, &StreamOptions::default()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn english_only_whisper_rejects_pair_requests() {
    let Some(path) = common::pair_english_only_model() else {
        eprintln!("skip .en rejection: set TRANSCRIBE_PAIR_EN_MODEL");
        return;
    };
    let model = Model::load(path).unwrap();
    assert_eq!(model.arch(), "whisper");
    let caps = model.capabilities();
    assert!(!caps.supports_language_candidates, "{caps:?}");
    assert_eq!(caps.languages, ["en"]);
    assert_incompatible_pair_is_rejected(&mut model.session().unwrap());
}

#[test]
fn non_whisper_rejects_pair_including_stream_begin() {
    let Some(path) = common::pair_non_whisper_model() else {
        eprintln!("skip non-Whisper rejection: set TRANSCRIBE_PAIR_UNSUPPORTED_MODEL");
        return;
    };
    let model = Model::load(path).unwrap();
    assert_ne!(model.arch(), "whisper");
    let caps = model.capabilities();
    assert!(!caps.supports_language_candidates, "{caps:?}");
    assert_incompatible_pair_is_rejected(&mut model.session().unwrap());
}

#[test]
fn candidate_strings_with_nul_are_rejected_by_all_safe_entry_points() {
    for path in common::pair_models("candidate_strings_with_nul_are_rejected_by_all_safe_entry_points") {
        let mut session = Model::load(path).unwrap().session().unwrap();
        let audio = [0.0; 1600];
        let options = pair(["pl\0en", "en"]);
        assert!(matches!(session.run(&audio, &options), Err(Error::Nul(_))));
        assert!(matches!(session.run_batch(&[&audio], &options), Err(Error::Nul(_))));
        assert!(matches!(session.stream(&options, &StreamOptions::default()), Err(Error::Nul(_))));
    }
}

#[test]
fn translation_keeps_detected_source_when_english_is_outside_pair() {
    let Some(german) = common::canonical_german_audio() else {
        eprintln!("skip pair translation: canonical German speech fixture unavailable");
        return;
    };
    for path in common::pair_models("translation_keeps_detected_source_when_english_is_outside_pair") {
        let model = Model::load(&path).unwrap();
        assert_pair_capability(&model);
        let caps = model.capabilities();
        assert!(caps.supports_translate, "{caps:?}");
        assert!(caps.languages.iter().any(|l| l == "de"), "{caps:?}");
        let mut session = model.session().unwrap();
        let options = RunOptions {
            task: Task::Translate,
            target_language: Some("en".into()),
            ..pair(["de", "pl"])
        };
        let result = session.run(&german, &options).unwrap();
        assert_eq!(result.language.as_deref(), Some("de"), "{path:?}: {result:?}");
        // The canonical german.wav describes the beach and sun. These English
        // words establish translation output independently of source detection.
        let text = result.text.to_lowercase();
        assert!(text.contains("beach") || text.contains("sun"), "{path:?}: {result:?}");
        let batch = session.run_batch(&[&german], &options).unwrap();
        assert_eq!(batch.len(), 1);
        let translated = batch.into_iter().next().unwrap().unwrap();
        assert_eq!(translated.language.as_deref(), Some("de"));
        let text = translated.text.to_lowercase();
        assert!(text.contains("beach") || text.contains("sun"), "{path:?}: {translated:?}");
    }
}

#[test]
fn empty_candidates_preserve_auto_and_single_language() {
    let Some(english) = common::pair_audio("en") else {
        eprintln!("skip Auto/Single preservation: English speech fixture unavailable");
        return;
    };
    for path in common::pair_models("empty_candidates_preserve_auto_and_single_language") {
        let mut session = Model::load(&path).unwrap().session().unwrap();
        for (options, expected) in [
            (RunOptions::default(), Some("en")),
            (
                RunOptions {
                    language: Some("en".into()),
                    ..Default::default()
                },
                None,
            ),
        ] {
            let result = session.run(&english, &options).unwrap();
            // A forced hint is not model-detected language evidence.
            assert_eq!(result.language.as_deref(), expected, "{path:?}: {result:?}");
        }
        let empty_hint = RunOptions {
            language: Some(String::new()),
            ..pair(["pl", "en"])
        };
        let result = session.run(&english, &empty_hint).unwrap();
        assert_eq!(result.language.as_deref(), Some("en"), "{path:?}: {result:?}");
    }
}
