//! Shared fixtures for the integration tests, mirroring the Python conftest.
//!
//! Two tiers: no-model tests always run; model-gated tests resolve a local
//! whisper-tiny.en + jfk.wav and return `None` (clean skip) when absent. The
//! "country" content assertion is specific to jfk.wav and holds for any English
//! ASR model.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Once;

/// Register backend modules before the first model load. Idempotent and a no-op
/// in compiled-in (static / plain `shared`) builds; in a `dynamic-backends`
/// build it loads the per-ISA CPU / GPU modules the native build installed —
/// without it a model load in that posture finds zero devices. Folded into the
/// fixture resolvers below so every model-gated test gets it for free.
fn ensure_backends() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        transcribe_cpp::init_backends_default().expect("init_backends_default");
    });
}

/// Repo root: walk up from this crate until we find the marker that only the
/// repo root carries (immune to how deep the crate dir is nested).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("include/transcribe.h").is_file() && p.join("ggml").is_dir())
        .expect("could not locate repo root from CARGO_MANIFEST_DIR")
        .to_path_buf()
}

/// The smoke model path, or `None` when the GGUF is not present.
pub fn smoke_model() -> Option<PathBuf> {
    ensure_backends();
    let path = std::env::var_os("TRANSCRIBE_SMOKE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf"));
    path.is_file().then_some(path)
}

/// The smoke audio as 16 kHz mono f32 PCM, or `None` when the WAV is absent.
pub fn smoke_audio() -> Option<Vec<f32>> {
    let path = std::env::var_os("TRANSCRIBE_SMOKE_AUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("samples/jfk.wav"));
    path.is_file().then(|| load_wav(&path))
}

/// The streaming smoke model (moonshine-streaming-tiny), or `None` if absent.
pub fn smoke_streaming_model() -> Option<PathBuf> {
    ensure_backends();
    let path = std::env::var_os("TRANSCRIBE_SMOKE_STREAMING_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repo_root().join("models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf")
        });
    path.is_file().then_some(path)
}

/// A feature-specific canary: env override or the in-repo GGUF, `None` when
/// neither is present (clean skip). CI exports the lightweight shared canaries;
/// heavyweight models such as Voxtral remain local-only.
fn family_model(env_var: &str, default_rel: &str) -> Option<PathBuf> {
    ensure_backends();
    let path = std::env::var_os(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join(default_rel));
    path.is_file().then_some(path)
}

/// Cache-aware parakeet streaming canary (accepts PARAKEET_STREAM).
pub fn smoke_parakeet_stream_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL",
        "models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf",
    )
}

/// Chunked/buffered parakeet streaming canary (accepts PARAKEET_BUFFERED_STREAM).
pub fn smoke_parakeet_buffered_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_SMOKE_PARAKEET_BUFFERED_MODEL",
        "models/parakeet-unified-en-0.6b/parakeet-unified-en-0.6b-Q8_0.gguf",
    )
}

/// Voxtral realtime streaming canary (accepts VOXTRAL_REALTIME_STREAM).
pub fn smoke_voxtral_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_SMOKE_VOXTRAL_MODEL",
        "models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf",
    )
}

/// Canary model whose generic PNC run parameter changes the prompt.
pub fn smoke_pnc_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_SMOKE_PNC_MODEL",
        "models/canary-180m-flash/canary-180m-flash-Q8_0.gguf",
    )
}

/// SenseVoice model whose generic ITN parameter changes text normalization.
pub fn smoke_itn_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_SMOKE_ITN_MODEL",
        "models/SenseVoiceSmall/SenseVoiceSmall-Q8_0.gguf",
    )
}

/// Both fixtures together; prints a skip note and returns `None` if either is
/// missing (so the caller can `return` early — the Rust equivalent of skip).
pub fn smoke_fixtures(test: &str) -> Option<(PathBuf, Vec<f32>)> {
    match (smoke_model(), smoke_audio()) {
        (Some(m), Some(a)) => Some((m, a)),
        _ => {
            eprintln!(
                "skip {test}: smoke model/audio not present (set TRANSCRIBE_SMOKE_MODEL / _AUDIO)"
            );
            None
        }
    }
}

/// Explicit multilingual Whisper fixtures, exercised independently so the
/// same public-result regressions cover GGUF and legacy `.bin` checkpoints.
pub fn pair_models(test: &str) -> Vec<PathBuf> {
    ensure_backends();
    let models: Vec<_> = ["TRANSCRIBE_PAIR_MODEL", "TRANSCRIBE_PAIR_BIN_MODEL"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .collect();
    if models.is_empty() {
        eprintln!(
            "skip {test}: set TRANSCRIBE_PAIR_MODEL (GGUF) / TRANSCRIBE_PAIR_BIN_MODEL (.bin)"
        );
    }
    models
}

/// Clean human speech for pair tests. Polish is an explicit external fixture;
/// English and German default to the existing canonical sample recordings.
pub fn pair_audio(language: &str) -> Option<Vec<f32>> {
    let (env_var, default_rel) = match language {
        "pl" => ("TRANSCRIBE_PAIR_AUDIO_PL", None),
        "en" => ("TRANSCRIBE_PAIR_AUDIO_EN", Some("samples/jfk.wav")),
        "de" => ("TRANSCRIBE_PAIR_AUDIO_DE", Some("samples/german.wav")),
        _ => panic!("no pair audio fixture for {language}"),
    };
    let path = std::env::var_os(env_var)
        .map(PathBuf::from)
        .or_else(|| default_rel.map(|p| repo_root().join(p)))?;
    path.is_file().then(|| load_wav(&path))
}

/// German sample with the canonical beach/sun transcript used by translation
/// assertions; deliberately independent of external detection-only fixtures.
pub fn canonical_german_audio() -> Option<Vec<f32>> {
    let path = repo_root().join("samples/german.wav");
    path.is_file().then(|| load_wav(&path))
}

pub fn pair_english_only_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_PAIR_EN_MODEL",
        "models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf",
    )
}

pub fn pair_non_whisper_model() -> Option<PathBuf> {
    family_model(
        "TRANSCRIBE_PAIR_UNSUPPORTED_MODEL",
        "models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf",
    )
}

fn load_wav(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "{path:?} must be mono");
    assert_eq!(spec.sample_rate, 16_000, "{path:?} must be 16 kHz");
    assert_eq!(spec.bits_per_sample, 16, "{path:?} must be 16-bit");
    reader
        .samples::<i16>()
        .map(|s| s.expect("wav sample") as f32 / 32768.0)
        .collect()
}
