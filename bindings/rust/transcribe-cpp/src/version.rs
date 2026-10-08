//! Version + ABI introspection and the load-time version gate.
//!
//! Pre-1.0 the on-disk ABI may break between minor releases, so the binding
//! and the linked library must agree on the base `MAJOR.MINOR.PATCH` and the
//! run/capability layouts. The gate runs once, lazily, before the first model
//! load. Packaging-only suffixes on the runtime string are tolerated — only
//! the leading release segment is compared.

use std::sync::OnceLock;

use transcribe_cpp_sys as sys;

use crate::error::{Error, Result};
use crate::result::owned_str;
use crate::types::AbiStruct;

/// The base version string this crate's bindings were built against.
///
/// Taken from this crate's own `Cargo.toml` (`CARGO_PKG_VERSION`), not the
/// generated FFI macros: a version-only bump must not churn the committed
/// bindings or the abihash (notes/releasing.md §8 P0 #1). The generators no
/// longer emit `TRANSCRIBE_VERSION_*`, so this is also the only source left.
pub fn compiled_version() -> String {
    base(env!("CARGO_PKG_VERSION")).to_string()
}

/// The `MAJOR.MINOR.PATCH` version string of the linked native library.
pub fn version() -> String {
    owned_str(unsafe { sys::transcribe_version() })
}

/// The short git commit the native library was built from (or "unknown").
pub fn version_commit() -> String {
    owned_str(unsafe { sys::transcribe_version_commit() })
}

/// The public-ABI digest the committed bindings were generated against.
pub fn header_hash() -> &'static str {
    sys::PUBLIC_HEADER_HASH
}

/// The native library's `sizeof` for a public ABI struct, or 0 if unknown.
pub fn abi_struct_size(which: AbiStruct) -> usize {
    unsafe { sys::transcribe_abi_struct_size(which.to_raw()) }
}

/// The native library's `alignof` for a public ABI struct, or 0 if unknown.
pub fn abi_struct_align(which: AbiStruct) -> usize {
    unsafe { sys::transcribe_abi_struct_align(which.to_raw()) }
}

/// Leading dotted-numeric release segment ("0.0.1.post1" -> "0.0.1").
fn base(v: &str) -> &str {
    let end = v
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(v.len());
    v[..end].trim_end_matches('.')
}

static GATE: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Run (once) the pre-1.0 version and run/capability ABI gate before model load.
pub(crate) fn ensure_compatible() -> Result<()> {
    let outcome = GATE.get_or_init(|| {
        let runtime = version();
        let compiled = compiled_version();
        if base(&runtime) != base(&compiled) {
            return Err(format!(
                "loaded transcribe library is {runtime}, but these bindings \
                 were generated for {compiled} (base versions must match \
                 pre-1.0)"
            ));
        }
        // Forks can extend the ABI without changing the base version. Check
        // before any init function can write a mismatched caller-owned struct.
        for (name, which, size, align) in [
            (
                "transcribe_run_params",
                AbiStruct::RunParams,
                std::mem::size_of::<sys::transcribe_run_params>(),
                std::mem::align_of::<sys::transcribe_run_params>(),
            ),
            (
                "transcribe_capabilities",
                AbiStruct::Capabilities,
                std::mem::size_of::<sys::transcribe_capabilities>(),
                std::mem::align_of::<sys::transcribe_capabilities>(),
            ),
        ] {
            let runtime_size = abi_struct_size(which);
            let runtime_align = abi_struct_align(which);
            if runtime_size != size || runtime_align != align {
                return Err(format!(
                    "loaded transcribe library {runtime} has {name} size/alignment \
                     {runtime_size}/{runtime_align}, but these bindings require \
                     {size}/{align}; use the matching patched native runtime"
                ));
            }
        }
        Ok(())
    });
    outcome.clone().map_err(Error::VersionMismatch)
}
