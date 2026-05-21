#![no_main]
//! Fuzz harness: feed arbitrary bytes to the YAML spec decoder, ensure no
//! panics. Decode errors are expected and acceptable; panics are not.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = graphnet_engine::ArchitectureSpec::from_yaml(s);
    }
});
