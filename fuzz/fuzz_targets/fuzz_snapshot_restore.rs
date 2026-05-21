#![no_main]
//! Fuzz harness: feed arbitrary bytes to the bincode snapshot decoder,
//! ensure no panics. Decode errors are expected; panics are not.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = graphnet_engine::restore(data);
});
