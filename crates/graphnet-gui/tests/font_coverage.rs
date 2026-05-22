//! Regression guard: every non-ASCII glyph used in main.rs must be
//! covered by one of the bundled fonts. Prevents anyone from adding
//! `🪞` (Unicode 12.0, not in Symbola) and shipping silent squares.
//!
//! Approach: read main.rs, extract every non-ASCII codepoint, look up
//! each codepoint in the bundled Symbola font's cmap table. Fail if
//! any are missing.
//!
//! If this test fails because you added a new glyph: either pick a
//! Symbola-compatible alternative, OR add a third fallback font with
//! the needed codepoint and update theme::install_phosphor_fonts.

use std::collections::BTreeSet;
use std::fs;

/// Parse the cmap (character → glyph index) of a TTF font. Returns the
/// set of supported codepoints. Hand-rolled minimal parser — avoids
/// pulling in a font-parsing dep just for one test.
fn read_ttf_cmap_codepoints(bytes: &[u8]) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    if bytes.len() < 12 {
        return out;
    }
    let read_u16 = |off: usize| -> u16 {
        u16::from_be_bytes([bytes[off], bytes[off + 1]])
    };
    let read_u32 = |off: usize| -> u32 {
        u32::from_be_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ])
    };
    let num_tables = read_u16(4) as usize;
    let mut cmap_offset: Option<usize> = None;
    for i in 0..num_tables {
        let off = 12 + i * 16;
        if &bytes[off..off + 4] == b"cmap" {
            cmap_offset = Some(read_u32(off + 8) as usize);
            break;
        }
    }
    let Some(cmap_off) = cmap_offset else {
        return out;
    };
    let num_sub = read_u16(cmap_off + 2) as usize;
    // Walk subtables, prefer format-4 (BMP) + format-12 (full Unicode).
    for i in 0..num_sub {
        let rec = cmap_off + 4 + i * 8;
        let sub_off = cmap_off + read_u32(rec + 4) as usize;
        let format = read_u16(sub_off);
        if format == 4 {
            // BMP segment array.
            let seg_count_x2 = read_u16(sub_off + 6) as usize;
            let seg_count = seg_count_x2 / 2;
            let end_codes_off = sub_off + 14;
            let start_codes_off = end_codes_off + seg_count_x2 + 2;
            for s in 0..seg_count {
                let end_code =
                    read_u16(end_codes_off + s * 2) as u32;
                let start_code =
                    read_u16(start_codes_off + s * 2) as u32;
                if start_code == 0xFFFF {
                    continue;
                }
                for cp in start_code..=end_code {
                    out.insert(cp);
                }
            }
        } else if format == 12 {
            // Full Unicode group array.
            let num_groups = read_u32(sub_off + 12) as usize;
            for g in 0..num_groups {
                let grp_off = sub_off + 16 + g * 12;
                let start = read_u32(grp_off);
                let end = read_u32(grp_off + 4);
                for cp in start..=end {
                    out.insert(cp);
                }
            }
        }
    }
    out
}

#[test]
fn every_non_ascii_glyph_used_is_covered_by_bundled_fonts() {
    let main = fs::read_to_string("src/main.rs")
        .or_else(|_| fs::read_to_string("crates/graphnet-gui/src/main.rs"))
        .expect("read main.rs");
    let chars: BTreeSet<char> =
        main.chars().filter(|c| (*c as u32) > 127).collect();

    let symbola = fs::read("assets/Symbola.ttf")
        .or_else(|_| fs::read("crates/graphnet-gui/assets/Symbola.ttf"))
        .expect("read Symbola.ttf");
    let symbola_cmap = read_ttf_cmap_codepoints(&symbola);

    // egui's default fonts also cover ASCII + Latin Extended +
    // common emoji via NotoEmoji-Regular. For now we only verify
    // Symbola coverage since that's our explicit fallback. Latin and
    // ASCII are excluded by the filter above.
    let mut missing = BTreeSet::new();
    for c in &chars {
        if !symbola_cmap.contains(&(*c as u32)) {
            missing.insert(*c);
        }
    }
    assert!(
        missing.is_empty(),
        "{} char(s) used in main.rs but NOT in bundled Symbola font: \
         {:?}\n\
         Fix: either swap for a Symbola-supported alternative, OR add \
         a third fallback font to theme::install_phosphor_fonts that \
         covers these codepoints.",
        missing.len(),
        missing.iter().map(|c| format!("{c}=U+{:04X}", *c as u32)).collect::<Vec<_>>()
    );
    eprintln!(
        "OK: {} unique non-ASCII chars in main.rs, all covered by Symbola.",
        chars.len()
    );
}
