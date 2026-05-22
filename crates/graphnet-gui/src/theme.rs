//! PlausiDen design tokens applied to egui — per the `feedback_plausiden_design_premium`
//! memory: deep indigo-black dark mode, blue→purple gradient accents,
//! 4/8/12/18/28/44 spacing scale, custom typography hierarchy.

use eframe::egui;

// --- Spacing scale (px) --------------------------------------------------
// Density pass iter 122: tightened ~30% to match JetBrains/VS-Code density.
// Owner: "notice how much smaller everything is compared to your app."

pub const SPACE_XS: f32 = 2.0;
pub const SPACE_SM: f32 = 4.0;
pub const SPACE_MD: f32 = 7.0;
pub const SPACE_LG: f32 = 10.0;
pub const SPACE_XL: f32 = 14.0;

// --- Typography (px) -----------------------------------------------------
// Iter 122: dropped further to match JetBrains menu/editor density.
// Previous pass (iter 26) was 18/14.5/12.5/11/9.5.

pub const SIZE_H1: f32 = 14.0;
pub const SIZE_H2: f32 = 12.0;
pub const SIZE_BODY: f32 = 11.0;
pub const SIZE_SMALL: f32 = 10.0;
pub const SIZE_TINY: f32 = 9.0;

// --- Rounded corners ------------------------------------------------------

pub const RADIUS_SM: f32 = 4.0;
pub const RADIUS_MD: f32 = 6.0;
pub const RADIUS_LG: f32 = 10.0;
pub const RADIUS_PILL: f32 = 999.0;

// --- Palette --------------------------------------------------------------

pub const BG: egui::Color32 = egui::Color32::from_rgb(0x0A, 0x0D, 0x14);
pub const BG_CARD: egui::Color32 = egui::Color32::from_rgb(0x12, 0x16, 0x22);
pub const BG_CARD_HOVER: egui::Color32 = egui::Color32::from_rgb(0x1A, 0x20, 0x30);

pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(0xF2, 0xF3, 0xF7);
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(0x8A, 0x91, 0xA8);
pub const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(0x4E, 0x55, 0x6A);

pub const ACCENT_BLUE: egui::Color32 = egui::Color32::from_rgb(0x5B, 0x8D, 0xEF);
pub const ACCENT_PURPLE: egui::Color32 = egui::Color32::from_rgb(0xA4, 0x5B, 0xEF);
pub const ACCENT_MID: egui::Color32 = egui::Color32::from_rgb(0x7E, 0x74, 0xEF);

pub const BORDER_SUBTLE: egui::Color32 = egui::Color32::from_rgb(0x1F, 0x26, 0x36);
pub const BORDER_ACCENT: egui::Color32 = egui::Color32::from_rgb(0x7E, 0x74, 0xEF);

pub fn op_color(tag: &str) -> egui::Color32 {
    match tag {
        "identity" => egui::Color32::from_rgb(0x72, 0x86, 0xD3),
        "dense" => egui::Color32::from_rgb(0x3F, 0x7D, 0x58),
        "hrr_bind" => egui::Color32::from_rgb(0xBC, 0x46, 0x4B),
        "permute" => egui::Color32::from_rgb(0xC8, 0x9F, 0x4A),
        "negate" => egui::Color32::from_rgb(0xA1, 0x4C, 0xB0),
        _ => TEXT_MUTED,
    }
}

/// Light-mode counterparts to the dark palette. Used when [`install_light`]
/// is called. WCAG-AA contrast preserved.
pub const BG_LIGHT: egui::Color32 = egui::Color32::from_rgb(0xFA, 0xFA, 0xFD);
pub const BG_CARD_LIGHT: egui::Color32 = egui::Color32::from_rgb(0xFF, 0xFF, 0xFF);
pub const BG_CARD_HOVER_LIGHT: egui::Color32 = egui::Color32::from_rgb(0xEE, 0xEE, 0xF6);
pub const TEXT_PRIMARY_LIGHT: egui::Color32 = egui::Color32::from_rgb(0x12, 0x16, 0x22);
pub const TEXT_MUTED_LIGHT: egui::Color32 = egui::Color32::from_rgb(0x55, 0x5C, 0x73);
pub const TEXT_DIM_LIGHT: egui::Color32 = egui::Color32::from_rgb(0x9A, 0xA0, 0xB3);
pub const BORDER_SUBTLE_LIGHT: egui::Color32 = egui::Color32::from_rgb(0xE0, 0xE3, 0xEC);

/// Theme variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Dark,
    Light,
}

pub fn install_mode(ctx: &egui::Context, mode: Mode) {
    match mode {
        Mode::Dark => install_dark(ctx),
        Mode::Light => install_light(ctx),
    }
}

#[allow(dead_code)]
pub fn install(ctx: &egui::Context) {
    install_dark(ctx);
}

/// Register the Phosphor icon font alongside the default fonts so we can
/// use phosphor icon glyphs anywhere in the UI.
fn install_phosphor_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

pub fn install_dark(ctx: &egui::Context) {
    install_phosphor_fonts(ctx);
    let mut style = (*ctx.style()).clone();

    use egui::FontFamily::Proportional;
    use egui::FontId;
    use egui::TextStyle as TS;
    style
        .text_styles
        .insert(TS::Heading, FontId::new(SIZE_H1, Proportional));
    style
        .text_styles
        .insert(TS::Body, FontId::new(SIZE_BODY, Proportional));
    style
        .text_styles
        .insert(TS::Button, FontId::new(SIZE_BODY, Proportional));
    style
        .text_styles
        .insert(TS::Small, FontId::new(SIZE_SMALL, Proportional));
    style.text_styles.insert(
        TS::Monospace,
        FontId::new(SIZE_SMALL, egui::FontFamily::Monospace),
    );

    style.spacing.item_spacing = egui::vec2(SPACE_SM, SPACE_SM);
    style.spacing.window_margin = egui::Margin::same(SPACE_LG);
    style.spacing.button_padding = egui::vec2(SPACE_LG, SPACE_MD);
    style.spacing.menu_margin = egui::Margin::same(SPACE_SM);

    style.visuals.dark_mode = true;
    style.visuals.override_text_color = Some(TEXT_PRIMARY);
    style.visuals.panel_fill = BG;
    style.visuals.window_fill = BG;
    style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);
    style.visuals.window_rounding = egui::Rounding::same(RADIUS_LG);

    let card_radius = egui::Rounding::same(RADIUS_MD);
    style.visuals.widgets.noninteractive.bg_fill = BG_CARD;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.noninteractive.rounding = card_radius;
    style.visuals.widgets.inactive.bg_fill = BG_CARD;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.inactive.rounding = card_radius;
    style.visuals.widgets.inactive.weak_bg_fill = BG_CARD;
    style.visuals.widgets.hovered.bg_fill = BG_CARD_HOVER;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, BORDER_ACCENT);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.hovered.rounding = card_radius;
    style.visuals.widgets.hovered.weak_bg_fill = BG_CARD_HOVER;
    style.visuals.widgets.active.bg_fill = ACCENT_MID;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_MID);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    style.visuals.widgets.active.rounding = card_radius;
    style.visuals.widgets.active.weak_bg_fill = ACCENT_MID;

    style.visuals.selection.bg_fill = ACCENT_MID.gamma_multiply(0.4);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT_MID);
    style.visuals.hyperlink_color = ACCENT_BLUE;
    style.visuals.faint_bg_color = BG_CARD;
    style.visuals.extreme_bg_color = BG;

    ctx.set_style(style);
}

pub fn install_light(ctx: &egui::Context) {
    install_phosphor_fonts(ctx);
    let mut style = (*ctx.style()).clone();
    use egui::FontFamily::Proportional;
    use egui::FontId;
    use egui::TextStyle as TS;
    style.text_styles.insert(TS::Heading, FontId::new(SIZE_H1, Proportional));
    style.text_styles.insert(TS::Body, FontId::new(SIZE_BODY, Proportional));
    style.text_styles.insert(TS::Button, FontId::new(SIZE_BODY, Proportional));
    style.text_styles.insert(TS::Small, FontId::new(SIZE_SMALL, Proportional));
    style
        .text_styles
        .insert(TS::Monospace, FontId::new(SIZE_SMALL, egui::FontFamily::Monospace));

    style.spacing.item_spacing = egui::vec2(SPACE_SM, SPACE_SM);
    style.spacing.window_margin = egui::Margin::same(SPACE_LG);
    style.spacing.button_padding = egui::vec2(SPACE_LG, SPACE_MD);
    style.spacing.menu_margin = egui::Margin::same(SPACE_SM);

    style.visuals.dark_mode = false;
    style.visuals.override_text_color = Some(TEXT_PRIMARY_LIGHT);
    style.visuals.panel_fill = BG_LIGHT;
    style.visuals.window_fill = BG_LIGHT;
    style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE_LIGHT);
    style.visuals.window_rounding = egui::Rounding::same(RADIUS_LG);

    let card_radius = egui::Rounding::same(RADIUS_MD);
    style.visuals.widgets.noninteractive.bg_fill = BG_CARD_LIGHT;
    style.visuals.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, BORDER_SUBTLE_LIGHT);
    style.visuals.widgets.noninteractive.fg_stroke =
        egui::Stroke::new(1.0, TEXT_PRIMARY_LIGHT);
    style.visuals.widgets.noninteractive.rounding = card_radius;
    style.visuals.widgets.inactive.bg_fill = BG_CARD_LIGHT;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE_LIGHT);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY_LIGHT);
    style.visuals.widgets.inactive.rounding = card_radius;
    style.visuals.widgets.inactive.weak_bg_fill = BG_CARD_LIGHT;
    style.visuals.widgets.hovered.bg_fill = BG_CARD_HOVER_LIGHT;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, BORDER_ACCENT);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY_LIGHT);
    style.visuals.widgets.hovered.rounding = card_radius;
    style.visuals.widgets.hovered.weak_bg_fill = BG_CARD_HOVER_LIGHT;
    style.visuals.widgets.active.bg_fill = ACCENT_MID;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_MID);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
    style.visuals.widgets.active.rounding = card_radius;
    style.visuals.widgets.active.weak_bg_fill = ACCENT_MID;

    style.visuals.selection.bg_fill = ACCENT_MID.gamma_multiply(0.25);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT_MID);
    style.visuals.hyperlink_color = ACCENT_BLUE;
    style.visuals.faint_bg_color = BG_CARD_LIGHT;
    style.visuals.extreme_bg_color = BG_LIGHT;

    ctx.set_style(style);
}

/// Return the BG color appropriate for the current mode.
#[allow(dead_code)]
pub fn bg_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => BG,
        Mode::Light => BG_LIGHT,
    }
}

#[allow(dead_code)]
pub fn bg_card_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => BG_CARD,
        Mode::Light => BG_CARD_LIGHT,
    }
}

#[allow(dead_code)]
pub fn bg_card_hover_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => BG_CARD_HOVER,
        Mode::Light => BG_CARD_HOVER_LIGHT,
    }
}

#[allow(dead_code)]
pub fn text_muted_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => TEXT_MUTED,
        Mode::Light => TEXT_MUTED_LIGHT,
    }
}

#[allow(dead_code)]
pub fn text_dim_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => TEXT_DIM,
        Mode::Light => TEXT_DIM_LIGHT,
    }
}

#[allow(dead_code)]
pub fn border_subtle_for(mode: Mode) -> egui::Color32 {
    match mode {
        Mode::Dark => BORDER_SUBTLE,
        Mode::Light => BORDER_SUBTLE_LIGHT,
    }
}

pub fn paint_gradient(painter: &egui::Painter, rect: egui::Rect) {
    let n = 64;
    let width = rect.width() / n as f32;
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let r = lerp(ACCENT_BLUE.r() as f32, ACCENT_PURPLE.r() as f32, t) as u8;
        let g = lerp(ACCENT_BLUE.g() as f32, ACCENT_PURPLE.g() as f32, t) as u8;
        let b = lerp(ACCENT_BLUE.b() as f32, ACCENT_PURPLE.b() as f32, t) as u8;
        let band = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + i as f32 * width, rect.min.y),
            egui::vec2(width + 1.0, rect.height()),
        );
        painter.rect_filled(band, 0.0, egui::Color32::from_rgb(r, g, b));
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
