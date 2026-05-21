//! GraphNet native GUI shell — PlausiDen-themed, GPU-accelerated (wgpu),
//! interactive. Iteration 2.

#![forbid(unsafe_code)]

mod theme;

use eframe::egui;
use graphnet_engine::{ArchSummary, Model, Operation, Stack};
use plausiden_hdc::Hypervector;

struct App {
    stack: Stack,
    input_seed: u64,
    input: Hypervector,
    last_output: Option<Hypervector>,
    last_latency_ms: Option<f64>,
    forwards: u64,
    dim: usize,
    template: &'static str,
}

const TEMPLATES: &[(&str, &str, usize)] = &[
    ("stack-tiny", "1 op · D=1k", 1_000),
    ("stack-standard", "3 ops · D=10k", 10_000),
    ("stack-dense-only", "4 dense · D=10k", 10_000),
    ("stack-fft-heavy", "3 hrr · D=1k (pow2)", 1_024),
];

impl App {
    fn new(ctx: &egui::Context) -> Self {
        theme::install(ctx);
        let mut app = Self {
            stack: Stack::new(10_000),
            input_seed: 42,
            input: Hypervector::random_seeded(10_000, 42),
            last_output: None,
            last_latency_ms: None,
            forwards: 0,
            dim: 10_000,
            template: "stack-standard",
        };
        app.load_template("stack-standard");
        app
    }

    fn load_template(&mut self, name: &'static str) {
        self.template = name;
        let dim = TEMPLATES
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, _, d)| *d)
            .unwrap_or(10_000);
        self.dim = dim;
        self.input = Hypervector::random_seeded(dim, self.input_seed);
        self.stack = Stack::new(dim);
        match name {
            "stack-tiny" => {
                self.stack.add_operation(Operation::Identity);
            }
            "stack-standard" => {
                self.stack.add_operation(Operation::Identity);
                self.stack.add_operation(Operation::Dense {
                    key: Hypervector::random_seeded(dim, 1),
                });
                self.stack.add_operation(Operation::HrrBind {
                    key: Hypervector::random_seeded(dim, 2),
                });
            }
            "stack-dense-only" => {
                for s in 1..=4 {
                    self.stack.add_operation(Operation::Dense {
                        key: Hypervector::random_seeded(dim, s),
                    });
                }
            }
            "stack-fft-heavy" => {
                for s in 10..13 {
                    self.stack.add_operation(Operation::HrrBind {
                        key: Hypervector::random_seeded(dim, s),
                    });
                }
            }
            _ => {}
        }
        self.last_output = None;
        self.last_latency_ms = None;
    }

    fn regenerate_input(&mut self) {
        self.input_seed = self.input_seed.wrapping_add(1);
        self.input = Hypervector::random_seeded(self.dim, self.input_seed);
    }

    fn add_op(&mut self, kind: &str) {
        let seed = self.stack.len() as u64 + 100;
        let op = match kind {
            "identity" => Operation::Identity,
            "dense" => Operation::Dense {
                key: Hypervector::random_seeded(self.dim, seed),
            },
            "hrr_bind" => Operation::HrrBind {
                key: Hypervector::random_seeded(self.dim, seed + 100),
            },
            _ => return,
        };
        self.stack.add_operation(op);
    }

    fn remove_op(&mut self, idx: usize) {
        if idx < self.stack.len() {
            self.stack.remove_operation(idx);
        }
    }

    fn arch_summary(&self) -> ArchSummary {
        self.stack.arch_summary()
    }

    fn run_forward(&mut self) {
        let started = std::time::Instant::now();
        if let Ok(out) = self.stack.forward(&self.input) {
            #[allow(clippy::cast_precision_loss)]
            let ms = started.elapsed().as_micros() as f64 / 1000.0;
            self.last_output = Some(out);
            self.last_latency_ms = Some(ms);
            self.forwards = self.forwards.saturating_add(1);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("hero")
            .frame(egui::Frame::none().fill(theme::BG))
            .resizable(false)
            .show_separator_line(false)
            .show(ctx, |ui| {
                let band_h = 80.0;
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), band_h),
                    egui::Sense::hover(),
                );
                theme::paint_gradient(ui.painter(), rect);
                let inner_rect = rect.shrink2(egui::vec2(theme::SPACE_XL, theme::SPACE_MD));
                let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(inner_rect));
                inner.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("PlausiDen / GraphNet")
                            .size(theme::SIZE_H2)
                            .color(egui::Color32::WHITE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_MD);
                    ui.label(
                        egui::RichText::new("Live REPL · GPU (wgpu)")
                            .size(theme::SIZE_SMALL)
                            .color(egui::Color32::from_white_alpha(190)),
                    );
                });
                inner.add_space(theme::SPACE_XS);
                inner.label(
                    egui::RichText::new(graphnet_engine::banner())
                        .size(theme::SIZE_TINY)
                        .color(egui::Color32::from_white_alpha(170))
                        .monospace(),
                );
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "● template: {}   ·   dim: {}   ·   ops: {}",
                        self.template,
                        self.dim,
                        self.stack.len()
                    ))
                    .size(theme::SIZE_SMALL)
                    .color(theme::TEXT_MUTED),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let latency = self
                        .last_latency_ms
                        .map(|ms| format!("last: {ms:.3} ms"))
                        .unwrap_or_else(|| "last: —".to_string());
                    ui.label(
                        egui::RichText::new(format!(
                            "forwards: {}   ·   {}",
                            self.forwards, latency
                        ))
                        .size(theme::SIZE_SMALL)
                        .color(theme::TEXT_MUTED),
                    );
                });
            });
        });

        egui::SidePanel::left("arch")
            .min_width(280.0)
            .max_width(340.0)
            .resizable(true)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG)
                    .inner_margin(egui::Margin::same(theme::SPACE_LG)),
            )
            .show(ctx, |ui| {
                section_heading(ui, "Templates");
                ui.add_space(theme::SPACE_SM);
                for (name, desc, _) in TEMPLATES {
                    let active = *name == self.template;
                    let chip_fill = if active {
                        theme::ACCENT_MID
                    } else {
                        theme::BG_CARD
                    };
                    let chip_stroke = if active {
                        theme::ACCENT_MID
                    } else {
                        theme::BORDER_SUBTLE
                    };
                    let resp = ui.add(
                        egui::Button::new(
                            egui::RichText::new(format!("{name}\n{desc}"))
                                .size(theme::SIZE_SMALL)
                                .color(theme::TEXT_PRIMARY),
                        )
                        .fill(chip_fill)
                        .stroke(egui::Stroke::new(1.0, chip_stroke))
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .min_size(egui::vec2(ui.available_width(), 44.0)),
                    );
                    if resp.clicked() {
                        self.load_template(name);
                    }
                    ui.add_space(theme::SPACE_XS);
                }

                ui.add_space(theme::SPACE_LG);
                section_heading(ui, "Architecture");
                ui.add_space(theme::SPACE_SM);
                let summary = self.arch_summary();
                card(ui, |ui| {
                    metric(ui, "family", &summary.family);
                    metric(ui, "input dim", &summary.input_dim.to_string());
                    metric(ui, "output dim", &summary.output_dim.to_string());
                    metric(ui, "ops", &summary.substructures.to_string());
                });

                ui.add_space(theme::SPACE_LG);
                section_heading(ui, "Operations");
                ui.add_space(theme::SPACE_SM);
                let ops: Vec<_> = self
                    .stack
                    .operations()
                    .iter()
                    .enumerate()
                    .map(|(i, op)| (i, op.tag().to_string()))
                    .collect();
                let mut to_remove: Option<usize> = None;
                for (idx, tag) in &ops {
                    if op_chip_with_remove(ui, *idx, tag) {
                        to_remove = Some(*idx);
                    }
                    ui.add_space(theme::SPACE_XS);
                }
                if let Some(i) = to_remove {
                    self.remove_op(i);
                }

                ui.add_space(theme::SPACE_SM);
                ui.horizontal(|ui| {
                    if mini_button(ui, "+ Identity", theme::op_color("identity")).clicked() {
                        self.add_op("identity");
                    }
                    if mini_button(ui, "+ Dense", theme::op_color("dense")).clicked() {
                        self.add_op("dense");
                    }
                    if mini_button(ui, "+ HrrBind", theme::op_color("hrr_bind")).clicked() {
                        self.add_op("hrr_bind");
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(theme::BG)
                    .inner_margin(egui::Margin::same(theme::SPACE_XL)),
            )
            .show(ctx, |ui| {
                section_heading(ui, "Input");
                ui.add_space(theme::SPACE_SM);
                let input_clone = self.input.clone();
                card(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("seed = {}", self.input_seed))
                                .size(theme::SIZE_BODY),
                        );
                    });
                    ui.add_space(theme::SPACE_SM);
                    hypervector_heatmap(ui, &input_clone, 80, 4.0);
                });
                ui.add_space(theme::SPACE_SM);
                let regen = ui.add(
                    egui::Button::new(
                        egui::RichText::new("regenerate input")
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_PRIMARY),
                    )
                    .fill(theme::BG_CARD_HOVER)
                    .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                    .rounding(egui::Rounding::same(theme::RADIUS_SM)),
                );
                if regen.clicked() {
                    self.regenerate_input();
                }

                ui.add_space(theme::SPACE_LG);
                let btn = ui.add(
                    egui::Button::new(
                        egui::RichText::new("▶  Run forward")
                            .size(theme::SIZE_BODY)
                            .strong(),
                    )
                    .fill(theme::ACCENT_MID)
                    .rounding(egui::Rounding::same(theme::RADIUS_MD))
                    .min_size(egui::vec2(ui.available_width().min(280.0), 48.0)),
                );
                if btn.clicked() {
                    self.run_forward();
                }

                ui.add_space(theme::SPACE_LG);
                if let Some(out) = self.last_output.clone() {
                    section_heading(ui, "Output");
                    ui.add_space(theme::SPACE_SM);
                    let latency = self.last_latency_ms;
                    card(ui, |ui| {
                        metric(ui, "dim", &out.dim().to_string());
                        if let Some(ms) = latency {
                            metric(ui, "latency", &format!("{ms:.3} ms"));
                        }
                        ui.add_space(theme::SPACE_SM);
                        hypervector_heatmap(ui, &out, 80, 4.0);
                    });
                } else {
                    ui.label(
                        egui::RichText::new("(click ‘Run forward’ to render output heatmap)")
                            .color(theme::TEXT_DIM)
                            .italics(),
                    );
                }
            });
    }
}

fn section_heading(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(theme::SIZE_TINY)
            .color(theme::TEXT_MUTED)
            .strong(),
    );
}

fn metric(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .color(theme::TEXT_MUTED)
                .size(theme::SIZE_SMALL),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value)
                    .color(theme::TEXT_PRIMARY)
                    .size(theme::SIZE_BODY)
                    .strong(),
            );
        });
    });
}

fn card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::none()
        .fill(theme::BG_CARD)
        .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
        .rounding(egui::Rounding::same(theme::RADIUS_MD))
        .inner_margin(egui::Margin::same(theme::SPACE_MD))
        .show(ui, add_contents);
}

fn mini_button(ui: &mut egui::Ui, text: &str, accent: egui::Color32) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(text)
                .size(theme::SIZE_SMALL)
                .color(accent)
                .strong(),
        )
        .fill(accent.gamma_multiply(0.18))
        .stroke(egui::Stroke::new(1.0, accent))
        .rounding(egui::Rounding::same(theme::RADIUS_SM)),
    )
}

fn op_chip_with_remove(ui: &mut egui::Ui, idx: usize, tag: &str) -> bool {
    let mut removed = false;
    let bg = theme::op_color(tag).gamma_multiply(0.25);
    let fg = theme::op_color(tag);
    egui::Frame::none()
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, fg))
        .rounding(egui::Rounding::same(theme::RADIUS_PILL))
        .inner_margin(egui::Margin::symmetric(theme::SPACE_MD, theme::SPACE_SM))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("{idx:02}"))
                        .color(theme::TEXT_MUTED)
                        .size(theme::SIZE_TINY)
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new(tag)
                        .color(fg)
                        .size(theme::SIZE_BODY)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let resp = ui.add(
                        egui::Button::new(
                            egui::RichText::new("×")
                                .size(theme::SIZE_BODY)
                                .color(theme::TEXT_MUTED),
                        )
                        .fill(egui::Color32::TRANSPARENT)
                        .stroke(egui::Stroke::NONE)
                        .min_size(egui::vec2(20.0, 20.0)),
                    );
                    if resp.clicked() {
                        removed = true;
                    }
                });
            });
        });
    removed
}

fn hypervector_heatmap(ui: &mut egui::Ui, v: &Hypervector, cols: usize, cell: f32) {
    let dim = v.dim();
    let rows = dim.div_ceil(cols);
    let total_w = cols as f32 * cell;
    let total_h = rows as f32 * cell;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(total_w, total_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let data = v.as_slice();
    for i in 0..dim {
        let r = i / cols;
        let c = i % cols;
        let x = rect.min.x + c as f32 * cell;
        let y = rect.min.y + r as f32 * cell;
        let colour = if data[i] > 0 {
            theme::ACCENT_BLUE
        } else {
            theme::ACCENT_PURPLE.gamma_multiply(0.55)
        };
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell - 1.0, cell - 1.0)),
            0.0,
            colour,
        );
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([800.0, 540.0])
            .with_title("GraphNet"),
        ..Default::default()
    };
    eframe::run_native(
        "GraphNet",
        options,
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn dummy_ctx() -> egui::Context {
        egui::Context::default()
    }

    #[test]
    fn app_constructs_with_default_template() {
        let app = App::new(&dummy_ctx());
        assert_eq!(app.dim, 10_000);
        assert_eq!(app.stack.len(), 3);
        assert_eq!(app.template, "stack-standard");
    }

    #[test]
    fn loading_each_template_yields_expected_op_count() {
        let mut app = App::new(&dummy_ctx());
        app.load_template("stack-tiny");
        assert_eq!(app.stack.len(), 1);
        app.load_template("stack-standard");
        assert_eq!(app.stack.len(), 3);
        app.load_template("stack-dense-only");
        assert_eq!(app.stack.len(), 4);
        app.load_template("stack-fft-heavy");
        assert_eq!(app.stack.len(), 3);
    }

    #[test]
    fn add_remove_op_round_trip() {
        let mut app = App::new(&dummy_ctx());
        let before = app.stack.len();
        app.add_op("dense");
        assert_eq!(app.stack.len(), before + 1);
        app.remove_op(before);
        assert_eq!(app.stack.len(), before);
    }

    #[test]
    fn regenerate_input_advances_seed() {
        let mut app = App::new(&dummy_ctx());
        let before = app.input_seed;
        app.regenerate_input();
        assert_eq!(app.input_seed, before + 1);
    }

    #[test]
    fn run_forward_advances_counter() {
        let mut app = App::new(&dummy_ctx());
        app.run_forward();
        assert_eq!(app.forwards, 1);
        assert!(app.last_output.is_some());
    }
}
