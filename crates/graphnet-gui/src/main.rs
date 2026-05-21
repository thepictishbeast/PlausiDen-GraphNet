//! GraphNet native GUI shell — PlausiDen-themed, GPU-accelerated, interactive.
//! Iter 3 adds: keyboard shortcuts, live continuous mode, cosine-similarity
//! gauge, Save/Load YAML buttons, slimmer hero.

#![forbid(unsafe_code)]

mod theme;

use eframe::egui;
use graphnet_engine::{
    stack_from_yaml, stack_to_yaml, ArchSummary, ForwardTrace, Model, Operation, Stack,
};
use plausiden_hdc::{cos_sim, Hypervector};

const YAML_PATH: &str = "graphnet-stack.yaml";

struct App {
    stack: Stack,
    input_seed: u64,
    input: Hypervector,
    last_output: Option<Hypervector>,
    last_trace: Option<ForwardTrace>,
    selected_op: Option<usize>,
    last_latency_ms: Option<f64>,
    last_cos_sim: Option<f64>,
    forwards: u64,
    dim: usize,
    template: &'static str,
    live: bool,
    live_fps: f64,
    live_last_frame: std::time::Instant,
    status_msg: Option<(String, std::time::Instant)>,
    mode: theme::Mode,
}

const TEMPLATES: &[(&str, &str, usize)] = &[
    ("stack-tiny", "1 op · D=1k", 1_000),
    ("stack-standard", "3 ops · D=10k", 10_000),
    ("stack-dense-only", "4 dense · D=10k", 10_000),
    ("stack-fft-heavy", "3 hrr · D=1k", 1_024),
];

impl App {
    fn new(ctx: &egui::Context) -> Self {
        theme::install_dark(ctx);
        let mut app = Self {
            stack: Stack::new(10_000),
            input_seed: 42,
            input: Hypervector::random_seeded(10_000, 42),
            last_output: None,
            last_trace: None,
            selected_op: None,
            last_latency_ms: None,
            last_cos_sim: None,
            forwards: 0,
            dim: 10_000,
            template: "stack-standard",
            live: false,
            live_fps: 0.0,
            live_last_frame: std::time::Instant::now(),
            status_msg: None,
            mode: theme::Mode::Dark,
        };
        app.load_template("stack-standard");
        app
    }

    fn toggle_mode(&mut self, ctx: &egui::Context) {
        self.mode = match self.mode {
            theme::Mode::Dark => theme::Mode::Light,
            theme::Mode::Light => theme::Mode::Dark,
        };
        theme::install_mode(ctx, self.mode);
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
        self.last_trace = None;
        self.selected_op = None;
        self.last_latency_ms = None;
        self.last_cos_sim = None;
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
        if let Ok(trace) = self.stack.forward_with_trace(&self.input) {
            #[allow(clippy::cast_precision_loss)]
            let ms = started.elapsed().as_micros() as f64 / 1000.0;
            self.last_cos_sim = cos_sim(&self.input, &trace.bundled).ok();
            self.last_output = Some(trace.bundled.clone());
            self.last_trace = Some(trace);
            self.last_latency_ms = Some(ms);
            self.forwards = self.forwards.saturating_add(1);
        }
    }

    fn save_yaml(&mut self) {
        match stack_to_yaml(&self.stack) {
            Ok(yaml) => match std::fs::write(YAML_PATH, &yaml) {
                Ok(_) => self.set_status(format!("saved → {YAML_PATH} ({} bytes)", yaml.len())),
                Err(e) => self.set_status(format!("save failed: {e}")),
            },
            Err(e) => self.set_status(format!("encode failed: {e}")),
        }
    }

    fn load_yaml(&mut self) {
        match std::fs::read_to_string(YAML_PATH) {
            Ok(yaml) => match stack_from_yaml(&yaml) {
                Ok(stack) => {
                    self.dim = stack.dim();
                    self.stack = stack;
                    self.input = Hypervector::random_seeded(self.dim, self.input_seed);
                    self.last_output = None;
                    self.last_latency_ms = None;
                    self.last_cos_sim = None;
                    self.set_status(format!("loaded ← {YAML_PATH}"));
                }
                Err(e) => self.set_status(format!("decode failed: {e}")),
            },
            Err(e) => self.set_status(format!("read failed: {e}")),
        }
    }

    fn set_status(&mut self, msg: String) {
        self.status_msg = Some((msg, std::time::Instant::now()));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.run_forward();
            }
            if i.key_pressed(egui::Key::R) {
                self.regenerate_input();
            }
            if i.key_pressed(egui::Key::L) {
                self.live = !self.live;
            }
            if i.key_pressed(egui::Key::Num1) {
                self.load_template("stack-tiny");
            }
            if i.key_pressed(egui::Key::Num2) {
                self.load_template("stack-standard");
            }
            if i.key_pressed(egui::Key::Num3) {
                self.load_template("stack-dense-only");
            }
            if i.key_pressed(egui::Key::Num4) {
                self.load_template("stack-fft-heavy");
            }
            if (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::S) {
                self.save_yaml();
            }
            if (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::O) {
                self.load_yaml();
            }
        });

        if self.live {
            self.run_forward();
            let now = std::time::Instant::now();
            let dt = now.duration_since(self.live_last_frame).as_secs_f64();
            if dt > 0.0 {
                self.live_fps = self.live_fps * 0.9 + (1.0 / dt) * 0.1;
            }
            self.live_last_frame = now;
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("hero")
            .frame(egui::Frame::none().fill(theme::BG))
            .resizable(false)
            .show_separator_line(false)
            .show(ctx, |ui| {
                let band_h = 60.0;
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), band_h),
                    egui::Sense::hover(),
                );
                theme::paint_gradient(ui.painter(), rect);
                let inner_rect = rect.shrink2(egui::vec2(theme::SPACE_XL, theme::SPACE_SM));
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
                        egui::RichText::new("Live REPL · GPU · v0.1.0")
                            .size(theme::SIZE_SMALL)
                            .color(egui::Color32::from_white_alpha(190)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let toggle = ui.add(
                            egui::Button::new(
                                egui::RichText::new(match self.mode {
                                    theme::Mode::Dark => "☀ light",
                                    theme::Mode::Light => "☾ dark",
                                })
                                .size(theme::SIZE_SMALL)
                                .color(egui::Color32::WHITE)
                                .strong(),
                            )
                            .fill(egui::Color32::from_white_alpha(40))
                            .stroke(egui::Stroke::new(
                                1.0,
                                egui::Color32::from_white_alpha(120),
                            ))
                            .rounding(egui::Rounding::same(theme::RADIUS_SM))
                            .min_size(egui::vec2(72.0, 28.0)),
                        );
                        if toggle.clicked() {
                            self.toggle_mode(ctx);
                        }
                        ui.add_space(theme::SPACE_MD);
                        ui.label(
                            egui::RichText::new(
                                "[Space] fwd · [R] regen · [L] live · [1-4] tpl · [⌘S/⌘O] yaml",
                            )
                            .size(theme::SIZE_TINY)
                            .color(egui::Color32::from_white_alpha(160))
                            .monospace(),
                        );
                    });
                });
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
                    let live_label = if self.live {
                        format!("● live @ {:.0} fps", self.live_fps)
                    } else {
                        String::new()
                    };
                    let mut right = format!("forwards: {}   ·   {}", self.forwards, latency);
                    if !live_label.is_empty() {
                        right = format!("{live_label}   ·   {right}");
                    }
                    ui.label(
                        egui::RichText::new(right)
                            .size(theme::SIZE_SMALL)
                            .color(if self.live {
                                theme::ACCENT_BLUE
                            } else {
                                theme::TEXT_MUTED
                            }),
                    );
                });
            });
            let now = std::time::Instant::now();
            let mut clear = false;
            if let Some((ref msg, t)) = self.status_msg {
                if now.duration_since(t).as_secs_f64() < 4.0 {
                    ui.label(
                        egui::RichText::new(msg)
                            .size(theme::SIZE_SMALL)
                            .color(theme::ACCENT_PURPLE),
                    );
                } else {
                    clear = true;
                }
            }
            if clear {
                self.status_msg = None;
            }
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
                for (i, (name, desc, _)) in TEMPLATES.iter().enumerate() {
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
                            egui::RichText::new(format!("{}  ·  {name}\n{desc}", i + 1))
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

                ui.add_space(theme::SPACE_MD);
                ui.horizontal(|ui| {
                    if mini_button(ui, "Save YAML", theme::ACCENT_MID).clicked() {
                        self.save_yaml();
                    }
                    if mini_button(ui, "Load YAML", theme::ACCENT_BLUE).clicked() {
                        self.load_yaml();
                    }
                });

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
                ui.horizontal(|ui| {
                    if mini_button(ui, "regenerate (R)", theme::TEXT_MUTED).clicked() {
                        self.regenerate_input();
                    }
                });

                ui.add_space(theme::SPACE_LG);
                ui.horizontal(|ui| {
                    let btn = ui.add(
                        egui::Button::new(
                            egui::RichText::new("▶  Run forward  (Space)")
                                .size(theme::SIZE_BODY)
                                .strong(),
                        )
                        .fill(theme::ACCENT_MID)
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .min_size(egui::vec2(240.0, 48.0)),
                    );
                    if btn.clicked() {
                        self.run_forward();
                    }

                    let live_label = if self.live {
                        "■ Stop live (L)"
                    } else {
                        "● Start live (L)"
                    };
                    let live_color = if self.live {
                        theme::ACCENT_PURPLE
                    } else {
                        theme::BG_CARD_HOVER
                    };
                    let live_btn = ui.add(
                        egui::Button::new(
                            egui::RichText::new(live_label)
                                .size(theme::SIZE_BODY)
                                .color(theme::TEXT_PRIMARY)
                                .strong(),
                        )
                        .fill(live_color)
                        .stroke(egui::Stroke::new(1.0, theme::ACCENT_PURPLE))
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .min_size(egui::vec2(180.0, 48.0)),
                    );
                    if live_btn.clicked() {
                        self.live = !self.live;
                    }
                });

                ui.add_space(theme::SPACE_LG);
                if let Some(out) = self.last_output.clone() {
                    section_heading(ui, "Output");
                    ui.add_space(theme::SPACE_SM);
                    let latency = self.last_latency_ms;
                    let sim = self.last_cos_sim;
                    card(ui, |ui| {
                        metric(ui, "dim", &out.dim().to_string());
                        if let Some(ms) = latency {
                            metric(ui, "latency", &format!("{ms:.3} ms"));
                        }
                        if let Some(s) = sim {
                            ui.add_space(theme::SPACE_SM);
                            cosine_similarity_bar(ui, s);
                        }
                        ui.add_space(theme::SPACE_SM);
                        hypervector_heatmap(ui, &out, 80, 4.0);
                    });

                    // Per-op output inspector (uses ForwardTrace).
                    if let Some(trace) = self.last_trace.clone() {
                        ui.add_space(theme::SPACE_LG);
                        section_heading(ui, "Per-op inspector");
                        ui.add_space(theme::SPACE_SM);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("View:")
                                    .color(theme::TEXT_MUTED)
                                    .size(theme::SIZE_SMALL),
                            );
                            for op_out in &trace.per_op {
                                let active = self.selected_op == Some(op_out.index);
                                let accent = theme::op_color(&op_out.tag);
                                let btn = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(format!(
                                            "[{}] {}",
                                            op_out.index, op_out.tag
                                        ))
                                        .size(theme::SIZE_SMALL)
                                        .color(if active {
                                            egui::Color32::WHITE
                                        } else {
                                            accent
                                        })
                                        .strong(),
                                    )
                                    .fill(if active {
                                        accent
                                    } else {
                                        accent.gamma_multiply(0.18)
                                    })
                                    .stroke(egui::Stroke::new(1.0, accent))
                                    .rounding(egui::Rounding::same(theme::RADIUS_SM)),
                                );
                                if btn.clicked() {
                                    self.selected_op = if active {
                                        None
                                    } else {
                                        Some(op_out.index)
                                    };
                                }
                            }
                        });
                        if let Some(idx) = self.selected_op {
                            if let Some(op_out) = trace.per_op.get(idx) {
                                ui.add_space(theme::SPACE_SM);
                                let sim_to_input = cos_sim(&self.input, &op_out.output).ok();
                                let sim_to_out = cos_sim(&op_out.output, &trace.bundled).ok();
                                card(ui, |ui| {
                                    metric(ui, "op", &format!("[{}] {}", op_out.index, op_out.tag));
                                    if let Some(s) = sim_to_input {
                                        metric(ui, "cos_sim → input", &format!("{s:+.3}"));
                                    }
                                    if let Some(s) = sim_to_out {
                                        metric(ui, "cos_sim → bundled", &format!("{s:+.3}"));
                                    }
                                    ui.add_space(theme::SPACE_SM);
                                    hypervector_heatmap(ui, &op_out.output, 80, 4.0);
                                });
                            }
                        }
                    }
                } else {
                    ui.label(
                        egui::RichText::new("(click ‘Run forward’ or press Space)")
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

fn cosine_similarity_bar(ui: &mut egui::Ui, sim: f64) {
    let t = ((sim + 1.0) / 2.0).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("cos_sim(input, output)")
                .color(theme::TEXT_MUTED)
                .size(theme::SIZE_SMALL),
        );
        ui.label(
            egui::RichText::new(format!("{sim:+.3}"))
                .color(theme::TEXT_PRIMARY)
                .size(theme::SIZE_BODY)
                .strong()
                .monospace(),
        );
    });
    let avail = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail, 8.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, theme::BG_CARD_HOVER);
    let fill_w = rect.width() * t as f32;
    let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height()));
    let colour = if sim > 0.0 {
        theme::ACCENT_BLUE
    } else {
        theme::ACCENT_PURPLE
    };
    painter.rect_filled(fill_rect, 4.0, colour);
    let half_x = rect.min.x + rect.width() * 0.5;
    painter.line_segment(
        [egui::pos2(half_x, rect.min.y), egui::pos2(half_x, rect.max.y)],
        egui::Stroke::new(1.0, theme::TEXT_DIM),
    );
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
    fn run_forward_records_cos_sim() {
        let mut app = App::new(&dummy_ctx());
        app.run_forward();
        assert!(app.last_output.is_some());
        assert!(app.last_cos_sim.is_some());
        let s = app.last_cos_sim.expect("set");
        assert!((-1.0..=1.0).contains(&s));
    }

    #[test]
    fn save_then_load_yaml_round_trips() {
        let mut app = App::new(&dummy_ctx());
        let before_len = app.stack.len();
        app.save_yaml();
        app.add_op("dense");
        assert_eq!(app.stack.len(), before_len + 1);
        app.load_yaml();
        assert_eq!(app.stack.len(), before_len);
        let _ = std::fs::remove_file(YAML_PATH);
    }
}
