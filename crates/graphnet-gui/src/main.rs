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

fn persistent_state_path() -> std::path::PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            std::path::PathBuf::from(home).join(".config")
        });
    base.join("graphnet").join("state.yaml")
}

struct App {
    stack: Stack,
    input_seed: u64,
    input: Hypervector,
    last_output: Option<Hypervector>,
    last_trace: Option<ForwardTrace>,
    selected_op: Option<usize>,
    last_latency_ms: Option<f64>,
    last_cos_sim: Option<f64>,
    cos_sim_history: std::collections::VecDeque<f64>,
    latency_history: std::collections::VecDeque<f64>,
    forwards: u64,
    dim: usize,
    template: &'static str,
    live: bool,
    live_fps: f64,
    live_last_frame: std::time::Instant,
    status_msg: Option<(String, std::time::Instant)>,
    mode: theme::Mode,
    show_help: bool,
    reseed_counter: u64,
    zoom_target: Option<ZoomTarget>,
    dim_slider: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZoomTarget {
    Input,
    Output,
    PerOp(usize),
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
            cos_sim_history: std::collections::VecDeque::with_capacity(128),
            latency_history: std::collections::VecDeque::with_capacity(128),
            forwards: 0,
            dim: 10_000,
            template: "stack-standard",
            live: false,
            live_fps: 0.0,
            live_last_frame: std::time::Instant::now(),
            status_msg: None,
            mode: theme::Mode::Dark,
            show_help: false,
            reseed_counter: 0,
            zoom_target: None,
            dim_slider: 10_000,
        };
        app.load_template("stack-standard");
        // Try to restore previous session.
        let _ = app.restore();
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
        self.dim_slider = dim;
    }

    /// Change the dimensionality on the fly. Drops all ops and the
    /// trace history since hypervectors at the old dim are now invalid.
    fn set_dim(&mut self, new_dim: usize) {
        if new_dim == self.dim {
            return;
        }
        self.dim = new_dim;
        self.input = Hypervector::random_seeded(new_dim, self.input_seed);
        self.stack = Stack::new(new_dim);
        self.last_output = None;
        self.last_trace = None;
        self.selected_op = None;
        self.last_latency_ms = None;
        self.last_cos_sim = None;
        self.cos_sim_history.clear();
        self.latency_history.clear();
        self.dim_slider = new_dim;
        self.set_status(format!("dim → {new_dim}; stack cleared"));
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

    /// Replace the key on a Dense/HrrBind op without removing it. No-op
    /// on Identity (no key to rotate).
    fn reseed_op(&mut self, idx: usize) {
        if idx >= self.stack.len() {
            return;
        }
        let tag = self.stack.operations()[idx].tag().to_string();
        self.reseed_counter = self.reseed_counter.wrapping_add(1);
        let new_seed = 9_000 + self.reseed_counter * 7 + idx as u64;
        let new_op = match tag.as_str() {
            "dense" => Operation::Dense {
                key: Hypervector::random_seeded(self.dim, new_seed),
            },
            "hrr_bind" => Operation::HrrBind {
                key: Hypervector::random_seeded(self.dim, new_seed),
            },
            _ => return,
        };
        self.stack.replace_operation(idx, new_op);
        self.set_status(format!("reseeded op [{idx}] {tag} → seed={new_seed}"));
    }

    fn arch_summary(&self) -> ArchSummary {
        self.stack.arch_summary()
    }

    fn run_forward(&mut self) {
        let started = std::time::Instant::now();
        if let Ok(trace) = self.stack.forward_with_trace(&self.input) {
            #[allow(clippy::cast_precision_loss)]
            let ms = started.elapsed().as_micros() as f64 / 1000.0;
            let sim = cos_sim(&self.input, &trace.bundled).ok();
            self.last_cos_sim = sim;
            if let Some(s) = sim {
                if self.cos_sim_history.len() >= 128 {
                    self.cos_sim_history.pop_front();
                }
                self.cos_sim_history.push_back(s);
            }
            if self.latency_history.len() >= 128 {
                self.latency_history.pop_front();
            }
            self.latency_history.push_back(ms);
            self.last_output = Some(trace.bundled.clone());
            self.last_trace = Some(trace);
            self.last_latency_ms = Some(ms);
            self.forwards = self.forwards.saturating_add(1);
        }
    }

    fn persist(&mut self) {
        let path = persistent_state_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(yaml) = stack_to_yaml(&self.stack) {
            let _ = std::fs::write(&path, yaml);
        }
    }

    fn restore(&mut self) -> bool {
        let path = persistent_state_path();
        let Ok(yaml) = std::fs::read_to_string(&path) else {
            return false;
        };
        match stack_from_yaml(&yaml) {
            Ok(stack) => {
                self.dim = stack.dim();
                self.stack = stack;
                self.input = Hypervector::random_seeded(self.dim, self.input_seed);
                self.last_output = None;
                self.last_trace = None;
                self.last_latency_ms = None;
                self.last_cos_sim = None;
                self.set_status(format!("restored ← {}", path.display()));
                true
            }
            Err(_) => false,
        }
    }

    fn reset_stack(&mut self) {
        self.stack = Stack::new(self.dim);
        self.last_output = None;
        self.last_trace = None;
        self.selected_op = None;
        self.last_latency_ms = None;
        self.last_cos_sim = None;
        self.cos_sim_history.clear();
        self.set_status(format!("stack reset (dim={})", self.dim));
    }

    fn save_yaml(&mut self) {
        match stack_to_yaml(&self.stack) {
            Ok(yaml) => match std::fs::write(YAML_PATH, &yaml) {
                Ok(_) => self.set_status(format!("saved → {YAML_PATH} ({} bytes)", yaml.len())),
                Err(e) => self.set_status(format!("save failed: {e}")),
            },
            Err(e) => self.set_status(format!("encode failed: {e}")),
        }
        // Mirror to persistent state so it survives restart.
        self.persist();
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
            if i.key_pressed(egui::Key::H) || i.key_pressed(egui::Key::F1) {
                self.show_help = !self.show_help;
            }
            if i.key_pressed(egui::Key::Escape) {
                self.show_help = false;
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

                    // Frame-budget indicator dot (green ≤16ms, amber ≤32ms, red >32ms).
                    if let Some(ms) = self.last_latency_ms {
                        let (colour, label) = if ms <= 16.0 {
                            (egui::Color32::from_rgb(0x3F, 0xB1, 0x6E), "60fps OK")
                        } else if ms <= 32.0 {
                            (theme::ACCENT_PURPLE, "30fps OK")
                        } else {
                            (
                                egui::Color32::from_rgb(0xE0, 0x6A, 0x5B),
                                "below 30fps",
                            )
                        };
                        ui.add_space(theme::SPACE_MD);
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(10.0, 10.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().circle_filled(rect.center(), 5.0, colour);
                        ui.label(
                            egui::RichText::new(label)
                                .size(theme::SIZE_TINY)
                                .color(theme::TEXT_MUTED),
                        );
                    }
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

                ui.add_space(theme::SPACE_MD);
                section_heading(ui, "Dim");
                ui.add_space(theme::SPACE_XS);
                ui.label(
                    egui::RichText::new(format!("{}", self.dim_slider))
                        .size(theme::SIZE_BODY)
                        .color(theme::TEXT_PRIMARY)
                        .strong(),
                );
                let mut slider_value = self.dim_slider;
                let slider_resp = ui.add(
                    egui::Slider::new(&mut slider_value, 256..=16_384)
                        .logarithmic(true)
                        .show_value(false)
                        .text(""),
                );
                self.dim_slider = slider_value;
                if slider_resp.drag_stopped() || slider_resp.lost_focus() {
                    self.set_dim(self.dim_slider);
                }
                ui.label(
                    egui::RichText::new("(drag-and-release to apply; stack clears)")
                        .size(theme::SIZE_TINY)
                        .color(theme::TEXT_DIM),
                );

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
                let mut to_reseed: Option<usize> = None;
                for (idx, tag) in &ops {
                    let action = op_chip_actions(ui, *idx, tag);
                    if action.remove {
                        to_remove = Some(*idx);
                    }
                    if action.reseed {
                        to_reseed = Some(*idx);
                    }
                    ui.add_space(theme::SPACE_XS);
                }
                if let Some(i) = to_remove {
                    self.remove_op(i);
                }
                if let Some(i) = to_reseed {
                    self.reseed_op(i);
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
                ui.add_space(theme::SPACE_SM);
                if mini_button(ui, "✕ Reset stack", theme::TEXT_MUTED).clicked() {
                    self.reset_stack();
                }
            });

        // Help overlay.
        if self.show_help {
            egui::Window::new("Keyboard shortcuts  (Esc / H to close)")
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.0, theme::BORDER_ACCENT))
                        .rounding(egui::Rounding::same(theme::RADIUS_LG))
                        .inner_margin(egui::Margin::same(theme::SPACE_XL)),
                )
                .show(ctx, |ui| {
                    let pairs = [
                        ("Space", "run forward"),
                        ("R", "regenerate input"),
                        ("L", "toggle live continuous mode"),
                        ("1 / 2 / 3 / 4", "load template"),
                        ("⌘S / Ctrl+S", "save YAML"),
                        ("⌘O / Ctrl+O", "load YAML"),
                        ("H / F1", "toggle this help"),
                        ("Esc", "close help"),
                    ];
                    egui::Grid::new("help_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_XL, theme::SPACE_SM])
                        .show(ui, |ui| {
                            for (key, action) in pairs {
                                ui.label(
                                    egui::RichText::new(key)
                                        .color(theme::ACCENT_BLUE)
                                        .size(theme::SIZE_BODY)
                                        .monospace()
                                        .strong(),
                                );
                                ui.label(
                                    egui::RichText::new(action)
                                        .color(theme::TEXT_PRIMARY)
                                        .size(theme::SIZE_BODY),
                                );
                                ui.end_row();
                            }
                        });
                });
        }

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
                    if hypervector_heatmap_clickable(ui, &input_clone, 80, 4.0) {
                        self.zoom_target = Some(ZoomTarget::Input);
                    }
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

                // Architecture graph viz.
                section_heading(ui, "Architecture");
                ui.add_space(theme::SPACE_SM);
                let op_tags: Vec<String> = self
                    .stack
                    .operations()
                    .iter()
                    .map(|op| op.tag().to_string())
                    .collect();
                card(ui, |ui| {
                    architecture_graph(ui, &op_tags);
                });

                if !self.cos_sim_history.is_empty() {
                    ui.add_space(theme::SPACE_LG);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            section_heading(ui, "cos_sim history");
                            ui.add_space(theme::SPACE_SM);
                            card(ui, |ui| {
                                sparkline(ui, self.cos_sim_history.iter().copied(), 80.0);
                            });
                        });
                        ui.add_space(theme::SPACE_MD);
                        ui.vertical(|ui| {
                            section_heading(ui, "latency history (ms)");
                            ui.add_space(theme::SPACE_SM);
                            card(ui, |ui| {
                                latency_sparkline(ui, self.latency_history.iter().copied(), 80.0);
                            });
                        });
                    });
                }

                ui.add_space(theme::SPACE_LG);
                let mut zoom_request: Option<ZoomTarget> = None;
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
                        if hypervector_heatmap_clickable(ui, &out, 80, 4.0) {
                            zoom_request = Some(ZoomTarget::Output);
                        }
                    });

                    // Per-op contribution bars (visual story: how much
                    // does each op influence the bundled output?).
                    if let Some(trace) = self.last_trace.clone() {
                        ui.add_space(theme::SPACE_LG);
                        section_heading(ui, "Per-op contribution");
                        ui.add_space(theme::SPACE_SM);
                        let contributions: Vec<(usize, String, f64)> = trace
                            .per_op
                            .iter()
                            .map(|op_out| {
                                let s = cos_sim(&op_out.output, &trace.bundled)
                                    .unwrap_or(0.0);
                                (op_out.index, op_out.tag.clone(), s)
                            })
                            .collect();
                        card(ui, |ui| {
                            contribution_bars(ui, &contributions);
                        });

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
                                    if hypervector_heatmap_clickable(
                                        ui, &op_out.output, 80, 4.0,
                                    ) {
                                        zoom_request = Some(ZoomTarget::PerOp(idx));
                                    }
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
                if let Some(z) = zoom_request {
                    self.zoom_target = Some(z);
                }
            });

        // Zoom modal.
        if let Some(target) = self.zoom_target {
            let hv = match target {
                ZoomTarget::Input => Some(self.input.clone()),
                ZoomTarget::Output => self.last_output.clone(),
                ZoomTarget::PerOp(idx) => self
                    .last_trace
                    .as_ref()
                    .and_then(|t| t.per_op.get(idx).map(|o| o.output.clone())),
            };
            let title = match target {
                ZoomTarget::Input => "Input — full view".to_string(),
                ZoomTarget::Output => "Output — full view".to_string(),
                ZoomTarget::PerOp(idx) => format!("Per-op [{idx}] — full view"),
            };
            if let Some(v) = hv {
                let mut close = false;
                egui::Window::new(title)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .collapsible(false)
                    .resizable(false)
                    .frame(
                        egui::Frame::none()
                            .fill(theme::BG_CARD)
                            .stroke(egui::Stroke::new(1.0, theme::BORDER_ACCENT))
                            .rounding(egui::Rounding::same(theme::RADIUS_LG))
                            .inner_margin(egui::Margin::same(theme::SPACE_XL)),
                    )
                    .show(ctx, |ui| {
                        // Larger cells: 6 px instead of 4.
                        let cols = (v.dim() as f32).sqrt().ceil() as usize;
                        let cols = cols.clamp(40, 160);
                        hypervector_heatmap(ui, &v, cols, 6.0);
                        ui.add_space(theme::SPACE_SM);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("dim = {}", v.dim()))
                                    .color(theme::TEXT_MUTED)
                                    .size(theme::SIZE_SMALL),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("Close (Esc)").clicked() {
                                        close = true;
                                    }
                                },
                            );
                        });
                    });
                if close
                    || ctx.input(|i| i.key_pressed(egui::Key::Escape))
                {
                    self.zoom_target = None;
                }
            } else {
                self.zoom_target = None;
            }
        }
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

#[derive(Default)]
struct OpChipAction {
    remove: bool,
    reseed: bool,
}

fn op_chip_actions(ui: &mut egui::Ui, idx: usize, tag: &str) -> OpChipAction {
    let mut action = OpChipAction::default();
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
                    let remove = ui.add(
                        egui::Button::new(
                            egui::RichText::new("×")
                                .size(theme::SIZE_BODY)
                                .color(theme::TEXT_MUTED),
                        )
                        .fill(egui::Color32::TRANSPARENT)
                        .stroke(egui::Stroke::NONE)
                        .min_size(egui::vec2(20.0, 20.0)),
                    );
                    if remove.clicked() {
                        action.remove = true;
                    }
                    // Reseed only meaningful for keyed ops.
                    if matches!(tag, "dense" | "hrr_bind") {
                        let reseed = ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("⟳")
                                        .size(theme::SIZE_SMALL)
                                        .color(fg),
                                )
                                .fill(egui::Color32::TRANSPARENT)
                                .stroke(egui::Stroke::NONE)
                                .min_size(egui::vec2(20.0, 20.0)),
                            )
                            .on_hover_text("reseed key");
                        if reseed.clicked() {
                            action.reseed = true;
                        }
                    }
                });
            });
        });
    action
}

/// Horizontal bars showing each op's cos_sim(op_output, bundled).
/// Shows at a glance which ops dominate the bundled output.
fn contribution_bars(ui: &mut egui::Ui, entries: &[(usize, String, f64)]) {
    if entries.is_empty() {
        ui.label(
            egui::RichText::new("(no ops)")
                .color(theme::TEXT_DIM)
                .size(theme::SIZE_SMALL),
        );
        return;
    }
    let row_h = 22.0_f32;
    let max_abs = entries
        .iter()
        .map(|(_, _, s)| s.abs())
        .fold(0.0_f64, f64::max)
        .max(0.01);
    let avail_w = ui.available_width();
    let label_w = 110.0;
    let value_w = 64.0;
    let bar_max_w = (avail_w - label_w - value_w - 20.0).max(40.0);

    for (idx, tag, sim) in entries {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(avail_w, row_h), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        // Label.
        let colour = theme::op_color(tag);
        painter.text(
            egui::pos2(rect.min.x + 4.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("[{idx}] {tag}"),
            egui::FontId::proportional(theme::SIZE_SMALL),
            colour,
        );
        // Bar.
        let bar_origin_x = rect.min.x + label_w;
        let bar_w = (sim.abs() / max_abs) as f32 * bar_max_w;
        let bar_rect = egui::Rect::from_min_size(
            egui::pos2(bar_origin_x, rect.center().y - 6.0),
            egui::vec2(bar_w, 12.0),
        );
        let bar_colour = if *sim >= 0.0 {
            colour
        } else {
            colour.gamma_multiply(0.5)
        };
        painter.rect_filled(bar_rect, 6.0, bar_colour);
        // Value text.
        painter.text(
            egui::pos2(rect.max.x - 4.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("{sim:+.3}"),
            egui::FontId::proportional(theme::SIZE_SMALL),
            theme::TEXT_PRIMARY,
        );
    }
}

/// Clickable variant: returns true if the user clicked the heatmap (to
/// request zoom). Otherwise mirrors `hypervector_heatmap`.
fn hypervector_heatmap_clickable(
    ui: &mut egui::Ui,
    v: &Hypervector,
    cols: usize,
    cell: f32,
) -> bool {
    let dim = v.dim();
    let rows = dim.div_ceil(cols);
    let total_w = cols as f32 * cell;
    let total_h = rows as f32 * cell;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(total_w, total_h), egui::Sense::click());
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
    // Hover tooltip.
    if let Some(pos) = response.hover_pos() {
        let local = pos - rect.min;
        let c = (local.x / cell).floor() as i64;
        let r = (local.y / cell).floor() as i64;
        if c >= 0 && r >= 0 && (c as usize) < cols && (r as usize) < rows {
            let idx = (r as usize) * cols + (c as usize);
            if idx < dim {
                response
                    .clone()
                    .on_hover_text_at_pointer(format!(
                        "index {idx}  →  {:+}  ·  click to zoom",
                        data[idx]
                    ));
            }
        }
    }
    response.clicked()
}

fn hypervector_heatmap(ui: &mut egui::Ui, v: &Hypervector, cols: usize, cell: f32) {
    let dim = v.dim();
    let rows = dim.div_ceil(cols);
    let total_w = cols as f32 * cell;
    let total_h = rows as f32 * cell;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(total_w, total_h), egui::Sense::hover());
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
    // Hover tooltip: pinpoint the cell under the cursor.
    if let Some(pos) = response.hover_pos() {
        let local = pos - rect.min;
        let c = (local.x / cell).floor() as i64;
        let r = (local.y / cell).floor() as i64;
        if c >= 0 && r >= 0 && (c as usize) < cols && (r as usize) < rows {
            let idx = (r as usize) * cols + (c as usize);
            if idx < dim {
                response.on_hover_text_at_pointer(format!(
                    "index {idx}  →  {:+}",
                    data[idx]
                ));
            }
        }
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

/// Draw the Stack as a flow graph: INPUT → [op chips in parallel] → BUNDLE → OUTPUT.
fn architecture_graph(ui: &mut egui::Ui, op_tags: &[String]) {
    let n_ops = op_tags.len().max(1);
    let row_h = 220.0_f32.min(80.0 + n_ops as f32 * 26.0);
    let width = ui.available_width().min(720.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, row_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let mid_y = rect.center().y;
    let left_x = rect.min.x + 60.0;
    let right_x = rect.max.x - 60.0;
    let chip_w = 90.0;
    let chip_h = 26.0;
    let middle_x = (left_x + right_x) * 0.5;

    // INPUT bubble.
    let in_pos = egui::pos2(left_x, mid_y);
    painter.circle_filled(in_pos, 28.0, theme::ACCENT_BLUE);
    painter.text(
        in_pos,
        egui::Align2::CENTER_CENTER,
        "INPUT",
        egui::FontId::proportional(theme::SIZE_TINY),
        egui::Color32::WHITE,
    );

    // BUNDLE bubble (closer to output).
    let bundle_pos = egui::pos2(right_x - 70.0, mid_y);
    painter.circle_filled(bundle_pos, 28.0, theme::ACCENT_PURPLE);
    painter.text(
        bundle_pos,
        egui::Align2::CENTER_CENTER,
        "BUNDLE",
        egui::FontId::proportional(theme::SIZE_TINY),
        egui::Color32::WHITE,
    );

    // OUTPUT bubble.
    let out_pos = egui::pos2(right_x, mid_y);
    painter.circle_filled(out_pos, 28.0, theme::ACCENT_MID);
    painter.text(
        out_pos,
        egui::Align2::CENTER_CENTER,
        "OUT",
        egui::FontId::proportional(theme::SIZE_TINY),
        egui::Color32::WHITE,
    );

    // BUNDLE → OUTPUT connector.
    painter.line_segment(
        [
            egui::pos2(bundle_pos.x + 28.0, mid_y),
            egui::pos2(out_pos.x - 28.0, mid_y),
        ],
        egui::Stroke::new(1.5, theme::TEXT_MUTED),
    );

    // Op chips stacked vertically between input and bundle.
    let op_total_h = n_ops as f32 * chip_h + (n_ops as f32 - 1.0).max(0.0) * 4.0;
    let start_y = mid_y - op_total_h * 0.5;

    if op_tags.is_empty() {
        // No ops yet — draw a dashed grey wire.
        painter.line_segment(
            [
                egui::pos2(in_pos.x + 28.0, mid_y),
                egui::pos2(bundle_pos.x - 28.0, mid_y),
            ],
            egui::Stroke::new(1.0, theme::TEXT_DIM),
        );
        painter.text(
            egui::pos2(middle_x, mid_y - 18.0),
            egui::Align2::CENTER_CENTER,
            "(no ops — add some on the left)",
            egui::FontId::proportional(theme::SIZE_TINY),
            theme::TEXT_DIM,
        );
        return;
    }

    for (i, tag) in op_tags.iter().enumerate() {
        let y = start_y + i as f32 * (chip_h + 4.0);
        let chip_rect = egui::Rect::from_center_size(
            egui::pos2(middle_x, y + chip_h * 0.5),
            egui::vec2(chip_w, chip_h),
        );
        let colour = theme::op_color(tag);
        painter.rect_filled(
            chip_rect,
            theme::RADIUS_PILL,
            colour.gamma_multiply(0.25),
        );
        painter.rect_stroke(
            chip_rect,
            theme::RADIUS_PILL,
            egui::Stroke::new(1.0, colour),
        );
        painter.text(
            chip_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("[{i}] {tag}"),
            egui::FontId::proportional(theme::SIZE_TINY),
            colour,
        );

        // Input → chip connector.
        painter.line_segment(
            [
                egui::pos2(in_pos.x + 28.0, mid_y),
                egui::pos2(chip_rect.min.x, chip_rect.center().y),
            ],
            egui::Stroke::new(1.0, theme::TEXT_MUTED),
        );
        // Chip → bundle connector.
        painter.line_segment(
            [
                egui::pos2(chip_rect.max.x, chip_rect.center().y),
                egui::pos2(bundle_pos.x - 28.0, mid_y),
            ],
            egui::Stroke::new(1.0, theme::TEXT_MUTED),
        );
    }
}

/// Plot recent latency values (positive ms). Auto-scales y-axis.
fn latency_sparkline(ui: &mut egui::Ui, values: impl Iterator<Item = f64>, height: f32) {
    let vs: Vec<f64> = values.collect();
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    if vs.is_empty() {
        return;
    }

    let max_v = vs.iter().copied().fold(0.0_f64, f64::max).max(1.0);

    // 16ms (60fps) reference line.
    if max_v > 16.0 {
        let y16 = rect.max.y - (16.0 / max_v) as f32 * rect.height();
        painter.line_segment(
            [egui::pos2(rect.min.x, y16), egui::pos2(rect.max.x, y16)],
            egui::Stroke::new(0.5, theme::ACCENT_PURPLE.gamma_multiply(0.5)),
        );
    }

    if vs.len() < 2 {
        return;
    }

    let step = rect.width() / (vs.len() - 1) as f32;
    let to_y = |v: f64| -> f32 {
        let t = (v / max_v) as f32;
        rect.max.y - t * rect.height()
    };

    let mut prev = egui::pos2(rect.min.x, to_y(vs[0]));
    for (i, v) in vs.iter().enumerate().skip(1) {
        let p = egui::pos2(rect.min.x + step * i as f32, to_y(*v));
        let colour = if *v <= 16.0 {
            theme::ACCENT_BLUE
        } else if *v <= 32.0 {
            theme::ACCENT_PURPLE
        } else {
            egui::Color32::from_rgb(0xE0, 0x6A, 0x5B)
        };
        painter.line_segment([prev, p], egui::Stroke::new(1.5, colour));
        prev = p;
    }

    // Annotate max.
    painter.text(
        egui::pos2(rect.max.x - 4.0, rect.min.y + 8.0),
        egui::Align2::RIGHT_TOP,
        format!("max {:.2} ms", max_v),
        egui::FontId::proportional(theme::SIZE_TINY),
        theme::TEXT_MUTED,
    );
}

/// Tiny inline plot of recent values, ∈ [-1, 1] expected.
fn sparkline(ui: &mut egui::Ui, values: impl Iterator<Item = f64>, height: f32) {
    let vs: Vec<f64> = values.collect();
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Zero-line in the middle.
    let mid_y = rect.center().y;
    painter.line_segment(
        [egui::pos2(rect.min.x, mid_y), egui::pos2(rect.max.x, mid_y)],
        egui::Stroke::new(0.5, theme::TEXT_DIM),
    );

    if vs.len() < 2 {
        return;
    }

    let step = rect.width() / (vs.len() - 1) as f32;
    let to_y = |v: f64| -> f32 {
        let t = ((v + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
        rect.max.y - t * rect.height()
    };

    let mut prev = egui::pos2(rect.min.x, to_y(vs[0]));
    for (i, v) in vs.iter().enumerate().skip(1) {
        let p = egui::pos2(rect.min.x + step * i as f32, to_y(*v));
        painter.line_segment([prev, p], egui::Stroke::new(1.5, theme::ACCENT_BLUE));
        prev = p;
    }

    // Highlight last value.
    let last = egui::pos2(rect.max.x, to_y(*vs.last().unwrap_or(&0.0)));
    painter.circle_filled(last, 3.0, theme::ACCENT_PURPLE);
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
