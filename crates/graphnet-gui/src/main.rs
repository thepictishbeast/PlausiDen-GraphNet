//! GraphNet native GUI shell — Phase 12 scaffold.
//!
//! Minimal eframe app that loads a Stack from the gallery + shows arch
//! summary + lets the user step through a forward pass. Lays the
//! foundation for the full Phase 12 interactive dashboard (3D graph,
//! intervention controls, resource gauges, time-travel debug).
//!
//! Run: `cargo run -p graphnet-gui --release`
//!
//! Phase 12 follow-up ticks add: wgpu 3D graph view, intervention
//! palette, resource panel, Tauri shell for distribution.

#![forbid(unsafe_code)]

use eframe::egui;
use graphnet_engine::{ArchSummary, Model, Operation, Stack};
use plausiden_hdc::Hypervector;

/// Application state.
struct App {
    stack: Stack,
    /// The deterministic input we feed on every forward.
    input: Hypervector,
    /// Most recent forward output (None until first run).
    last_output: Option<Hypervector>,
    /// Forward counter.
    forwards: u64,
}

impl App {
    fn new() -> Self {
        let stack = Stack::new(1_000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Dense {
                key: Hypervector::random_seeded(1_000, 1),
            });
        let input = Hypervector::random_seeded(1_000, 42);
        Self {
            stack,
            input,
            last_output: None,
            forwards: 0,
        }
    }

    fn arch_summary(&self) -> ArchSummary {
        self.stack.arch_summary()
    }

    fn run_forward(&mut self) {
        if let Ok(out) = self.stack.forward(&self.input) {
            self.last_output = Some(out);
            self.forwards = self.forwards.saturating_add(1);
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let summary = self.arch_summary();

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("GraphNet");
                ui.separator();
                ui.label(graphnet_engine::banner());
            });
        });

        egui::SidePanel::left("left").show(ctx, |ui| {
            ui.heading("Architecture");
            ui.label(format!("family: {}", summary.family));
            ui.label(format!("input dim: {}", summary.input_dim));
            ui.label(format!("output dim: {}", summary.output_dim));
            ui.label(format!("substructures: {}", summary.substructures));
            for note in &summary.notes {
                ui.label(note);
            }
            ui.separator();
            ui.heading("Operations");
            for (idx, op) in self.stack.operations().iter().enumerate() {
                ui.label(format!("[{idx}] {}", op.tag()));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Forward pass");
            ui.label(format!("forwards run: {}", self.forwards));
            if ui.button("▶ Run forward").clicked() {
                self.run_forward();
            }
            if let Some(out) = &self.last_output {
                ui.separator();
                ui.label(format!("output dim: {}", out.dim()));
                ui.label(format!(
                    "first 32 bipolar values: {:?}",
                    &out.as_slice()[..out.dim().min(32)]
                ));
            }
        });
    }
}

/// Entry point.
///
/// BUG ASSUMPTION: requires a graphical environment (X11 / Wayland / macOS /
/// Windows). Headless CI builds the binary but cannot run it.
fn main() -> Result<(), eframe::Error> {
    eframe::run_native(
        "GraphNet",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn app_constructs_with_default_stack() {
        let app = App::new();
        assert_eq!(app.stack.dim(), 1_000);
        assert_eq!(app.stack.len(), 2);
        assert_eq!(app.forwards, 0);
        assert!(app.last_output.is_none());
    }

    #[test]
    fn run_forward_advances_counter_and_captures_output() {
        let mut app = App::new();
        app.run_forward();
        assert_eq!(app.forwards, 1);
        assert!(app.last_output.is_some());
        let out = app.last_output.as_ref().expect("just set");
        assert_eq!(out.dim(), 1_000);
    }

    #[test]
    fn arch_summary_reports_two_ops() {
        let app = App::new();
        let summary = app.arch_summary();
        assert_eq!(summary.family, "stack");
        assert_eq!(summary.substructures, 2);
    }
}
