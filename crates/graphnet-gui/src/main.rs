//! GraphNet native GUI shell — PlausiDen-themed, GPU-accelerated, interactive.
//! Iter 3 adds: keyboard shortcuts, live continuous mode, cosine-similarity
//! gauge, Save/Load YAML buttons, slimmer hero.

#![forbid(unsafe_code)]

mod theme;

use eframe::egui;
use graphnet_engine::{
    flop_estimate, stack_from_yaml, stack_to_yaml, ArchSummary, ForwardTrace, Model, Operation,
    ResourceMonitor, ResourceSample, Stack,
};
use plausiden_hdc::{cos_sim, hamming, Hypervector};

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
    walkthrough_step: Option<usize>,
    demo: Option<DemoState>,
    spawn_time: std::time::Instant,
    /// Yaw rotation of the 3D arch graph, in radians.
    arch_yaw: f32,
    /// Pitch rotation of the 3D arch graph (vertical drag).
    arch_pitch: f32,
    /// Roll rotation of the 3D arch graph (Shift+drag).
    arch_roll: f32,
    /// Auto-rotate the 3D arch graph?
    arch_autorotate: bool,
    /// Arch graph zoom factor (1.0 = baseline, 0.5..3.0).
    arch_zoom: f32,
    /// Op index currently being dragged in the sidebar (for reorder).
    drag_source: Option<usize>,
    /// Time of the most recent forward — animates particle flow on connectors.
    last_forward_at: Option<std::time::Instant>,
    /// Ring buffer of "action" log lines (timestamp + severity + text),
    /// shown in the right-hand panel for interaction feedback.
    action_log: Vec<LogEntry>,
    /// Which left-panel mode the tool palette has selected.
    tool_mode: ToolMode,
    /// Undo stack: snapshots of Stack BEFORE each mutating action.
    undo_stack: Vec<Stack>,
    /// Redo stack: snapshots popped by undo, restorable via redo.
    redo_stack: Vec<Stack>,
    /// Template search filter (#717).
    template_filter: String,
    /// Live host CPU / RAM monitor.
    resource_monitor: ResourceMonitor,
    last_sample: Option<ResourceSample>,
    last_sample_at: std::time::Instant,
    /// Unlocked achievements (slug → Instant of unlock).
    achievements: std::collections::HashMap<&'static str, std::time::Instant>,
    /// Tracking sets for achievement criteria.
    op_kinds_seen: std::collections::HashSet<String>,
    templates_seen: std::collections::HashSet<String>,
    /// Index of the current active objective (0..OBJECTIVES.len()).
    current_objective: usize,
    /// Whether each objective is completed (parallel to OBJECTIVES).
    objective_done: Vec<bool>,
    /// Colormap for hypervector heatmaps (#710).
    colormap: Colormap,
    /// Show the templates popup modal? (#745)
    show_templates_popup: bool,
    /// Show the console/REPL pane? Toggled by backtick (#747).
    show_console: bool,
    /// Console input buffer.
    console_input: String,
    /// Console history of (command, output) lines.
    console_history: Vec<(String, String)>,
    /// Console history cursor for up-arrow recall (-1 = no recall active).
    console_history_cursor: i32,
    /// Timestamp of the user's last input action (key/click/forward).
    last_user_action: std::time::Instant,
    /// Currently-active adaptive-tutorial hint (None if no hint).
    adaptive_hint: Option<&'static str>,
    /// Font-scale multiplier — user-adjustable in Settings (#732).
    font_scale: f32,
    /// Demo pace in seconds per template.
    demo_pace_sec: f64,
    /// Most recently mutated op index + timestamp (drives diff halo #718).
    recent_change: Option<(usize, std::time::Instant)>,
    /// Workspace tab (#743).
    workspace: Workspace,
    /// Zoom-modal cell-size multiplier (#721).
    zoom_modal_scale: f32,
    /// A/B compare: snapshot of a stack to compare against current (#711).
    snapshot_stack: Option<Stack>,
    /// 4 stack slots A/B/C/D for multi-stack comparison (#722). None = empty slot.
    slots: [Option<Stack>; 4],
    /// Which slot is currently active for editing (matches App.stack).
    active_slot: usize,
    /// Audio output stream + sink (#723). None when audio is disabled.
    audio: Option<AudioState>,
    /// User-facing audio enable toggle.
    audio_enabled: bool,
    /// Show the left arch panel? (User can collapse it via [Tab].)
    show_left_panel: bool,
    /// Show the right action/contrib panel?
    show_right_panel: bool,
    /// Show the floating stats window? (#741)
    show_floating_stats: bool,
    /// Show the floating mini-help window? (#741)
    show_floating_minihelp: bool,
    /// Training pipeline state (#757).
    train: TrainState,
    /// Loaded image path (for status display) (#758).
    loaded_image: Option<std::path::PathBuf>,
    /// When was the most recent 3D op-node click? Used to flash the node.
    last_node_click_at: Option<std::time::Instant>,
    /// Which op was last clicked in the 3D graph (for the flash).
    last_node_clicked: Option<usize>,
}

#[derive(Debug, Clone)]
struct TrainState {
    /// Target hypervector — what we want the stack to produce from `App.input`.
    target: Option<Hypervector>,
    /// Loss over time: 1 - cos_sim(actual, target), one entry per training step.
    loss_history: Vec<f64>,
    /// Which training mode is selected.
    mode: TrainMode,
    /// How many bits to flip per step (perturbation magnitude).
    perturb_bits: usize,
    /// Train counter.
    steps: u64,
    /// Random seed for perturbations (advanced per step).
    rng_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainMode {
    /// Random-perturb a Dense/HrrBind key; accept if loss improved.
    HillClimb,
    /// Same as HillClimb but accepts worse with probability decreasing in delta.
    SimulatedAnneal,
    /// Replace a random key entirely each step (lazy baseline).
    Random,
}

impl TrainMode {
    fn all() -> &'static [TrainMode] {
        &[
            TrainMode::HillClimb,
            TrainMode::SimulatedAnneal,
            TrainMode::Random,
        ]
    }
    fn label(self) -> &'static str {
        match self {
            TrainMode::HillClimb => "hill-climb",
            TrainMode::SimulatedAnneal => "simulated anneal",
            TrainMode::Random => "random",
        }
    }
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            target: None,
            loss_history: Vec::new(),
            mode: TrainMode::HillClimb,
            perturb_bits: 50,
            steps: 0,
            rng_seed: 1,
        }
    }
}

struct AudioState {
    /// Stub for now — real rodio output requires libasound2-dev on Linux.
    /// When available, hold OutputStream + OutputStreamHandle here.
    _placeholder: (),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workspace {
    Edit,
    Live,
    Compare,
    Train,
}

impl Workspace {
    fn all() -> &'static [Workspace] {
        &[Workspace::Edit, Workspace::Live, Workspace::Compare, Workspace::Train]
    }
    fn label(self) -> &'static str {
        match self {
            Workspace::Edit => "📝 Edit",
            Workspace::Live => "▶ Live",
            Workspace::Compare => "⇄ Compare",
            Workspace::Train => "🎓 Train",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Colormap {
    /// Default: blue (positive) / purple-dim (negative). Pure bipolar.
    Bipolar,
    /// Viridis-style: dark purple → teal → yellow.
    Viridis,
    /// Plasma-style: dark purple → magenta → orange → yellow.
    Plasma,
    /// Monochrome: black (negative) → white (positive).
    Mono,
}

impl Colormap {
    fn all() -> &'static [Colormap] {
        &[
            Colormap::Bipolar,
            Colormap::Viridis,
            Colormap::Plasma,
            Colormap::Mono,
        ]
    }
    fn label(self) -> &'static str {
        match self {
            Colormap::Bipolar => "bipolar",
            Colormap::Viridis => "viridis",
            Colormap::Plasma => "plasma",
            Colormap::Mono => "mono",
        }
    }
    /// Map a bipolar value (-1 or +1) to a colour for this colormap.
    fn map(self, v: i8) -> egui::Color32 {
        match self {
            Colormap::Bipolar => {
                if v > 0 {
                    theme::ACCENT_BLUE
                } else {
                    theme::ACCENT_PURPLE.gamma_multiply(0.55)
                }
            }
            Colormap::Viridis => {
                if v > 0 {
                    egui::Color32::from_rgb(0xFD, 0xE7, 0x25) // yellow
                } else {
                    egui::Color32::from_rgb(0x44, 0x01, 0x54) // dark purple
                }
            }
            Colormap::Plasma => {
                if v > 0 {
                    egui::Color32::from_rgb(0xF0, 0xF9, 0x21) // pale yellow
                } else {
                    egui::Color32::from_rgb(0x0D, 0x08, 0x87) // deep blue-violet
                }
            }
            Colormap::Mono => {
                if v > 0 {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_rgb(0x14, 0x18, 0x22)
                }
            }
        }
    }
}

/// One challenge in the objectives catalog (#720).
struct Objective {
    title: &'static str,
    description: &'static str,
    /// Function that takes the App state and returns `true` if complete.
    check: fn(&App) -> bool,
}

const OBJECTIVES: &[Objective] = &[
    Objective {
        title: "1. First Contact",
        description: "Press Space (or click ▶ Run forward) to run your first forward.",
        check: |a| a.forwards >= 1,
    },
    Objective {
        title: "2. Add an Op",
        description: "Add any operation to the stack via + Identity / + Dense / + HrrBind / + Permute / + Negate.",
        check: |a| a.stack.len() > 3, // standard template has 3 ops
    },
    Objective {
        title: "3. Try HrrBind",
        description: "Use the FFT-based hrr_bind operation — load fft-heavy (3) or add it manually.",
        check: |a| a.op_kinds_seen.contains("hrr_bind"),
    },
    Objective {
        title: "4. Five Forwards",
        description: "Run forward five times. Watch the cos_sim history sparkline build up.",
        check: |a| a.forwards >= 5,
    },
    Objective {
        title: "5. Live Mode",
        description: "Press L to start live continuous mode. Stay in it for a few seconds.",
        check: |a| a.forwards >= 50,
    },
    Objective {
        title: "6. Decorrelate",
        description: "Build a stack where cos_sim(input, output) ≤ 0.05 (orthogonal).",
        check: |a| a.last_cos_sim.is_some_and(|s| s.abs() <= 0.05),
    },
    Objective {
        title: "7. Save Your Work",
        description: "Save a YAML config to disk (⌘S / Ctrl+S).",
        check: |a| a.achievements.contains_key("demo_runner") || a.action_log.iter().any(|e| e.msg.starts_with("saved →")),
    },
    Objective {
        title: "8. Hyperdimensional",
        description: "Bump the dim to 16,384 in Settings and run a forward there.",
        check: |a| a.dim >= 16_000 && a.forwards >= 1,
    },
    Objective {
        title: "9. All Op Kinds",
        description: "Use every op kind: identity, dense, hrr_bind, permute, negate.",
        check: |a| a.op_kinds_seen.len() >= 5,
    },
    Objective {
        title: "10. Template Connoisseur",
        description: "Load every example config (1 through 0).",
        check: |a| a.templates_seen.len() >= TEMPLATES.len(),
    },
];

#[derive(Debug, Clone)]
struct LogEntry {
    at: std::time::Instant,
    severity: LogSeverity,
    msg: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogSeverity {
    Info,
    Success,
    Warn,
    Error,
}

impl LogSeverity {
    fn color(self) -> egui::Color32 {
        match self {
            LogSeverity::Info => theme::TEXT_PRIMARY,
            LogSeverity::Success => egui::Color32::from_rgb(0x4D, 0xC4, 0x82),
            LogSeverity::Warn => egui::Color32::from_rgb(0xE0, 0xB1, 0x5B),
            LogSeverity::Error => egui::Color32::from_rgb(0xE0, 0x6A, 0x5B),
        }
    }
    fn glyph(self) -> &'static str {
        match self {
            LogSeverity::Info => "ⓘ",
            LogSeverity::Success => "✓",
            LogSeverity::Warn => "⚠",
            LogSeverity::Error => "✕",
        }
    }
}

struct HvSummary {
    fingerprint: String,
    percent_positive: f64,
    binary_prefix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolMode {
    Templates,
    Edit,
    Inspect,
    Compare,
    Settings,
    Help,
}

impl ToolMode {
    fn icon(self) -> &'static str {
        match self {
            ToolMode::Templates => "📋",
            ToolMode::Edit => "✏",
            ToolMode::Inspect => "🔍",
            ToolMode::Compare => "⇄",
            ToolMode::Settings => "⚙",
            ToolMode::Help => "❓",
        }
    }
    fn label(self) -> &'static str {
        match self {
            ToolMode::Templates => "Templates (open popup)",
            ToolMode::Edit => "Edit stack",
            ToolMode::Inspect => "Inspect (selected op)",
            ToolMode::Compare => "Compare (A/B/C/D slots)",
            ToolMode::Settings => "Settings",
            ToolMode::Help => "Help (open overlay)",
        }
    }
}

#[derive(Debug, Clone)]
struct DemoState {
    template_idx: usize,
    started_at: std::time::Instant,
    /// Seconds per template.
    pace_sec: f64,
    paused: bool,
    /// Time-offset paused at, for resuming with correct elapsed.
    paused_offset: f64,
}

const WALKTHROUGH_STEPS: &[(&str, &str)] = &[
    (
        "Welcome to GraphNet",
        "A live REPL + graphing calculator for HDC neural networks. \
         No training, no GPU model files — just compose ops and watch the \
         network behave in real time. Press → for the tour.",
    ),
    (
        "1. Pick a template (or blank)",
        "Click + New… in the left panel for the templates popup with 10 \
         example configs + blank-stack option, or press number keys 1-9/0 \
         for direct selection. Each template has an explanation on hover.",
    ),
    (
        "2. Run a forward",
        "Press SPACE (or click ▶ Run forward). The Stack applies every op \
         in parallel and bundles the outputs into one HDC vector. The Output \
         card and 3D arch graph animate to show data flowing.",
    ),
    (
        "3. Mutate the network live",
        "Keyboard: A/D/F/P/N adds Identity/Dense/HrrBind/Permute/Negate. \
         Backspace removes the selected op. Right-click any chip for \
         reseed/duplicate/move/convert/remove. Drag chips to reorder. \
         Drag the Dim slider in Settings (or type a number).",
    ),
    (
        "4. Inspect per-op behaviour",
        "After a forward, the Per-op contribution bars + Per-op inspector \
         in the right panel show each operation's individual output and its \
         cos_sim to input + bundled. Click chips in the 3D arch graph or \
         the inspector to drill in.",
    ),
    (
        "5. 3D architecture viewport",
        "Drag the 3D graph for yaw/pitch. Shift+drag for roll. Scroll wheel \
         zooms. F resets rotation. Click an op node to select; the inline \
         editor under the graph lets you reseed / duplicate / move / remove \
         / convert. Right-click an op chip in the sidebar for the full \
         context menu. Each op kind has a unique symbolic glyph in its 3D \
         node.",
    ),
    (
        "6. Live mode + objectives",
        "Press L to start live continuous mode (FPS shown). The right panel \
         tracks: cos_sim history, latency history, achievements (12 badges), \
         and a 10-step Objectives card that guides discovery.",
    ),
    (
        "7. Save / Load / Share / Console",
        "⌘S / ⌘O for save/load YAML (native file dialogs). ⌘E for PNG \
         export. Drag-drop a .yaml on the window to load. Press ` (backtick) \
         for the REPL/console — type 'help' for commands. Auto-saves to \
         ~/.config/graphnet/state.yaml.",
    ),
    (
        "8. Help is always there",
        "Press H or F1 anytime — shows shortcuts, achievements grid, \
         experiment recipes. Adaptive hints surface a banner if you're \
         idle for 30 seconds with stuck state. Esc closes any modal.",
    ),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZoomTarget {
    Input,
    Output,
    PerOp(usize),
}

struct Template {
    name: &'static str,
    summary: &'static str,
    explanation: &'static str,
    dim: usize,
    /// Op kind tags applied in order: "identity" / "dense" / "hrr_bind".
    ops: &'static [&'static str],
}

const TEMPLATES: &[Template] = &[
    Template {
        name: "minimal",
        summary: "1 identity · D=1k",
        explanation: "Smallest possible Stack — one passthrough op. \
                      Output equals input. Useful for sanity-checking the pipeline.",
        dim: 1_000,
        ops: &["identity"],
    },
    Template {
        name: "standard",
        summary: "id + dense + hrr · D=10k",
        explanation: "The canonical heterogeneous Stack: one identity, one dense \
                      projection, one HRR (FFT-based) binding. Demonstrates all three \
                      op kinds bundled into a single output.",
        dim: 10_000,
        ops: &["identity", "dense", "hrr_bind"],
    },
    Template {
        name: "echo-state",
        summary: "3 identity + 1 dense",
        explanation: "Three identity ops + one dense projection. Output is dominated \
                      by the input (high cos_sim to input) with a small dense perturbation. \
                      Models a 'reservoir' that mostly preserves the signal.",
        dim: 10_000,
        ops: &["identity", "identity", "identity", "dense"],
    },
    Template {
        name: "mixture-of-4",
        summary: "4 dense projections",
        explanation: "Four parallel dense projections with different random keys. \
                      Each op produces an independent transformation; the bundle \
                      averages them — analogous to a mixture-of-experts with equal weights.",
        dim: 10_000,
        ops: &["dense", "dense", "dense", "dense"],
    },
    Template {
        name: "fft-heavy",
        summary: "3 hrr (FFT bindings)",
        explanation: "Three HRR (Holographic Reduced Representation) bindings via FFT \
                      circular convolution. Highest per-forward cost; watch the latency \
                      sparkline. D=1024 (power of 2 for FFT efficiency).",
        dim: 1_024,
        ops: &["hrr_bind", "hrr_bind", "hrr_bind"],
    },
    Template {
        name: "noise-resilience",
        summary: "8 mixed ops",
        explanation: "Eight ops mixing identity + dense + hrr_bind. Even with this much \
                      cross-talk, the bundled output retains meaningful similarity to the \
                      input — demonstrates HDC's noise robustness.",
        dim: 10_000,
        ops: &[
            "identity",
            "dense",
            "hrr_bind",
            "identity",
            "dense",
            "hrr_bind",
            "dense",
            "identity",
        ],
    },
    Template {
        name: "dense-cascade",
        summary: "6 dense ops · D=10k",
        explanation: "Six dense projections — denser network, higher latency. Compare \
                      cos_sim(input, output) against the 4-projection mixture: more ops \
                      generally drives output toward the centroid of the key space.",
        dim: 10_000,
        ops: &["dense", "dense", "dense", "dense", "dense", "dense"],
    },
    Template {
        name: "wide-D",
        summary: "id + dense · D=16k",
        explanation: "Same as standard but at D=16,384. Wider hypervectors → more \
                      capacity (more distinguishable random vectors) at the cost of \
                      latency and memory.",
        dim: 16_384,
        ops: &["identity", "dense"],
    },
    Template {
        name: "positional-encode",
        summary: "4 permute · D=10k",
        explanation: "Four circular permutations at staggered shifts. Each op \
                      rotates the input by a different amount; the bundle is the \
                      mean of those rotations. Models a positional-encoding step \
                      in HDC: bind(item, permute(key, position)).",
        dim: 10_000,
        ops: &["permute", "permute", "permute", "permute"],
    },
    Template {
        name: "anti-correlation",
        summary: "dense + negate · D=10k",
        explanation: "Demonstrates Negate (cos_sim(v, -v) = -1). A dense projection \
                      followed by a negated dense projection. The bundle of v and -v \
                      lies on the equator (cos_sim ≈ 0). Useful for null-space \
                      experiments.",
        dim: 10_000,
        ops: &["dense", "negate", "dense"],
    },
];

const FIRST_RUN_PATH: &str = ".graphnet_seen_walkthrough";

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
            walkthrough_step: None,
            demo: None,
            spawn_time: std::time::Instant::now(),
            arch_yaw: 0.6,
            arch_pitch: 0.15,
            arch_roll: 0.0,
            arch_autorotate: false,
            arch_zoom: 1.0,
            drag_source: None,
            last_forward_at: None,
            action_log: Vec::new(),
            tool_mode: ToolMode::Edit,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            template_filter: String::new(),
            resource_monitor: ResourceMonitor::new(),
            last_sample: None,
            last_sample_at: std::time::Instant::now(),
            achievements: std::collections::HashMap::new(),
            op_kinds_seen: std::collections::HashSet::new(),
            templates_seen: std::collections::HashSet::new(),
            current_objective: 0,
            objective_done: vec![false; OBJECTIVES.len()],
            colormap: Colormap::Bipolar,
            show_templates_popup: false,
            show_console: false,
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_cursor: -1,
            last_user_action: std::time::Instant::now(),
            adaptive_hint: None,
            font_scale: 1.0,
            demo_pace_sec: 2.5,
            recent_change: None,
            workspace: Workspace::Edit,
            zoom_modal_scale: 1.0,
            snapshot_stack: None,
            show_left_panel: true,
            show_right_panel: true,
            show_floating_stats: false,
            show_floating_minihelp: false,
            train: TrainState::default(),
            loaded_image: None,
            last_node_click_at: None,
            last_node_clicked: None,
            slots: [None, None, None, None],
            active_slot: 0,
            audio: None,
            audio_enabled: false,
        };
        app.load_template("standard");
        // Try to restore previous session.
        let _ = app.restore();
        // Run an initial forward so the user sees an output IMMEDIATELY
        // — addresses "i can't tell if anything's working" at startup.
        app.run_forward();
        // First-run walkthrough.
        let walkthrough_marker = persistent_state_path()
            .parent()
            .map(|p| p.join(FIRST_RUN_PATH))
            .unwrap_or_else(|| std::path::PathBuf::from(FIRST_RUN_PATH));
        if !walkthrough_marker.exists() {
            app.walkthrough_step = Some(0);
        }
        app
    }

    fn dismiss_walkthrough(&mut self) {
        self.walkthrough_step = None;
        let marker = persistent_state_path()
            .parent()
            .map(|p| p.join(FIRST_RUN_PATH))
            .unwrap_or_else(|| std::path::PathBuf::from(FIRST_RUN_PATH));
        if let Some(parent) = marker.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&marker, b"1");
    }

    fn start_demo(&mut self) {
        self.demo = Some(DemoState {
            template_idx: 0,
            started_at: std::time::Instant::now(),
            pace_sec: self.demo_pace_sec,
            paused: false,
            paused_offset: 0.0,
        });
        self.set_status(format!(
            "Demo started — cycling templates ({:.1}s each)",
            self.demo_pace_sec
        ));
    }

    fn demo_pause_toggle(&mut self) {
        if let Some(d) = self.demo.as_mut() {
            if d.paused {
                d.paused = false;
                d.started_at = std::time::Instant::now()
                    - std::time::Duration::from_secs_f64(d.paused_offset);
                self.set_status("demo resumed".to_string());
            } else {
                d.paused = true;
                d.paused_offset = d.started_at.elapsed().as_secs_f64();
                self.set_status("demo paused".to_string());
            }
        }
    }

    fn demo_skip(&mut self, delta: i32) {
        if let Some(d) = self.demo.as_mut() {
            let n = TEMPLATES.len() as i32;
            let new_idx =
                ((d.template_idx as i32 + delta).clamp(0, n - 1)) as usize;
            d.template_idx = new_idx;
            d.started_at = std::time::Instant::now()
                - std::time::Duration::from_secs_f64(new_idx as f64 * d.pace_sec);
            d.paused_offset = new_idx as f64 * d.pace_sec;
        }
    }

    fn stop_demo(&mut self) {
        self.demo = None;
        self.set_status("Demo stopped".to_string());
    }

    fn tick_demo(&mut self) {
        let Some(demo) = self.demo.clone() else {
            return;
        };
        if demo.paused {
            return;
        }
        let elapsed = demo.started_at.elapsed().as_secs_f64();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_idx = (elapsed / demo.pace_sec) as usize;
        if target_idx >= TEMPLATES.len() {
            self.set_status("Demo complete — try templates yourself".to_string());
            self.demo = None;
            return;
        }
        if target_idx != demo.template_idx {
            let t = TEMPLATES[target_idx].name;
            self.load_template(t);
            self.run_forward();
            self.set_status(format!(
                "Demo step {}/{}: {} — {}",
                target_idx + 1,
                TEMPLATES.len(),
                t,
                TEMPLATES[target_idx].summary
            ));
            if let Some(d) = self.demo.as_mut() {
                d.template_idx = target_idx;
            }
        }
    }

    fn toggle_mode(&mut self, ctx: &egui::Context) {
        self.mode = match self.mode {
            theme::Mode::Dark => theme::Mode::Light,
            theme::Mode::Light => theme::Mode::Dark,
        };
        theme::install_mode(ctx, self.mode);
    }

    fn load_template(&mut self, name: &'static str) {
        self.push_undo();
        let template = TEMPLATES
            .iter()
            .find(|t| t.name == name)
            .unwrap_or(&TEMPLATES[1]);
        self.template = template.name;
        let dim = template.dim;
        self.dim = dim;
        self.input = Hypervector::random_seeded(dim, self.input_seed);
        self.stack = Stack::new(dim);
        for (i, kind) in template.ops.iter().enumerate() {
            let seed = (i as u64) + 1;
            let op = match *kind {
                "identity" => Operation::Identity,
                "dense" => Operation::Dense {
                    key: Hypervector::random_seeded(dim, seed),
                },
                "hrr_bind" => Operation::HrrBind {
                    key: Hypervector::random_seeded(dim, seed + 100),
                },
                _ => continue,
            };
            self.stack.add_operation(op);
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
        self.push_undo();
        let new_idx = self.stack.len();
        let seed = self.stack.len() as u64 + 100;
        self.recent_change = Some((new_idx, std::time::Instant::now()));
        let op = match kind {
            "identity" => Operation::Identity,
            "dense" => Operation::Dense {
                key: Hypervector::random_seeded(self.dim, seed),
            },
            "hrr_bind" => Operation::HrrBind {
                key: Hypervector::random_seeded(self.dim, seed + 100),
            },
            "permute" => Operation::Permute {
                shift: (self.stack.len() * 7 + 13) % self.dim.max(1),
            },
            "negate" => Operation::Negate,
            _ => return,
        };
        self.stack.add_operation(op);
    }

    fn remove_op(&mut self, idx: usize) {
        if idx < self.stack.len() {
            self.push_undo();
            self.stack.remove_operation(idx);
        }
    }

    /// Replace the key on a Dense/HrrBind op, or rotate Permute shift.
    fn reseed_op(&mut self, idx: usize) {
        if idx >= self.stack.len() {
            return;
        }
        self.push_undo();
        self.recent_change = Some((idx, std::time::Instant::now()));
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
            "permute" => Operation::Permute {
                shift: ((new_seed as usize).wrapping_mul(13)) % self.dim.max(1),
            },
            _ => return,
        };
        self.stack.replace_operation(idx, new_op);
        self.set_status(format!("reseeded op [{idx}] {tag} → seed={new_seed}"));
    }

    fn arch_summary(&self) -> ArchSummary {
        self.stack.arch_summary()
    }

    /// Compute the appropriate adaptive hint based on the user's current
    /// idle time + state. Returns None if no hint applies.
    fn compute_adaptive_hint(&self) -> Option<&'static str> {
        let idle = self.last_user_action.elapsed().as_secs();
        if idle < 30 {
            return None; // user is active
        }
        // Most-specific cases first.
        if self.stack.is_empty() {
            return Some("Stack is empty — press '+ New…' or any number 1-9 to load a template");
        }
        if self.forwards == 0 {
            return Some("Press SPACE (or click ▶ Run forward) to execute the stack");
        }
        if !self.show_help && self.forwards < 3 {
            return Some("Stuck? Press H for help, or backtick (\\`) for the console");
        }
        if !self.live && self.forwards >= 10 {
            return Some("Try LIVE mode — press L to run forward continuously");
        }
        if self.op_kinds_seen.len() < 5 && self.forwards >= 5 {
            return Some("You haven't tried every op kind yet — press A/D/F/P/N to add one");
        }
        if self.templates_seen.len() < TEMPLATES.len() && self.forwards >= 20 {
            return Some("Try another template — press '+ New…' or 1-0 to browse");
        }
        None
    }

    fn run_forward(&mut self) {
        if self.stack.len() == 0 {
            self.log(
                LogSeverity::Warn,
                "forward skipped — stack has zero ops".to_string(),
            );
            return;
        }
        let started = std::time::Instant::now();
        match self.stack.forward_with_trace(&self.input) {
            Err(e) => {
                self.log(LogSeverity::Error, format!("forward error: {e}"));
                return;
            }
            Ok(_) => {}
        }
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
            self.last_forward_at = Some(std::time::Instant::now());
            self.check_achievements();
            self.play_forward_tone(sim);
        }
    }

    /// Parse + execute a REPL command. Returns user-visible output.
    fn run_console_cmd(&mut self, cmd: &str) -> String {
        let cmd = cmd.trim();
        if cmd.is_empty() {
            return String::new();
        }
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts.as_slice() {
            ["help"] | ["?"] => {
                "Commands:\n  fwd / forward          run a forward\n  \
                 live                   toggle live mode\n  \
                 add <kind>             add op (identity/dense/hrr_bind/permute/negate)\n  \
                 rm <idx> | remove <idx>\n  \
                 reseed <idx>\n  \
                 dim <n>                set dim (256..16384)\n  \
                 template <name>        load template\n  \
                 regen                  regenerate input\n  \
                 reset                  clear stack\n  \
                 undo / redo\n  \
                 save / load\n  \
                 png                    export PNG\n  \
                 stat                   print stack summary\n  \
                 clear                  clear console history".to_string()
            }
            ["fwd"] | ["forward"] => {
                self.run_forward();
                format!("ran forward (#{} · {:.3} ms)",
                    self.forwards,
                    self.last_latency_ms.unwrap_or(0.0))
            }
            ["live"] => {
                self.live = !self.live;
                format!("live = {}", self.live)
            }
            ["add", kind] => {
                let kind_s = (*kind).to_string();
                self.add_op(&kind_s);
                format!("added {kind}")
            }
            ["rm", idx_s] | ["remove", idx_s] => {
                match idx_s.parse::<usize>() {
                    Ok(i) => {
                        self.remove_op(i);
                        format!("removed [{i}]")
                    }
                    Err(_) => format!("invalid index: {idx_s}"),
                }
            }
            ["reseed", idx_s] => match idx_s.parse::<usize>() {
                Ok(i) => {
                    self.reseed_op(i);
                    format!("reseeded [{i}]")
                }
                Err(_) => format!("invalid index: {idx_s}"),
            },
            ["dim", n_s] => match n_s.parse::<usize>() {
                Ok(n) => {
                    self.set_dim(n.clamp(256, 16_384));
                    format!("dim = {}", self.dim)
                }
                Err(_) => format!("invalid number: {n_s}"),
            },
            ["template", name] => {
                let name_owned = (*name).to_string();
                if let Some(t) = TEMPLATES.iter().find(|t| t.name == name_owned) {
                    self.load_template(t.name);
                    format!("loaded template '{}'", t.name)
                } else {
                    format!(
                        "unknown template '{name}'. Try one of: {}",
                        TEMPLATES.iter().map(|t| t.name).collect::<Vec<_>>().join(", ")
                    )
                }
            }
            ["regen"] => {
                self.regenerate_input();
                format!("regenerated input (seed = {})", self.input_seed)
            }
            ["reset"] => {
                self.reset_stack();
                "stack cleared".to_string()
            }
            ["undo"] => {
                self.undo();
                "undo".to_string()
            }
            ["redo"] => {
                self.redo();
                "redo".to_string()
            }
            ["save"] => {
                self.save_yaml();
                "save dialog".to_string()
            }
            ["load"] => {
                self.load_yaml();
                "load dialog".to_string()
            }
            ["png"] => {
                self.export_png();
                "png export".to_string()
            }
            ["stat"] => {
                let ops: Vec<&str> = self.stack.operations().iter().map(|op| op.tag()).collect();
                format!(
                    "template={} · dim={} · ops={} · forwards={} · cos_sim={:?}",
                    self.template,
                    self.dim,
                    ops.join(","),
                    self.forwards,
                    self.last_cos_sim
                )
            }
            ["clear"] => {
                self.console_history.clear();
                "console cleared".to_string()
            }
            _ => format!("unknown command: '{cmd}' (try 'help')"),
        }
    }

    /// Toggle audio output (#723). Stubbed until libasound2-dev is on the
    /// build host — when reinstated, lazily initialize the rodio output
    /// stream here.
    fn toggle_audio(&mut self) {
        self.audio_enabled = !self.audio_enabled;
        if self.audio_enabled && self.audio.is_none() {
            self.audio = Some(AudioState { _placeholder: () });
            self.log(
                LogSeverity::Warn,
                "audio: enabled (stub — needs libasound2-dev for real sound)".to_string(),
            );
        } else if self.audio_enabled {
            self.log(LogSeverity::Info, "audio: enabled (stub)".to_string());
        } else {
            self.log(LogSeverity::Info, "audio: muted".to_string());
        }
    }

    /// Stub: when rodio is reinstated, play a short tone here.
    fn play_forward_tone(&self, _sim: Option<f64>) {
        // Intentionally silent until libasound2-dev is available on the
        // build host. self.audio_enabled gates the future call site.
    }

    /// Run one training step against the current target (#757).
    fn train_step(&mut self) {
        let Some(target) = self.train.target.clone() else {
            self.log(
                LogSeverity::Warn,
                "no target set — click 'set current output as target' first".to_string(),
            );
            return;
        };
        if self.stack.len() == 0 {
            self.log(LogSeverity::Warn, "empty stack — add ops first".to_string());
            return;
        }
        // Find a keyed op to perturb (Dense, HrrBind, Permute).
        let keyed_indices: Vec<usize> = self
            .stack
            .operations()
            .iter()
            .enumerate()
            .filter_map(|(i, op)| {
                matches!(op.tag(), "dense" | "hrr_bind" | "permute").then_some(i)
            })
            .collect();
        if keyed_indices.is_empty() {
            self.log(LogSeverity::Warn, "no keyed ops to train".to_string());
            return;
        }
        self.train.rng_seed = self.train.rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rng = self.train.rng_seed;
        let idx_pick = keyed_indices[(rng as usize) % keyed_indices.len()];
        // Compute current loss.
        let cur_out = match self.stack.forward(&self.input) {
            Ok(o) => o,
            Err(_) => return,
        };
        let cur_loss = if cur_out.dim() == target.dim() {
            1.0 - cos_sim(&cur_out, &target).unwrap_or(0.0)
        } else {
            return;
        };
        // Build candidate op with a perturbation.
        let candidate_op = match self.train.mode {
            TrainMode::Random => match self.stack.operations()[idx_pick].tag() {
                "dense" => Operation::Dense {
                    key: Hypervector::random_seeded(self.dim, rng),
                },
                "hrr_bind" => Operation::HrrBind {
                    key: Hypervector::random_seeded(self.dim, rng),
                },
                "permute" => Operation::Permute {
                    shift: (rng as usize) % self.dim.max(1),
                },
                _ => return,
            },
            TrainMode::HillClimb | TrainMode::SimulatedAnneal => {
                match &self.stack.operations()[idx_pick] {
                    Operation::Dense { key } => {
                        let mut new_key: Vec<i8> = key.as_slice().to_vec();
                        let mut r = rng;
                        for _ in 0..self.train.perturb_bits {
                            r = r.wrapping_mul(6364136223846793005).wrapping_add(11);
                            let i = (r as usize) % new_key.len();
                            new_key[i] = -new_key[i];
                        }
                        Operation::Dense {
                            key: Hypervector::from_bipolar(new_key)
                                .unwrap_or_else(|| key.clone()),
                        }
                    }
                    Operation::HrrBind { key } => {
                        let mut new_key: Vec<i8> = key.as_slice().to_vec();
                        let mut r = rng;
                        for _ in 0..self.train.perturb_bits {
                            r = r.wrapping_mul(6364136223846793005).wrapping_add(11);
                            let i = (r as usize) % new_key.len();
                            new_key[i] = -new_key[i];
                        }
                        Operation::HrrBind {
                            key: Hypervector::from_bipolar(new_key)
                                .unwrap_or_else(|| key.clone()),
                        }
                    }
                    Operation::Permute { shift } => Operation::Permute {
                        shift: (shift + (rng as usize % 7) + 1) % self.dim.max(1),
                    },
                    _ => return,
                }
            }
        };
        // Try the candidate.
        let prev_op = self.stack.replace_operation(idx_pick, candidate_op);
        let new_loss = match self.stack.forward(&self.input) {
            Ok(o) if o.dim() == target.dim() => {
                1.0 - cos_sim(&o, &target).unwrap_or(0.0)
            }
            _ => {
                // Revert + bail.
                self.stack.replace_operation(idx_pick, prev_op);
                return;
            }
        };
        let accept = match self.train.mode {
            TrainMode::HillClimb => new_loss < cur_loss,
            TrainMode::Random => true,
            TrainMode::SimulatedAnneal => {
                if new_loss < cur_loss {
                    true
                } else {
                    let delta = new_loss - cur_loss;
                    let temp = (1.0 / (1.0 + self.train.steps as f64 / 100.0)).max(0.01);
                    let prob = (-delta / temp).exp();
                    let r = (rng % 1_000_000) as f64 / 1_000_000.0;
                    r < prob
                }
            }
        };
        if !accept {
            self.stack.replace_operation(idx_pick, prev_op);
        }
        self.train.loss_history.push(if accept { new_loss } else { cur_loss });
        if self.train.loss_history.len() > 2_000 {
            self.train.loss_history.remove(0);
        }
        self.train.steps = self.train.steps.saturating_add(1);
    }

    /// Load an image file → quantize to bipolar hypervector → set as input.
    /// (#758 computer vision support.)
    fn load_image_as_input(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "bmp"])
            .pick_file()
        else {
            self.log(LogSeverity::Warn, "image load cancelled".to_string());
            return;
        };
        let img = match image::open(&path) {
            Ok(i) => i,
            Err(e) => {
                self.log(LogSeverity::Error, format!("image decode failed: {e}"));
                return;
            }
        };
        // Convert to greyscale 8-bit then bipolar at the current dim.
        let grey = img.to_luma8();
        let pixels: Vec<u8> = grey.into_raw();
        if pixels.is_empty() {
            self.log(LogSeverity::Error, "empty image".to_string());
            return;
        }
        // Resample pixel array down to self.dim by stride sampling, then
        // threshold at 128.
        let data: Vec<i8> = (0..self.dim)
            .map(|i| {
                let src = (i as f64) * (pixels.len() as f64) / (self.dim as f64);
                let p = pixels[src as usize];
                if p >= 128 {
                    1i8
                } else {
                    -1
                }
            })
            .collect();
        match Hypervector::from_bipolar(data) {
            Some(hv) => {
                self.input = hv;
                self.loaded_image = Some(path.clone());
                self.log(
                    LogSeverity::Success,
                    format!("image → input: {}", path.display()),
                );
            }
            None => {
                self.log(LogSeverity::Error, "quantize failed".to_string());
            }
        }
    }

    /// Compute a human-readable summary of a hypervector (#753).
    fn hv_summary(v: &Hypervector) -> HvSummary {
        let data = v.as_slice();
        let positive = data.iter().filter(|x| **x > 0).count();
        let n = data.len();
        let percent_pos = if n == 0 { 0.0 } else { positive as f64 / n as f64 };
        // Blake3 fingerprint (truncated to 8 hex chars).
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let hash = blake3::hash(bytes);
        let fp = hash.to_hex().to_string()[..8].to_string();
        // First 24 components as binary string.
        let prefix: String = data
            .iter()
            .take(24)
            .map(|x| if *x > 0 { '1' } else { '0' })
            .collect();
        HvSummary {
            fingerprint: fp,
            percent_positive: percent_pos,
            binary_prefix: prefix,
        }
    }

    fn export_png(&mut self) {
        // Use scrot if available; fall back to import (ImageMagick).
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let path = std::path::PathBuf::from(home).join(format!("graphnet-{ts}.png"));
        let outcome = std::process::Command::new("scrot")
            .args(["-u", "-z"])
            .arg(&path)
            .output()
            .or_else(|_| {
                std::process::Command::new("import")
                    .args(["-window", "GraphNet"])
                    .arg(&path)
                    .output()
            });
        match outcome {
            Ok(out) if out.status.success() => {
                self.log(LogSeverity::Success, format!("📷 saved → {}", path.display()));
            }
            Ok(out) => {
                self.log(
                    LogSeverity::Error,
                    format!(
                        "screenshot failed: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    ),
                );
            }
            Err(e) => {
                self.log(LogSeverity::Error, format!("screenshot tool missing: {e}"));
            }
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
        self.push_undo();
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
        let path = rfd::FileDialog::new()
            .add_filter("YAML", &["yaml", "yml"])
            .set_file_name("graphnet-stack.yaml")
            .set_directory(std::env::current_dir().unwrap_or_else(|_| ".".into()))
            .save_file();
        let Some(path) = path else {
            self.log(LogSeverity::Warn, "save cancelled".to_string());
            return;
        };
        match stack_to_yaml(&self.stack) {
            Ok(yaml) => match std::fs::write(&path, &yaml) {
                Ok(_) => self.log(
                    LogSeverity::Success,
                    format!("saved → {} ({} bytes)", path.display(), yaml.len()),
                ),
                Err(e) => self.log(LogSeverity::Error, format!("save failed: {e}")),
            },
            Err(e) => self.log(LogSeverity::Error, format!("encode failed: {e}")),
        }
        self.persist();
    }

    fn load_yaml(&mut self) {
        let path = rfd::FileDialog::new()
            .add_filter("YAML", &["yaml", "yml"])
            .set_directory(std::env::current_dir().unwrap_or_else(|_| ".".into()))
            .pick_file();
        let Some(path) = path else {
            self.log(LogSeverity::Warn, "load cancelled".to_string());
            return;
        };
        match std::fs::read_to_string(&path) {
            Ok(yaml) => match stack_from_yaml(&yaml) {
                Ok(stack) => {
                    self.dim = stack.dim();
                    self.stack = stack;
                    self.input = Hypervector::random_seeded(self.dim, self.input_seed);
                    self.last_output = None;
                    self.last_trace = None;
                    self.last_latency_ms = None;
                    self.last_cos_sim = None;
                    self.log(LogSeverity::Success, format!("loaded ← {}", path.display()));
                }
                Err(e) => self.log(LogSeverity::Error, format!("decode failed: {e}")),
            },
            Err(e) => self.log(LogSeverity::Error, format!("read failed: {e}")),
        }
    }

    fn set_status(&mut self, msg: String) {
        self.log(LogSeverity::Info, msg);
    }

    fn unlock(&mut self, slug: &'static str, label: &str, icon: &str) {
        if self.achievements.contains_key(slug) {
            return;
        }
        self.achievements.insert(slug, std::time::Instant::now());
        // Celebrate with a success-level log entry.
        let msg = format!("🏆 Achievement unlocked: {icon} {label}");
        // Recursion-safe: write directly to action_log without calling
        // self.log (we set the status from here too).
        self.status_msg = Some((msg.clone(), std::time::Instant::now()));
        self.action_log.push(LogEntry {
            at: std::time::Instant::now(),
            severity: LogSeverity::Success,
            msg,
        });
        if self.action_log.len() > 128 {
            self.action_log.remove(0);
        }
    }

    /// Generate context-aware hints based on current state.
    fn smart_suggestions(&self) -> Vec<String> {
        let mut hints = Vec::new();
        // Count op kinds.
        let mut counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for op in self.stack.operations() {
            *counts.entry(op.tag()).or_insert(0) += 1;
        }
        // Suggest diversity if one kind dominates.
        if let Some((dominant, &count)) = counts.iter().max_by_key(|&(_, c)| *c) {
            if count >= 4 && self.stack.len() >= 4 {
                let missing: Vec<&str> = ["identity", "dense", "hrr_bind", "permute", "negate"]
                    .iter()
                    .copied()
                    .filter(|k| !counts.contains_key(k))
                    .collect();
                if !missing.is_empty() {
                    hints.push(format!(
                        "Stack has {count}× {dominant} — try adding {} for diversity",
                        missing[0]
                    ));
                }
            }
        }
        // High cos_sim ⇒ stack is mostly a pass-through.
        if let Some(s) = self.last_cos_sim {
            if s > 0.85 {
                hints.push(format!(
                    "cos_sim {s:+.2} — output is close to input. \
                     Add hrr_bind or dense to decorrelate."
                ));
            }
        }
        // Latency budget.
        if let Some(ms) = self.last_latency_ms {
            if ms > 16.0 {
                hints.push(format!(
                    "Last forward took {ms:.1} ms (>16 ms = below 60 fps). \
                     Drop dim or remove ops for live mode."
                ));
            }
        }
        // Empty stack.
        if self.stack.is_empty() {
            hints.push(
                "Stack is empty. Pick a template (1-9) or click an + Add button.".to_string(),
            );
        }
        // Suggest a feature not yet used.
        if self.forwards >= 3 && !self.live {
            hints.push("Press L to start live continuous mode — watch the FPS counter.".to_string());
        }
        if self.forwards >= 5 && !self.achievements.contains_key("all_op_kinds") {
            let missing: Vec<&str> = ["identity", "dense", "hrr_bind", "permute", "negate"]
                .iter()
                .copied()
                .filter(|k| !self.op_kinds_seen.contains(*k))
                .collect();
            if !missing.is_empty() {
                hints.push(format!(
                    "You haven't tried op kind {} yet — add one to unlock 🎭 Generalist",
                    missing[0]
                ));
            }
        }
        hints
    }

    /// Get the key hypervector of a Dense/HrrBind op (if any).
    fn op_key(&self, idx: usize) -> Option<Hypervector> {
        let op = self.stack.operations().get(idx)?;
        match op {
            Operation::Dense { key } => Some(key.clone()),
            Operation::HrrBind { key } => Some(key.clone()),
            _ => None,
        }
    }

    /// Check criteria after a forward / mutation and unlock anything new.
    fn check_achievements(&mut self) {
        // Record state into tracking sets.
        for op in self.stack.operations() {
            self.op_kinds_seen.insert(op.tag().to_string());
        }
        self.templates_seen.insert(self.template.to_string());

        if self.forwards >= 1 {
            self.unlock("first_forward", "First Forward", "🚀");
        }
        if self.forwards >= 10 {
            self.unlock("ten_forwards", "Warming Up", "🔥");
        }
        if self.forwards >= 100 {
            self.unlock("hundred_forwards", "Iteration Champion", "💯");
        }
        if self.forwards >= 1000 {
            self.unlock("thousand_forwards", "Live & Loving It", "⚡");
        }
        if self.op_kinds_seen.len() >= 5 {
            self.unlock("all_op_kinds", "Generalist", "🎭");
        }
        if self.templates_seen.len() >= TEMPLATES.len() {
            self.unlock("all_templates", "Template Connoisseur", "📚");
        }
        if self.stack.len() >= 10 {
            self.unlock("ten_op_stack", "Tower Builder", "🏗");
        }
        if self.dim >= 16_000 {
            self.unlock("high_dim", "Hyperdimensional", "🌌");
        }
        if self.dim <= 256 {
            self.unlock("low_dim", "Minimalist", "🌱");
        }
        if self.demo.is_some() {
            self.unlock("demo_runner", "Demo Watcher", "🎬");
        }
        if self.undo_stack.len() >= 5 {
            self.unlock("undoer", "Revisionist", "↶");
        }
        if !self.cos_sim_history.is_empty() {
            let max_sim = self
                .cos_sim_history
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_sim > 0.85 {
                self.unlock("high_sim", "Echo Chamber", "🪞");
            }
        }
        // Advance objective if the current one completes.
        self.check_objectives();
    }

    fn check_objectives(&mut self) {
        // Mark any newly-completed objectives.
        for (i, obj) in OBJECTIVES.iter().enumerate() {
            if !self.objective_done[i] && (obj.check)(self) {
                self.objective_done[i] = true;
                let msg = format!("🎯 Objective complete: {}", obj.title);
                self.status_msg = Some((msg.clone(), std::time::Instant::now()));
                self.action_log.push(LogEntry {
                    at: std::time::Instant::now(),
                    severity: LogSeverity::Success,
                    msg,
                });
                if self.action_log.len() > 128 {
                    self.action_log.remove(0);
                }
            }
        }
        // Advance current_objective to the next incomplete one.
        while self.current_objective < OBJECTIVES.len()
            && self.objective_done[self.current_objective]
        {
            self.current_objective += 1;
        }
    }

    fn log(&mut self, severity: LogSeverity, msg: String) {
        self.status_msg = Some((msg.clone(), std::time::Instant::now()));
        self.action_log.push(LogEntry {
            at: std::time::Instant::now(),
            severity,
            msg,
        });
        if self.action_log.len() > 128 {
            self.action_log.remove(0);
        }
    }

    /// Push the current Stack onto the undo stack before mutating it.
    /// Clears the redo stack (any redo branch is invalidated by a new edit).
    fn push_undo(&mut self) {
        self.undo_stack.push(self.stack.clone());
        if self.undo_stack.len() > 64 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(std::mem::replace(&mut self.stack, prev));
            self.dim = self.stack.dim();
            self.dim_slider = self.dim;
            self.last_output = None;
            self.last_trace = None;
            self.selected_op = None;
            self.set_status(format!("↶ undo (depth {})", self.undo_stack.len()));
        } else {
            self.set_status("↶ nothing to undo".to_string());
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(std::mem::replace(&mut self.stack, next));
            self.dim = self.stack.dim();
            self.dim_slider = self.dim;
            self.last_output = None;
            self.last_trace = None;
            self.selected_op = None;
            self.set_status(format!("↷ redo (depth {})", self.redo_stack.len()));
        } else {
            self.set_status("↷ nothing to redo".to_string());
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Force-maximize on EVERY frame until the WM honors it. Some WMs
        // ignore both with_maximized() AND single-shot Maximized — sending
        // continuously until self.forwards exceeds 3 ensures it sticks.
        if self.forwards <= 3 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
        let mut advance_walkthrough = false;
        let mut undo_request = false;
        let mut redo_request = false;
        let mut png_request = false;
        let any_input = ctx.input(|i| {
            i.keys_down.iter().count() > 0 || i.pointer.any_down() || i.pointer.is_moving()
        });
        if any_input {
            self.last_user_action = std::time::Instant::now();
        }
        self.adaptive_hint = self.compute_adaptive_hint();
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
            let template_keys = [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
                egui::Key::Num5,
                egui::Key::Num6,
                egui::Key::Num7,
                egui::Key::Num8,
                egui::Key::Num9,
                egui::Key::Num0,
            ];
            for (idx, key) in template_keys.iter().enumerate() {
                if i.key_pressed(*key) {
                    if let Some(t) = TEMPLATES.get(idx) {
                        self.load_template(t.name);
                    }
                }
            }
            if (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::S) {
                self.save_yaml();
            }
            if (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::O) {
                self.load_yaml();
            }
            if (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::E) {
                png_request = true;
            }
            if i.key_pressed(egui::Key::Backtick) {
                self.show_console = !self.show_console;
            }
            // Tab toggles left panel; Shift+Tab toggles right panel.
            if i.key_pressed(egui::Key::Tab) {
                if i.modifiers.shift {
                    self.show_right_panel = !self.show_right_panel;
                } else {
                    self.show_left_panel = !self.show_left_panel;
                }
            }
            if (i.modifiers.command || i.modifiers.ctrl)
                && !i.modifiers.shift
                && i.key_pressed(egui::Key::Z)
            {
                undo_request = true;
            }
            if (i.modifiers.command || i.modifiers.ctrl)
                && i.modifiers.shift
                && i.key_pressed(egui::Key::Z)
            {
                redo_request = true;
            }
            if i.key_pressed(egui::Key::H) || i.key_pressed(egui::Key::F1) {
                self.show_help = !self.show_help;
            }
            // More shortcuts (#749).
            if !i.modifiers.command && !i.modifiers.ctrl {
                if i.key_pressed(egui::Key::A) {
                    self.add_op("identity");
                }
                if i.key_pressed(egui::Key::D) {
                    self.add_op("dense");
                }
                if i.key_pressed(egui::Key::F) {
                    self.add_op("hrr_bind"); // F = Fft
                }
                if i.key_pressed(egui::Key::P) {
                    self.add_op("permute");
                }
                if i.key_pressed(egui::Key::N) {
                    self.add_op("negate");
                }
                if i.key_pressed(egui::Key::Backspace)
                    && self.selected_op.is_some()
                {
                    if let Some(idx) = self.selected_op {
                        self.remove_op(idx);
                        self.selected_op = None;
                    }
                }
            }
            if i.key_pressed(egui::Key::Escape) {
                self.show_help = false;
                self.zoom_target = None;
                if self.walkthrough_step.is_some() {
                    self.walkthrough_step = None;
                }
            }
            if i.key_pressed(egui::Key::ArrowRight) && self.walkthrough_step.is_some() {
                advance_walkthrough = true;
            }
        });
        if undo_request {
            self.undo();
        }
        if redo_request {
            self.redo();
        }
        if png_request {
            self.export_png();
        }
        if advance_walkthrough {
            let next = self.walkthrough_step.map(|s| s + 1).unwrap_or(0);
            if next >= WALKTHROUGH_STEPS.len() {
                self.dismiss_walkthrough();
            } else {
                self.walkthrough_step = Some(next);
            }
        }
        // Drag-and-drop YAML load.
        let dropped: Vec<std::path::PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for path in dropped {
            match std::fs::read_to_string(&path) {
                Ok(yaml) => match stack_from_yaml(&yaml) {
                    Ok(stack) => {
                        self.dim = stack.dim();
                        self.stack = stack;
                        self.input = Hypervector::random_seeded(self.dim, self.input_seed);
                        self.last_output = None;
                        self.last_trace = None;
                        self.last_latency_ms = None;
                        self.last_cos_sim = None;
                        self.set_status(format!("dropped ← {}", path.display()));
                    }
                    Err(e) => self.set_status(format!("dropped file: decode failed: {e}")),
                },
                Err(e) => self.set_status(format!("dropped file: read failed: {e}")),
            }
        }

        // Demo tick.
        if self.demo.is_some() {
            self.tick_demo();
            ctx.request_repaint();
        }

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
                let band_h = 56.0;
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), band_h),
                    egui::Sense::hover(),
                );
                // SOLID dark band — text reads cleanly. (Previous gradient hero
                // had poor contrast per owner feedback.)
                let hero_bg = egui::Color32::from_rgb(0x07, 0x09, 0x10);
                ui.painter().rect_filled(rect, 0.0, hero_bg);
                // Thin 3px gradient accent strip at the BOTTOM.
                let strip = egui::Rect::from_min_max(
                    egui::pos2(rect.min.x, rect.max.y - 3.0),
                    rect.max,
                );
                theme::paint_gradient(ui.painter(), strip);

                let inner_rect = rect.shrink2(egui::vec2(theme::SPACE_LG, theme::SPACE_SM));
                let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(inner_rect));
                inner.horizontal_centered(|ui| {
                    // Small gradient pill as brand mark.
                    let (pill_rect, _) =
                        ui.allocate_exact_size(egui::vec2(6.0, 26.0), egui::Sense::hover());
                    theme::paint_gradient(ui.painter(), pill_rect);
                    ui.add_space(theme::SPACE_SM);
                    ui.label(
                        egui::RichText::new("GraphNet")
                            .size(theme::SIZE_H2)
                            .color(egui::Color32::WHITE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_MD);
                    // Workspace tabs (#743).
                    for ws in Workspace::all() {
                        let active = self.workspace == *ws;
                        let resp = ui.add(
                            egui::Button::new(
                                egui::RichText::new(ws.label())
                                    .size(theme::SIZE_SMALL)
                                    .color(if active {
                                        egui::Color32::WHITE
                                    } else {
                                        egui::Color32::from_rgb(0xA0, 0xA8, 0xB8)
                                    })
                                    .strong(),
                            )
                            .fill(if active {
                                theme::ACCENT_MID
                            } else {
                                egui::Color32::from_rgb(0x15, 0x1A, 0x26)
                            })
                            .stroke(if active {
                                egui::Stroke::new(1.0, theme::ACCENT_MID)
                            } else {
                                egui::Stroke::NONE
                            })
                            .rounding(egui::Rounding::same(theme::RADIUS_SM))
                            .min_size(egui::vec2(0.0, 28.0)),
                        );
                        if resp.clicked() {
                            self.workspace = *ws;
                            self.set_status(format!("workspace → {}", ws.label()));
                            // Workspace tabs ACTUALLY change behavior now.
                            match ws {
                                Workspace::Edit => {
                                    self.tool_mode = ToolMode::Edit;
                                    self.live = false;
                                }
                                Workspace::Live => {
                                    self.tool_mode = ToolMode::Edit;
                                    self.live = true; // auto-start live mode
                                    // Maximize 3D real estate by collapsing left panel.
                                    self.show_left_panel = false;
                                }
                                Workspace::Compare => {
                                    self.tool_mode = ToolMode::Compare;
                                    self.live = false;
                                    self.show_left_panel = true;
                                }
                                Workspace::Train => {
                                    self.tool_mode = ToolMode::Edit;
                                    self.live = false;
                                    self.show_left_panel = true;
                                }
                            }
                        }
                        ui.add_space(2.0);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let hero_btn = |ui: &mut egui::Ui,
                                        text: &str,
                                        active: bool|
                         -> egui::Response {
                            let fg = if active {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_rgb(0xD8, 0xDC, 0xE8)
                            };
                            let fill = if active {
                                theme::ACCENT_MID
                            } else {
                                egui::Color32::from_rgb(0x15, 0x1A, 0x26)
                            };
                            let stroke_col = if active {
                                theme::ACCENT_MID
                            } else {
                                egui::Color32::from_rgb(0x32, 0x3A, 0x50)
                            };
                            ui.add(
                                egui::Button::new(
                                    egui::RichText::new(text)
                                        .size(theme::SIZE_SMALL)
                                        .color(fg)
                                        .strong(),
                                )
                                .fill(fill)
                                .stroke(egui::Stroke::new(1.0, stroke_col))
                                .rounding(egui::Rounding::same(theme::RADIUS_PILL))
                                .min_size(egui::vec2(0.0, 24.0)),
                            )
                        };
                        if hero_btn(ui, " ❓ Help ", self.show_help).clicked() {
                            self.show_help = !self.show_help;
                        }
                        ui.add_space(theme::SPACE_XS);
                        // ⌨ Console toggle button — owner asked for visibility.
                        if hero_btn(ui, " ⌨ Console ", self.show_console).clicked() {
                            self.show_console = !self.show_console;
                        }
                        ui.add_space(theme::SPACE_XS);
                        let demo_active = self.demo.is_some();
                        let demo_lbl = if demo_active { " ⏹ Stop demo " } else { " ▶ Demo " };
                        if hero_btn(ui, demo_lbl, demo_active).clicked() {
                            if demo_active {
                                self.stop_demo();
                            } else {
                                self.start_demo();
                            }
                        }
                        if demo_active {
                            // Demo transport controls: ⏮ ⏯ ⏭ + progress.
                            if hero_btn(ui, " ⏮ ", false).clicked() {
                                self.demo_skip(-1);
                            }
                            let paused = self.demo.as_ref().is_some_and(|d| d.paused);
                            let pp_lbl = if paused { " ▶ " } else { " ⏸ " };
                            if hero_btn(ui, pp_lbl, false).clicked() {
                                self.demo_pause_toggle();
                            }
                            if hero_btn(ui, " ⏭ ", false).clicked() {
                                self.demo_skip(1);
                            }
                            // Progress text.
                            if let Some(d) = &self.demo {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{}/{}",
                                        d.template_idx + 1,
                                        TEMPLATES.len()
                                    ))
                                    .size(theme::SIZE_TINY)
                                    .color(theme::TEXT_MUTED)
                                    .monospace(),
                                );
                            }
                        }
                        ui.add_space(theme::SPACE_XS);
                        let theme_lbl = match self.mode {
                            theme::Mode::Dark => " ☀ Light ",
                            theme::Mode::Light => " ☾ Dark ",
                        };
                        if hero_btn(ui, theme_lbl, false).clicked() {
                            self.toggle_mode(ctx);
                        }
                    });
                });
            });

        // Sample CPU / RAM every 750ms (sysinfo is moderately expensive).
        if self.last_sample_at.elapsed().as_millis() > 750 {
            self.last_sample = Some(self.resource_monitor.sample());
            self.last_sample_at = std::time::Instant::now();
        }

        // Adaptive hint banner (#719) — only when stuck.
        if let Some(hint) = self.adaptive_hint {
            egui::TopBottomPanel::top("adaptive_hint")
                .frame(
                    egui::Frame::none()
                        .fill(theme::ACCENT_PURPLE.gamma_multiply(0.25))
                        .inner_margin(egui::Margin::symmetric(theme::SPACE_LG, theme::SPACE_SM))
                        .stroke(egui::Stroke::new(1.0, theme::ACCENT_PURPLE)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("💡")
                                .size(theme::SIZE_BODY)
                                .color(theme::ACCENT_PURPLE),
                        );
                        ui.label(
                            egui::RichText::new(hint)
                                .size(theme::SIZE_SMALL)
                                .color(theme::TEXT_PRIMARY),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("×").clicked() {
                                    self.last_user_action = std::time::Instant::now();
                                }
                            },
                        );
                    });
                });
        }

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let flops = flop_estimate(self.dim, self.stack.len());
                let (cpu_txt, ram_txt) = if let Some(s) = &self.last_sample {
                    (
                        format!("{:.0}%", s.host_cpu_load * 100.0),
                        format!("{} MB", s.proc_ram_used / 1024 / 1024),
                    )
                } else {
                    ("—".to_string(), "—".to_string())
                };
                // Workspace badge (left-most).
                ui.label(
                    egui::RichText::new(self.workspace.label())
                        .size(theme::SIZE_SMALL)
                        .color(theme::ACCENT_BLUE)
                        .strong(),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new(format!(
                        "{}  ·  dim {}  ·  ops {}  ·  flops {flops:.1e}",
                        self.template,
                        self.dim,
                        self.stack.len()
                    ))
                    .size(theme::SIZE_SMALL)
                    .color(theme::TEXT_MUTED),
                );
                if let Some(idx) = self.selected_op {
                    if let Some(op) = self.stack.operations().get(idx) {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("✎ [{}] {}", idx, op.tag()))
                                .size(theme::SIZE_SMALL)
                                .color(theme::op_color(op.tag()))
                                .strong(),
                        );
                    }
                }
                if self.workspace == Workspace::Train && self.train.steps > 0 {
                    ui.separator();
                    let loss = self
                        .train
                        .loss_history
                        .last()
                        .map(|l| format!("loss {l:.4}"))
                        .unwrap_or_default();
                    ui.label(
                        egui::RichText::new(format!("🎓 step {} · {}", self.train.steps, loss))
                            .size(theme::SIZE_SMALL)
                            .color(theme::ACCENT_PURPLE)
                            .strong(),
                    );
                }
                ui.add_space(theme::SPACE_MD);
                ui.label(
                    egui::RichText::new(format!("CPU {cpu_txt}  ·  RAM {ram_txt}"))
                        .size(theme::SIZE_SMALL)
                        .color(theme::ACCENT_BLUE)
                        .monospace(),
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

        // Blender/FreeCAD-style tool palette: icon-only column on the far
        // left for mode switching.
        egui::SidePanel::left("tool_palette")
            .exact_width(44.0)
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_CARD)
                    .inner_margin(egui::Margin::symmetric(theme::SPACE_XS, theme::SPACE_SM)),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    let modes = [
                        ToolMode::Templates,
                        ToolMode::Edit,
                        ToolMode::Inspect,
                        ToolMode::Compare,
                        ToolMode::Settings,
                        ToolMode::Help,
                    ];
                    for mode in modes {
                        let active = self.tool_mode == mode;
                        let resp = ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(mode.icon())
                                        .size(18.0)
                                        .color(if active {
                                            egui::Color32::WHITE
                                        } else {
                                            theme::TEXT_MUTED
                                        }),
                                )
                                .fill(if active {
                                    theme::ACCENT_MID
                                } else {
                                    egui::Color32::TRANSPARENT
                                })
                                .stroke(if active {
                                    egui::Stroke::new(1.0, theme::ACCENT_MID)
                                } else {
                                    egui::Stroke::NONE
                                })
                                .rounding(egui::Rounding::same(theme::RADIUS_SM))
                                .min_size(egui::vec2(32.0, 32.0)),
                            )
                            .on_hover_text(mode.label());
                        if resp.clicked() {
                            // Templates + Help are ACTIONS, not panel modes —
                            // they trigger popups and snap the panel back to
                            // Edit so the user has a useful view to work in.
                            match mode {
                                ToolMode::Templates => {
                                    self.show_templates_popup = true;
                                    self.tool_mode = ToolMode::Edit;
                                }
                                ToolMode::Help => {
                                    self.show_help = true;
                                    self.tool_mode = ToolMode::Edit;
                                }
                                _ => {
                                    self.tool_mode = mode;
                                }
                            }
                        }
                        ui.add_space(theme::SPACE_XS);
                    }
                });
            });

        if self.show_left_panel {
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
                // IDE-style header bar with title + close × button.
                ui.horizontal(|ui| {
                    let mode_label = self.tool_mode.label();
                    ui.label(
                        egui::RichText::new(mode_label)
                            .size(theme::SIZE_TINY)
                            .color(theme::ACCENT_PURPLE)
                            .strong(),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .small_button("×")
                                .on_hover_text("close panel (Tab to re-open)")
                                .clicked()
                            {
                                self.show_left_panel = false;
                            }
                        },
                    );
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .max_height(f32::INFINITY)
                    .show(ui, |ui| {
                // Mode-aware left panel content (label moved to header above).
                let _mode_label = self.tool_mode.label();
                ui.add_space(theme::SPACE_SM);
                if self.tool_mode == ToolMode::Settings {
                    section_heading(ui, "Behaviour");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        // Audio toggle.
                        let label = if self.audio_enabled {
                            "🔊 audio: on"
                        } else {
                            "🔇 audio: off"
                        };
                        if ui.button(label).clicked() {
                            self.toggle_audio();
                        }
                        ui.add_space(theme::SPACE_SM);
                        ui.label(
                            egui::RichText::new("input seed")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        let mut seed_val = self.input_seed;
                        let seed_resp = ui.add(
                            egui::DragValue::new(&mut seed_val)
                                .speed(1)
                                .range(0..=u64::MAX),
                        );
                        if seed_resp.changed() {
                            self.input_seed = seed_val;
                            self.input =
                                Hypervector::random_seeded(self.dim, self.input_seed);
                        }
                        ui.add_space(theme::SPACE_SM);
                        ui.label(
                            egui::RichText::new("demo pace (seconds per template)")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.demo_pace_sec, 0.5..=10.0)
                                .step_by(0.5)
                                .suffix(" s"),
                        );
                    });

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Appearance");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        // Font scale slider.
                        ui.label(
                            egui::RichText::new("font scale")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        let mut scale = self.font_scale;
                        let scale_resp = ui.add(
                            egui::Slider::new(&mut scale, 0.75..=1.5)
                                .step_by(0.05)
                                .text("×"),
                        );
                        if scale_resp.drag_stopped() || scale_resp.lost_focus() {
                            self.font_scale = scale;
                            // Re-install the theme — would need theme.rs hook;
                            // for now log + persist for next launch.
                            self.set_status(format!("font scale → {:.2}× (restart to apply)", scale));
                        }
                        ui.add_space(theme::SPACE_SM);
                        let mode_label = if self.mode == theme::Mode::Dark { "Dark" } else { "Light" };
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("theme")
                                    .color(theme::TEXT_MUTED)
                                    .size(theme::SIZE_SMALL),
                            );
                            if ui.button(mode_label).clicked() {
                                self.toggle_mode(ctx);
                            }
                        });
                        ui.checkbox(&mut self.arch_autorotate, "arch graph auto-rotate");
                        ui.checkbox(&mut self.live, "live continuous mode");
                        ui.checkbox(&mut self.show_floating_stats, "📊 floating stats window");
                        ui.checkbox(
                            &mut self.show_floating_minihelp,
                            "⌨ floating shortcuts window",
                        );
                        ui.add_space(theme::SPACE_SM);
                        ui.label(
                            egui::RichText::new("heatmap colormap")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        ui.horizontal_wrapped(|ui| {
                            for &cmap in Colormap::all() {
                                let active = self.colormap == cmap;
                                let resp = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(cmap.label())
                                            .size(theme::SIZE_SMALL)
                                            .color(if active {
                                                egui::Color32::WHITE
                                            } else {
                                                theme::TEXT_PRIMARY
                                            })
                                            .strong(),
                                    )
                                    .fill(if active {
                                        theme::ACCENT_MID
                                    } else {
                                        theme::BG_CARD_HOVER
                                    })
                                    .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                                    .rounding(egui::Rounding::same(theme::RADIUS_SM)),
                                );
                                if resp.clicked() {
                                    self.colormap = cmap;
                                    self.set_status(format!("colormap → {}", cmap.label()));
                                }
                            }
                        });
                    });
                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "Dim");
                    ui.add_space(theme::SPACE_XS);
                    let mut slider_value = self.dim_slider;
                    let slider_resp = ui.add(
                        egui::Slider::new(&mut slider_value, 256..=16_384)
                            .logarithmic(true)
                            .text("D"),
                    );
                    self.dim_slider = slider_value;
                    if slider_resp.drag_stopped() || slider_resp.lost_focus() {
                        self.set_dim(self.dim_slider);
                    }
                    return;
                }
                if self.tool_mode == ToolMode::Compare {
                    // Multi-stack slots A/B/C/D (#722).
                    section_heading(ui, "Slots (A/B/C/D)");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "Save the current stack into one of 4 slots. Recall a slot \
                                 to make it the active stack.",
                            )
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_PRIMARY),
                        );
                        ui.add_space(theme::SPACE_SM);
                        let mut to_save: Option<usize> = None;
                        let mut to_recall: Option<usize> = None;
                        let mut to_clear: Option<usize> = None;
                        let slots_snapshot: Vec<Option<(usize, usize)>> = self
                            .slots
                            .iter()
                            .map(|s| s.as_ref().map(|st| (st.len(), st.dim())))
                            .collect();
                        let active = self.active_slot;
                        for (i, slot_info) in slots_snapshot.iter().enumerate() {
                            let slot: &Option<(usize, usize)> = slot_info;
                            let letter = (b'A' + i as u8) as char;
                            let is_active = active == i;
                            let mut active_clicked = false;
                            ui.horizontal(|ui| {
                                let label = if is_active {
                                    format!("● {letter}")
                                } else {
                                    format!("○ {letter}")
                                };
                                let resp = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(label)
                                            .size(theme::SIZE_SMALL)
                                            .color(if is_active {
                                                egui::Color32::WHITE
                                            } else {
                                                theme::TEXT_PRIMARY
                                            })
                                            .strong(),
                                    )
                                    .fill(if is_active {
                                        theme::ACCENT_MID
                                    } else {
                                        theme::BG_CARD_HOVER
                                    })
                                    .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                                    .rounding(egui::Rounding::same(theme::RADIUS_SM))
                                    .min_size(egui::vec2(32.0, 26.0)),
                                );
                                if resp.clicked() {
                                    active_clicked = true;
                                }
                                if mini_button(ui, "save", theme::ACCENT_BLUE).clicked() {
                                    to_save = Some(i);
                                }
                                let has = slot.is_some();
                                ui.add_enabled_ui(has, |ui| {
                                    if mini_button(ui, "recall", theme::ACCENT_PURPLE).clicked() {
                                        to_recall = Some(i);
                                    }
                                    if mini_button(ui, "✕", theme::TEXT_MUTED).clicked() {
                                        to_clear = Some(i);
                                    }
                                });
                                if let Some((ops, dim)) = slot {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "({ops} ops · D={dim})"
                                        ))
                                        .size(theme::SIZE_TINY)
                                        .color(theme::TEXT_MUTED),
                                    );
                                }
                            });
                            if active_clicked {
                                to_recall = Some(i);
                            }
                        }
                        if let Some(i) = to_save {
                            self.slots[i] = Some(self.stack.clone());
                            self.set_status(format!(
                                "saved current stack to slot {}",
                                (b'A' + i as u8) as char
                            ));
                        }
                        if let Some(i) = to_recall {
                            if let Some(s) = self.slots[i].clone() {
                                self.push_undo();
                                self.dim = s.dim();
                                self.stack = s;
                                self.dim_slider = self.dim;
                                self.active_slot = i;
                                self.set_status(format!(
                                    "recalled slot {} as active stack",
                                    (b'A' + i as u8) as char
                                ));
                            }
                        }
                        if let Some(i) = to_clear {
                            self.slots[i] = None;
                            self.set_status(format!(
                                "cleared slot {}",
                                (b'A' + i as u8) as char
                            ));
                        }
                    });

                    // Stack composition (#746): chain slots A → B → C → D.
                    let occupied: Vec<usize> = self
                        .slots
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| if s.is_some() { Some(i) } else { None })
                        .collect();
                    if occupied.len() >= 2 {
                        ui.add_space(theme::SPACE_MD);
                        section_heading(ui, "Composition (chain)");
                        ui.add_space(theme::SPACE_SM);
                        card(ui, |ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Chain: input → {} → output",
                                    occupied
                                        .iter()
                                        .map(|i| (b'A' + *i as u8) as char)
                                        .collect::<String>()
                                        .chars()
                                        .map(|c| format!("[{c}]"))
                                        .collect::<Vec<_>>()
                                        .join(" → ")
                                ))
                                .size(theme::SIZE_SMALL)
                                .color(theme::ACCENT_PURPLE)
                                .strong(),
                            );
                            // Compute chained output: feed input → slot[0] → slot[1] → …
                            let mut current = self.input.clone();
                            let mut last_dim = self.dim;
                            let mut chain_ok = true;
                            for &i in &occupied {
                                if let Some(s) = &self.slots[i] {
                                    if s.dim() != current.dim() {
                                        chain_ok = false;
                                        break;
                                    }
                                    match s.forward(&current) {
                                        Ok(out) => {
                                            current = out;
                                            last_dim = s.dim();
                                        }
                                        Err(_) => {
                                            chain_ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if chain_ok {
                                let final_sim =
                                    cos_sim(&self.input, &current).unwrap_or(0.0);
                                ui.add_space(theme::SPACE_SM);
                                metric(ui, "chain length", &occupied.len().to_string());
                                metric(ui, "output dim", &last_dim.to_string());
                                metric(
                                    ui,
                                    "cos_sim(input, chained)",
                                    &format!("{final_sim:+.3}"),
                                );
                                ui.add_space(theme::SPACE_SM);
                                cosine_similarity_bar(ui, final_sim);
                            } else {
                                ui.label(
                                    egui::RichText::new(
                                        "(slot dims don't align — can't chain)",
                                    )
                                    .color(theme::TEXT_DIM)
                                    .italics()
                                    .size(theme::SIZE_SMALL),
                                );
                            }
                        });
                    }

                    ui.add_space(theme::SPACE_MD);
                    section_heading(ui, "A/B compare");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "Snapshot the current stack as 'A', then mutate the live \
                                 stack to create 'B'. Run a forward to compare outputs.",
                            )
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_PRIMARY),
                        );
                        ui.add_space(theme::SPACE_SM);
                        ui.horizontal(|ui| {
                            if mini_button(ui, "📸 Snapshot as A", theme::ACCENT_BLUE).clicked() {
                                self.snapshot_stack = Some(self.stack.clone());
                                self.set_status("snapshotted current stack as A".to_string());
                            }
                            if self.snapshot_stack.is_some() {
                                if mini_button(ui, "↺ Restore A", theme::ACCENT_PURPLE).clicked() {
                                    if let Some(a) = self.snapshot_stack.clone() {
                                        self.push_undo();
                                        self.stack = a;
                                        self.dim = self.stack.dim();
                                        self.dim_slider = self.dim;
                                        self.set_status("restored stack from snapshot A".to_string());
                                    }
                                }
                                if mini_button(ui, "✕ Clear A", theme::TEXT_MUTED).clicked() {
                                    self.snapshot_stack = None;
                                    self.set_status("cleared snapshot A".to_string());
                                }
                            }
                        });
                    });
                    if let Some(a) = &self.snapshot_stack {
                        ui.add_space(theme::SPACE_MD);
                        section_heading(ui, "A vs B");
                        ui.add_space(theme::SPACE_SM);
                        // Run forwards on both and compute cos_sim.
                        let a_out = a.forward(&self.input).ok();
                        let b_out = self.stack.forward(&self.input).ok();
                        card(ui, |ui| {
                            metric(ui, "A ops", &a.len().to_string());
                            metric(ui, "B ops", &self.stack.len().to_string());
                            metric(ui, "A dim", &a.dim().to_string());
                            metric(ui, "B dim", &self.stack.dim().to_string());
                            if let (Some(a_o), Some(b_o)) = (&a_out, &b_out) {
                                if a_o.dim() == b_o.dim() {
                                    let sim = cos_sim(a_o, b_o).unwrap_or(0.0);
                                    ui.add_space(theme::SPACE_SM);
                                    cosine_similarity_bar(ui, sim);
                                } else {
                                    ui.label(
                                        egui::RichText::new(
                                            "(dim mismatch — can't compute cos_sim)",
                                        )
                                        .color(theme::TEXT_DIM)
                                        .italics(),
                                    );
                                }
                            }
                        });
                    }
                    return;
                }
                // Train workspace content (#757).
                if self.workspace == Workspace::Train {
                    section_heading(ui, "🎓 Training");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        // Mode picker.
                        ui.label(
                            egui::RichText::new("mode")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        ui.horizontal_wrapped(|ui| {
                            for &m in TrainMode::all() {
                                let active = self.train.mode == m;
                                let resp = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(m.label())
                                            .size(theme::SIZE_SMALL)
                                            .color(if active {
                                                egui::Color32::WHITE
                                            } else {
                                                theme::TEXT_PRIMARY
                                            })
                                            .strong(),
                                    )
                                    .fill(if active {
                                        theme::ACCENT_MID
                                    } else {
                                        theme::BG_CARD_HOVER
                                    })
                                    .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                                    .rounding(egui::Rounding::same(theme::RADIUS_SM)),
                                );
                                if resp.clicked() {
                                    self.train.mode = m;
                                    self.set_status(format!("training mode → {}", m.label()));
                                }
                            }
                        });
                        ui.add_space(theme::SPACE_SM);
                        // Target setup.
                        ui.horizontal(|ui| {
                            if mini_button(ui, "🎯 set output as target", theme::ACCENT_BLUE)
                                .on_hover_text("freeze the current output as the target the stack should learn to produce")
                                .clicked()
                            {
                                if let Some(out) = &self.last_output {
                                    self.train.target = Some(out.clone());
                                    self.train.loss_history.clear();
                                    self.train.steps = 0;
                                    self.set_status("target set to current output".to_string());
                                } else {
                                    self.log(
                                        LogSeverity::Warn,
                                        "no output yet — run a forward first".to_string(),
                                    );
                                }
                            }
                            if mini_button(ui, "✕ clear target", theme::TEXT_MUTED).clicked() {
                                self.train.target = None;
                                self.train.loss_history.clear();
                                self.train.steps = 0;
                            }
                        });
                        ui.add_space(theme::SPACE_SM);
                        // Perturb bits slider.
                        ui.label(
                            egui::RichText::new("bits to flip per step")
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.train.perturb_bits, 1..=500)
                                .logarithmic(true)
                                .text("bits"),
                        );
                    });

                    ui.add_space(theme::SPACE_MD);
                    // Step controls.
                    ui.horizontal(|ui| {
                        if mini_button(ui, "⏵ step", theme::ACCENT_MID).clicked() {
                            self.train_step();
                        }
                        if mini_button(ui, "⏩ 10 steps", theme::ACCENT_BLUE).clicked() {
                            for _ in 0..10 {
                                self.train_step();
                            }
                        }
                        if mini_button(ui, "⏭ 100 steps", theme::ACCENT_PURPLE).clicked() {
                            for _ in 0..100 {
                                self.train_step();
                            }
                        }
                    });

                    if !self.train.loss_history.is_empty() {
                        ui.add_space(theme::SPACE_MD);
                        section_heading(ui, "📉 loss curve");
                        ui.add_space(theme::SPACE_SM);
                        card(ui, |ui| {
                            metric(ui, "steps", &self.train.steps.to_string());
                            if let Some(last) = self.train.loss_history.last() {
                                metric(ui, "current loss", &format!("{last:.4}"));
                            }
                            if let Some(first) = self.train.loss_history.first() {
                                metric(ui, "initial loss", &format!("{first:.4}"));
                            }
                            ui.add_space(theme::SPACE_SM);
                            loss_sparkline(
                                ui,
                                self.train.loss_history.iter().copied(),
                                100.0,
                            );
                        });
                    }
                    return;
                }

                if self.tool_mode == ToolMode::Inspect {
                    section_heading(ui, "Inspect");
                    ui.add_space(theme::SPACE_SM);
                    card(ui, |ui| {
                        if let Some(idx) = self.selected_op {
                            if let Some(op) = self.stack.operations().get(idx) {
                                ui.label(
                                    egui::RichText::new(format!("Op [{idx}] · {}", op.tag()))
                                        .size(theme::SIZE_BODY)
                                        .color(theme::op_color(op.tag()))
                                        .strong(),
                                );
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "Click an op in the 3D graph to inspect it here.",
                                )
                                .color(theme::TEXT_MUTED)
                                .italics()
                                .size(theme::SIZE_SMALL),
                            );
                        }
                    });
                    return;
                }
                if self.tool_mode == ToolMode::Help {
                    ui.label(
                        egui::RichText::new("Help overlay is open. Press H or Esc to close.")
                            .color(theme::TEXT_MUTED)
                            .size(theme::SIZE_SMALL),
                    );
                    return;
                }
                ui.horizontal(|ui| {
                    section_heading(ui, "Templates");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if mini_button(ui, "+ New…", theme::ACCENT_MID).clicked() {
                            self.show_templates_popup = true;
                        }
                    });
                });
                ui.add_space(theme::SPACE_SM);
                // Compact "current" indicator instead of full list.
                card(ui, |ui| {
                    ui.label(
                        egui::RichText::new(format!("● {}", self.template))
                            .size(theme::SIZE_BODY)
                            .color(theme::ACCENT_BLUE)
                            .strong(),
                    );
                    let summary = TEMPLATES
                        .iter()
                        .find(|t| t.name == self.template)
                        .map(|t| t.summary)
                        .unwrap_or("custom");
                    ui.label(
                        egui::RichText::new(summary)
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_MUTED),
                    );
                });
                // Compact-list also kept (for keyboard 1-0 + clarity), but
                // shrunk to one row with chevron expand.
                ui.add_space(theme::SPACE_XS);
                // Template search filter (#717) — only when explicitly typed.
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🔍")
                            .color(theme::TEXT_MUTED)
                            .size(theme::SIZE_SMALL),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.template_filter)
                            .hint_text("filter… (or click + New)")
                            .desired_width(ui.available_width()),
                    );
                });
                ui.add_space(theme::SPACE_XS);
                let filter = self.template_filter.to_lowercase();
                let filtered: Vec<(usize, &Template)> = TEMPLATES
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| {
                        !filter.is_empty()
                            && (t.name.to_lowercase().contains(&filter)
                                || t.summary.to_lowercase().contains(&filter)
                                || t.explanation.to_lowercase().contains(&filter))
                    })
                    .collect();
                for (i, template) in filtered.iter().map(|(i, t)| (*i, *t)) {
                    let active = template.name == self.template;
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
                    let resp = ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(format!(
                                    "{}  ·  {}\n{}",
                                    i + 1,
                                    template.name,
                                    template.summary
                                ))
                                .size(theme::SIZE_SMALL)
                                .color(theme::TEXT_PRIMARY),
                            )
                            .fill(chip_fill)
                            .stroke(egui::Stroke::new(1.0, chip_stroke))
                            .rounding(egui::Rounding::same(theme::RADIUS_MD))
                            .min_size(egui::vec2(ui.available_width(), 36.0)),
                        )
                        .on_hover_text(template.explanation);
                    if resp.clicked() {
                        self.load_template(template.name);
                    }
                    ui.add_space(theme::SPACE_XS);
                }

                ui.add_space(theme::SPACE_MD);
                ui.horizontal(|ui| {
                    if mini_button(ui, "💾  Save", theme::ACCENT_MID).clicked() {
                        self.save_yaml();
                    }
                    if mini_button(ui, "📂  Load", theme::ACCENT_BLUE).clicked() {
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
                // DragValue allows BOTH drag AND typing the literal number.
                let mut typed_value = self.dim_slider;
                ui.horizontal(|ui| {
                    let drag_resp = ui.add(
                        egui::DragValue::new(&mut typed_value)
                            .range(256..=16_384)
                            .speed(50.0)
                            .suffix(" dims"),
                    );
                    if drag_resp.drag_stopped() || drag_resp.lost_focus() {
                        self.set_dim(typed_value);
                    }
                });
                self.dim_slider = typed_value;
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
                    egui::RichText::new("(drag, scroll, or type — release to apply)")
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
                let mut to_duplicate: Option<usize> = None;
                let mut to_move_up: Option<usize> = None;
                let mut to_move_down: Option<usize> = None;
                let mut to_convert: Option<(usize, &'static str)> = None;
                let mut drag_drop_target: Option<usize> = None;
                let drag_active = self.drag_source.is_some();
                let recent_change = self.recent_change;
                for (idx, tag) in &ops {
                    let key_preview = self.op_key(*idx);
                    // Diff halo: pulse for 1.0s on the most-recently-mutated op.
                    let halo = recent_change.and_then(|(rc_idx, t)| {
                        if rc_idx == *idx {
                            let age = t.elapsed().as_secs_f32();
                            if age < 1.0 {
                                Some(1.0 - age)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    if halo.is_some() {
                        ctx.request_repaint();
                    }
                    let action = op_chip_actions_with_drag(
                        ui,
                        *idx,
                        tag,
                        self.drag_source == Some(*idx),
                        drag_active,
                        key_preview.as_ref(),
                        halo,
                    );
                    if action.remove {
                        to_remove = Some(*idx);
                    }
                    if action.reseed {
                        to_reseed = Some(*idx);
                    }
                    if action.duplicate {
                        to_duplicate = Some(*idx);
                    }
                    if action.move_up {
                        to_move_up = Some(*idx);
                    }
                    if action.move_down {
                        to_move_down = Some(*idx);
                    }
                    if let Some(k) = action.convert_to {
                        to_convert = Some((*idx, k));
                    }
                    if action.start_drag {
                        self.drag_source = Some(*idx);
                    }
                    if action.drop_here {
                        drag_drop_target = Some(*idx);
                    }
                    ui.add_space(theme::SPACE_XS);
                }
                // Drag released without a target → cancel.
                if !ctx.input(|i| i.pointer.any_down()) && self.drag_source.is_some() {
                    if let Some(to) = drag_drop_target {
                        if let Some(from) = self.drag_source {
                            if from != to {
                                self.stack.move_operation(from, to);
                                self.set_status(format!(
                                    "moved op [{from}] → [{to}]"
                                ));
                            }
                        }
                    }
                    self.drag_source = None;
                }
                if let Some(i) = to_remove {
                    self.remove_op(i);
                }
                if let Some(i) = to_reseed {
                    self.reseed_op(i);
                }
                if let Some(i) = to_duplicate {
                    if let Some(op) = self.stack.operations().get(i).cloned() {
                        self.push_undo();
                        self.stack.insert_operation(i + 1, op);
                        self.set_status(format!("duplicated op [{i}]"));
                    }
                }
                if let Some(i) = to_move_up {
                    if i > 0 {
                        self.push_undo();
                        self.stack.move_operation(i, i - 1);
                        self.set_status(format!("moved op [{i}] → [{}]", i - 1));
                    }
                }
                if let Some(i) = to_move_down {
                    if i + 1 < self.stack.len() {
                        self.push_undo();
                        self.stack.move_operation(i, i + 1);
                        self.set_status(format!("moved op [{i}] → [{}]", i + 1));
                    }
                }
                if let Some((i, new_kind)) = to_convert {
                    let seed = (i as u64).wrapping_mul(31).wrapping_add(1_000);
                    let new_op = match new_kind {
                        "identity" => Operation::Identity,
                        "dense" => Operation::Dense {
                            key: Hypervector::random_seeded(self.dim, seed),
                        },
                        "hrr_bind" => Operation::HrrBind {
                            key: Hypervector::random_seeded(self.dim, seed + 100),
                        },
                        "permute" => Operation::Permute {
                            shift: ((seed as usize).wrapping_mul(7) + 13)
                                % self.dim.max(1),
                        },
                        "negate" => Operation::Negate,
                        _ => Operation::Identity,
                    };
                    self.push_undo();
                    self.stack.replace_operation(i, new_op);
                    self.set_status(format!("converted op [{i}] → {new_kind}"));
                }

                ui.add_space(theme::SPACE_SM);
                ui.horizontal(|ui| {
                    if mini_button(ui, "➕ Id", theme::op_color("identity")).clicked() {
                        self.add_op("identity");
                    }
                    if mini_button(ui, "➕ Dense", theme::op_color("dense")).clicked() {
                        self.add_op("dense");
                    }
                    if mini_button(ui, "➕ Hrr", theme::op_color("hrr_bind")).clicked() {
                        self.add_op("hrr_bind");
                    }
                });
                ui.add_space(theme::SPACE_XS);
                ui.horizontal(|ui| {
                    if mini_button(ui, "➕ Permute", theme::op_color("permute")).clicked() {
                        self.add_op("permute");
                    }
                    if mini_button(ui, "➕ Negate", theme::op_color("negate")).clicked() {
                        self.add_op("negate");
                    }
                });
                ui.add_space(theme::SPACE_SM);
                if mini_button(ui, "🗑  Reset stack", theme::TEXT_MUTED).clicked() {
                    self.reset_stack();
                }
                    }); // close ScrollArea::vertical
            });
        } // show_left_panel

        // Floating windows (#741) — Blender-style draggable/resizable.
        if self.show_floating_stats {
            let mut open = self.show_floating_stats;
            egui::Window::new("📊 Live stats")
                .open(&mut open)
                .resizable(true)
                .default_size([240.0, 200.0])
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.0, theme::ACCENT_BLUE))
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .inner_margin(egui::Margin::same(theme::SPACE_MD)),
                )
                .show(ctx, |ui| {
                    metric(ui, "template", self.template);
                    metric(ui, "dim", &self.dim.to_string());
                    metric(ui, "ops", &self.stack.len().to_string());
                    metric(ui, "forwards", &self.forwards.to_string());
                    if let Some(ms) = self.last_latency_ms {
                        metric(ui, "last latency", &format!("{ms:.2} ms"));
                    }
                    if let Some(s) = self.last_cos_sim {
                        metric(ui, "cos_sim(in,out)", &format!("{s:+.3}"));
                    }
                });
            self.show_floating_stats = open;
        }
        if self.show_floating_minihelp {
            let mut open = self.show_floating_minihelp;
            egui::Window::new("⌨ Shortcuts")
                .open(&mut open)
                .resizable(true)
                .default_size([300.0, 280.0])
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.0, theme::ACCENT_PURPLE))
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .inner_margin(egui::Margin::same(theme::SPACE_MD)),
                )
                .show(ctx, |ui| {
                    let entries = [
                        ("Space", "run forward"),
                        ("R", "regen input"),
                        ("L", "live mode"),
                        ("A / D / F", "+ id / dense / hrr"),
                        ("P / N", "+ permute / negate"),
                        ("Backspace", "remove selected"),
                        ("1-0", "load template"),
                        ("Tab", "toggle left panel"),
                        ("⇧Tab", "toggle right panel"),
                        ("⌘S / ⌘O", "save / load YAML"),
                        ("⌘E", "PNG export"),
                        ("⌘Z / ⌘⇧Z", "undo / redo"),
                        ("`", "console"),
                        ("H / F1", "help overlay"),
                        ("Esc", "close modal"),
                    ];
                    egui::Grid::new("mini_help_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_LG, 2.0])
                        .show(ui, |ui| {
                            for (k, a) in entries {
                                ui.label(
                                    egui::RichText::new(k)
                                        .color(theme::ACCENT_BLUE)
                                        .monospace()
                                        .strong(),
                                );
                                ui.label(
                                    egui::RichText::new(a)
                                        .color(theme::TEXT_PRIMARY)
                                        .size(theme::SIZE_SMALL),
                                );
                                ui.end_row();
                            }
                        });
                });
            self.show_floating_minihelp = open;
        }

        // Console / REPL pane (#747) — bottom-docked when shown.
        if self.show_console {
            let mut submit: Option<String> = None;
            egui::TopBottomPanel::bottom("console")
                .resizable(true)
                .default_height(220.0)
                .min_height(120.0)
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.0, theme::ACCENT_BLUE))
                        .inner_margin(egui::Margin::same(theme::SPACE_MD)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("⌨ Console")
                                .size(theme::SIZE_BODY)
                                .color(theme::ACCENT_BLUE)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new("(backtick toggles · type 'help')")
                                .size(theme::SIZE_TINY)
                                .color(theme::TEXT_MUTED),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("close").clicked() {
                                    self.show_console = false;
                                }
                            },
                        );
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .max_height(120.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for (cmd, out) in &self.console_history {
                                ui.label(
                                    egui::RichText::new(format!("> {cmd}"))
                                        .size(theme::SIZE_SMALL)
                                        .color(theme::ACCENT_BLUE)
                                        .monospace(),
                                );
                                if !out.is_empty() {
                                    ui.label(
                                        egui::RichText::new(out)
                                            .size(theme::SIZE_SMALL)
                                            .color(theme::TEXT_PRIMARY)
                                            .monospace(),
                                    );
                                }
                                ui.add_space(theme::SPACE_XS);
                            }
                        });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(">")
                                .color(theme::ACCENT_BLUE)
                                .strong()
                                .monospace(),
                        );
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut self.console_input)
                                .desired_width(ui.available_width() - 60.0)
                                .font(egui::TextStyle::Monospace),
                        );
                        if resp.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        {
                            submit = Some(std::mem::take(&mut self.console_input));
                            self.console_history_cursor = -1;
                            resp.request_focus();
                        }
                        // Up arrow → recall previous command.
                        if resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::ArrowUp))
                        {
                            let n = self.console_history.len() as i32;
                            if n > 0 {
                                self.console_history_cursor =
                                    (self.console_history_cursor + 1).min(n - 1);
                                let idx =
                                    (n - 1 - self.console_history_cursor).max(0) as usize;
                                if let Some((cmd, _)) =
                                    self.console_history.get(idx)
                                {
                                    self.console_input = cmd.clone();
                                }
                            }
                        }
                        if resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::ArrowDown))
                        {
                            if self.console_history_cursor > 0 {
                                self.console_history_cursor -= 1;
                                let n = self.console_history.len() as i32;
                                let idx =
                                    (n - 1 - self.console_history_cursor).max(0) as usize;
                                if let Some((cmd, _)) =
                                    self.console_history.get(idx)
                                {
                                    self.console_input = cmd.clone();
                                }
                            } else if self.console_history_cursor == 0 {
                                self.console_history_cursor = -1;
                                self.console_input.clear();
                            }
                        }
                        if ui.button("run").clicked() {
                            submit = Some(std::mem::take(&mut self.console_input));
                            self.console_history_cursor = -1;
                        }
                    });
                });
            if let Some(cmd) = submit {
                let out = self.run_console_cmd(&cmd);
                if !cmd.trim().is_empty() {
                    self.console_history.push((cmd, out));
                    if self.console_history.len() > 100 {
                        self.console_history.remove(0);
                    }
                }
            }
        }

        // Templates popup modal (#745).
        if self.show_templates_popup {
            let mut load: Option<&'static str> = None;
            let mut close = false;
            egui::Window::new("Choose a template")
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(true)
                .default_size([720.0, 520.0])
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.5, theme::ACCENT_PURPLE))
                        .rounding(egui::Rounding::same(theme::RADIUS_LG))
                        .inner_margin(egui::Margin::same(theme::SPACE_LG)),
                )
                .show(ctx, |ui| {
                    ui.label(
                        egui::RichText::new("Start a new experiment")
                            .size(theme::SIZE_H1)
                            .color(theme::ACCENT_BLUE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_XS);
                    ui.label(
                        egui::RichText::new("Pick a template or start from a blank stack.")
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_MUTED),
                    );
                    ui.add_space(theme::SPACE_MD);
                    egui::ScrollArea::vertical()
                        .max_height(380.0)
                        .show(ui, |ui| {
                            // Blank stack option as first card.
                            let blank = ui.add(
                                egui::Button::new(
                                    egui::RichText::new("⬚  Blank stack\nStart from nothing — add ops manually.")
                                        .size(theme::SIZE_SMALL)
                                        .color(theme::TEXT_PRIMARY),
                                )
                                .fill(theme::BG_CARD_HOVER)
                                .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                                .rounding(egui::Rounding::same(theme::RADIUS_MD))
                                .min_size(egui::vec2(ui.available_width(), 50.0)),
                            );
                            if blank.clicked() {
                                load = Some("__blank__");
                            }
                            ui.add_space(theme::SPACE_SM);
                            for (i, template) in TEMPLATES.iter().enumerate() {
                                let card = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(format!(
                                            "{}.  {}\n{}\n\n{}",
                                            i + 1,
                                            template.name,
                                            template.summary,
                                            template.explanation,
                                        ))
                                        .size(theme::SIZE_SMALL)
                                        .color(theme::TEXT_PRIMARY),
                                    )
                                    .fill(theme::BG_CARD_HOVER)
                                    .stroke(egui::Stroke::new(1.0, theme::BORDER_SUBTLE))
                                    .rounding(egui::Rounding::same(theme::RADIUS_MD))
                                    .min_size(egui::vec2(ui.available_width(), 84.0)),
                                );
                                if card.clicked() {
                                    load = Some(template.name);
                                }
                                ui.add_space(theme::SPACE_XS);
                            }
                        });
                    ui.add_space(theme::SPACE_MD);
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                });
            if let Some(name) = load {
                if name == "__blank__" {
                    self.push_undo();
                    self.template = "blank";
                    self.stack = Stack::new(self.dim);
                    self.set_status("loaded blank stack".to_string());
                } else {
                    self.load_template(name);
                }
                self.show_templates_popup = false;
            }
            if close
                || ctx.input(|i| i.key_pressed(egui::Key::Escape))
            {
                self.show_templates_popup = false;
            }
        }

        // Walkthrough overlay — first-run tutorial WITH spotlight on the
        // current step's relevant UI region (#726).
        if let Some(step) = self.walkthrough_step {
            // Darken everything except a spotlight rect for the focused region.
            let screen = ctx.screen_rect();
            let spotlight_rect = walkthrough_spotlight_rect(step, screen);
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("walkthrough_scrim"),
            ));
            // Four scrim rects covering everything except the spotlight.
            let scrim = egui::Color32::from_black_alpha(160);
            // top
            painter.rect_filled(
                egui::Rect::from_min_max(
                    screen.min,
                    egui::pos2(screen.max.x, spotlight_rect.min.y),
                ),
                0.0,
                scrim,
            );
            // bottom
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(screen.min.x, spotlight_rect.max.y),
                    screen.max,
                ),
                0.0,
                scrim,
            );
            // left
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(screen.min.x, spotlight_rect.min.y),
                    egui::pos2(spotlight_rect.min.x, spotlight_rect.max.y),
                ),
                0.0,
                scrim,
            );
            // right
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(spotlight_rect.max.x, spotlight_rect.min.y),
                    egui::pos2(screen.max.x, spotlight_rect.max.y),
                ),
                0.0,
                scrim,
            );
            // Spotlight outline.
            painter.rect_stroke(
                spotlight_rect,
                egui::Rounding::same(theme::RADIUS_MD),
                egui::Stroke::new(2.0, theme::ACCENT_PURPLE),
            );
            let (title, body) = WALKTHROUGH_STEPS
                .get(step)
                .copied()
                .unwrap_or((WALKTHROUGH_STEPS[0].0, WALKTHROUGH_STEPS[0].1));
            let mut close = false;
            let mut advance = false;
            egui::Window::new(format!("Walkthrough {}/{}", step + 1, WALKTHROUGH_STEPS.len()))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .collapsible(false)
                .resizable(false)
                .frame(
                    egui::Frame::none()
                        .fill(theme::BG_CARD)
                        .stroke(egui::Stroke::new(1.5, theme::ACCENT_PURPLE))
                        .rounding(egui::Rounding::same(theme::RADIUS_LG))
                        .inner_margin(egui::Margin::same(theme::SPACE_XL)),
                )
                .show(ctx, |ui| {
                    ui.set_min_width(540.0);
                    ui.label(
                        egui::RichText::new(title)
                            .size(theme::SIZE_H1)
                            .color(theme::ACCENT_BLUE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_MD);
                    ui.label(
                        egui::RichText::new(body)
                            .size(theme::SIZE_BODY)
                            .color(theme::TEXT_PRIMARY),
                    );
                    ui.add_space(theme::SPACE_LG);
                    ui.horizontal(|ui| {
                        if ui
                            .button(
                                egui::RichText::new("Skip tour")
                                    .size(theme::SIZE_SMALL)
                                    .color(theme::TEXT_MUTED),
                            )
                            .clicked()
                        {
                            close = true;
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                let label = if step + 1 >= WALKTHROUGH_STEPS.len() {
                                    "Finish ✓"
                                } else {
                                    "Next →"
                                };
                                let next = ui.add(
                                    egui::Button::new(
                                        egui::RichText::new(label)
                                            .size(theme::SIZE_BODY)
                                            .color(egui::Color32::WHITE)
                                            .strong(),
                                    )
                                    .fill(theme::ACCENT_MID)
                                    .rounding(egui::Rounding::same(theme::RADIUS_MD))
                                    .min_size(egui::vec2(120.0, 36.0)),
                                );
                                if next.clicked() {
                                    advance = true;
                                }
                            },
                        );
                    });
                    ui.add_space(theme::SPACE_SM);
                    ui.label(
                        egui::RichText::new(format!(
                            "Press → to advance · Esc to dismiss · Step {}/{}",
                            step + 1,
                            WALKTHROUGH_STEPS.len()
                        ))
                        .size(theme::SIZE_TINY)
                        .color(theme::TEXT_DIM),
                    );
                });
            if advance {
                let next = step + 1;
                if next >= WALKTHROUGH_STEPS.len() {
                    self.dismiss_walkthrough();
                } else {
                    self.walkthrough_step = Some(next);
                }
            }
            if close {
                self.dismiss_walkthrough();
            }
        }

        // Help overlay: shortcuts AND experiment recipes.
        if self.show_help {
            let avail_h = ctx.screen_rect().height();
            let mut help_open = self.show_help;
            egui::Window::new("GraphNet help")
                .open(&mut help_open) // ← gives the title-bar × button
                .max_height(avail_h - 80.0)
                .max_width(720.0)
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
                    ui.set_min_width(540.0);
                    // Big "Close" button + tutorial/demo launch row at top.
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("✕ Close")
                                        .size(theme::SIZE_BODY)
                                        .strong(),
                                )
                                .fill(theme::ACCENT_PURPLE)
                                .min_size(egui::vec2(90.0, 32.0)),
                            )
                            .clicked()
                        {
                            self.show_help = false;
                        }
                        if mini_button(ui, "📖 Replay tutorial", theme::ACCENT_BLUE)
                            .clicked()
                        {
                            self.walkthrough_step = Some(0);
                            self.show_help = false;
                        }
                        if mini_button(ui, "🎬 Run demo", theme::ACCENT_MID).clicked() {
                            if self.demo.is_none() {
                                self.start_demo();
                            }
                            self.show_help = false;
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(avail_h - 220.0)
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Keyboard shortcuts")
                            .size(theme::SIZE_H2)
                            .color(theme::ACCENT_BLUE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_SM);
                    let pairs = [
                        ("Space", "run forward"),
                        ("R", "regenerate input"),
                        ("L", "toggle live continuous mode"),
                        ("1 / 2 … 9 / 0", "load template"),
                        ("⌘S / Ctrl+S", "save YAML"),
                        ("⌘O / Ctrl+O", "load YAML"),
                        ("H / F1", "toggle this help"),
                        ("Esc", "close modal"),
                    ];
                    egui::Grid::new("help_shortcuts")
                        .num_columns(2)
                        .spacing([theme::SPACE_XL, theme::SPACE_XS])
                        .show(ui, |ui| {
                            for (key, action) in pairs {
                                ui.label(
                                    egui::RichText::new(key)
                                        .color(theme::ACCENT_BLUE)
                                        .size(theme::SIZE_SMALL)
                                        .monospace()
                                        .strong(),
                                );
                                ui.label(
                                    egui::RichText::new(action)
                                        .color(theme::TEXT_PRIMARY)
                                        .size(theme::SIZE_SMALL),
                                );
                                ui.end_row();
                            }
                        });

                    ui.add_space(theme::SPACE_LG);
                    ui.label(
                        egui::RichText::new(format!(
                            "Achievements ({}/{})",
                            self.achievements.len(),
                            11
                        ))
                        .size(theme::SIZE_H2)
                        .color(theme::ACCENT_BLUE)
                        .strong(),
                    );
                    ui.add_space(theme::SPACE_SM);
                    let all_badges: &[(&str, &str, &str)] = &[
                        ("first_forward", "🚀", "First Forward — run any forward"),
                        ("ten_forwards", "🔥", "Warming Up — 10 forwards"),
                        ("hundred_forwards", "💯", "Iteration Champion — 100 forwards"),
                        ("thousand_forwards", "⚡", "Live & Loving It — 1000 forwards"),
                        ("all_op_kinds", "🎭", "Generalist — use all 5 op kinds"),
                        ("all_templates", "📚", "Template Connoisseur — load every template"),
                        ("ten_op_stack", "🏗", "Tower Builder — 10+ ops in one stack"),
                        ("high_dim", "🌌", "Hyperdimensional — D ≥ 16,000"),
                        ("low_dim", "🌱", "Minimalist — D ≤ 256"),
                        ("demo_runner", "🎬", "Demo Watcher — run the auto-demo"),
                        ("undoer", "↶", "Revisionist — 5+ undo operations"),
                        ("high_sim", "🪞", "Echo Chamber — cos_sim > 0.85"),
                    ];
                    egui::Grid::new("achievements_grid")
                        .num_columns(2)
                        .spacing([theme::SPACE_MD, theme::SPACE_XS])
                        .show(ui, |ui| {
                            for (slug, icon, description) in all_badges {
                                let unlocked = self.achievements.contains_key(slug);
                                let icon_col = if unlocked {
                                    egui::Color32::from_rgb(0xE0, 0xB1, 0x5B)
                                } else {
                                    theme::TEXT_DIM
                                };
                                let text_col = if unlocked {
                                    theme::TEXT_PRIMARY
                                } else {
                                    theme::TEXT_DIM
                                };
                                ui.label(
                                    egui::RichText::new(*icon)
                                        .size(theme::SIZE_H2)
                                        .color(icon_col),
                                );
                                ui.label(
                                    egui::RichText::new(*description)
                                        .size(theme::SIZE_SMALL)
                                        .color(text_col),
                                );
                                ui.end_row();
                            }
                        });

                    ui.add_space(theme::SPACE_LG);
                    ui.label(
                        egui::RichText::new("Try this experiment recipes:")
                            .size(theme::SIZE_H2)
                            .color(theme::ACCENT_PURPLE)
                            .strong(),
                    );
                    ui.add_space(theme::SPACE_SM);
                    let recipes = [
                        "Press 2 → standard config. Press Space 5 times. Watch \
                         the cos_sim sparkline build up. The output is mostly \
                         orthogonal to input — expected for HDC binding.",
                        "Press 6 → noise-resilience (8 mixed ops). cos_sim(in, out) \
                         drops further. Now press R a few times — see how the \
                         shape of the output heatmap changes with each new input.",
                        "Press L for live mode. Then drag the Dim slider from 10k \
                         to 16k. Watch latency spike on the sparkline + the \
                         frame-budget dot turn amber/red.",
                        "Press 9 → positional-encode. Then click each Permute \
                         chip in the 3D arch graph — see how each one contributes \
                         a rotated copy of the input.",
                        "Press 10 (key 0) → anti-correlation. The bundled output's \
                         cos_sim to input should be near 0 — the dense and -dense \
                         legs cancel.",
                        "Click the ▶ Demo button in the hero. Watch it cycle \
                         through every config. Each step shows a different network \
                         shape.",
                    ];
                    for (i, recipe) in recipes.iter().enumerate() {
                        ui.label(
                            egui::RichText::new(format!("{}.  {}", i + 1, recipe))
                                .size(theme::SIZE_SMALL)
                                .color(theme::TEXT_PRIMARY),
                        );
                        ui.add_space(theme::SPACE_XS);
                    }
                    }); // close ScrollArea
                });
            // honor the title-bar × button.
            if !help_open {
                self.show_help = false;
            }
        }

        // Right panel: action log + sparklines + per-op inspector.
        // Pulls the noise out of the central panel + uses previously empty space.
        if self.show_right_panel {
        egui::SidePanel::right("right")
            .min_width(320.0)
            .max_width(420.0)
            .resizable(true)
            .frame(
                egui::Frame::none()
                    .fill(theme::BG)
                    .inner_margin(egui::Margin::same(theme::SPACE_LG)),
            )
            .show(ctx, |ui| {
                // IDE-style header bar with close ×.
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Inspector")
                            .size(theme::SIZE_TINY)
                            .color(theme::ACCENT_PURPLE)
                            .strong(),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .small_button("×")
                                .on_hover_text("close panel (Shift+Tab to re-open)")
                                .clicked()
                            {
                                self.show_right_panel = false;
                            }
                        },
                    );
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .max_height(f32::INFINITY)
                    .show(ui, |ui| {
                        section_heading(ui, "Action log");
                        ui.add_space(theme::SPACE_SM);
                        card(ui, |ui| {
                            if self.action_log.is_empty() {
                                ui.label(
                                    egui::RichText::new("(no actions yet — try pressing Space)")
                                        .color(theme::TEXT_DIM)
                                        .italics()
                                        .size(theme::SIZE_SMALL),
                                );
                            } else {
                                let now = std::time::Instant::now();
                                for entry in self.action_log.iter().rev().take(12) {
                                    let age = now.duration_since(entry.at).as_secs_f32();
                                    let alpha = (1.0 - (age / 60.0).clamp(0.0, 0.7)).max(0.3);
                                    let base = entry.severity.color();
                                    let colour = egui::Color32::from_rgba_unmultiplied(
                                        base.r(),
                                        base.g(),
                                        base.b(),
                                        (alpha * 255.0) as u8,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{age:>4.1}s"))
                                                .size(theme::SIZE_TINY)
                                                .color(theme::TEXT_MUTED)
                                                .monospace(),
                                        );
                                        ui.label(
                                            egui::RichText::new(entry.severity.glyph())
                                                .size(theme::SIZE_SMALL)
                                                .color(base),
                                        );
                                        ui.label(
                                            egui::RichText::new(&entry.msg)
                                                .size(theme::SIZE_SMALL)
                                                .color(colour),
                                        );
                                    });
                                }
                            }
                        });

                        // Active objective card (#720).
                        ui.add_space(theme::SPACE_MD);
                        let n_done = self.objective_done.iter().filter(|x| **x).count();
                        section_heading(
                            ui,
                            &format!("🎯 Objective {}/{}", n_done, OBJECTIVES.len()),
                        );
                        ui.add_space(theme::SPACE_SM);
                        card(ui, |ui| {
                            if let Some(obj) = OBJECTIVES.get(self.current_objective) {
                                ui.label(
                                    egui::RichText::new(obj.title)
                                        .size(theme::SIZE_BODY)
                                        .color(theme::ACCENT_BLUE)
                                        .strong(),
                                );
                                ui.add_space(theme::SPACE_XS);
                                ui.label(
                                    egui::RichText::new(obj.description)
                                        .size(theme::SIZE_SMALL)
                                        .color(theme::TEXT_PRIMARY),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("🏆 All objectives complete!")
                                        .size(theme::SIZE_BODY)
                                        .color(theme::ACCENT_PURPLE)
                                        .strong(),
                                );
                            }
                            ui.add_space(theme::SPACE_SM);
                            // Progress dots.
                            ui.horizontal(|ui| {
                                for (i, done) in self.objective_done.iter().enumerate() {
                                    let colour = if *done {
                                        theme::ACCENT_MID
                                    } else if i == self.current_objective {
                                        theme::ACCENT_BLUE
                                    } else {
                                        theme::TEXT_DIM
                                    };
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().circle_filled(rect.center(), 4.0, colour);
                                }
                            });
                        });

                        // Smart suggestions — context-aware hints (#709).
                        let hints = self.smart_suggestions();
                        if !hints.is_empty() {
                            ui.add_space(theme::SPACE_LG);
                            section_heading(ui, "💡 Suggestions");
                            ui.add_space(theme::SPACE_SM);
                            card(ui, |ui| {
                                for hint in &hints {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            egui::RichText::new("•")
                                                .size(theme::SIZE_BODY)
                                                .color(theme::ACCENT_PURPLE),
                                        );
                                        ui.label(
                                            egui::RichText::new(hint)
                                                .size(theme::SIZE_SMALL)
                                                .color(theme::TEXT_PRIMARY),
                                        );
                                    });
                                    ui.add_space(theme::SPACE_XS);
                                }
                            });
                        }

                        if !self.cos_sim_history.is_empty() {
                            ui.add_space(theme::SPACE_LG);
                            section_heading(ui, "cos_sim history");
                            ui.add_space(theme::SPACE_SM);
                            card(ui, |ui| {
                                sparkline(ui, self.cos_sim_history.iter().copied(), 70.0);
                            });

                            ui.add_space(theme::SPACE_MD);
                            section_heading(ui, "latency history (ms)");
                            ui.add_space(theme::SPACE_SM);
                            card(ui, |ui| {
                                latency_sparkline(
                                    ui,
                                    self.latency_history.iter().copied(),
                                    70.0,
                                );
                            });
                        }

                        // Per-op contribution + inspector — moved from central.
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
                            ui.label(
                                egui::RichText::new("Click a chip below or in the 3D graph:")
                                    .color(theme::TEXT_MUTED)
                                    .size(theme::SIZE_SMALL),
                            );
                            ui.add_space(theme::SPACE_XS);
                            let mut selected_change: Option<Option<usize>> = None;
                            ui.horizontal_wrapped(|ui| {
                                for op_out in &trace.per_op {
                                    let active = self.selected_op == Some(op_out.index);
                                    let accent = theme::op_color(&op_out.tag);
                                    let btn = ui.add(
                                        egui::Button::new(
                                            egui::RichText::new(format!(
                                                "[{}] {}",
                                                op_out.index, op_out.tag
                                            ))
                                            .size(theme::SIZE_TINY)
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
                                        selected_change = Some(if active {
                                            None
                                        } else {
                                            Some(op_out.index)
                                        });
                                    }
                                }
                            });
                            if let Some(s) = selected_change {
                                self.selected_op = s;
                            }
                            if let Some(idx) = self.selected_op {
                                if let Some(op_out) = trace.per_op.get(idx) {
                                    ui.add_space(theme::SPACE_SM);
                                    let sim_to_input =
                                        cos_sim(&self.input, &op_out.output).ok();
                                    let sim_to_out =
                                        cos_sim(&op_out.output, &trace.bundled).ok();
                                    let inspector_output = op_out.output.clone();
                                    let inspector_index = op_out.index;
                                    let inspector_tag = op_out.tag.clone();
                                    let mut zoom_here = false;
                                    card(ui, |ui| {
                                        metric(
                                            ui,
                                            "op",
                                            &format!("[{inspector_index}] {inspector_tag}"),
                                        );
                                        if let Some(s) = sim_to_input {
                                            metric(
                                                ui,
                                                "cos_sim → input",
                                                &format!("{s:+.3}"),
                                            );
                                        }
                                        if let Some(s) = sim_to_out {
                                            metric(
                                                ui,
                                                "cos_sim → bundled",
                                                &format!("{s:+.3}"),
                                            );
                                        }
                                        ui.add_space(theme::SPACE_SM);
                                        if hypervector_heatmap_clickable(
                                            ui,
                                            &inspector_output,
                                            60,
                                            3.0,
                                        ) {
                                            zoom_here = true;
                                        }
                                    });
                                    if zoom_here {
                                        self.zoom_target = Some(ZoomTarget::PerOp(idx));
                                    }
                                }
                            }
                        }
                    });
            });
        } // show_right_panel

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(theme::BG)
                    .inner_margin(egui::Margin::same(theme::SPACE_XL)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .max_height(f32::INFINITY)
                    .show(ui, |ui| {
                // Two-column layout: Input (left) + Output preview (right)
                // when there's a recent forward. Single column otherwise.
                let input_clone = self.input.clone();
                let has_output = self.last_output.is_some();
                let mut zoom_request_local: Option<ZoomTarget> = None;
                let mut regen_clicked = false;
                let mut load_image_clicked = false;
                ui.columns(if has_output { 2 } else { 1 }, |cols| {
                    // Left col: input.
                    let left = &mut cols[0];
                    section_heading(left, "Input");
                    left.add_space(theme::SPACE_SM);
                    let input_summary = Self::hv_summary(&input_clone);
                    card(left, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("seed = {}", self.input_seed))
                                    .size(theme::SIZE_BODY),
                            );
                            if mini_button(ui, "🎲", theme::TEXT_MUTED)
                                .on_hover_text("regenerate input (R)")
                                .clicked()
                            {
                                regen_clicked = true;
                            }
                            if mini_button(ui, "🖼 image…", theme::ACCENT_BLUE)
                                .on_hover_text(
                                    "load PNG/JPG/BMP → quantize to bipolar input (#758)",
                                )
                                .clicked()
                            {
                                load_image_clicked = true;
                            }
                            let _ = &mut load_image_clicked;
                        });
                        if let Some(p) = &self.loaded_image {
                            ui.label(
                                egui::RichText::new(format!(
                                    "📷 {}",
                                    p.file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("?")
                                ))
                                .size(theme::SIZE_TINY)
                                .color(theme::ACCENT_PURPLE),
                            );
                        }
                        ui.label(
                            egui::RichText::new(format!(
                                "fp {} · {:.1}% +1",
                                input_summary.fingerprint,
                                input_summary.percent_positive * 100.0
                            ))
                            .size(theme::SIZE_TINY)
                            .color(theme::TEXT_MUTED)
                            .monospace(),
                        )
                        .on_hover_text(
                            "fp = blake3 fingerprint of the hypervector\n\
                             % +1 = fraction of components with positive sign",
                        );
                        ui.label(
                            egui::RichText::new(format!("…{}…", input_summary.binary_prefix))
                                .size(theme::SIZE_TINY)
                                .color(theme::ACCENT_BLUE)
                                .monospace(),
                        )
                        .on_hover_text("First 24 components as binary (1 = +1, 0 = -1)");
                        ui.add_space(theme::SPACE_SM);
                        if hypervector_heatmap_clickable_cmap(ui, &input_clone, 100, 2.0, self.colormap) {
                            zoom_request_local = Some(ZoomTarget::Input);
                        }
                    });
                    // Right col: latest output preview.
                    if has_output {
                        let right = &mut cols[1];
                        let out = self.last_output.clone().expect("checked");
                        let out_summary = Self::hv_summary(&out);
                        // Game-feel: green halo on high cos_sim, red on inverted.
                        let glow_intensity = self.last_forward_at
                            .map(|t| {
                                let age = t.elapsed().as_secs_f32();
                                if age < 0.8 {
                                    Some(1.0 - age / 0.8)
                                } else {
                                    None
                                }
                            })
                            .flatten();
                        let glow_color = match (self.last_cos_sim, glow_intensity) {
                            (Some(s), Some(intensity)) if s > 0.85 => Some((
                                egui::Color32::from_rgb(0x4D, 0xC4, 0x82),
                                intensity,
                            )),
                            (Some(s), Some(intensity)) if s < -0.5 => Some((
                                egui::Color32::from_rgb(0xE0, 0x6A, 0x5B),
                                intensity,
                            )),
                            _ => None,
                        };
                        section_heading(right, "Output (latest)");
                        right.add_space(theme::SPACE_SM);
                        card(right, |ui| {
                            if let Some((c, intensity)) = glow_color {
                                ui.painter().rect_stroke(
                                    ui.max_rect().expand(2.0),
                                    egui::Rounding::same(theme::RADIUS_MD),
                                    egui::Stroke::new(
                                        3.0,
                                        egui::Color32::from_rgba_unmultiplied(
                                            c.r(),
                                            c.g(),
                                            c.b(),
                                            (intensity * 220.0) as u8,
                                        ),
                                    ),
                                );
                                ctx.request_repaint();
                            }
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("dim = {}", out.dim()))
                                        .size(theme::SIZE_BODY),
                                );
                                if let Some(s) = self.last_cos_sim {
                                    ui.label(
                                        egui::RichText::new(format!("· cos_sim {s:+.3}"))
                                            .size(theme::SIZE_SMALL)
                                            .color(theme::TEXT_MUTED),
                                    );
                                }
                            });
                            ui.label(
                                egui::RichText::new(format!(
                                    "fp {} · {:.1}% +1",
                                    out_summary.fingerprint,
                                    out_summary.percent_positive * 100.0
                                ))
                                .size(theme::SIZE_TINY)
                                .color(theme::TEXT_MUTED)
                                .monospace(),
                            );
                            ui.label(
                                egui::RichText::new(format!("…{}…", out_summary.binary_prefix))
                                    .size(theme::SIZE_TINY)
                                    .color(theme::ACCENT_PURPLE)
                                    .monospace(),
                            );
                            ui.add_space(theme::SPACE_SM);
                            if hypervector_heatmap_clickable_cmap(ui, &out, 100, 2.0, self.colormap) {
                                zoom_request_local = Some(ZoomTarget::Output);
                            }
                        });
                    }
                });
                if let Some(zt) = zoom_request_local {
                    self.zoom_target = Some(zt);
                }
                if regen_clicked {
                    self.regenerate_input();
                }
                if load_image_clicked {
                    self.load_image_as_input();
                }

                ui.add_space(theme::SPACE_LG);
                // Animated pulse on the Run-forward button when in live mode.
                let pulse_amp = if self.live {
                    let t = self.spawn_time.elapsed().as_secs_f32();
                    ((t * 4.0).sin() * 0.5 + 0.5) * 0.5 + 0.5
                } else {
                    1.0
                };
                let pulsed_fill = egui::Color32::from_rgba_unmultiplied(
                    theme::ACCENT_MID.r(),
                    theme::ACCENT_MID.g(),
                    theme::ACCENT_MID.b(),
                    (pulse_amp * 255.0) as u8,
                );
                ui.horizontal(|ui| {
                    let btn = ui.add(
                        egui::Button::new(
                            egui::RichText::new("▶  Run forward")
                                .size(theme::SIZE_BODY)
                                .strong(),
                        )
                        .fill(pulsed_fill)
                        .rounding(egui::Rounding::same(theme::RADIUS_MD))
                        .min_size(egui::vec2(130.0, 32.0)),
                    );
                    if btn.clicked() {
                        self.run_forward();
                    }
                    if self.live {
                        ctx.request_repaint();
                    }

                    let live_label = if self.live { "⏹ Stop live" } else { "▶ Live" };
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
                        .min_size(egui::vec2(95.0, 32.0)),
                    );
                    if live_btn.clicked() {
                        self.live = !self.live;
                    }
                    ui.label(
                        egui::RichText::new("  (Space / L)")
                            .size(theme::SIZE_SMALL)
                            .color(theme::TEXT_MUTED),
                    );
                });

                ui.add_space(theme::SPACE_LG);

                // Architecture graph viz (3D-projected, rotatable).
                ui.horizontal(|ui| {
                    section_heading(ui, "Architecture (3D)");
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.checkbox(&mut self.arch_autorotate, "auto-rotate");
                        },
                    );
                });
                ui.add_space(theme::SPACE_SM);
                let op_tags: Vec<String> = self
                    .stack
                    .operations()
                    .iter()
                    .map(|op| op.tag().to_string())
                    .collect();
                if self.arch_autorotate {
                    self.arch_yaw += 0.008;
                    self.arch_pitch = 0.2 + (self.arch_yaw * 0.5).sin() * 0.25;
                    ctx.request_repaint();
                }
                let mut yaw = self.arch_yaw;
                let mut pitch = self.arch_pitch;
                let selected = self.selected_op;
                let mut graph_click: Option<usize> = None;
                // Particle phase: rolling 0..1 driven by elapsed time since
                // last forward. Particles travel for ~0.8s after each fwd.
                let particle_phase = self
                    .last_forward_at
                    .map(|t| {
                        let elapsed = t.elapsed().as_secs_f32();
                        if elapsed < 0.8 {
                            Some(elapsed / 0.8)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(None);
                if particle_phase.is_some() {
                    ctx.request_repaint();
                }
                let zoom = self.arch_zoom;
                let autorotate_state = self.arch_autorotate;
                let mut roll = self.arch_roll;
                let mut toolbar_action: Option<ArchToolAction> = None;
                card(ui, |ui| {
                    let contrib_vec: Option<Vec<f64>> = self.last_trace.as_ref().map(|t| {
                        t.per_op
                            .iter()
                            .map(|o| cos_sim(&o.output, &t.bundled).unwrap_or(0.0))
                            .collect()
                    });
                    let contrib_slice = contrib_vec.as_deref();
                    // Node click flash intensity (0..1) decays over 0.5s.
                    let click_flash = self.last_node_click_at.and_then(|t| {
                        let age = t.elapsed().as_secs_f32();
                        if age < 0.5 {
                            Some((self.last_node_clicked?, 1.0 - age / 0.5))
                        } else {
                            None
                        }
                    });
                    if click_flash.is_some() {
                        ctx.request_repaint();
                    }
                    let (clicked, new_yaw, new_pitch, new_roll, action) =
                        architecture_graph_3d(
                            ui,
                            &op_tags,
                            selected,
                            yaw,
                            pitch,
                            roll,
                            zoom,
                            autorotate_state,
                            particle_phase,
                            contrib_slice,
                            click_flash,
                        );
                    if let Some(idx) = clicked {
                        graph_click = Some(idx);
                    }
                    yaw = new_yaw;
                    pitch = new_pitch;
                    roll = new_roll;
                    toolbar_action = action;
                });
                self.arch_yaw = yaw;
                self.arch_pitch = pitch;
                self.arch_roll = roll;

                // Inline editor for the currently-selected op (3D graph).
                // Floats just below the architecture card.
                if let Some(sel_idx) = self.selected_op {
                    if sel_idx < self.stack.len() {
                        let tag = self.stack.operations()[sel_idx].tag().to_string();
                        let colour = theme::op_color(&tag);
                        ui.add_space(theme::SPACE_SM);
                        card(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("✎ Selected [{sel_idx}] {tag}"))
                                        .size(theme::SIZE_BODY)
                                        .color(colour)
                                        .strong(),
                                );
                                ui.add_space(theme::SPACE_MD);
                                if matches!(tag.as_str(), "dense" | "hrr_bind" | "permute") {
                                    if mini_button(ui, "⟳ reseed", colour).clicked() {
                                        self.reseed_op(sel_idx);
                                    }
                                }
                                if mini_button(ui, "⎘ duplicate", theme::ACCENT_BLUE).clicked() {
                                    if let Some(op) = self.stack.operations().get(sel_idx).cloned()
                                    {
                                        self.push_undo();
                                        self.stack.insert_operation(sel_idx + 1, op);
                                        self.set_status(format!(
                                            "duplicated op [{sel_idx}]"
                                        ));
                                    }
                                }
                                if mini_button(ui, "↑ up", theme::TEXT_MUTED).clicked()
                                    && sel_idx > 0
                                {
                                    self.push_undo();
                                    self.stack.move_operation(sel_idx, sel_idx - 1);
                                    self.selected_op = Some(sel_idx - 1);
                                }
                                if mini_button(ui, "↓ down", theme::TEXT_MUTED).clicked()
                                    && sel_idx + 1 < self.stack.len()
                                {
                                    self.push_undo();
                                    self.stack.move_operation(sel_idx, sel_idx + 1);
                                    self.selected_op = Some(sel_idx + 1);
                                }
                                if mini_button(ui, "× remove", theme::ACCENT_PURPLE).clicked() {
                                    self.remove_op(sel_idx);
                                    self.selected_op = None;
                                }
                            });
                            ui.add_space(theme::SPACE_XS);
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new("convert to:")
                                        .color(theme::TEXT_MUTED)
                                        .size(theme::SIZE_SMALL),
                                );
                                for new_kind in
                                    ["identity", "dense", "hrr_bind", "permute", "negate"]
                                {
                                    if new_kind == tag {
                                        continue;
                                    }
                                    if mini_button(ui, new_kind, theme::op_color(new_kind))
                                        .clicked()
                                    {
                                        let seed = (sel_idx as u64)
                                            .wrapping_mul(31)
                                            .wrapping_add(1_000);
                                        let new_op = match new_kind {
                                            "identity" => Operation::Identity,
                                            "dense" => Operation::Dense {
                                                key: Hypervector::random_seeded(self.dim, seed),
                                            },
                                            "hrr_bind" => Operation::HrrBind {
                                                key: Hypervector::random_seeded(
                                                    self.dim,
                                                    seed + 100,
                                                ),
                                            },
                                            "permute" => Operation::Permute {
                                                shift: ((seed as usize).wrapping_mul(7) + 13)
                                                    % self.dim.max(1),
                                            },
                                            "negate" => Operation::Negate,
                                            _ => Operation::Identity,
                                        };
                                        self.push_undo();
                                        self.stack.replace_operation(sel_idx, new_op);
                                        self.set_status(format!(
                                            "converted [{sel_idx}] → {new_kind}"
                                        ));
                                    }
                                }
                            });
                        });
                    }
                }

                // Quick-add palette right under the 3D graph — drag-drop UX
                // for adding new ops without leaving the central view.
                ui.add_space(theme::SPACE_SM);
                card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("➕ Add op to graph:")
                                .size(theme::SIZE_SMALL)
                                .color(theme::TEXT_MUTED),
                        );
                        for kind in ["identity", "dense", "hrr_bind", "permute", "negate"] {
                            if mini_button(ui, kind, theme::op_color(kind))
                                .on_hover_text(format!("click to add {kind} to the stack"))
                                .clicked()
                            {
                                self.add_op(kind);
                            }
                        }
                    });
                });

                if let Some(action) = toolbar_action {
                    match action {
                        ArchToolAction::Reset => {
                            self.arch_yaw = 0.6;
                            self.arch_pitch = 0.15;
                            self.arch_roll = 0.0;
                            self.arch_zoom = 1.0;
                            self.set_status("3D view reset".to_string());
                        }
                        ArchToolAction::ToggleAutorotate => {
                            self.arch_autorotate = !self.arch_autorotate;
                            self.set_status(format!(
                                "auto-rotate: {}",
                                if self.arch_autorotate { "on" } else { "off" }
                            ));
                        }
                        ArchToolAction::ZoomIn => {
                            self.arch_zoom = (self.arch_zoom * 1.2).min(3.0);
                            self.set_status(format!("zoom: {:.2}x", self.arch_zoom));
                        }
                        ArchToolAction::ZoomOut => {
                            self.arch_zoom = (self.arch_zoom / 1.2).max(0.5);
                            self.set_status(format!("zoom: {:.2}x", self.arch_zoom));
                        }
                    }
                }
                if let Some(idx) = graph_click {
                    self.selected_op = if self.selected_op == Some(idx) {
                        None
                    } else {
                        Some(idx)
                    };
                    self.last_node_click_at = Some(std::time::Instant::now());
                    self.last_node_clicked = Some(idx);
                    ctx.request_repaint();
                }

                // (cos_sim / latency sparklines moved to the right panel.)

                ui.add_space(theme::SPACE_LG);
                let mut zoom_request: Option<ZoomTarget> = None;
                if let Some(out) = self.last_output.clone() {
                    section_heading(ui, "Output");
                    ui.add_space(theme::SPACE_SM);
                    let latency = self.last_latency_ms;
                    let sim = self.last_cos_sim;
                    let hd = hamming(&self.input, &out).ok();
                    let bits_diff = self
                        .input
                        .as_slice()
                        .iter()
                        .zip(out.as_slice())
                        .filter(|(a, b)| a != b)
                        .count();
                    let mean_mag: f64 = {
                        #[allow(clippy::cast_precision_loss)]
                        let n = out.dim() as f64;
                        out.as_slice().iter().map(|x| f64::from(*x).abs()).sum::<f64>() / n
                    };
                    card(ui, |ui| {
                        metric(ui, "dim", &out.dim().to_string());
                        if let Some(ms) = latency {
                            metric(ui, "latency", &format!("{ms:.3} ms"));
                        }
                        metric(
                            ui,
                            "bits differ vs input",
                            &format!("{bits_diff} / {}", out.dim()),
                        );
                        if let Some(h) = hd {
                            metric(ui, "hamming distance", &format!("{h:.4}"));
                        }
                        metric(ui, "mean |value|", &format!("{mean_mag:.4}"));
                        if let Some(s) = sim {
                            ui.add_space(theme::SPACE_SM);
                            cosine_similarity_bar(ui, s);
                        }
                        ui.add_space(theme::SPACE_SM);
                        if hypervector_heatmap_clickable_cmap(ui, &out, 100, 2.0, self.colormap) {
                            zoom_request = Some(ZoomTarget::Output);
                        }
                    });

                    // (Per-op contribution + inspector moved to the right panel.)
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
                        // Pinch/scroll-zoom support (#721).
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y + i.zoom_delta() * 100.0);
                        if scroll.abs() > 0.5 {
                            self.zoom_modal_scale =
                                (self.zoom_modal_scale * (1.0 + scroll / 800.0))
                                    .clamp(0.3, 8.0);
                        }
                        let base_cell = 6.0_f32;
                        let cols = (v.dim() as f32).sqrt().ceil() as usize;
                        let cols = cols.clamp(40, 160);
                        let cell = (base_cell * self.zoom_modal_scale).clamp(1.0, 30.0);
                        egui::ScrollArea::both()
                            .max_height(ui.available_height() - 50.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                hypervector_heatmap(ui, &v, cols, cell);
                            });
                        ui.add_space(theme::SPACE_SM);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "dim = {} · zoom {:.2}× (scroll to adjust)",
                                    v.dim(),
                                    self.zoom_modal_scale
                                ))
                                .color(theme::TEXT_MUTED)
                                .size(theme::SIZE_SMALL),
                            );
                            if ui.button("Reset zoom").clicked() {
                                self.zoom_modal_scale = 1.0;
                            }
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
        let label_resp = ui.label(
            egui::RichText::new(label)
                .color(theme::TEXT_MUTED)
                .size(theme::SIZE_SMALL),
        );
        // Show ⓘ info tooltip for AI/HDC terms.
        if let Some(info) = ai_term_info(label) {
            label_resp.on_hover_text(info);
        }
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

/// Returns a plain-English explanation for an AI/HDC term, if known.
fn ai_term_info(label: &str) -> Option<&'static str> {
    let l = label.trim().to_lowercase();
    if l.contains("cos_sim") {
        return Some(
            "Cosine similarity — angle between two hypervectors.\n\
             +1.0  = identical direction\n\
              0.0  = orthogonal (uncorrelated, the HDC norm for random pairs)\n\
             -1.0  = opposite direction (negate)",
        );
    }
    if l.contains("hamming") {
        return Some(
            "Hamming distance — fraction of components that differ between\n\
             two bipolar hypervectors. 0.0 = identical, 0.5 = uncorrelated,\n\
             1.0 = pure negation.",
        );
    }
    if l.contains("dim") {
        return Some(
            "Dimensionality — number of components in each hypervector.\n\
             HDC capacity grows with D. 10,000 is a typical default;\n\
             1,024 is FFT-friendly for HrrBind; 16,384 for max capacity.",
        );
    }
    if l.contains("bits differ") {
        return Some(
            "How many positions changed sign between input and bundled output.\n\
             For uncorrelated random hypervectors this hovers near dim/2.",
        );
    }
    if l.contains("mean |value|") {
        return Some(
            "Mean magnitude across components. Pure bipolar = 1.0; the FFT-based\n\
             HrrBind path can produce slightly off values due to re-quantization.",
        );
    }
    if l.contains("family") {
        return Some(
            "Architecture family. 'stack' = heterogeneous bundle of ops on a\n\
             shared hypervector. Stack-of-stacks recursion lands in PlausiDen-Stack.",
        );
    }
    if l.contains("ops") || l == "op" || l.contains("operation") {
        return Some(
            "Number of operations in this Stack. Each op transforms the input\n\
             hypervector independently; the bundle is their majority sum.",
        );
    }
    if l.contains("latency") {
        return Some("Wall-clock time to run one forward pass through this Stack.")
    }
    if l.contains("forwards") {
        return Some("How many times you've run the Stack since launch.")
    }
    None
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
    // Animated hover: brighter fill + bolder stroke on hover.
    let resp = ui.add(
        egui::Button::new(
            egui::RichText::new(text)
                .size(theme::SIZE_SMALL)
                .color(accent)
                .strong(),
        )
        .fill(accent.gamma_multiply(0.18))
        .stroke(egui::Stroke::new(1.0, accent))
        .rounding(egui::Rounding::same(theme::RADIUS_SM)),
    );
    if resp.hovered() {
        // Subtle highlight ring around the button to "lift" it visually.
        ui.painter().rect_stroke(
            resp.rect.expand(1.5),
            egui::Rounding::same(theme::RADIUS_SM),
            egui::Stroke::new(1.0, accent.gamma_multiply(0.6)),
        );
    }
    resp
}

/// Ease in-out cubic, t in [0, 1] → eased value in [0, 1].
#[allow(dead_code)]
fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let s = 2.0 * t - 2.0;
        1.0 + s * s * s / 2.0
    }
}

#[derive(Default)]
struct OpChipAction {
    remove: bool,
    reseed: bool,
    start_drag: bool,
    drop_here: bool,
    /// Replace this op with a different kind.
    convert_to: Option<&'static str>,
    /// Duplicate this op directly after itself.
    duplicate: bool,
    /// Move up / down in the stack.
    move_up: bool,
    move_down: bool,
}

/// Op chip with drag-and-drop reorder support.
/// `is_being_dragged`: this chip is the drag source (visual feedback).
/// `drag_active`: SOME chip is being dragged (controls drop-target hover state).
/// Return the spotlight rect for a given walkthrough step.
/// Coordinates are heuristic guesses keyed to the panel layout — each step
/// targets the region the text talks about.
fn walkthrough_spotlight_rect(step: usize, screen: egui::Rect) -> egui::Rect {
    // Defaults to the centre.
    match step {
        // Step 0: Welcome — full window.
        0 => screen,
        // Step 1: Templates list (left panel, top).
        1 => egui::Rect::from_min_size(
            egui::pos2(screen.min.x + 44.0, screen.min.y + 56.0),
            egui::vec2(320.0, 240.0),
        ),
        // Step 2: Run forward button (central panel, mid).
        2 => egui::Rect::from_min_size(
            egui::pos2(screen.center().x - 200.0, screen.center().y - 30.0),
            egui::vec2(400.0, 60.0),
        ),
        // Step 3: Operations sidebar.
        3 => egui::Rect::from_min_size(
            egui::pos2(screen.min.x + 44.0, screen.center().y),
            egui::vec2(320.0, screen.height() * 0.4),
        ),
        // Step 4: Per-op contribution / inspector (right panel).
        4 => egui::Rect::from_min_size(
            egui::pos2(screen.max.x - 360.0, screen.center().y - 100.0),
            egui::vec2(340.0, 300.0),
        ),
        // Step 5: 3D arch graph (central panel, bottom).
        5 => egui::Rect::from_min_size(
            egui::pos2(screen.center().x - 280.0, screen.max.y - 380.0),
            egui::vec2(560.0, 340.0),
        ),
        // Step 6: Live mode + objectives — right panel objective card.
        6 => egui::Rect::from_min_size(
            egui::pos2(screen.max.x - 360.0, screen.min.y + 240.0),
            egui::vec2(340.0, 180.0),
        ),
        // Step 7: Save/Load buttons + console — top right hero region.
        7 => egui::Rect::from_min_size(
            egui::pos2(screen.max.x - 320.0, screen.min.y),
            egui::vec2(320.0, 56.0),
        ),
        // Step 8: Help button.
        _ => egui::Rect::from_min_size(
            egui::pos2(screen.max.x - 90.0, screen.min.y + 12.0),
            egui::vec2(78.0, 32.0),
        ),
    }
}

fn op_chip_actions_with_drag(
    ui: &mut egui::Ui,
    idx: usize,
    tag: &str,
    is_being_dragged: bool,
    drag_active: bool,
    key_preview: Option<&Hypervector>,
    halo: Option<f32>,
) -> OpChipAction {
    let mut action = OpChipAction::default();
    let bg = theme::op_color(tag).gamma_multiply(if is_being_dragged { 0.6 } else { 0.25 });
    let fg = theme::op_color(tag);
    let frame = egui::Frame::none()
        .fill(bg)
        .stroke(egui::Stroke::new(
            if is_being_dragged { 2.0 } else { 1.0 },
            fg,
        ))
        .rounding(egui::Rounding::same(theme::RADIUS_PILL))
        .inner_margin(egui::Margin::symmetric(theme::SPACE_MD, theme::SPACE_SM));

    let resp = frame
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("⋮ {idx:02}"))
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
                    if matches!(tag, "dense" | "hrr_bind" | "permute") {
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
        })
        .response;
    // Diff halo — recent-mutation pulse.
    if let Some(intensity) = halo {
        let halo_col = egui::Color32::from_rgba_unmultiplied(
            0xE0,
            0xB1,
            0x5B,
            (intensity * 200.0) as u8,
        );
        ui.painter().rect_stroke(
            resp.rect.expand(3.0),
            egui::Rounding::same(theme::RADIUS_PILL),
            egui::Stroke::new(2.5, halo_col),
        );
    }

    // Whole-row drag detection.
    let row_resp = ui.interact(resp.rect, egui::Id::new(("op_chip_drag", idx)), egui::Sense::drag());
    if row_resp.drag_started() {
        action.start_drag = true;
    }
    if drag_active && row_resp.hovered() {
        action.drop_here = true;
        // Visual feedback: thick stroke on the drop target.
        ui.painter().rect_stroke(
            resp.rect,
            egui::Rounding::same(theme::RADIUS_PILL),
            egui::Stroke::new(2.0, theme::ACCENT_PURPLE),
        );
    }
    // Right-click → context menu (#708).
    row_resp.context_menu(|ui| {
        ui.set_min_width(180.0);
        ui.label(
            egui::RichText::new(format!("Op [{idx}] · {tag}"))
                .size(theme::SIZE_SMALL)
                .color(theme::ACCENT_BLUE)
                .strong(),
        );
        ui.separator();
        if ui.button("⟳  Reseed key").clicked() {
            action.reseed = true;
            ui.close_menu();
        }
        if ui.button("⎘  Duplicate").clicked() {
            action.duplicate = true;
            ui.close_menu();
        }
        if ui.button("↑  Move up").clicked() {
            action.move_up = true;
            ui.close_menu();
        }
        if ui.button("↓  Move down").clicked() {
            action.move_down = true;
            ui.close_menu();
        }
        ui.separator();
        ui.label(
            egui::RichText::new("Convert to:")
                .size(theme::SIZE_TINY)
                .color(theme::TEXT_MUTED),
        );
        for new_kind in ["identity", "dense", "hrr_bind", "permute", "negate"] {
            if new_kind == tag {
                continue;
            }
            if ui.button(format!("→  {new_kind}")).clicked() {
                action.convert_to = Some(new_kind);
                ui.close_menu();
            }
        }
        ui.separator();
        if ui
            .add(
                egui::Button::new(
                    egui::RichText::new("×  Remove")
                        .color(egui::Color32::from_rgb(0xE0, 0x6A, 0x5B)),
                ),
            )
            .clicked()
        {
            action.remove = true;
            ui.close_menu();
        }
    });

    // Hover → show key preview popup for keyed ops (#712).
    if let Some(key) = key_preview {
        if row_resp.hovered() || resp.hovered() {
            egui::show_tooltip_for(
                ui.ctx(),
                egui::LayerId::new(
                    egui::Order::Tooltip,
                    egui::Id::new(("op_chip_tip", idx)),
                ),
                egui::Id::new(("op_chip_tip", idx)),
                &resp.rect,
                |ui| {
                    ui.set_max_width(220.0);
                    ui.label(
                        egui::RichText::new(format!("[{idx}] {tag} key"))
                            .size(theme::SIZE_SMALL)
                            .color(theme::ACCENT_BLUE)
                            .strong(),
                    );
                    hypervector_heatmap(ui, key, 32, 4.0);
                    ui.label(
                        egui::RichText::new(format!("D = {}", key.dim()))
                            .size(theme::SIZE_TINY)
                            .color(theme::TEXT_MUTED),
                    );
                },
            );
        }
    }
    action
}

#[allow(dead_code)]
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
                    // Reseed meaningful for keyed ops and permute.
                    if matches!(tag, "dense" | "hrr_bind" | "permute") {
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

/// Clickable + colormap variant.
fn hypervector_heatmap_clickable_cmap(
    ui: &mut egui::Ui,
    v: &Hypervector,
    cols: usize,
    cell: f32,
    cmap: Colormap,
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
        let colour = cmap.map(data[i]);
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell - 1.0, cell - 1.0)),
            0.0,
            colour,
        );
    }
    if let Some(pos) = response.hover_pos() {
        let local = pos - rect.min;
        let c = (local.x / cell).floor() as i64;
        let r = (local.y / cell).floor() as i64;
        if c >= 0 && r >= 0 && (c as usize) < cols && (r as usize) < rows {
            let idx = (r as usize) * cols + (c as usize);
            if idx < dim {
                response.clone().on_hover_text_at_pointer(format!(
                    "index {idx}  →  {:+}  ·  click to zoom",
                    data[idx]
                ));
            }
        }
    }
    response.clicked()
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
    hypervector_heatmap_with_cmap(ui, v, cols, cell, Colormap::Bipolar);
}

fn hypervector_heatmap_with_cmap(
    ui: &mut egui::Ui,
    v: &Hypervector,
    cols: usize,
    cell: f32,
    cmap: Colormap,
) {
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
        let colour = cmap.map(data[i]);
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

/// Draw the Stack as a 3D-perspective-projected flow graph.
/// INPUT (left) → op nodes spread on a YZ-disc → BUNDLE → OUTPUT (right).
/// User can drag horizontally to rotate the yaw.
/// Returns (clicked_op_idx, new_yaw_after_drag).
#[derive(Debug, Clone, Copy)]
enum ArchToolAction {
    Reset,
    ToggleAutorotate,
    ZoomIn,
    ZoomOut,
}

fn architecture_graph_3d(
    ui: &mut egui::Ui,
    op_tags: &[String],
    selected: Option<usize>,
    yaw_in: f32,
    pitch_in: f32,
    roll_in: f32,
    zoom: f32,
    autorotate: bool,
    particle_phase: Option<f32>,
    contributions: Option<&[f64]>,
    click_flash: Option<(usize, f32)>,
) -> (Option<usize>, f32, f32, f32, Option<ArchToolAction>) {
    let n_ops = op_tags.len();
    let height = 360.0_f32;
    let width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    // Drag: horizontal = yaw, vertical = pitch.
    // Shift+drag horizontal = roll.
    let mut yaw = yaw_in;
    let mut pitch = pitch_in;
    let mut roll = roll_in;
    let mut zoom_adjust: f32 = 0.0;
    if response.dragged() {
        let mods = ui.input(|i| i.modifiers);
        if mods.shift {
            roll += response.drag_delta().x * 0.02;
        } else {
            yaw += response.drag_delta().x * 0.02;
            pitch += response.drag_delta().y * 0.015;
            pitch = pitch.clamp(-1.2, 1.2);
        }
    }
    // Scroll-wheel zoom when cursor is over the viewport.
    if response.hovered() {
        let (scroll_y, zoom_delta) =
            ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if scroll_y.abs() > 0.5 || (zoom_delta - 1.0).abs() > 0.01 {
            zoom_adjust = scroll_y * 0.005 + (zoom_delta - 1.0) * 0.5;
        }
    }
    // Numpad / digit axis-aligned snaps (only when viewport hovered).
    if response.hovered() {
        ui.input(|i| {
            if i.key_pressed(egui::Key::F) {
                // F = focus reset (camera back to default but keep zoom).
                yaw = 0.6;
                pitch = 0.15;
                roll = 0.0;
            }
        });
    }

    // 3D camera: input at x=-1, bundle at x=+0.8, output at x=+1.2.
    // Ops spread around a unit circle in the YZ plane at x=0.
    let cx = rect.center().x;
    let cy = rect.center().y;
    let scale = ((rect.width().min(rect.height()) * 0.32).min(180.0)) * zoom;

    // Perspective project a 3D point (x,y,z) → 2D screen point.
    // Apply yaw (Y axis) then pitch (X axis) then perspective.
    let project = |x: f32, y: f32, z: f32| -> (egui::Pos2, f32) {
        // 1) Yaw around Y.
        let (sy, cyaw) = yaw.sin_cos();
        let xr = x * cyaw + z * sy;
        let zr1 = -x * sy + z * cyaw;
        // 2) Pitch around X.
        let (sp, cp) = pitch.sin_cos();
        let yr1 = y * cp - zr1 * sp;
        let zr = y * sp + zr1 * cp;
        // 3) Roll around Z (camera Z-axis).
        let (sr, cr) = roll.sin_cos();
        let xr2 = xr * cr - yr1 * sr;
        let yr2 = xr * sr + yr1 * cr;
        // 4) Perspective.
        let cam_z = -3.0_f32;
        let depth = zr - cam_z;
        let persp = 2.4 / depth.max(0.1);
        let sx = cx + xr2 * scale * persp;
        let sy_screen = cy + yr2 * scale * persp;
        (egui::pos2(sx, sy_screen), depth)
    };

    // Build all node positions first so we can sort by depth.
    enum Node<'a> {
        Input,
        Bundle,
        Output,
        Op(usize, &'a str),
    }
    let mut nodes: Vec<(Node, egui::Pos2, f32)> = Vec::new();
    let (p_in, d_in) = project(-1.0, 0.0, 0.0);
    nodes.push((Node::Input, p_in, d_in));
    let (p_bundle, d_bundle) = project(0.8, 0.0, 0.0);
    nodes.push((Node::Bundle, p_bundle, d_bundle));
    let (p_out, d_out) = project(1.2, 0.0, 0.0);
    nodes.push((Node::Output, p_out, d_out));

    let mut op_screen: Vec<(egui::Pos2, f32)> = Vec::with_capacity(n_ops);
    for (i, tag) in op_tags.iter().enumerate() {
        // Spread on a circle in the YZ plane.
        let angle = if n_ops == 1 {
            0.0
        } else {
            std::f32::consts::TAU * (i as f32) / (n_ops as f32)
        };
        let (sa, ca) = angle.sin_cos();
        let y = sa * 0.55;
        let z = ca * 0.55;
        let (p, d) = project(0.0, y, z);
        op_screen.push((p, d));
        nodes.push((Node::Op(i, tag.as_str()), p, d));
    }

    // Connectors drawn first, back-to-front.
    for (i, (op_pos, op_d)) in op_screen.iter().enumerate() {
        // Connector colour: muted unless selected.
        let is_selected = selected == Some(i);
        // Line weight by per-op contribution magnitude (when trace exists).
        let contrib_mag = contributions
            .and_then(|c| c.get(i))
            .map(|s| s.abs().min(1.0) as f32)
            .unwrap_or(0.0);
        let colour = if is_selected {
            theme::op_color(&op_tags[i])
        } else if contrib_mag > 0.1 {
            // Blend op color with text muted by contribution magnitude.
            let op_c = theme::op_color(&op_tags[i]);
            egui::Color32::from_rgba_unmultiplied(
                ((op_c.r() as f32 * contrib_mag
                    + theme::TEXT_MUTED.r() as f32 * (1.0 - contrib_mag))
                    as u8),
                ((op_c.g() as f32 * contrib_mag
                    + theme::TEXT_MUTED.g() as f32 * (1.0 - contrib_mag))
                    as u8),
                ((op_c.b() as f32 * contrib_mag
                    + theme::TEXT_MUTED.b() as f32 * (1.0 - contrib_mag))
                    as u8),
                255,
            )
        } else {
            theme::TEXT_MUTED
        };
        let stroke_w = if is_selected {
            1.8
        } else {
            1.0 + contrib_mag * 2.0
        };
        // INPUT → op
        painter.line_segment([p_in, *op_pos], egui::Stroke::new(stroke_w, colour));
        // op → BUNDLE
        painter.line_segment([*op_pos, p_bundle], egui::Stroke::new(stroke_w, colour));
        let _ = op_d;

        // Particle animation: a glowing dot travels along the path.
        if let Some(phase) = particle_phase {
            let op_colour = theme::op_color(&op_tags[i]);
            // Phase 0..0.5 → in→op; 0.5..1.0 → op→bundle.
            let (from, to, sub_phase) = if phase < 0.5 {
                (p_in, *op_pos, phase * 2.0)
            } else {
                (*op_pos, p_bundle, (phase - 0.5) * 2.0)
            };
            let particle_pos = egui::pos2(
                from.x + (to.x - from.x) * sub_phase,
                from.y + (to.y - from.y) * sub_phase,
            );
            // Glow halo.
            painter.circle_filled(particle_pos, 7.0, op_colour.gamma_multiply(0.3));
            painter.circle_filled(particle_pos, 4.5, op_colour);
            painter.circle_filled(particle_pos, 2.0, egui::Color32::WHITE);
        }
    }
    painter.line_segment(
        [p_bundle, p_out],
        egui::Stroke::new(1.5, theme::TEXT_MUTED),
    );
    // Particle from bundle to output for the second half of the animation.
    if let Some(phase) = particle_phase {
        if phase > 0.5 {
            let sub_phase = (phase - 0.5) * 2.0;
            let particle_pos = egui::pos2(
                p_bundle.x + (p_out.x - p_bundle.x) * sub_phase,
                p_bundle.y + (p_out.y - p_bundle.y) * sub_phase,
            );
            painter.circle_filled(particle_pos, 7.0, theme::ACCENT_PURPLE.gamma_multiply(0.3));
            painter.circle_filled(particle_pos, 4.5, theme::ACCENT_PURPLE);
            painter.circle_filled(particle_pos, 2.0, egui::Color32::WHITE);
        }
    }

    // Sort all nodes back-to-front by depth (greater depth = further away,
    // drawn first).
    nodes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut clicked: Option<usize> = None;
    let click_pos = if response.clicked() {
        response.interact_pointer_pos()
    } else {
        None
    };
    // Right-click pos for 3D context menu (handled by caller via the
    // selected_op/right-click flow — wire later for full context menus).
    let _right_click_pos: Option<egui::Pos2> =
        if response.secondary_clicked() {
            response.interact_pointer_pos()
        } else {
            None
        };
    // Hover position for node-hover highlight + tooltip.
    let hover_pos = response.hover_pos();

    for (kind, pos, depth) in &nodes {
        let size_mul = (3.0 / depth.max(0.1)).clamp(0.5, 1.6);
        match kind {
            Node::Input => {
                let r = 24.0 * size_mul;
                shaded_node(&painter, *pos, r, theme::ACCENT_BLUE, "INPUT", size_mul);
            }
            Node::Bundle => {
                let r = 24.0 * size_mul;
                shaded_node(&painter, *pos, r, theme::ACCENT_PURPLE, "BUNDLE", size_mul);
            }
            Node::Output => {
                let r = 24.0 * size_mul;
                shaded_node(&painter, *pos, r, theme::ACCENT_MID, "OUT", size_mul);
            }
            Node::Op(i, tag) => {
                let colour = theme::op_color(tag);
                let is_selected = selected == Some(*i);
                let r = 22.0 * size_mul;
                if is_selected {
                    // Selected: solid colour with subtle outer glow.
                    painter.circle_filled(*pos, r + 4.0, colour.gamma_multiply(0.25));
                    shaded_node(&painter, *pos, r, colour, &format!("[{i}]"), size_mul);
                } else {
                    // Dim: radial-gradient on dimmed colour.
                    let dim_col = colour.gamma_multiply(0.7);
                    shaded_node(&painter, *pos, r, dim_col, &format!("[{i}]"), size_mul);
                }
                // Symbolic glyph in the upper-right of the node (#756).
                draw_op_glyph(&painter, *pos, r, tag, size_mul);
                // Click flash halo — bright white pulse on freshly-clicked node.
                if let Some((flash_idx, intensity)) = click_flash {
                    if flash_idx == *i {
                        painter.circle_stroke(
                            *pos,
                            r + 6.0 + (1.0 - intensity) * 12.0,
                            egui::Stroke::new(
                                3.0 * intensity,
                                egui::Color32::from_rgba_unmultiplied(
                                    255,
                                    255,
                                    255,
                                    (intensity * 220.0) as u8,
                                ),
                            ),
                        );
                    }
                }
                painter.circle_stroke(
                    *pos,
                    r,
                    egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, colour),
                );

                if let Some(cp) = click_pos {
                    let dx = cp.x - pos.x;
                    let dy = cp.y - pos.y;
                    if (dx * dx + dy * dy).sqrt() <= r + 4.0 {
                        clicked = Some(*i);
                    }
                }
                // Right-click on a 3D node → set as selected, then the
                // existing inline-editor card handles the action menu.
                if let Some(rcp) = _right_click_pos {
                    let dx = rcp.x - pos.x;
                    let dy = rcp.y - pos.y;
                    if (dx * dx + dy * dy).sqrt() <= r + 4.0 {
                        clicked = Some(*i);
                    }
                }
                // Hover state — bright ring + tooltip.
                if let Some(hp) = hover_pos {
                    let dx = hp.x - pos.x;
                    let dy = hp.y - pos.y;
                    if (dx * dx + dy * dy).sqrt() <= r + 4.0 {
                        painter.circle_stroke(
                            *pos,
                            r + 3.0,
                            egui::Stroke::new(
                                1.5,
                                egui::Color32::from_white_alpha(180),
                            ),
                        );
                    }
                }
            }
        }
    }

    // Hovering toolbar (top-left of viewport).
    let toolbar_origin = egui::pos2(rect.min.x + 8.0, rect.min.y + 8.0);
    let mut tool_action: Option<ArchToolAction> = None;
    let mut x = toolbar_origin.x;
    let mut tool_button = |label: &str, hover: &str, active: bool| -> bool {
        let btn_size = egui::vec2(30.0, 26.0);
        let btn_rect = egui::Rect::from_min_size(egui::pos2(x, toolbar_origin.y), btn_size);
        let resp = ui.interact(btn_rect, egui::Id::new(("arch_tool", label)), egui::Sense::click());
        let fill = if active {
            theme::ACCENT_MID
        } else if resp.hovered() {
            egui::Color32::from_rgb(0x2A, 0x32, 0x44)
        } else {
            egui::Color32::from_rgb(0x15, 0x1A, 0x26)
        };
        let stroke_col = if active {
            theme::ACCENT_MID
        } else {
            egui::Color32::from_rgb(0x2A, 0x32, 0x44)
        };
        let p = ui.painter();
        p.rect_filled(btn_rect, theme::RADIUS_SM, fill);
        p.rect_stroke(btn_rect, theme::RADIUS_SM, egui::Stroke::new(1.0, stroke_col));
        p.text(
            btn_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(14.0),
            egui::Color32::WHITE,
        );
        if resp.hovered() {
            resp.on_hover_text(hover);
        }
        x += btn_size.x + 4.0;
        ui.input(|i| i.pointer.any_click() && i.pointer.interact_pos().map(|p| btn_rect.contains(p)).unwrap_or(false))
    };
    if tool_button("⟲", "Reset view (yaw/pitch/zoom)", false) {
        tool_action = Some(ArchToolAction::Reset);
    }
    if tool_button("🔄", "Toggle auto-rotate", autorotate) {
        tool_action = Some(ArchToolAction::ToggleAutorotate);
    }
    if tool_button("+", "Zoom in", false) {
        tool_action = Some(ArchToolAction::ZoomIn);
    }
    if tool_button("−", "Zoom out", false) {
        tool_action = Some(ArchToolAction::ZoomOut);
    }

    // Hint text — updated for new shortcuts.
    painter.text(
        egui::pos2(rect.min.x + 6.0, rect.max.y - 6.0),
        egui::Align2::LEFT_BOTTOM,
        "drag = yaw/pitch · shift+drag = roll · scroll = zoom · F = reset rotation · click op = select",
        egui::FontId::proportional(theme::SIZE_TINY),
        theme::TEXT_DIM,
    );

    // Apply the scroll-wheel zoom adjustment via the action channel — the
    // caller already routes zoom through ArchToolAction. Translate here.
    if zoom_adjust.abs() > 0.001 && tool_action.is_none() {
        tool_action = Some(if zoom_adjust > 0.0 {
            ArchToolAction::ZoomIn
        } else {
            ArchToolAction::ZoomOut
        });
    }

    (clicked, yaw, pitch, roll, tool_action)
}

/// Draw a small symbolic glyph for an op kind. Reinforces the "creature
/// has a personality" feel beyond the index/tag label (#756).
fn draw_op_glyph(
    painter: &egui::Painter,
    pos: egui::Pos2,
    r: f32,
    tag: &str,
    size_mul: f32,
) {
    let glyph_r = r * 0.55;
    let pos = egui::pos2(pos.x, pos.y + r * 0.55);
    let stroke = egui::Stroke::new(1.4 * size_mul, egui::Color32::WHITE);
    match tag {
        "identity" => {
            // Horizontal arrow → (passthrough)
            painter.line_segment(
                [
                    egui::pos2(pos.x - glyph_r * 0.6, pos.y),
                    egui::pos2(pos.x + glyph_r * 0.6, pos.y),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(pos.x + glyph_r * 0.3, pos.y - glyph_r * 0.3),
                    egui::pos2(pos.x + glyph_r * 0.6, pos.y),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(pos.x + glyph_r * 0.3, pos.y + glyph_r * 0.3),
                    egui::pos2(pos.x + glyph_r * 0.6, pos.y),
                ],
                stroke,
            );
        }
        "dense" => {
            // Diamond/crystal facets
            let pts = [
                egui::pos2(pos.x, pos.y - glyph_r * 0.5),
                egui::pos2(pos.x + glyph_r * 0.5, pos.y),
                egui::pos2(pos.x, pos.y + glyph_r * 0.5),
                egui::pos2(pos.x - glyph_r * 0.5, pos.y),
            ];
            for i in 0..4 {
                painter.line_segment([pts[i], pts[(i + 1) % 4]], stroke);
            }
            painter.line_segment([pts[0], pts[2]], stroke);
        }
        "hrr_bind" => {
            // Spiral approximation (3 arcs)
            let n = 24;
            let mut prev = pos;
            for k in 1..=n {
                let t = (k as f32) / (n as f32);
                let theta = t * std::f32::consts::TAU * 1.5;
                let rr = glyph_r * 0.5 * (1.0 - t * 0.5);
                let p = egui::pos2(pos.x + theta.cos() * rr, pos.y + theta.sin() * rr);
                if k > 1 {
                    painter.line_segment([prev, p], stroke);
                }
                prev = p;
            }
        }
        "permute" => {
            // Curved arrow loop
            let n = 16;
            let mut prev = egui::pos2(pos.x + glyph_r * 0.5, pos.y);
            for k in 1..=n {
                let t = (k as f32) / (n as f32);
                let theta = t * std::f32::consts::TAU * 0.8;
                let p = egui::pos2(
                    pos.x + glyph_r * 0.5 * theta.cos(),
                    pos.y + glyph_r * 0.5 * theta.sin(),
                );
                painter.line_segment([prev, p], stroke);
                prev = p;
            }
            // arrowhead
            painter.line_segment(
                [
                    prev,
                    egui::pos2(prev.x + glyph_r * 0.18, prev.y - glyph_r * 0.18),
                ],
                stroke,
            );
        }
        "negate" => {
            // Sine-wave inverter
            let n = 16;
            let mut prev = egui::pos2(pos.x - glyph_r * 0.6, pos.y);
            for k in 1..=n {
                let t = (k as f32) / (n as f32);
                let x = pos.x - glyph_r * 0.6 + t * glyph_r * 1.2;
                let y = pos.y + (t * std::f32::consts::TAU).sin() * glyph_r * 0.35;
                let p = egui::pos2(x, y);
                painter.line_segment([prev, p], stroke);
                prev = p;
            }
        }
        _ => {}
    }
}

/// Paint a node with a fake radial gradient (3D-sphere illusion).
/// Outer disc is darker; inner discs are progressively brighter and
/// offset up-left to suggest a light source at upper-left.
fn shaded_node(
    painter: &egui::Painter,
    pos: egui::Pos2,
    r: f32,
    colour: egui::Color32,
    text: &str,
    size_mul: f32,
) {
    // Outer (rim, darker).
    painter.circle_filled(pos, r, colour.gamma_multiply(0.7));
    // Mid (base colour).
    painter.circle_filled(pos, r * 0.85, colour);
    // Highlight 1.
    painter.circle_filled(
        egui::pos2(pos.x - r * 0.2, pos.y - r * 0.2),
        r * 0.55,
        colour.gamma_multiply(1.25),
    );
    // Highlight 2 (small bright spot).
    painter.circle_filled(
        egui::pos2(pos.x - r * 0.35, pos.y - r * 0.35),
        r * 0.18,
        egui::Color32::from_white_alpha(180),
    );
    // Label.
    painter.text(
        pos,
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(theme::SIZE_TINY * size_mul),
        egui::Color32::WHITE,
    );
}

/// 2D flow graph (kept for fallback / unit tests).
#[allow(dead_code)]
fn architecture_graph(
    ui: &mut egui::Ui,
    op_tags: &[String],
    selected: Option<usize>,
) -> Option<usize> {
    let n_ops = op_tags.len().max(1);
    let row_h = 220.0_f32.min(80.0 + n_ops as f32 * 26.0);
    let width = ui.available_width().min(720.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, row_h), egui::Sense::click());
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
        return None;
    }

    let mut clicked_idx: Option<usize> = None;
    let click_pos = if response.clicked() {
        response.interact_pointer_pos()
    } else {
        None
    };

    for (i, tag) in op_tags.iter().enumerate() {
        let y = start_y + i as f32 * (chip_h + 4.0);
        let chip_rect = egui::Rect::from_center_size(
            egui::pos2(middle_x, y + chip_h * 0.5),
            egui::vec2(chip_w, chip_h),
        );
        let colour = theme::op_color(tag);
        let is_selected = selected == Some(i);
        let fill = if is_selected {
            colour
        } else {
            colour.gamma_multiply(0.25)
        };
        let text_colour = if is_selected {
            egui::Color32::WHITE
        } else {
            colour
        };
        painter.rect_filled(chip_rect, theme::RADIUS_PILL, fill);
        painter.rect_stroke(
            chip_rect,
            theme::RADIUS_PILL,
            egui::Stroke::new(if is_selected { 2.0 } else { 1.0 }, colour),
        );
        painter.text(
            chip_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("[{i}] {tag}"),
            egui::FontId::proportional(theme::SIZE_TINY),
            text_colour,
        );

        // Input → chip connector.
        let connector_colour = if is_selected {
            colour
        } else {
            theme::TEXT_MUTED
        };
        let connector_stroke = if is_selected { 1.6 } else { 1.0 };
        painter.line_segment(
            [
                egui::pos2(in_pos.x + 28.0, mid_y),
                egui::pos2(chip_rect.min.x, chip_rect.center().y),
            ],
            egui::Stroke::new(connector_stroke, connector_colour),
        );
        painter.line_segment(
            [
                egui::pos2(chip_rect.max.x, chip_rect.center().y),
                egui::pos2(bundle_pos.x - 28.0, mid_y),
            ],
            egui::Stroke::new(connector_stroke, connector_colour),
        );

        // Click hit-test.
        if let Some(p) = click_pos {
            if chip_rect.contains(p) {
                clicked_idx = Some(i);
            }
        }
    }

    clicked_idx
}

/// Plot training loss curve. Loss in [0, 2] approximately; auto-scale.
fn loss_sparkline(ui: &mut egui::Ui, values: impl Iterator<Item = f64>, height: f32) {
    let vs: Vec<f64> = values.collect();
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    if vs.is_empty() {
        return;
    }
    let max_v = vs.iter().copied().fold(0.0_f64, f64::max).max(0.1);
    let min_v = vs.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    let range = (max_v - min_v).max(0.01);
    let step = if vs.len() > 1 {
        rect.width() / (vs.len() - 1) as f32
    } else {
        rect.width()
    };
    let to_y = |v: f64| -> f32 {
        let t = ((v - min_v) / range) as f32;
        rect.max.y - t * rect.height()
    };
    let mut prev = egui::pos2(rect.min.x, to_y(vs[0]));
    for (i, v) in vs.iter().enumerate().skip(1) {
        let p = egui::pos2(rect.min.x + step * i as f32, to_y(*v));
        painter.line_segment([prev, p], egui::Stroke::new(1.5, theme::ACCENT_BLUE));
        prev = p;
    }
    // Mark min reached.
    let min_y = to_y(min_v);
    painter.line_segment(
        [
            egui::pos2(rect.min.x, min_y),
            egui::pos2(rect.max.x, min_y),
        ],
        egui::Stroke::new(0.5, theme::ACCENT_PURPLE),
    );
    painter.text(
        egui::pos2(rect.max.x - 4.0, rect.min.y + 8.0),
        egui::Align2::RIGHT_TOP,
        format!("min {min_v:.3} · max {max_v:.3}"),
        egui::FontId::proportional(theme::SIZE_TINY),
        theme::TEXT_MUTED,
    );
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
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([960.0, 600.0])
            .with_maximized(true)
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
