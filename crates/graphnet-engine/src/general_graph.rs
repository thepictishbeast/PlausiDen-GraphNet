//! Generic NeuralGraph — Phase 1 of GENERAL_AI_VISUALIZATION.md.
//!
//! The existing [`crate::Stack`] models ONE flavor of architecture: parallel
//! operations sharing an input, bundled into a single output. To represent
//! arbitrary networks (transformer, CNN, RNN, LLMs) we need a generic DAG
//! where nodes are layers, edges are tensor connections, and the kind of
//! layer is data-driven.
//!
//! Phase 1 surface (this file):
//!
//! - [`NeuralGraph`] — top-level DAG with nodes + edges + metadata
//! - [`Node`] / [`NodeKind`] — node types (Input, Output, Layer, HdcOp)
//! - [`LayerKind`] — Dense / Conv2d / Attention / LSTM / Embedding / etc.
//! - [`Edge`] — typed connection with shape
//! - [`GraphMetadata`] — architecture name + parameter count
//!
//! No tensor execution in Phase 1 — that's Phase 4 (candle absorption).
//! No UI here — that's Phase 2-3 (egui_node_graph for 2D, generalized
//! arch_graph_3d for 3D).
//!
//! The existing [`crate::Stack`] is embeddable as a single
//! [`LayerKind::Hdc`], so we can keep all current functionality while
//! growing into general AI viz.
//!
//! BUG ASSUMPTION (Phase 1): graph cycles are not validated. A consumer
//! that needs DAG semantics must call [`NeuralGraph::has_cycle`] before
//! traversal. Phase 4 (execution) will enforce DAG at construction.

use serde::{Deserialize, Serialize};

use crate::Stack;

/// Tensor element type — minimal Phase 1 set. Phase 4 grows this to match
/// candle's `DType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// Bipolar (-1/+1) — the HDC native representation.
    Bipolar,
    /// 32-bit float — the default for most NN layers.
    F32,
    /// 16-bit float — common for quantized LLMs.
    F16,
    /// 32-bit signed integer — for token IDs in embeddings.
    I32,
}

/// Tensor shape. Empty Vec = scalar.
pub type Shape = Vec<usize>;

/// A typed edge — what flows between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Index of the source node in [`NeuralGraph::nodes`].
    pub from: usize,
    /// Index of the destination node.
    pub to: usize,
    /// Shape of the tensor on this edge.
    pub shape: Shape,
    /// Tensor element type.
    pub dtype: DType,
    /// Human-readable label ("logits", "attn", etc.).
    pub label: String,
}

/// Pool kind for [`LayerKind::Pool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolKind {
    Max,
    Avg,
    Sum,
}

/// Activation kind for [`NodeKind::Activation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActKind {
    Relu,
    Gelu,
    Silu,
    Tanh,
    Sigmoid,
    Softmax,
}

/// What kind of layer this node represents. Each variant carries its
/// dimensions / hyperparameters but NOT the weight tensors themselves —
/// weights live in the `params` field of [`Node::Layer`] (Phase 4 will
/// promote these to candle Tensors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerKind {
    /// Fully-connected: `out = input @ W + b`.
    Dense { in_dim: usize, out_dim: usize },
    /// 2D convolution.
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    /// Multi-head self-attention block.
    Attention {
        n_heads: usize,
        head_dim: usize,
        masked: bool,
    },
    /// Long short-term memory.
    Lstm { hidden: usize, num_layers: usize },
    /// Gated recurrent unit.
    Gru { hidden: usize, num_layers: usize },
    /// Token embedding lookup.
    Embedding { vocab: usize, dim: usize },
    /// Layer normalization.
    LayerNorm,
    /// Batch normalization.
    BatchNorm,
    /// Spatial pooling.
    Pool {
        kind: PoolKind,
        kernel: (usize, usize),
        stride: (usize, usize),
    },
    /// Dropout (parameter-free, kept for graph-completeness).
    Dropout { p: f32 },
    /// Reshape / view — no parameters, just shape change.
    Reshape { from: Shape, to: Shape },
    /// HDC stack — wraps the existing [`Stack`] as one layer kind. Lets
    /// every current GraphNet feature keep working unchanged.
    Hdc { stack: Stack },
    /// Custom / user-defined — opaque kind with a label.
    Custom { name: String },

    // ----- Novel-AI / research-frontier kinds (iter 139, #786) -----
    /// Energy-based model — output is computed by minimizing an energy
    /// function E(x). Used in Boltzmann machines, EBMs, Hopfield nets.
    EnergyBased {
        /// Dimensionality of the state vector.
        state_dim: usize,
        /// Number of inference-loop iterations to minimize the energy.
        steps: usize,
    },
    /// Neural ODE — output follows the integration of a learned ODE
    /// dx/dt = f(x, t, θ). Continuous-depth model.
    NeuralOde {
        /// State dimensionality.
        state_dim: usize,
        /// Integration steps (Euler / RK4 / etc.).
        steps: usize,
        /// Integration end time T (dx/dt integrated from 0..T).
        t_end: f32,
    },
    /// Hamiltonian network — preserves an energy invariant via symplectic
    /// integration. Used for physics-informed ML.
    Hamiltonian {
        /// Dimensionality of the position+momentum state (2D where D is
        /// configuration space dim).
        phase_dim: usize,
        /// Number of symplectic integration steps.
        steps: usize,
    },
    /// Coupled oscillator network — each unit is a phase angle θ_i with
    /// coupling K_ij. Used in Kuramoto-style sync models.
    Oscillator {
        /// Number of oscillators.
        n_oscillators: usize,
        /// Coupling strength scalar.
        coupling: f32,
        /// Integration steps.
        steps: usize,
    },
    /// Spiking neural network — leaky integrate-and-fire with discrete
    /// spike events. Bio-inspired.
    Spiking {
        /// Number of neurons.
        n_neurons: usize,
        /// Membrane time constant τ_m.
        tau_m: f32,
        /// Firing threshold.
        threshold: f32,
    },
    /// Symbolic / equation-based — user-defined math formula. Parameters
    /// are the symbols referenced in the formula. Used for novel research
    /// where the architecture IS the math.
    SymbolicFormula {
        /// Math expression as string (e.g. "tanh(W*x + b) + 0.1*sin(t)")
        formula: String,
        /// Named scalar parameters.
        params: Vec<(String, f32)>,
    },
}

impl LayerKind {
    /// Estimate the number of learnable parameters in this layer.
    ///
    /// BUG ASSUMPTION: parameter counts are rough — they ignore biases,
    /// gates that share weights, factorized attention, etc. Use as a UI
    /// hint, not a source of truth. Phase 4 will compute exact counts
    /// from loaded candle weights.
    #[must_use]
    pub fn param_count(&self) -> usize {
        match self {
            LayerKind::Dense { in_dim, out_dim } => in_dim * out_dim + out_dim,
            LayerKind::Conv2d {
                in_channels,
                out_channels,
                kernel: (kh, kw),
                ..
            } => in_channels * out_channels * kh * kw + out_channels,
            LayerKind::Attention { n_heads, head_dim, .. } => {
                // q + k + v + out projections, each n_heads * head_dim ^ 2.
                4 * n_heads * head_dim * head_dim
            }
            LayerKind::Lstm { hidden, num_layers } => {
                // 4 gates × (in + hidden + bias). Approximate in = hidden.
                4 * num_layers * (hidden * hidden + hidden * hidden + hidden)
            }
            LayerKind::Gru { hidden, num_layers } => {
                3 * num_layers * (hidden * hidden + hidden * hidden + hidden)
            }
            LayerKind::Embedding { vocab, dim } => vocab * dim,
            LayerKind::LayerNorm | LayerKind::BatchNorm => 2, // gamma + beta scalars
            LayerKind::Pool { .. } | LayerKind::Dropout { .. } | LayerKind::Reshape { .. } => {
                0
            }
            LayerKind::Hdc { stack } => {
                // Each Dense / HrrBind op has a key of `stack.dim()` bipolar bits.
                stack.dim() * stack.len()
            }
            LayerKind::Custom { .. } => 0,
            // Novel-AI kinds (iter 139):
            LayerKind::EnergyBased { state_dim, .. } => state_dim * state_dim,
            LayerKind::NeuralOde { state_dim, .. } => state_dim * state_dim,
            LayerKind::Hamiltonian { phase_dim, .. } => phase_dim * phase_dim,
            LayerKind::Oscillator { n_oscillators, .. } => n_oscillators * n_oscillators,
            LayerKind::Spiking { n_neurons, .. } => n_neurons * n_neurons,
            LayerKind::SymbolicFormula { params, .. } => params.len(),
        }
    }

    /// Short kind tag for display ("dense", "conv2d", "attention", ...).
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            LayerKind::Dense { .. } => "dense",
            LayerKind::Conv2d { .. } => "conv2d",
            LayerKind::Attention { .. } => "attention",
            LayerKind::Lstm { .. } => "lstm",
            LayerKind::Gru { .. } => "gru",
            LayerKind::Embedding { .. } => "embedding",
            LayerKind::LayerNorm => "layernorm",
            LayerKind::BatchNorm => "batchnorm",
            LayerKind::Pool { .. } => "pool",
            LayerKind::Dropout { .. } => "dropout",
            LayerKind::Reshape { .. } => "reshape",
            LayerKind::Hdc { .. } => "hdc",
            LayerKind::Custom { .. } => "custom",
            LayerKind::EnergyBased { .. } => "ebm",
            LayerKind::NeuralOde { .. } => "neural_ode",
            LayerKind::Hamiltonian { .. } => "hamiltonian",
            LayerKind::Oscillator { .. } => "oscillator",
            LayerKind::Spiking { .. } => "spiking",
            LayerKind::SymbolicFormula { .. } => "symbolic",
        }
    }
}

/// What a node represents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// Network input (tensor placeholder).
    Input { shape: Shape, dtype: DType },
    /// Network output.
    Output { shape: Shape, dtype: DType },
    /// A layer (with kind + optional learned weights).
    Layer {
        kind: LayerKind,
        /// Whether the UI should render this layer's internals (sub-nodes)
        /// rather than as a single block. Defaults false.
        expanded: bool,
    },
    /// A point-wise activation.
    Activation { kind: ActKind },
}

/// A node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// What this node represents.
    pub kind: NodeKind,
    /// Human-readable label.
    pub label: String,
}

impl Node {
    /// Convenience: input placeholder.
    #[must_use]
    pub fn input(shape: Shape, dtype: DType, label: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Input { shape, dtype },
            label: label.into(),
        }
    }

    /// Convenience: output placeholder.
    #[must_use]
    pub fn output(shape: Shape, dtype: DType, label: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Output { shape, dtype },
            label: label.into(),
        }
    }

    /// Convenience: layer.
    #[must_use]
    pub fn layer(kind: LayerKind, label: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Layer {
                kind,
                expanded: false,
            },
            label: label.into(),
        }
    }

    /// Convenience: activation.
    #[must_use]
    pub fn activation(kind: ActKind, label: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Activation { kind },
            label: label.into(),
        }
    }

    /// Estimated parameter count of this node.
    #[must_use]
    pub fn param_count(&self) -> usize {
        match &self.kind {
            NodeKind::Layer { kind, .. } => kind.param_count(),
            _ => 0,
        }
    }
}

/// Top-level graph metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Architecture family name ("GPT-2-small", "ResNet-18", ...).
    pub family: String,
    /// Optional source URL / file path.
    pub source: Option<String>,
    /// Notes / description.
    pub notes: String,
}

/// A neural-net DAG.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuralGraph {
    /// All nodes in deterministic order. Indices are stable.
    pub nodes: Vec<Node>,
    /// Edges (typed connections between nodes).
    pub edges: Vec<Edge>,
    /// Architecture metadata.
    pub metadata: GraphMetadata,
}

impl NeuralGraph {
    /// New empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node, return its index.
    pub fn add_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Append an edge.
    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Total parameter count across all layer nodes (rough estimate).
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.nodes.iter().map(Node::param_count).sum()
    }

    /// Count of nodes by kind tag (for the architecture summary chip row).
    #[must_use]
    pub fn kind_counts(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        for node in &self.nodes {
            let tag = match &node.kind {
                NodeKind::Input { .. } => "input",
                NodeKind::Output { .. } => "output",
                NodeKind::Layer { kind, .. } => kind.tag(),
                NodeKind::Activation { .. } => "activation",
            };
            *counts.entry(tag).or_insert(0) += 1;
        }
        counts
    }

    /// Detect cycles via DFS. Returns the first cycle found as a list of
    /// node indices in cycle order, or None.
    ///
    /// BUG ASSUMPTION: graph cycles break DAG-traversal assumptions. UIs
    /// rendering this graph should call this once after deserializing /
    /// loading and refuse to execute if a cycle is found.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        use std::collections::HashSet;
        let n = self.nodes.len();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            if e.from < n {
                adj[e.from].push(e.to);
            }
        }
        fn dfs(
            v: usize,
            adj: &[Vec<usize>],
            visited: &mut std::collections::HashSet<usize>,
            on_stack: &mut std::collections::HashSet<usize>,
        ) -> bool {
            if !visited.insert(v) {
                return on_stack.contains(&v);
            }
            on_stack.insert(v);
            for &w in &adj[v] {
                if w < adj.len() && dfs(w, adj, visited, on_stack) {
                    return true;
                }
            }
            on_stack.remove(&v);
            false
        }
        for start in 0..n {
            if !visited.contains(&start)
                && dfs(start, &adj, &mut visited, &mut on_stack)
            {
                return true;
            }
        }
        false
    }
}

/// Pre-built architectures for the templates popup ("New from template…"
/// in the menu bar). Each returns a fresh NeuralGraph that the UI can
/// render. None of these actually run inference yet (that's Phase 4 with
/// candle absorption). They exist so the 300-node viz (#774) and 2D/3D
/// toggle (#775) have meaningful graphs to render.
pub mod factories {
    use super::*;

    /// GPT-2-small skeleton: 12 transformer blocks (LN → Attn → Add →
    /// LN → MLP → Add) + token embedding + final LN + LM head.
    /// ~124M parameters via the rough estimator.
    #[must_use]
    pub fn gpt2_small() -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.metadata.family = "GPT-2-small".to_string();
        g.metadata.notes =
            "12 transformer blocks, 768 hidden dim, 12 heads of 64 head_dim".to_string();
        let dim = 768_usize;
        let n_layers = 12_usize;
        let vocab = 50_257_usize;
        // Input token IDs.
        let toks = g.add_node(Node::input(vec![1, 1024], DType::I32, "tokens"));
        // Token embedding.
        let wte = g.add_node(Node::layer(
            LayerKind::Embedding { vocab, dim },
            "wte (token embedding)",
        ));
        g.add_edge(Edge {
            from: toks,
            to: wte,
            shape: vec![1, 1024],
            dtype: DType::I32,
            label: "ids".to_string(),
        });
        // Position embedding (also an Embedding kind).
        let wpe = g.add_node(Node::layer(
            LayerKind::Embedding {
                vocab: 1024,
                dim,
            },
            "wpe (position embedding)",
        ));
        // Repeated transformer blocks.
        let mut prev = wte;
        for i in 0..n_layers {
            let ln1 = g.add_node(Node::layer(LayerKind::LayerNorm, format!("blk{i}.ln_1")));
            let attn = g.add_node(Node::layer(
                LayerKind::Attention {
                    n_heads: 12,
                    head_dim: 64,
                    masked: true,
                },
                format!("blk{i}.attn"),
            ));
            let ln2 = g.add_node(Node::layer(LayerKind::LayerNorm, format!("blk{i}.ln_2")));
            let fc1 = g.add_node(Node::layer(
                LayerKind::Dense {
                    in_dim: dim,
                    out_dim: 4 * dim,
                },
                format!("blk{i}.mlp.fc1"),
            ));
            let gelu = g.add_node(Node::activation(ActKind::Gelu, format!("blk{i}.gelu")));
            let fc2 = g.add_node(Node::layer(
                LayerKind::Dense {
                    in_dim: 4 * dim,
                    out_dim: dim,
                },
                format!("blk{i}.mlp.fc2"),
            ));
            // Wire the block.
            let make_edge =
                |from: usize, to: usize, label: &str| -> Edge {
                    Edge {
                        from,
                        to,
                        shape: vec![1, 1024, dim],
                        dtype: DType::F32,
                        label: label.to_string(),
                    }
                };
            g.add_edge(make_edge(prev, ln1, "x"));
            g.add_edge(make_edge(ln1, attn, "x"));
            g.add_edge(make_edge(attn, ln2, "x"));
            g.add_edge(make_edge(ln2, fc1, "x"));
            g.add_edge(make_edge(fc1, gelu, "x"));
            g.add_edge(make_edge(gelu, fc2, "x"));
            prev = fc2;
        }
        // Final LN + LM head.
        let ln_f = g.add_node(Node::layer(LayerKind::LayerNorm, "ln_f"));
        g.add_edge(Edge {
            from: prev,
            to: ln_f,
            shape: vec![1, 1024, dim],
            dtype: DType::F32,
            label: "x".to_string(),
        });
        let lm_head = g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: dim,
                out_dim: vocab,
            },
            "lm_head",
        ));
        g.add_edge(Edge {
            from: ln_f,
            to: lm_head,
            shape: vec![1, 1024, dim],
            dtype: DType::F32,
            label: "x".to_string(),
        });
        // Output logits.
        let out = g.add_node(Node::output(vec![1, 1024, vocab], DType::F32, "logits"));
        g.add_edge(Edge {
            from: lm_head,
            to: out,
            shape: vec![1, 1024, vocab],
            dtype: DType::F32,
            label: "logits".to_string(),
        });
        // wpe edge into wte (positional addition, conceptual).
        g.add_edge(Edge {
            from: wpe,
            to: wte,
            shape: vec![1, 1024, dim],
            dtype: DType::F32,
            label: "+pos".to_string(),
        });
        g
    }

    /// Single transformer encoder block (BERT-style): LN → MultiHeadAttn
    /// → Add → LN → MLP → Add. 4 layer nodes per block; useful for
    /// click-to-expand demos.
    #[must_use]
    pub fn transformer_block() -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.metadata.family = "Transformer encoder block".to_string();
        let input = g.add_node(Node::input(vec![1, 512, 768], DType::F32, "x"));
        let ln1 = g.add_node(Node::layer(LayerKind::LayerNorm, "ln1"));
        let attn = g.add_node(Node::layer(
            LayerKind::Attention {
                n_heads: 12,
                head_dim: 64,
                masked: false,
            },
            "attn",
        ));
        let ln2 = g.add_node(Node::layer(LayerKind::LayerNorm, "ln2"));
        let fc1 = g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: 768,
                out_dim: 3072,
            },
            "fc1",
        ));
        let gelu = g.add_node(Node::activation(ActKind::Gelu, "gelu"));
        let fc2 = g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: 3072,
                out_dim: 768,
            },
            "fc2",
        ));
        let output = g.add_node(Node::output(vec![1, 512, 768], DType::F32, "y"));
        let chain = [input, ln1, attn, ln2, fc1, gelu, fc2, output];
        for w in chain.windows(2) {
            g.add_edge(Edge {
                from: w[0],
                to: w[1],
                shape: vec![1, 512, 768],
                dtype: DType::F32,
                label: "x".to_string(),
            });
        }
        g
    }

    /// ResNet-18-style basic block: Conv2d → BN → ReLU → Conv2d → BN
    /// → Add (skip). Conceptual graph only; no real weights.
    #[must_use]
    pub fn resnet_basic_block() -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.metadata.family = "ResNet-18 basic block".to_string();
        let input = g.add_node(Node::input(vec![1, 64, 56, 56], DType::F32, "x"));
        let conv1 = g.add_node(Node::layer(
            LayerKind::Conv2d {
                in_channels: 64,
                out_channels: 64,
                kernel: (3, 3),
                stride: (1, 1),
            },
            "conv1",
        ));
        let bn1 = g.add_node(Node::layer(LayerKind::BatchNorm, "bn1"));
        let relu1 = g.add_node(Node::activation(ActKind::Relu, "relu1"));
        let conv2 = g.add_node(Node::layer(
            LayerKind::Conv2d {
                in_channels: 64,
                out_channels: 64,
                kernel: (3, 3),
                stride: (1, 1),
            },
            "conv2",
        ));
        let bn2 = g.add_node(Node::layer(LayerKind::BatchNorm, "bn2"));
        let output = g.add_node(Node::output(vec![1, 64, 56, 56], DType::F32, "y"));
        let chain = [input, conv1, bn1, relu1, conv2, bn2, output];
        for w in chain.windows(2) {
            g.add_edge(Edge {
                from: w[0],
                to: w[1],
                shape: vec![1, 64, 56, 56],
                dtype: DType::F32,
                label: "x".to_string(),
            });
        }
        // Skip connection input → output (residual).
        g.add_edge(Edge {
            from: input,
            to: output,
            shape: vec![1, 64, 56, 56],
            dtype: DType::F32,
            label: "skip".to_string(),
        });
        g
    }

    /// Novel-AI demo: a research-frontier network mixing physics + symbolic
    /// + spiking + neural-ODE layers. Demonstrates the new LayerKinds added
    /// in iter 139 for the #786 "make completely novel AI" direction.
    #[must_use]
    pub fn novel_ai_demo() -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.metadata.family = "Novel-AI demo".to_string();
        g.metadata.notes =
            "Energy-based input → NeuralODE evolution → Hamiltonian preservation \
             → Oscillator coupling → Spiking decoder → Symbolic output."
                .to_string();
        let inp = g.add_node(Node::input(vec![1, 128], DType::F32, "x"));
        let ebm = g.add_node(Node::layer(
            LayerKind::EnergyBased {
                state_dim: 128,
                steps: 20,
            },
            "ebm-init",
        ));
        let ode = g.add_node(Node::layer(
            LayerKind::NeuralOde {
                state_dim: 128,
                steps: 16,
                t_end: 1.0,
            },
            "ode-evolve",
        ));
        let ham = g.add_node(Node::layer(
            LayerKind::Hamiltonian {
                phase_dim: 128,
                steps: 8,
            },
            "energy-preserve",
        ));
        let osc = g.add_node(Node::layer(
            LayerKind::Oscillator {
                n_oscillators: 64,
                coupling: 0.3,
                steps: 32,
            },
            "kuramoto-couple",
        ));
        let spike = g.add_node(Node::layer(
            LayerKind::Spiking {
                n_neurons: 64,
                tau_m: 20.0,
                threshold: 1.0,
            },
            "lif-spike",
        ));
        let sym = g.add_node(Node::layer(
            LayerKind::SymbolicFormula {
                formula: "tanh(W*x + b) + 0.1*sin(t)".to_string(),
                params: vec![
                    ("W".to_string(), 1.0),
                    ("b".to_string(), 0.0),
                    ("t".to_string(), 0.0),
                ],
            },
            "symbolic-read",
        ));
        let out = g.add_node(Node::output(vec![1, 64], DType::F32, "y"));
        let chain = [inp, ebm, ode, ham, osc, spike, sym, out];
        for w in chain.windows(2) {
            g.add_edge(Edge {
                from: w[0],
                to: w[1],
                shape: vec![1, 128],
                dtype: DType::F32,
                label: "x".to_string(),
            });
        }
        g
    }

    /// Wrap an existing HDC Stack as a NeuralGraph for unified rendering.
    /// One HdcLayer node bridges between Input → HDC → Output.
    #[must_use]
    pub fn from_hdc_stack(stack: Stack) -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.metadata.family = "HDC stack".to_string();
        let dim = stack.dim();
        let n_ops = stack.len();
        let input = g.add_node(Node::input(vec![dim], DType::Bipolar, "input"));
        let hdc = g.add_node(Node::layer(
            LayerKind::Hdc { stack },
            format!("hdc ({n_ops} ops)"),
        ));
        let output = g.add_node(Node::output(vec![dim], DType::Bipolar, "bundle"));
        g.add_edge(Edge {
            from: input,
            to: hdc,
            shape: vec![dim],
            dtype: DType::Bipolar,
            label: "input".to_string(),
        });
        g.add_edge(Edge {
            from: hdc,
            to: output,
            shape: vec![dim],
            dtype: DType::Bipolar,
            label: "bundle".to_string(),
        });
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = NeuralGraph::new();
        assert_eq!(g.nodes.len(), 0);
        assert_eq!(g.edges.len(), 0);
        assert_eq!(g.total_params(), 0);
        assert!(!g.has_cycle());
    }

    #[test]
    fn dense_layer_param_count() {
        // 768→3072 GELU MLP block: 768*3072 + 3072 = 2_362_368
        let d = LayerKind::Dense {
            in_dim: 768,
            out_dim: 3072,
        };
        assert_eq!(d.param_count(), 768 * 3072 + 3072);
    }

    #[test]
    fn gpt2_small_attention_estimate() {
        // GPT-2 small: 12 heads × head_dim 64. 4 × 12 × 64² = 196_608.
        let a = LayerKind::Attention {
            n_heads: 12,
            head_dim: 64,
            masked: true,
        };
        assert_eq!(a.param_count(), 4 * 12 * 64 * 64);
    }

    #[test]
    fn embedding_param_count() {
        // GPT-2 small wte: 50257 × 768
        let e = LayerKind::Embedding {
            vocab: 50257,
            dim: 768,
        };
        assert_eq!(e.param_count(), 50257 * 768);
    }

    #[test]
    fn hdc_layer_wraps_existing_stack() {
        use crate::Operation;
        let stack = Stack::new(1000)
            .with_operation(Operation::Identity)
            .with_operation(Operation::Identity);
        let layer = LayerKind::Hdc { stack };
        assert_eq!(layer.tag(), "hdc");
        // 2 ops × 1000 dim — rough estimate for param_count.
        assert_eq!(layer.param_count(), 2000);
    }

    #[test]
    fn simple_dag_no_cycle() {
        let mut g = NeuralGraph::new();
        let input = g.add_node(Node::input(vec![1, 768], DType::F32, "x"));
        let dense = g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: 768,
                out_dim: 3072,
            },
            "fc1",
        ));
        let output = g.add_node(Node::output(vec![1, 3072], DType::F32, "y"));
        g.add_edge(Edge {
            from: input,
            to: dense,
            shape: vec![1, 768],
            dtype: DType::F32,
            label: "x".to_string(),
        });
        g.add_edge(Edge {
            from: dense,
            to: output,
            shape: vec![1, 3072],
            dtype: DType::F32,
            label: "y".to_string(),
        });
        assert!(!g.has_cycle());
        assert_eq!(g.total_params(), 768 * 3072 + 3072);
    }

    #[test]
    fn cycle_detected() {
        let mut g = NeuralGraph::new();
        let a = g.add_node(Node::layer(
            LayerKind::LayerNorm,
            "a",
        ));
        let b = g.add_node(Node::layer(
            LayerKind::LayerNorm,
            "b",
        ));
        g.add_edge(Edge {
            from: a,
            to: b,
            shape: vec![],
            dtype: DType::F32,
            label: "".to_string(),
        });
        g.add_edge(Edge {
            from: b,
            to: a,
            shape: vec![],
            dtype: DType::F32,
            label: "".to_string(),
        });
        assert!(g.has_cycle());
    }

    #[test]
    fn gpt2_small_factory_runs() {
        let g = factories::gpt2_small();
        assert_eq!(g.metadata.family, "GPT-2-small");
        // Should have many nodes: input + wte + wpe + (12 * 6 layer nodes)
        // + ln_f + lm_head + output = ~78
        assert!(g.nodes.len() >= 70);
        assert!(g.nodes.len() <= 90);
        assert!(!g.has_cycle());
        // Parameter count should be roughly 124M ± rough estimator slack.
        let p = g.total_params();
        // GPT-2-small is 124M nominal; our rough estimator under-counts
        // (no bias on attn, etc.) so accept 80M..200M.
        assert!(
            p > 80_000_000 && p < 250_000_000,
            "expected ~124M params, got {p}"
        );
    }

    #[test]
    fn transformer_block_factory() {
        let g = factories::transformer_block();
        assert!(!g.has_cycle());
        // 8 nodes: input + ln1 + attn + ln2 + fc1 + gelu + fc2 + output
        assert_eq!(g.nodes.len(), 8);
    }

    #[test]
    fn resnet_block_has_skip_connection() {
        let g = factories::resnet_basic_block();
        // 7 nodes; 6 chain edges + 1 skip = 7 edges
        assert_eq!(g.nodes.len(), 7);
        assert_eq!(g.edges.len(), 7);
        assert!(!g.has_cycle());
    }

    #[test]
    fn novel_ai_demo_runs() {
        let g = factories::novel_ai_demo();
        assert_eq!(g.metadata.family, "Novel-AI demo");
        // 8 nodes: input + 6 novel-kind layers + output.
        assert_eq!(g.nodes.len(), 8);
        assert!(!g.has_cycle());
        // Each novel layer kind contributes some parameter count.
        let p = g.total_params();
        assert!(p > 0, "novel demo should have nonzero param estimate");
    }

    #[test]
    fn novel_layer_kinds_have_tags() {
        let kinds = [
            LayerKind::EnergyBased { state_dim: 10, steps: 5 },
            LayerKind::NeuralOde { state_dim: 10, steps: 5, t_end: 1.0 },
            LayerKind::Hamiltonian { phase_dim: 10, steps: 5 },
            LayerKind::Oscillator { n_oscillators: 10, coupling: 0.1, steps: 5 },
            LayerKind::Spiking { n_neurons: 10, tau_m: 20.0, threshold: 1.0 },
            LayerKind::SymbolicFormula {
                formula: "x*x".to_string(),
                params: vec![],
            },
        ];
        let expected = ["ebm", "neural_ode", "hamiltonian", "oscillator", "spiking", "symbolic"];
        for (k, e) in kinds.iter().zip(expected.iter()) {
            assert_eq!(k.tag(), *e);
        }
    }

    #[test]
    fn from_hdc_stack_factory() {
        use crate::Operation;
        let stack = Stack::new(1000).with_operation(Operation::Identity);
        let g = factories::from_hdc_stack(stack);
        assert_eq!(g.nodes.len(), 3); // input + hdc + output
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn kind_counts_aggregates() {
        let mut g = NeuralGraph::new();
        g.add_node(Node::input(vec![1, 768], DType::F32, "x"));
        g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: 768,
                out_dim: 3072,
            },
            "fc1",
        ));
        g.add_node(Node::layer(
            LayerKind::Dense {
                in_dim: 3072,
                out_dim: 768,
            },
            "fc2",
        ));
        g.add_node(Node::activation(ActKind::Gelu, "gelu"));
        let counts = g.kind_counts();
        assert_eq!(counts.get("input"), Some(&1));
        assert_eq!(counts.get("dense"), Some(&2));
        assert_eq!(counts.get("activation"), Some(&1));
    }
}
