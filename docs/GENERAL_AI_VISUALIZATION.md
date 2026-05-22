# General AI visualization — design doc

Owner direction: "find more ways of representing neural nets [...] make sure
the AIs that are represented are all AI types including LLMs and make sure
we can modify an actual neural net and visualize the way it looks. say with
300 nodes. [...] you should be displaying the different layers and the[y]
should be represented by nodes themselves so you can click them and expand
them. and the net should have both 2D and 3D representations."

This is a major expansion. The current app visualizes only HDC stacks
(parallel ops, single bundled output). To represent *any* neural net we
need a generic compute-graph model, layer-as-node abstraction, and the
ability to load + edit real models.

## Phases

### Phase 1 — Generic graph model (foundation)
Replace the implicit Stack-only graph with a general DAG:
```rust
struct NeuralGraph {
    nodes: Vec<Node>,           // generic compute nodes
    edges: Vec<Edge>,           // typed connections (tensor shape)
    metadata: GraphMetadata,    // architecture name, parameter count
}

enum Node {
    Input { shape: Vec<usize>, dtype: DType, label: String },
    Output { shape: Vec<usize>, dtype: DType, label: String },
    Layer {
        kind: LayerKind,         // Dense, Conv2d, Attention, LSTM, …
        params: HashMap<String, Tensor>,
        expand: bool,            // user-toggleable nested view
        sub_nodes: Vec<usize>,   // indices into nodes; rendered when expanded
    },
    Activation { kind: ActKind },
    Reshape { from: Vec<usize>, to: Vec<usize> },
    HdcOp { op: graphnet_engine::Operation }, // current stack ops live here
}

enum LayerKind {
    Dense { in: usize, out: usize },
    Conv2d { in_ch: usize, out_ch: usize, kernel: (usize, usize) },
    Attention { heads: usize, head_dim: usize, masked: bool },
    Lstm { hidden: usize, layers: usize },
    Gru  { hidden: usize, layers: usize },
    Embedding { vocab: usize, dim: usize },
    LayerNorm,
    BatchNorm,
    Pool { kind: PoolKind, kernel: (usize, usize) },
    Hdc { stack: graphnet_engine::Stack },  // existing HDC stack as a layer
}
```

### Phase 2 — Layer-as-node viz (click to expand)
A 300-node net is rendered as ~10-20 high-level nodes (each layer is one
3D node). Click a node → it expands inline showing internals (e.g. clicking
"Attention" reveals heads + q/k/v projections). egui's `CollapsingHeader`
adapted to 3D.

Hierarchy levels:
1. Network-level (architecture overview, 10-20 layer nodes)
2. Layer-level (heads of attention, channels of conv)
3. Neuron-level (only at small scale, e.g. <100 neurons per layer)
4. Connection-level (weight matrix as a heatmap when an edge is selected)

### Phase 3 — Dual 2D / 3D representations
- **2D** = traditional "block diagram" view (rows of boxes, top-to-bottom
  data flow). Render with `egui_plot::Plot` or hand-painted rects.
- **3D** = current arch_graph_3d generalized — each node positioned in 3D
  space, edges as splines. Layers stacked along the z-axis; nodes within
  a layer arranged radially.
- Toggle button in hero: `2D | 3D`. Layout switches with animation.

### Phase 4 — Load real models
Absorb `candle-core` + `candle-nn` + `safetensors` (Hugging Face's Rust
ML stack):
- `safetensors::SafeTensors::deserialize(bytes)` reads HF model weights
- `candle_core::Tensor` is the unit
- `candle_transformers::models::llama` / `bert` / `gpt2` have ready-made
  architectures
- Build a `NeuralGraph` from a loaded model by traversing the candle
  model's forward call

GGUF support via `candle_core::quantized::gguf_file` for quantized LLMs.

### Phase 5 — Editable weights + live inference
Once a model is loaded:
- Right-click any weight tensor → "Inspect heatmap" (already works for
  HDC keys, generalize to tensors)
- Right-click → "Replace tensor" → numeric perturbation slider
- Press Space → run forward through the live-edited model
- Compare output to baseline (using existing A/B compare panel)

### Phase 6 — LLM-specific views
- Attention heatmap (per-head attention weights across tokens)
- Token embedding similarity heatmap
- Layer-by-layer hidden-state trajectory
- Activation patching (zero out a head, see output change)

## FOSS to absorb (replacement candidates)

| Current | Replace with | Why |
|---|---|---|
| Hand-painted sparklines | `egui_plot` | Real plot library, log axes, multi-series, mouse interactions, zoom. Already in egui ecosystem. |
| Hand-rolled 3D projection | `egui_node_graph` for 2D + keep 3D | Mature node-graph editor with drag/connect/expand. Saves rolling our own. |
| Manual HDC math | Keep `plausiden-hdc` | This is purpose-built and good. |
| (none) Model loading | `candle-core` + `candle-nn` + `candle-transformers` + `safetensors` | HF's Rust ML stack. Drop-in for loading any HF model. |
| (none) Tensor ops | `candle-core` `Tensor` | Already imports cleanly into eframe. |
| (none) Generic graph data | `petgraph` | Battle-tested directed-graph library with topo-sort, traversal, layout. |
| `ui_smoke_test.sh` xdotool | Add `enigo` (Rust input automation) as an in-app test mode | xdotool focus leaks; enigo can drive the app from inside its own process. |
| File dialogs (`rfd`) | Keep `rfd` | Works fine. |
| Plotting in central panel | `egui_plot` | Same as sparklines. |
| Image input (`image`) | Keep `image` | Works fine. |
| Audio (`rodio` deferred) | Keep `rodio` once libasound2-dev lands | Works on most Linux. |

## Suggested implementation order

1. **Iter 122**: Absorb `egui_plot` — replace existing sparklines, get real
   plot widget. Bounded.
2. **Iter 123-130**: Phase 1 — `NeuralGraph` + `Node`/`Edge`/`LayerKind`
   data structures + serialize/deserialize. No UI changes yet; just the
   model.
3. **Iter 131-135**: Phase 2 — render `NeuralGraph` in 2D using
   `egui_node_graph`. Click to expand layers.
4. **Iter 136-140**: Phase 3 — adapt existing 3D viz to render
   `NeuralGraph` (each Node is a 3D node, each Edge a spline).
   2D/3D toggle in hero.
5. **Iter 141-150**: Phase 4 — absorb candle/safetensors. Load HF models.
   Build `NeuralGraph` from candle model traversal.
6. **Iter 151+**: Phase 5-6 — live editing + LLM-specific views.

## Scope realism

300 nodes rendered in 3D: with current shaded_node code, 300 nodes at 60fps
should be fine on the wgpu backend (the painter has been tested with ~50
nodes; 300 is 6× but still O(N) per frame, well under budget).

Loading a real LLM: candle's llama-7b is ~13GB weights. Won't fit on most
machines. Start with **GPT-2-small** (124M params, ~500MB) or
**distilbert-base-uncased** (66M, ~250MB) as initial targets.

Full LLM inference in Rust on CPU is slow but feasible at <1B params.
GPU via wgpu compute shaders is the next absorb (candle has CUDA
support; wgpu compute is in candle's roadmap).

## What this doesn't change

- The existing HDC stack remains the default + "HDC layer" within the
  general graph. Templates still work.
- The current 3D viewport code stays — it gets generalized to render
  generic Nodes (not just HDC ops).
- The training pipeline (#757) extends to gradient descent on candle
  tensors, not just bit-flip on HDC keys.

This is a 3-4 month effort. Open question for owner: prioritize
**generic-graph foundation first** (so anything fits) or **LLM loading
first** (so we can demo on real models sooner)?
