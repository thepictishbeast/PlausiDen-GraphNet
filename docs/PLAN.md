# GraphNet — Build Plan

Live REPL + interactive graphical environment for neural-network architecture
work. Primary focus: PlausiDen Stack architecture (HDC / VSA substrate). First-
class secondary support: standard LLMs (transformer-based), graph neural networks,
state-space models, and as many other AI families as we can wrap.

Tagline: *"A graphing calculator for AI — see real work happening in real time,
edit values and architecture, watch effects propagate instantly, in 2D and 3D."*

## Status

Specification + plan. Implementation not started. Captured 2026-05-17 from
owner direction across multiple session segments.

## 1. Naming

**GraphNet** is the chosen name. Owner picked it; previous "Scope" suggestion
is retired.

Conflict note: "GraphNet" is also used for graph neural networks (GNNs) in
some published research. To minimise confusion the project ships as
`PlausiDen-GraphNet` everywhere — the `PlausiDen-` prefix disambiguates from
GNN research. The Python package is `plausiden_graphnet`; the Rust crate is
`graphnet-engine`. We never publish anything just as "GraphNet" alone.

## 2. What GraphNet does — full feature spec

### 2.1 Multi-architecture support

GraphNet is **AI-family-agnostic** at the introspection layer. v1 ships with:

| Architecture family | v1 | v2+ |
|---|---|---|
| **PlausiDen Stack** (HDC / VSA + heterogeneous ops) | ✓ first-class | — |
| **Transformer LLMs** (HuggingFace `transformers` compatible) | ✓ | — |
| **State-space models** (Mamba, RWKV, S4) | ✓ | — |
| **Graph neural networks** (PyG-compatible) | — | ✓ |
| **CNNs** (image classifiers, vision backbones) | — | ✓ |
| **Diffusion models** | — | ✓ |
| **Mixture of Experts** (Mixtral, DBRX, DeepSeek) | — | ✓ |
| **Multimodal** (CLIP, LLaVA-style) | — | ✓ |
| **Reinforcement learning agents** (actor-critic, PPO) | — | ✓ |
| **Recurrent nets** (LSTM, GRU — legacy) | — | ✓ |

The introspection API is uniform — every model exposes `.modules()`, `.weights()`,
`.activations()`, `.intervene()`. Per-family adapters translate to the
respective frameworks (PyTorch, JAX, etc.).

### 2.2 Stacking different AI types

Owner's explicit ask: "allow them to be stacked." GraphNet lets users compose
heterogeneous models into a pipeline or composition graph:

- **Pipeline**: input → Model A → output A → Model B → output B → ...
- **Parallel composition**: input → [Model A | Model B | Model C] → aggregator → output
- **Routing composition**: input → router → one-of-N models → output
- **Hierarchical**: outer model contains inner models as substructures

This is the user-facing surface of the Stack architecture but generalised:
any model can be a Stack component, regardless of its internal family.

A practical example:
```
input image →
  CLIP image encoder (transformer) →
    PlausiDen Stack (HDC reasoning) →
      Mamba (sequence model) →
        decoder LLM (transformer) →
          output text
```

GraphNet handles tensor-shape translation between adjacent components and
shows the full pipeline as an editable graph.

### 2.3 Live interactive REPL

Jupyter-hosted Python REPL with reactive semantics. Owner's verbatim ask:
*"like a graphing calculator or REPL or something but for AI where you can
see real work happening in real time and the values you change you can see
take effect instantly."*

Concrete capabilities:

- Load a model: `g = graphnet.load("model.pt")` or `g = graphnet.build(spec)`
- Run a forward pass with full state capture
- Modify ANY weight / bias / hyperparameter — change propagates immediately
- Modify ANY architectural decision — add a layer, remove an attention head,
  swap an activation function — without a restart
- Watch any tensor live as the model runs
- Stream inputs through the model continuously; visualisation updates at
  display framerate
- Step backwards (rewind) using state snapshots
- Diff two model configurations side-by-side

### 2.4 Graphical representation — 2D and 3D, editable, rotatable

Owner's verbatim ask: *"have graphical representations of the AI thats editable
and rotateabel and where you can sellect inputs and outputs."*

#### 2.4.1 2D view (Jupyter-embedded)

- **Graph layout** of the architecture: nodes = layers/operations, edges =
  tensor flow. Built on `graphviz` + custom layout, rendered as SVG.
- **Heatmaps**: any tensor as a 2D heatmap; per-dimension activation values.
- **Histograms**: weight distributions, activation distributions, gradient
  distributions.
- **Time-series plots**: any scalar over forward-pass steps or training steps.
- All updateable live.

#### 2.4.2 3D view (rotatable, interactive)

- **3D architecture graph**: same architecture rendered in 3D space.
  Nodes float, edges connect in 3D, user can orbit / pan / zoom.
- **Click to select** any node — opens edit panel for that operation's
  parameters / weights / hyperparameters.
- **Drag to reposition**: layout is force-directed by default but the user
  can pin nodes.
- **Color encoding**: node colour shows activation magnitude or one of:
  utilisation, error contribution, parameter count, compute cost.
- **Cross-section view**: slice the architecture along a chosen axis to see
  intermediate state at that depth.
- **Tensor visualisation in 3D**: high-dimensional tensors projected to 3D
  via PCA / t-SNE / UMAP, with controls to flip projection and time-evolve.
- **Hypervector visualisation**: HDC vectors (D=10,000) shown as 3D point
  clouds (PCA-projected) with binding/unbinding operations animated.

Rendering options (decide at Phase 4):

| Backend | Pros | Cons | Decision factor |
|---|---|---|---|
| **plotly 3D** | works in Jupyter; rotation/zoom native; Python-only | not GPU-accelerated; clunky for >10k nodes | v1 default |
| **k3d-jupyter** | GPU-accelerated WebGL in Jupyter; nice for point clouds | smaller community; some quirks | secondary v1 |
| **three.js via ipywidget** | browser-native 3D; arbitrary scenes; rotateable | bigger build; custom widget | v2 if needed |
| **bevy** (Rust) | native; full 3D engine; fast | heavyweight dep; harder to embed | only for standalone GUI |
| **wgpu + egui** (Rust) | native; lightweight 3D; pairs with egui dock | more code to write | for Phase 6 native GUI |

v1 uses plotly 3D + k3d-jupyter combined; v6 switches to wgpu/egui for the
native shell.

### 2.5 Selectable inputs and outputs

Owner's ask. GraphNet exposes the model as a typed graph with explicit
input/output ports:

- **Input selector**: pick from a registered input library (text strings,
  images, hypervectors, custom-loaded tensors) OR provide raw input. Drag
  the chosen input onto an input port in the 2D/3D view.
- **Output selector**: tap into any intermediate tensor as the "output of
  interest." The visualisation centres on that tensor. Multiple outputs can
  be selected simultaneously.
- **Probe points**: mark any tensor as a probe; GraphNet collects + records
  it on every forward.

### 2.6 Resource usage display (RAM / GPU / CPU + cost)

Owner's ask. Live instrumentation panel shows:

| Metric | What's shown |
|---|---|
| **RAM (host)** | Current usage, peak, per-tensor allocation breakdown |
| **VRAM (GPU)** | Per-GPU current/peak, per-tensor breakdown if on GPU |
| **CPU utilisation** | Per-core busy fraction, integrated load |
| **GPU utilisation** | Per-GPU SM occupancy, memory bandwidth utilisation |
| **FLOPs** | Counted per forward pass, both theoretical (analytical) and measured |
| **Wall-time** | Per-layer, per-operation, hot-path breakdown |
| **Energy** (estimated) | Joules per forward pass via nvidia-smi power draw |
| **$ cost** (estimated) | Joules → kWh → local electricity cost OR cloud
                            instance $/hr × wall-time |

Per-architecture comparison: load two models, run identical inputs, see
side-by-side resource cost.

Source libraries:
- `psutil` for CPU + RAM
- `pynvml` for NVIDIA GPU (nvidia-ml-py wrapper)
- `pyrocm`-equivalent for AMD (if available)
- Custom Rust instrumentation for FLOP counts in HDC ops
- Analytical FLOP counters (e.g., `fvcore.nn.FlopCountAnalysis`) for
  transformer layers

### 2.7 Export and save configs

Owner's ask. Three serialization layers:

| Layer | Format | Purpose |
|---|---|---|
| **Architecture spec** | YAML / JSON | "This model has these layers in this shape" — human-readable, version-controllable, shareable |
| **Trained weights** | safetensors / numpy npz | The actual numerical state |
| **Session state** | binary (bincode) | Full GraphNet session including selected probes, viz settings, watched tensors, intervention history |

Owner shares the architecture spec; weights ride along separately when
relevant. A session export reproduces the exact viz the user was looking at.

Plus:
- **Diff format**: two configs → human-readable diff
- **Rollback**: intervention history is a stack; undo/redo any change
- **Sharing**: optional URL-shortened export (paste link, recipient opens
  in GraphNet on their machine)

### 2.8 Test, audit, debug — built-in

Owner's verbatim ask: *"make sure you test, audit and debug GraphNet, make
sure you prove it works and make it perfect."*

GraphNet ships with its own test + audit + debug infrastructure:

- **Unit tests** in `graphnet-engine` (Rust) for every op + intervention path
- **Property tests** via proptest for HDC operations (binding/unbinding
  identity, bundling associativity, permutation invariance)
- **Integration tests** in Python for the full REPL surface
- **Visual regression tests** using PlausiDen-Crawler against GraphNet's
  Jupyter UI — desktop + mobile, light + dark themes
- **Performance benchmark suite** — every Phase ships with benches; perf
  regressions are CI failures
- **Mutation testing** via `cargo mutants` on the Rust core
- **Fuzzing** via `cargo fuzz` on the deserialisation paths
- **AVP-2 checklist** applied — every public API has a BUG ASSUMPTION
  comment, every unsafe block has a SAFETY proof, every public function has
  a test, no `unwrap`/`expect` outside test code
- **AVP-2 audit reports** generated from CI, signed via attest pipeline
  (per AVP §8b doctrine + the attest infrastructure already in Forge)

### 2.9 "Anything else I can think of" — additional features

These were not explicitly asked for but follow from the vision. Ship as
appropriate per phase.

- **Live training mode**: not just inference; show training step-by-step,
  visualise gradient flow, watch loss surface evolution.
- **Comparison mode**: load two architectures side-by-side, run identical
  inputs, diff every intermediate state.
- **Recording + playback**: capture a session (inputs + interventions + state
  evolution) and replay it. Useful for reproducing bugs and sharing demos.
- **Plugin system**: third parties write custom operation modes; GraphNet
  loads them via a `PluginOp` trait + dynamic linking.
- **Counterfactual analysis**: for a chosen output, ask "what's the minimum
  change to input X that would shift output Y by Δ?" — drives interventions.
- **Sensitivity analysis**: rank model parameters by their contribution to
  output variance.
- **Adversarial input crafter**: built-in tooling for finding adversarial
  examples against the model under inspection.
- **Hardware advisor**: given an architecture, recommend GPU class + RAM
  amount + estimated cost-to-train.
- **Architecture suggestions**: an LLM-judge (using an external API or local
  small model) reads the architecture + outputs + offers structural
  suggestions ("this layer is redundant", "consider adding skip-connection
  here").
- **Energy / carbon dashboard**: integrate estimated kWh + grid carbon
  intensity → CO₂-equivalent per forward pass.
- **Multi-language inputs**: provide inputs in different formats (text,
  audio, image, hypervector) and watch how the model handles each.
- **Tutorial system**: shipped notebooks for "first 30 minutes" — Stack
  101, Transformer introspection 101, HDC 101.
- **Documentation generator**: from a loaded model, generate a markdown
  doc describing the architecture suitable for paper appendix.
- **Notebook → script export**: capture a Jupyter exploration as a
  standalone `.py` script that reproduces the result.

## 3. Architecture (technical)

```
                ┌──────────────────────────────────────────────────────┐
                │  PlausiDen-GraphNet Frontend                          │
                │                                                       │
                │  ┌─────────────────┐  ┌──────────────────────────┐  │
                │  │ Jupyter / IPython│  │ Visualisations            │  │
                │  │ REPL              │  │ - 2D graph (graphviz/svg)│  │
                │  │ Python objects:   │  │ - 3D graph (plotly/k3d) │  │
                │  │   GraphNet, Stack │  │ - heatmaps               │  │
                │  │   Model, Probe    │  │ - histograms             │  │
                │  │   Intervention    │  │ - resource gauges        │  │
                │  └────────┬──────────┘  └────────────┬─────────────┘  │
                │           │                          │                 │
                └───────────┼──────────────────────────┼─────────────────┘
                            │                          │
                            │ PyO3 FFI                 │ WebSocket / IPC
                            │                          │
              ┌─────────────▼──────────────────────────▼────────────┐
              │  graphnet-engine (Rust)                              │
              │                                                       │
              │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
              │  │ Model trait │  │ Intervention │  │ Adapters    │  │
              │  │ + state     │  │ Engine       │  │  - HDC/Stack│  │
              │  │ tracker     │  │              │  │  - Torch    │  │
              │  └─────────────┘  └──────────────┘  │  - JAX      │  │
              │  ┌─────────────┐  ┌──────────────┐  │  - Mamba    │  │
              │  │ Resource    │  │ Recording /  │  │  - ...      │  │
              │  │ Monitor     │  │ Playback     │  └────────────┘  │
              │  └─────────────┘  └──────────────┘                   │
              └────────────────────────┬───────────────────────────────┘
                                       │
                ┌──────────────────────▼────────────────────────────┐
                │  lfi_vsa_core (Rust, existing)                     │
                │  HDC primitives, Stack execution                    │
                └─────────────────────────────────────────────────────┘
                
                ┌─────────────────────────────────────────────────────┐
                │  External AI runtimes (loaded as adapters)           │
                │  - PyTorch (via tch-rs or pyo3-torch)               │
                │  - JAX (via Python bridge)                          │
                │  - HuggingFace transformers                         │
                │  - Mamba (mamba_ssm)                                │
                │  - ...                                              │
                └─────────────────────────────────────────────────────┘
```

## 4. Language choice — rationale + FOSS picks

Owner: *"if a pure rust build is too much to ask for for something to rapidly
prototype then find FOSS projects you can usd for the GUI parts and backend
and such."*

Decision: **Hybrid Rust + Python**, using high-quality FOSS for the GUI parts
that would be expensive to build from scratch.

| Layer | Tech | License | Reason |
|---|---|---|---|
| Core HDC + Stack execution | Rust + `lfi_vsa_core` | (PlausiDen) | Owned; performance-critical |
| Python bindings | `pyo3` 0.22+ | Apache-2.0 / MIT | Industry standard |
| Numpy interop | `numpy` (in Python), `numpy` feature of pyo3 | BSD-3 | Required for tensor exchange |
| REPL host | IPython / Jupyter | BSD-3 | The standard ML iteration UX |
| 2D plots | matplotlib | PSF-license | Universal, well-documented |
| Interactive 2D | plotly | MIT | Jupyter-native interactivity |
| 3D Jupyter-embedded | k3d-jupyter | MIT | GPU-accelerated 3D in notebooks |
| Graph layout | graphviz (system) + `graphviz` python | EPL / MIT | Industry-standard graph drawing |
| Architecture graph | networkx (manipulation) + graphviz (render) | BSD-3 | Solid combination |
| Native GUI (Phase 6) | egui + wgpu via tauri OR pure egui | Apache-2.0 / MIT | Mature Rust GUI stack |
| Transformer adapter | huggingface `transformers` | Apache-2.0 | Standard for LLM work |
| Mamba adapter | `mamba_ssm` | Apache-2.0 | Official Mamba impl |
| HuggingFace integration | `transformers` + `safetensors` | Apache-2.0 | Ecosystem standard |
| Resource monitoring | `psutil` + `pynvml` | BSD + BSD | De facto choices |
| Serialization | `bincode` + `serde` + `serde_yaml` + `safetensors` | MIT / Apache-2.0 | Standard Rust |

Total third-party FOSS list is large but every entry is permissively licensed
and widely battle-tested. No surprises. This addresses the owner's "find FOSS
projects" instruction directly.

## 5. Public API surface (Python)

```python
import graphnet as gn

# === Loading models ===

# PlausiDen Stack
s = gn.load_stack("path/to/stack.bin")

# Transformer LLM
llm = gn.load_transformer("gpt2")  # HuggingFace shortcut
llm = gn.load_transformer("/path/to/local/model")

# Mamba
m = gn.load_mamba("state-spaces/mamba-130m")

# Custom architecture from spec
spec = gn.architecture.from_yaml("my-arch.yaml")
model = gn.build(spec)

# === Stacking models ===

pipeline = gn.Pipeline([
    gn.load_transformer("clip-vit-base"),
    gn.load_stack("reasoning.bin"),
    gn.load_mamba("mamba-130m"),
])

# === Running ===

inp = gn.Input.image("photo.png")  # or .text, .hypervector, .tensor
out = pipeline.forward(inp)
print(out)

# Continuous mode
pipeline.run_continuous(input_stream=gn.streams.random_images(n=1000))

# === Probes ===

p = pipeline.models[1].probe("layer_3.attention.output")  # tap a tensor
p.watch()  # opens live heatmap

# === Intervention ===

# Modify a weight
pipeline.models[0].weights["transformer.h.0.attn.c_attn.weight"][0, 0] = 0.0

# Add a new operation to a Stack (Stack architecture only)
pipeline.models[1].stacks[0].add_operation("fft_bind", init="random")

# Remove a transformer attention head
pipeline.models[0].remove_head(layer=3, head=2)

# === Visualisation ===

pipeline.viz.architecture(mode="2d")  # opens 2D graph
pipeline.viz.architecture(mode="3d")  # opens 3D rotatable graph
pipeline.viz.heatmap("models[1].output")
pipeline.viz.resource_gauges()  # live RAM/GPU/CPU/$
pipeline.viz.compare(other_pipeline)  # side-by-side diff

# === Export ===

pipeline.export.architecture("my-arch.yaml")  # spec only
pipeline.export.weights("my-arch.safetensors")  # weights only
pipeline.export.session("my-session.gn")  # full session

# === Resource accounting ===

cost = pipeline.estimate.cost(inputs_per_day=10_000)
print(cost)
# Cost(electricity=$3.40/day, cloud_a100=$24.00/day, ...)

# === Tests + audit ===

audit = pipeline.audit()
print(audit)
# Audit(layers=42, params=124M, FLOPs=...,
#       warnings=[...],
#       recommendations=[...])
```

## 6. Repository structure

```
github.com/thepictishbeast/PlausiDen-GraphNet  (new, FSL-1.1-MIT)

PlausiDen-GraphNet/
├── README.md (AVP warning + GraphNet intro)
├── LICENSE (FSL-1.1-MIT)
├── Cargo.toml (workspace)
├── pyproject.toml (Python package config)
│
├── crates/
│   ├── graphnet-engine/      # Core Rust engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model/         # Model trait + impls
│   │       ├── intervene/     # Intervention API
│   │       ├── monitor/       # Resource monitoring
│   │       ├── record/        # Session recording/playback
│   │       ├── stack/         # Stack architecture support
│   │       └── adapters/      # External AI framework adapters
│   │
│   ├── graphnet-bindings/    # PyO3 Python bindings
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   └── graphnet-gui/          # Native GUI (Phase 6)
│       └── (egui + wgpu)
│
├── python/
│   └── graphnet/
│       ├── __init__.py
│       ├── viz/                # Visualisation layer
│       │   ├── plot_2d.py
│       │   ├── plot_3d.py
│       │   └── widgets.py
│       ├── adapters/            # Python-side framework adapters
│       │   ├── transformers.py
│       │   ├── mamba.py
│       │   └── jax.py
│       ├── streams/             # Input streams (random, dataset, file, ...)
│       └── audit/               # Built-in audit toolkit
│
├── examples/                    # Jupyter notebooks
│   ├── 01_first_stack.ipynb
│   ├── 02_load_a_transformer.ipynb
│   ├── 03_stack_them.ipynb
│   ├── 04_3d_visualisation.ipynb
│   ├── 05_intervention.ipynb
│   ├── 06_resource_costs.ipynb
│   ├── 07_export_import.ipynb
│   └── 99_full_tutorial.ipynb
│
├── tests/
│   ├── rust/                    # Rust integration tests
│   ├── python/                  # Python integration tests
│   └── visual/                  # Visual-regression tests (PlausiDen-Crawler)
│
├── benches/                     # Criterion benchmarks
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ADAPTERS.md              # How to write a new adapter
│   ├── INTERVENTION_API.md
│   ├── VISUALISATION.md
│   ├── EXPORT_FORMAT.md
│   └── AVP2_AUDIT.md
│
└── .github/workflows/
    ├── ci.yml                   # fmt + clippy + test + python tests
    ├── audit.yml                # AVP-2 audit run
    ├── visual.yml               # Crawler-driven visual tests
    └── bench.yml                # Performance benchmark
```

## 7. Build phases — full plan

Each phase has a success criterion + estimated effort. The numbers are
calendar-rough; actual depends on focus + scope creep.

### Phase 0: Repo scaffold + CI
~1 day. Create the repo, set up Cargo workspace, set up Python package
config, set up CI workflows (fmt + clippy + test + visual), set up
benchmark runner. No GraphNet code yet.

### Phase 1: graphnet-engine Rust core (1–2 weeks)
- `Model` trait: `forward`, `intervene`, `snapshot`, `restore`, `subscribe`
- `Stack` + `StackOfStacks` impls (depending on `lfi_vsa_core`)
- In-memory state tracking
- Snapshot/restore using bincode
- Subscriber pattern for live tensor watching
- Unit tests on toy Stacks
- Criterion benches

Success: Rust-only example builds a 3-operation Stack, runs 1000 forwards,
takes snapshots, intervenes, restores, all benched.

### Phase 2: PyO3 bindings (3–5 days)
- `graphnet` Python package, generated from `graphnet-engine` via PyO3
- Hypervector as numpy-buffer-compatible
- Stack as Python class
- Probe + Intervention APIs callable from Python
- Tests: full Phase 1 scenario driven from Python

Success: `pip install -e .`, open Python REPL, replicate Phase 1.

### Phase 3: Jupyter integration + basic 2D viz (1 week)
- IPython magics
- `_repr_html_` for live HTML tables (gates, magnitudes, etc.)
- matplotlib heatmap support
- graphviz/networkx architecture graph (2D, SVG)
- All updateable live via `subscribe` channel

Success: Notebook example shows live updates as user runs forwards
interactively.

### Phase 4: 3D visualisation (rotatable, interactive) (1–2 weeks)
- plotly 3D scatter for hypervector PCA projections
- plotly 3D graph for architecture
- k3d-jupyter for high-density point clouds
- Click-to-select operation panels
- Drag-to-reposition; pin-as-anchor
- Cross-section view (slice along depth/layer axis)

Success: Owner opens a notebook, sees the model in 3D, rotates it, clicks a
layer, sees its weights, edits one, watches the change reflected in the
output heatmap.

### Phase 5: Multi-architecture adapters (2–3 weeks)
- Transformer adapter (HuggingFace `transformers`)
- Mamba adapter (`mamba_ssm`)
- Generic `nn.Module` adapter (PyTorch any)
- Adapter API documented for third-party additions

Success: Load a 7B GPT-2 + a 130M Mamba + a Stack, compose them in a
Pipeline, run inputs through, visualise full graph end-to-end.

### Phase 6: Live continuous-execution mode (1 week)
- `run_continuous(model, input_stream)` runs in a background thread
- Visualisations update at ~30 FPS
- Intervention from REPL propagates immediately

Success: 5 minutes of continuous execution; mid-stream intervention; visible
behavioural change.

### Phase 7: Architectural mutation (1–2 weeks)
- Add/remove/swap operations at runtime
- Spawn/merge/delete Stacks
- Entropy gradient sliders
- Undo/redo intervention history (rollback)

Success: Mutate a Stack mid-execution, watch the model adapt; undo,
re-mutate.

### Phase 8: Resource accounting + cost estimation (1 week)
- Hook psutil + pynvml + analytical FLOP counters
- Live gauges in the notebook viz
- `pipeline.estimate.cost(...)` API
- Energy estimation via NVIDIA SMI power draw
- Side-by-side resource comparison

Success: Owner can see at a glance: this model takes X joules per forward,
costs $Y per million inferences at current electricity rates.

### Phase 9: Export/import + session recording (1 week)
- YAML architecture spec emission + parsing
- Weights via safetensors
- Full session bincode roundtrip
- Session recording/playback
- Architecture diff format

Success: Owner exports a config, shares with us, we load it, see the same
state, can intervene and re-export.

### Phase 10: Test/audit/debug suite (1–2 weeks)
- Property tests on HDC ops
- Visual regression tests via PlausiDen-Crawler
- Mutation testing via cargo mutants
- Fuzz testing via cargo fuzz
- AVP-2 audit pipeline integrated into CI
- Benchmark suite with regression alerts

Success: Every Phase 1–9 contribution lands with tests; mutation survival
< 5%; visual diffs near zero across runs; AVP-2 audit reports
auto-generated.

### Phase 11: "Anything else" features (rolling, post-Phase-10)
- Live training mode
- Comparison mode
- Counterfactual analysis
- Sensitivity analysis
- Plugin system
- Tutorial notebooks polished
- Documentation generator
- Hardware advisor
- Energy/carbon dashboard

These ship as they're built; order driven by owner feedback after the core
is real.

### Phase 12: Native GUI shell (later, optional)
- Tauri-shelled app OR pure-egui native
- Embedded Python interpreter for REPL + Rust engine for performance
- Distributable as .app / .exe / AppImage

Success: Owner double-clicks an icon, gets GraphNet desktop UI, no terminal.

## 8. Cross-cutting requirements

These apply throughout, per PlausiDen doctrine.

- **Light + dark themes** (per forge-default-themes-a11y) — every visual
  element ships in both
- **WCAG 2.1 AA accessibility** — semantic HTML where applicable, keyboard
  navigation, visible focus rings, contrast
- **Mobile + desktop tested** — even though GraphNet is fundamentally a
  desktop tool, the Jupyter viz layer must not break on mobile
- **AVP-2 doctrine compliance** — every commit guilty until proven innocent,
  BUG ASSUMPTION comments on public funcs, no unwrap in non-test code,
  zeroize secrets, ed25519-only crypto, TLS 1.3 only
- **ISO standards** — 8601 timestamps, 639-1 language codes, etc.
- **PlausiDen design language** — gradient heroes, soft shadows, custom
  typography, 4/8/12/18/28/44 spacing scale (per design-premium doctrine)
- **FSL-1.1-MIT license** (per repo-presentation memory)
- **Public repo** with proper AVP warning banner
- **CI on ubuntu-latest** (per 2026-05-17 migration)
- **main = active dev, master = validated tip** (per AVP §8b)

## 9. Test plan — proving it works

Test infrastructure ships alongside the code. No "we'll add tests later."

### 9.1 Unit tests (Rust)
- Every public function in `graphnet-engine`
- Property tests for HDC operations:
  - `unbind(bind(a, k), k) ≈ a` (with noise tolerance)
  - `bundle(a, b) = bundle(b, a)` (commutativity)
  - `bind(bind(a, k1), k2) = bind(bind(a, k2), k1)` (commutativity)
  - Permutation cycle: `Π^D(x) = x` for some D
- Snapshot/restore round-trip identity
- Intervention reversibility

### 9.2 Integration tests (Python)
- Full REPL scenarios: load → intervene → observe → restore
- Multi-architecture pipelines: transformer + Stack + Mamba composition
- Export → import → identity
- Continuous mode startup + shutdown + intervention mid-stream

### 9.3 Visual regression tests
- Driven by PlausiDen-Crawler's chromiumoxide runner
- Captures Jupyter notebook viz outputs
- Compares against golden screenshots
- Desktop + mobile viewports, light + dark themes
- Fails CI on visual diff above tolerance

### 9.4 Performance benchmarks
- Per-operation latency (HDC bind/bundle/unbind, transformer forward, ...)
- Per-architecture forward latency
- Memory allocation patterns
- Run on every push; regression alerts to GitHub PR

### 9.5 Mutation testing
- `cargo mutants` on graphnet-engine
- Target: < 5% mutant survival
- Surviving mutants flagged as test-quality regressions

### 9.6 Fuzz testing
- `cargo fuzz` on YAML spec parsing
- `cargo fuzz` on bincode session restore
- Goal: zero panics on any input

### 9.7 AVP-2 audit reports
- Each Phase ships with an audit report appended to docs/AVP2_AUDIT.md
- 36-pass minimum on critical-path code (engine + intervention + serdes)
- Signed via attest infrastructure (per Forge attest pipeline)

## 10. Open questions

1. **Adapter compatibility versions.** HuggingFace transformers updates
   frequently; adapter must pin a known-good version. Tracking strategy?
2. **GPU support timing.** Phase 1-9 are CPU-only. When does GPU support
   land — Phase 11 or earlier?
3. **Multi-process model.** If a user loads a 70B LLM that doesn't fit on
   one GPU, GraphNet needs multi-process / tensor-parallel awareness. Out
   of scope for v1? Or required?
4. **Persistent storage.** Sessions can be large. Where do they live by
   default? Owner-controlled config; sane default needed.
5. **Telemetry.** Do we collect any (anonymous) usage data to drive
   prioritisation? Default off, opt-in? Owner decides.
6. **Tutorial-first or REPL-first.** The first launch experience —
   notebook tutorial or empty REPL? Both, with `graphnet hello` opening
   the tutorial.

## 11. Sequencing with Stack architecture

GraphNet and the Stack architecture (`STACK_ARCHITECTURE.md`) are paired
tools. Build interleaved, not sequential:

| Time | GraphNet phase | Stack phase | Why interleaved |
|---|---|---|---|
| Week 1 | 0 (scaffold) | 0 (HDC foundation verify) | Both start at the same baseline |
| Week 2 | 1 (engine) | 1 (single Stack) | GraphNet engine USES the Stack impl |
| Week 3 | 2 (PyO3) | 1 cont. | Python access to Stack |
| Week 4 | 3 (2D viz) | 2 (gated Stack) | Viz helps debug gating decisions |
| Week 5–6 | 4 (3D viz) | 3 (stack-of-stacks) | 3D is essential for stack-of-stacks |
| Week 7–8 | 5 (adapters) | 4 (entropy gradient) | Adapters bring in LLMs to compare |
| Week 9 | 6 (continuous) | 5 (self-modification) | Continuous mode needed to watch
                                                       self-modification dynamics |
| Week 10+ | 7+ (mutation) | ongoing | Mutual reinforcement |

## 12. First action

Begin Phase 0. Concrete first commits in order:

1. Create `/home/user/Development/PlausiDen/PlausiDen-GraphNet/` working tree
2. `git init`, set FSL-1.1-MIT LICENSE, AVP-2 README banner
3. Cargo workspace with `graphnet-engine`, `graphnet-bindings`, future `graphnet-gui`
4. Python `pyproject.toml` + `setup.py` for the `graphnet` package
5. Empty `lib.rs` files + `__init__.py` placeholders
6. `.github/workflows/ci.yml` (fmt + clippy + test + python tests)
7. First commit pushed to GitHub (new repo `thepictishbeast/PlausiDen-GraphNet`)

Then Phase 1.

## 13. Dependencies — full FOSS map, per feature

Per owner direction: "if a pure rust build is too much to ask for find FOSS
projects." Below is the per-feature dependency choice with rationale + license.

### 13.1 Rust crates (graphnet-engine + bindings + gui)

| Feature | Crate | License | Why this one |
|---|---|---|---|
| HDC primitives | `lfi_vsa_core` (workspace local) | (PlausiDen) | Already exists, battle-tested |
| FFT (for HRR binding) | `rustfft` 6.x | MIT/Apache-2.0 | Fastest pure-Rust FFT, no deps |
| Linear algebra | `nalgebra` 0.33 | Apache-2.0 / BSD-3 | Mature, broad |
| Tensor library | `ndarray` 0.16 | MIT/Apache-2.0 | Numpy-compatible memory layout |
| Tensor + autodiff (optional) | `candle-core` + `candle-nn` 0.7 | MIT/Apache-2.0 | HF Rust framework; minimal deps |
| Alt tensor + autodiff | `burn` 0.14 | MIT/Apache-2.0 | More featureful; bigger dep tree |
| Python bindings | `pyo3` 0.22 + `pyo3-build-config` | Apache-2.0/MIT | Industry standard |
| Numpy bridge | `numpy` (pyo3 feature) | BSD-3 | Required |
| Async runtime | `tokio` 1.x | MIT | For continuous-execution + WebSocket |
| Multi-thread | `rayon` 1.10 | MIT/Apache-2.0 | Data-parallel HDC ops |
| Channels | `crossbeam-channel` 0.5 | MIT/Apache-2.0 | Lock-free subscribe channels |
| Logging | `tracing` 0.1 + `tracing-subscriber` 0.3 | MIT | Per AVP-2 doctrine; structured logs |
| Log persistence | `tracing-appender` 0.2 | MIT | Rolling file appender |
| Log JSON | `tracing-subscriber` json feature | MIT | Machine-parseable logs |
| Telemetry (optional) | `opentelemetry` 0.27 | Apache-2.0 | Distributed tracing; off by default |
| Serialization | `serde` 1.x + `serde_json` + `serde_yaml` | MIT/Apache-2.0 | Standard Rust |
| Binary serialization | `bincode` 2.x | MIT | Session snapshot/restore |
| Safetensors | `safetensors` 0.4 | Apache-2.0 | HuggingFace weight format |
| YAML | `serde_yaml` 0.9 | MIT/Apache-2.0 | Architecture spec format |
| Hashing | `blake3` 1.x | CC0/Apache-2.0 | Fast content hashing for cache keys |
| UUID | `uuid` 1.x + v4 | Apache-2.0/MIT | Session IDs |
| Time | `chrono` 0.4 + `time` 0.3 | MIT/Apache-2.0 | ISO 8601 timestamps (per ISO doctrine) |
| Errors | `thiserror` 2.x + `anyhow` 1.x | MIT/Apache-2.0 | Standard Rust |
| HTTP server | `axum` 0.7 | MIT | For WebSocket bridge to frontend |
| WebSocket | `tokio-tungstenite` 0.27 | MIT | Used in Crawler already; consistency |
| Native GUI | `egui` 0.29 + `eframe` 0.29 | MIT | Immediate-mode GUI; mature |
| 3D rendering (native) | `wgpu` 23 + `bevy_egui` (if needed) | MIT/Apache-2.0 | Cross-platform GPU |
| Graph layout | `petgraph` 0.7 | MIT/Apache-2.0 | Standard Rust graph data structures |
| GPU memory query | `nvml-wrapper` 0.10 | MIT | NVIDIA management; safe Rust wrapper |
| System info | `sysinfo` 0.32 | MIT | CPU/RAM monitoring |
| Resource: ROCm | (no Rust binding) | — | Use FFI to ROCm SMI lib if AMD GPU |
| Snapshot diff | `similar` 2.x | Apache-2.0 | For diff visualisation |
| Random | `rand` 0.8 + `rand_chacha` | MIT/Apache-2.0 | For entropy gradient stochasticity |
| Property testing | `proptest` 1.x | MIT/Apache-2.0 | HDC algebraic property tests |
| Mutation testing | `cargo-mutants` (dev-time) | MIT/Apache-2.0 | AVP-2 mutation testing |
| Fuzz testing | `cargo-fuzz` + `libfuzzer-sys` | MIT/Apache-2.0 | Deserialisation fuzzing |
| Benchmarking | `criterion` 0.5 | Apache-2.0/MIT | Standard Rust bench harness |

### 13.2 Python packages (graphnet python package)

| Feature | Package | License | Why this one |
|---|---|---|---|
| Numpy interop | `numpy` >=1.26 | BSD-3 | Universal |
| REPL host | `ipython` >=8.20 + `jupyter` >=4 | BSD-3 | Standard ML REPL |
| Jupyter widget | `ipywidgets` >=8.1 | BSD-3 | For interactive controls |
| 2D plotting | `matplotlib` >=3.9 | PSF | Universal |
| Interactive 2D | `plotly` >=5.24 | MIT | Hover + zoom + click |
| 3D in Jupyter | `plotly` 3D + `k3d` >=2.16 | MIT/MIT | Two backends; switch per need |
| Alt 3D | `pythreejs` >=2.4 | BSD-3 | Three.js in Jupyter; alternative |
| Graph rendering | `graphviz` >=0.20 + `pygraphviz` >=1.12 | MIT/BSD-3 | Layout algorithms |
| Graph data | `networkx` >=3.3 | BSD-3 | Standard graph library |
| Resource monitor | `psutil` >=6.0 | BSD-3 | CPU + RAM cross-platform |
| GPU monitor (NVIDIA) | `pynvml` >=12 | BSD-3 | NVIDIA management library |
| GPU monitor (AMD) | `rocm-smi-lib` (Python binding) | MIT | If AMD GPUs present |
| FLOP analysis | `fvcore` >=0.1.6 + `ptflops` >=0.7 | Apache-2.0/MIT | Two complementary tools |
| HuggingFace LLMs | `transformers` >=4.45 | Apache-2.0 | Industry standard |
| HF tokenizers | `tokenizers` >=0.20 | Apache-2.0 | Standard |
| PyTorch | `torch` >=2.5 | BSD-3 | Required by transformers + mamba |
| JAX | `jax` >=0.4.34 + `jaxlib` | Apache-2.0 | For JAX-side models (optional) |
| Mamba | `mamba-ssm` >=2.2 | Apache-2.0 | Official Mamba implementation |
| GNN adapter (v2+) | `torch-geometric` >=2.6 | MIT | Standard GNN framework |
| Dimensionality reduction | `scikit-learn` >=1.5 | BSD-3 | PCA, t-SNE |
| UMAP | `umap-learn` >=0.5 | BSD-3 | Better for HDC vector projection |
| Wavelet transforms | `pywavelets` >=1.7 | MIT | Multi-resolution; FFT alternative |
| Symbolic maths | `sympy` >=1.13 | BSD-3 | For derivation display + LaTeX rendering |
| LaTeX rendering | `matplotlib` mathtext + `nbconvert` | PSF | Display equations |
| Live coding | `nbformat` >=5.10 + `jupyter-server` | BSD-3 | Notebook export/import |
| Visual regression | `pixelmatch-py` >=0.3 | MIT | Image diff |
| Property testing | `hypothesis` >=6.115 | MPL-2.0 | Python equivalent of proptest |
| Pytest | `pytest` >=8.3 + `pytest-asyncio` | MIT | Test runner |
| Docs (later) | `mkdocs-material` >=9.5 | MIT | Standard docs site |

### 13.3 Per-feature FOSS choice map (collated)

For each owner-asked feature, the exact FOSS that does it:

| Feature | FOSS chosen | Why |
|---|---|---|
| 2D architecture graph | graphviz + networkx + plotly | Layout algorithms + interactivity |
| **3D rotatable architecture graph** | plotly 3D OR k3d-jupyter | Both work in Jupyter; k3d for >10k nodes |
| Click-to-select node panel | ipywidgets + plotly callbacks | Native widget infra |
| Drag-to-reposition | plotly drawmode (limited) → custom widget needed in Phase 4 | — |
| Live heatmap | matplotlib + plotly + GraphNet subscribe channel | — |
| Live time-series | plotly streaming | — |
| Hypervector PCA viz | scikit-learn PCA + plotly 3D scatter | — |
| Hypervector UMAP viz | umap-learn + plotly | UMAP better for HDC structure preservation |
| Wavelet view | pywavelets + matplotlib | — |
| Spectral (FFT) view | numpy.fft + matplotlib + interactive plotly | — |
| **Equation rendering** | matplotlib mathtext + sympy LaTeX | For maths display |
| Resource gauges | psutil + pynvml + ipywidgets Gauge | — |
| Cost estimator | analytical formulas + electricity-rate config | — |
| Architecture spec format | serde_yaml + jsonschema | YAML for human-edit |
| Weights format | safetensors | HF-compatible |
| Session format | bincode + zstd compression | Compact |
| Diff viewer | similar (Rust) + difflib (Python) | — |
| Recording/playback | bincode session + replay queue | — |
| Native GUI shell | egui + wgpu + tauri (later) | Mature Rust GUI |
| Plugin system | `libloading` 0.8 + dyn-trait pattern | Dynamic Rust libraries |
| LLM judge (suggestions) | OpenRouter API OR local Mistral 7B via llama.cpp | Configurable |

## 14. Logging + debugging architecture

Owner direction: *"put lots of debugging logs that can be checked later."*

Structured tracing via `tracing` (Rust) + `logging` (Python), aggregated to
disk + queryable.

### 14.1 Log streams

| Stream | Source | Default level | Format | Storage |
|---|---|---|---|---|
| `engine.*` | graphnet-engine Rust core | INFO | JSON | `logs/engine-YYYY-MM-DDTHH.json` (rotating) |
| `intervene.*` | every intervention API call | DEBUG | JSON | `logs/intervene-YYYY-MM-DDTHH.json` |
| `forward.*` | every forward pass | DEBUG | JSON | `logs/forward-YYYY-MM-DDTHH.json` |
| `viz.*` | visualisation layer | INFO | JSON | `logs/viz-YYYY-MM-DDTHH.json` |
| `adapter.<name>.*` | per adapter (transformer / mamba / ...) | INFO | JSON | `logs/adapter-<name>-YYYY-MM-DDTHH.json` |
| `monitor.*` | resource monitor samples | INFO | JSON | `logs/monitor-YYYY-MM-DDTHH.json` |
| `audit.*` | AVP-2 audit findings | INFO | JSON | `logs/audit-YYYY-MM-DDTHH.json` |

All logs:
- Tagged with `session_id` (UUID v4)
- Tagged with `trace_id` (per forward pass)
- Tagged with `span_id` (per operation within a forward)
- Wall-clock + monotonic ns timestamps (ISO 8601 per ISO doctrine)
- Caller location (`file:line`)

### 14.2 Log retention

- Hourly rotation; gzip after rotation
- Default retention 30 days; configurable
- Total cap (default 5 GiB) — oldest deleted when exceeded
- Logs written to `~/.graphnet/logs/<session_id>/` by default;
  configurable to project-local

### 14.3 Replay from logs

Critical capability: *check later* implies replayability.

```python
import graphnet as gn

# Open a past session by log directory
session = gn.replay.open("~/.graphnet/logs/abc-123/")

# Inspect interventions made
for i in session.interventions:
    print(i.timestamp, i.target, i.action)

# Reconstruct the model state at a specific point
model = session.model_at(intervention_index=5)

# See the inputs that went through
for forward in session.forwards:
    print(forward.timestamp, forward.input_shape, forward.output_shape)
```

### 14.4 Debug REPL hooks

When `GRAPHNET_DEBUG=1`:
- Every forward pass dumps full intermediate state to log
- Every intervention dumps before+after diff
- Every visualisation update is logged
- Resource samples every 100ms (vs. 1s default)
- Stack trace on every error (not just panic)

When `GRAPHNET_TRACE=op:bind`:
- Only logs binding operations
- Useful for narrow debugging

### 14.5 Performance profiling

- Built-in `gn.profile.start()` / `gn.profile.stop()` — wraps `cargo flamegraph`-
  equivalent or `perf` on Linux, Instruments on macOS
- Outputs flamegraph SVG + per-op latency table
- Integrates with the live resource panel

## 15. GUI-first UX philosophy

Owner direction: *"i really need it to be intuitive but fully capable... i
dont want to have to type a lot and use commands but also have that option
available."*

Design principle: **every operation has a GUI path AND a REPL path. Neither
is hidden; both are first-class. The GUI path is the default for new users;
the REPL path is the default for repeated work.**

### 15.1 Default launch experience

```
$ graphnet hello
```

Opens a Jupyter notebook tutorial titled "GraphNet in 5 minutes":
- Step 1: Load a Stack with one click → preset palette of starter
  architectures
- Step 2: Run a sample input via drag-and-drop → see the output
- Step 3: Look at the 3D graph → rotate it
- Step 4: Click a layer → see its weights
- Step 5: Drag a slider to change a weight → watch the output change

User never types a line of code unless they want to. At the end of step 5,
there's a "Show me the code" button that reveals the equivalent REPL
sequence — onboarding to the power-user mode.

### 15.2 Three interaction modes

| Mode | Used by | What it looks like |
|---|---|---|
| **Pure GUI** | First-time users, mom-class | Drag/drop nodes, click to configure, sliders for parameters, no code visible |
| **Mixed** | Most use | GUI for navigation + viz; REPL for batch work, custom queries |
| **Pure REPL** | Power users, automation | Notebook with full Python API; viz on demand via `.viz` |

### 15.3 GUI-first operation patterns

Every operation that exists in the REPL has a GUI gesture:

| Operation | REPL | GUI |
|---|---|---|
| Load model | `gn.load("file.pt")` | File → Open, OR drag file onto canvas |
| Forward pass | `model.forward(x)` | Drag input onto input port; auto-runs |
| Continuous mode | `model.run_continuous(stream)` | Click ▶️ button; pause with ⏸️ |
| Probe a tensor | `model.probe("path.to.x")` | Right-click any node → "Probe" |
| Heatmap a probe | `probe.viz_heatmap()` | Probe automatically heatmapped in panel |
| Intervene on a weight | `model.weights[k] = v` | Click weight cell → edit value → Enter |
| Add a Stack op | `stack.add_operation("fft_bind")` | Right-click Stack → "Add Op" → pick from menu |
| Remove an op | `stack.remove_operation(i)` | Click op → Delete key |
| Change entropy | `stack.tau = 0.5` | Slider on Stack node, 0.0–1.0 |
| Compare models | `gn.compare(m1, m2)` | Drag two models onto compare canvas |
| Export config | `model.export("file.yaml")` | File → Export → choose format |
| Import config | `gn.import("file.yaml")` | File → Import, OR drag file onto canvas |
| Snapshot | `model.snapshot()` | Ctrl/Cmd + S |
| Restore | `model.restore(snap)` | Ctrl/Cmd + Z (most recent) OR History panel |
| Audit | `model.audit()` | Click 🔍 Audit button → report opens in side panel |

## 16. HDC operations — complete GUI counterpart map

Per owner direction: *"have all HDC commands available with GUI counterparts."*

All HDC primitives exposed in the REPL appear as draggable nodes in the GUI.
Connect nodes with edges; the system runs the operation when a complete
path exists from input to output.

| HDC operation | REPL | GUI node | Visual feedback |
|---|---|---|---|
| Random hypervector | `gn.hd.random(D=10000)` | "Random" node, slider for D | Generates new vec on click |
| Bind | `gn.hd.bind(a, b)` | "⊗ Bind" node, 2 input ports, 1 output | Output magnitude meter |
| Bundle | `gn.hd.bundle(a, b, ...)` | "+ Bundle" node, N input ports, 1 output | Output similarity to each input shown |
| Unbind | `gn.hd.unbind(c, k)` | "⊗⁻¹ Unbind" node, 2 input ports (vec, key), 1 output | Cosine sim with expected operand |
| Permute | `gn.hd.permute(v, n=1)` | "Π Permute" node, input + shift slider | Animated rotation visualization |
| Inverse | `gn.hd.inverse(v)` | "Inv" node, 1 input, 1 output | Shows v ⊗ v⁻¹ ≈ identity |
| Negate | `gn.hd.negate(v)` | "Neg" node, 1 input, 1 output | Bipolar flip visualization |
| Cosine similarity | `gn.hd.cos_sim(a, b)` | "cos" node, 2 inputs, scalar output | Big number display + gauge |
| Hamming distance | `gn.hd.hamming(a, b)` | "Δ Hamming" node, 2 inputs, scalar output | Big number display |
| Threshold (binarize) | `gn.hd.threshold(v)` | "T Threshold" node, input + threshold slider | Histogram before/after |
| Cleanup (NN search) | `gn.hd.cleanup(v, codebook)` | "🔍 Cleanup" node, vec + codebook input | Top-K candidates with sim scores |
| HRR / FFT bind | `gn.hd.hrr_bind(a, b)` | "⊗ᶠ FFT Bind" node, 2 inputs, 1 output | FFT magnitude spectrum side-by-side |
| HRR / FFT unbind | `gn.hd.hrr_unbind(c, k)` | "⊗⁻¹ᶠ FFT Unbind" node | Spectrum view |
| Tensor product bind | `gn.hd.tensor_bind(a, b)` | "⊗ₜ Tensor Bind" (creates higher-dim) | Matrix view |
| Bipolar XOR (efficient bind) | `gn.hd.xor_bind(a, b)` | "⊕ XOR Bind" node | Bit-flip count visualization |
| Sparse projection | `gn.hd.sparse_project(v, density)` | "↓ Sparse" node, density slider | Sparsity bar |
| Position encoding | `gn.hd.position(v, pos)` | "Pos" node, vec + position | Animated position-key binding |
| Sequence encoding | `gn.hd.encode_sequence(seq)` | "Seq" node, input = list of vecs | Visualised binding chain |
| Set encoding | `gn.hd.encode_set(items)` | "Set" node, input = list of vecs | Bundle aggregation viz |
| Pair encoding | `gn.hd.encode_pair(a, b, key_a, key_b)` | "Pair" node, 4 inputs | Composite vec output |

Stack-level operations (also exposed):

| Stack operation | REPL | GUI node |
|---|---|---|
| Create empty Stack | `gn.Stack(D=10000)` | "Empty Stack" panel; D slider |
| Add Dense op | `stack.add_dense()` | Drag "Dense" op into Stack panel |
| Add FFT op | `stack.add_fft()` | Drag "FFT" op into Stack panel |
| Add FNet-style mixing | `stack.add_fnet()` | Drag "FNet" op into Stack panel |
| Add Gated routing | `stack.add_gated()` | Drag "Gate" op |
| Add Aggregator | `stack.add_aggregator()` | Drag "Aggregator" op |
| Add Identity / skip | `stack.add_identity()` | Drag "Identity" op |
| Set entropy | `stack.tau = 0.3` | Slider on Stack panel header |
| Connect Stacks | `stack1 >> stack2` | Drag output port → input port |
| Build Stack-of-Stacks | `gn.StackOfStacks([s1, s2, ...])` | "Compose" panel; drag Stacks in |

## 17. Advanced mathematics — graphical maths support

Owner direction: *"cover advanced maths and make it easy to use the maths
graphically."*

A dedicated **Maths Panel** in the GUI exposes mathematical operations as
visual primitives. Every operation has live visualisation.

### 17.1 Mathematical operations exposed

**Linear algebra**
- Matrix multiply (2D + 3D tensor product visualisation)
- SVD (singular value decomposition) — bar chart of singular values
- Eigendecomposition — eigenvalue plot + eigenvector display
- QR decomposition — matrix views
- Matrix inverse (with condition number warning if ill-conditioned)
- Pseudoinverse (Moore-Penrose) for non-square matrices
- Cholesky decomposition (for symmetric positive-definite)
- LU decomposition with permutation matrix display
- Matrix rank computation with rank-deficient warning

**Spectral analysis**
- FFT / iFFT — magnitude + phase spectra side-by-side
- 2D FFT — image spectrum view
- DCT (discrete cosine transform) — for compression viz
- Wavelet transform (Daubechies, Haar, Symlet) — multi-resolution plot
- Spectrogram (STFT) — time-frequency heatmap
- Hilbert transform — analytic signal envelope view

**Probability + statistics**
- Gaussian sampling (parameterised; live histogram)
- Multivariate Gaussian with covariance ellipsoid display
- KDE (kernel density estimation) — smooth density viz
- Histograms with adjustable bins
- Q-Q plots
- Hypothesis test (t-test, KS, chi-squared) with p-value display
- Information theory: entropy, mutual information, KL divergence — numeric + plot

**Dimensionality reduction (live)**
- PCA — variance-explained plot + scatter
- t-SNE — animated convergence
- UMAP — interactive 3D embedding
- Isomap, LLE, Spectral embedding — comparative views
- Random projection (for HDC)

**Optimization**
- Gradient descent visualised on a 2D / 3D loss surface
- Momentum, Adam, RMSProp animated trajectories side-by-side
- Newton's method visualisation
- Conjugate gradient
- Trust region methods

**Calculus on tensors**
- Numerical gradient (finite difference) visualisation
- Jacobian as heatmap
- Hessian as 3D surface
- Divergence + curl for vector fields

**Topology + manifolds**
- Persistent homology (for understanding HDC vector cluster structure)
- Manifold embedding visualisation
- Geodesic distance

**Symbolic maths (display only)**
- Equation rendering via LaTeX
- Derivation chains (sympy-derived) shown step-by-step
- Symbolic simplification + display

### 17.2 GUI for maths

The Maths Panel layout:

```
┌─ Maths Panel ─────────────────────────────────────────────┐
│ [ Linear ] [ Spectral ] [ Stats ] [ DimRed ] [ Optim ] ... │
│                                                            │
│ ┌────────────────┐  ┌─────────────────────────────────┐   │
│ │ Operation tree │  │ Live visualisation              │   │
│ │ - Matrix mul   │  │                                 │   │
│ │ - SVD          │  │   [3D rotatable plot here]      │   │
│ │ - Eigendecomp  │  │                                 │   │
│ │ ...            │  │                                 │   │
│ └────────────────┘  └─────────────────────────────────┘   │
│ ┌────────────────────────────────────────────────────┐    │
│ │ Inputs (drag tensors here)                          │    │
│ │ A: [tensor of shape (768, 768)]                     │    │
│ │ B: [tensor of shape (768, 100)]                     │    │
│ └────────────────────────────────────────────────────┘    │
│ Live derivation:                                           │
│    AB = A · B  where (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ                  │
│ Result: [tensor of shape (768, 100)] [view ▼]             │
└────────────────────────────────────────────────────────────┘
```

Click any operation in the tree, drag in tensors, see the result + the
mathematical derivation rendered in LaTeX, with a 3D plot if the operation
produces something visualisable.

For tensors that come from the loaded model (weights, activations), drag
straight from the architecture graph into the Maths Panel inputs.

### 17.3 GUI-first equation editor

A WYSIWYG equation editor where users compose mathematical expressions by
dragging tensor symbols + operator symbols, not by typing LaTeX. The result
is executable code; clicking "Run" applies it to the current tensors.

For users who prefer typing: a LaTeX input field exists. Both produce the
same internal representation.

## 18. Future-proofing strategy

Owner direction: *"try to future proof it as much as possible."*

### 18.1 Hardware abstraction

- Compute backend trait — `CpuBackend`, `CudaBackend`, `MetalBackend`,
  `VulkanBackend`, `WgpuBackend`, future `TpuBackend` / `QuantumBackend`
- All HDC ops dispatch through the backend trait
- Per-tensor backend choice; not per-model
- Adding a new backend = implementing the trait, not rewriting users

### 18.2 Model family abstraction

- `Adapter` trait — any AI family wraps to GraphNet's `Model` interface
- v1 adapters: HDC/Stack, Transformer, Mamba
- Adding a new adapter = implementing the trait + tests
- Plugin-loadable adapters (libloading) so adapters can ship outside core

### 18.3 Serialisation forward-compatibility

- Architecture spec YAML uses versioned schema (`version: "1.0"`); migration
  paths required for breaking changes
- Weight format = safetensors (HF de-facto standard; extremely stable)
- Session format = bincode + a `format_version` field; loader migrations
  shipped with each breaking version
- Spec schema published in `docs/SPEC_SCHEMA.md` so third parties can author
  configs

### 18.4 API stability promise

- Public Python API (`graphnet.*`) follows semver
- 0.x = breaking allowed; 1.0+ = breaking requires major bump
- Deprecation cycle: warn for 1 minor, remove on next major
- Stay in 0.x until Phase 11 ships

### 18.5 Plugin architecture

- Custom operations in `Stack` are pluggable via `Operation` trait
- Custom adapters via `Adapter` trait (loaded via libloading)
- Custom visualisations via Python `viz` plugin discovery (entry points)
- Custom maths operations via Python `maths` plugin discovery

### 18.6 LLM-driven REPL (future)

When a strong local LLM is available, optional natural-language interface:
*"show me what the attention pattern looks like at layer 5 when I feed it
'the cat sat on the mat'"* — LLM judge translates to GraphNet API calls,
runs, shows results.

Off by default; opt-in when configured with an LLM endpoint.

### 18.7 Quantum-readiness (speculative)

If quantum computing ever becomes practical for ML:
- HDC binding via QFT (Quantum Fourier Transform) is natively quantum-friendly
- The `Backend` trait already abstracts; quantum is just another backend
- Architecture survives the substrate change

### 18.8 Cross-platform binaries

- Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64) all
  supported from day 1
- Python wheels published for all three
- Native GUI ships as platform-specific binary in Phase 12

### 18.9 Internationalization

- Strings extracted to translation tables from v1
- LaTeX rendering supports Unicode mathematical symbols
- Locale-aware number formatting (per ISO 8601 for dates, per locale for
  decimal separators)

### 18.10 Long-term documentation

- Each Phase ships with a Phase Audit Report (`docs/audits/phase-N.md`)
- ARCHITECTURE.md updated per Phase
- Design Decision Log (`docs/adr/`) per significant choice
- Tutorial notebooks versioned alongside major releases

## 19. Loop directive

The build runs autonomously per owner direction. Loop firing pattern:

**Per tick (one bounded shippable unit):**

1. CronList — confirm loop is alive
2. Check task list for next pending GraphNet-related task
3. Pick the lowest task that's not blocked
4. Implement the bounded unit
5. Mandatory pre-push:
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets --locked`
   - `cargo test --workspace --locked`
   - `pytest python/tests/`
   - If anything fails: fix, don't push
6. Commit (Co-Authored-By: Claude Opus 4.7) + push origin main
7. Mark task as completed
8. Append log entry to `~/.graphnet/build-log.md`

**Stop conditions** (loop exits cleanly when ALL hold):
1. All Phase 0–10 tasks completed
2. CI green on origin/main HEAD
3. Visual regression tests passing
4. AVP-2 audit report shows no STRICT findings on critical-path code
5. README, ARCHITECTURE.md, tutorial notebooks all current

**Stop-and-report conditions** (loop exits non-cleanly, reports to owner):
1. Hard compile error not resolvable in 3 ticks
2. Test failure on shipped code (regression)
3. CI failure on a SHA the loop pushed
4. Disk full / network unreachable
5. Owner adds a task marked priority:owner-decision

**Never:**
- Force-push to main or master
- Skip pre-push gates
- Push to chronic-failure repos (forge-audit / SkillShots label_consistency)
- Touch out-of-scope (Sacred*, plausiden.com, Mailroom, Vault, Shield)

## 20. Where this doc lives + maintenance

This doc lives at `/home/user/Development/PlausiDen/PlausiDen-AI/lfi_vsa_core/
GRAPHNET_BUILD_PLAN.md` and is the source of truth for GraphNet planning.

When implementation begins, this doc is mirrored to
`/home/user/Development/PlausiDen/PlausiDen-GraphNet/docs/PLAN.md` for
co-location with the implementation.

Updates to the plan during implementation:
- Phase scope changes → update relevant Phase section + bump revision footer
- New owner direction → add new section + reference it from Phase that
  incorporates it
- Lessons learned → append to a `## 21. Build log` section as we go
- Don't delete sections — strike-through with reason

---

*Revision: 2026-05-17 (initial). Author: Claude Opus 4.7 from owner direction
across session 49b47b3b-5d2a-4b32-87ad-c5bca20dccc4.*

---

## 21. Additional features — triple-check pass

Per owner direction *"try to think if anything else i forgot or didnt think
if to add in GraphNet"*, the following ship as features. Each was missing or
under-specified in §1–§20.

### 21.1 Time-travel debugging

Neural network execution is hard to debug because state is huge and ephemeral.
GraphNet ships with first-class time-travel:

- **Step**: single-step through a forward pass, layer-by-layer or op-by-op.
  At each step, full intermediate state is visualised + queryable.
- **Backstep**: undo a step. Return to the state before the last op ran.
- **Continue**: resume forward to next breakpoint or end.
- **Replay**: re-execute from any saved snapshot.
- **Watchpoints**: register a callback that fires when a named tensor's
  value crosses a threshold (e.g., "alert when attention.head[3] mean > 0.8").
- **Breakpoints**: pause execution before a named operation runs.
- **Conditional breakpoints**: break only when a condition is true (e.g.,
  "break at layer 5 when input is over 100 tokens").
- **Backwards execution**: re-run a previous forward pass — bytewise
  deterministic given the same RNG seed.

REPL API:
```python
debugger = model.debug()
debugger.add_breakpoint("stacks[2].operations[1]")
debugger.add_watchpoint(
    target="stacks[0].output",
    condition=lambda v: v.mean() > 0.5,
)
out = model.forward(x)  # pauses at breakpoint
debugger.step()         # single op forward
debugger.backstep()     # undo it
debugger.continue_()    # run to next break
```

GUI: a toolbar with ▶️ ⏸️ ⏭️ ⏮️ buttons; breakpoint/watchpoint manager in
side panel.

### 21.2 Crash recovery + autosave

Sessions get long; crashes are unforgivable.

- **Autosave**: full session state written to disk every 30 seconds (config).
- **WAL-style append-only history**: every intervention is also written to a
  separate journal file — fastest path to "what did I just do."
- **Lost-recovery dialog**: on next start after a crash, GraphNet detects
  unflushed session, offers to restore. Like Photoshop's recovery.
- **Concurrent-write safety**: lock file with PID; warn if another process
  is editing the same session.

### 21.3 Architecture template library

Owner uses "starter architectures" frequently. Ship a curated library:

| Template | Description |
|---|---|
| `stack-tiny` | 1-stack, 3 ops, D=1024, toy-task starter |
| `stack-standard` | Stack-of-stacks, D=10,000, suitable for real tasks |
| `stack-deep` | 8-level stack-of-stacks for advanced experiments |
| `transformer-tiny` | 4-layer transformer; for quick smoke-tests |
| `transformer-gpt2-small` | GPT-2 124M; standard baseline |
| `mamba-130m` | Mamba 130M; standard baseline |
| `fnet-base` | FNet (FFT-mixing transformer); for spectral comparison |
| `hybrid-stack-transformer` | Stack + transformer hybrid; experimental |
| `hybrid-stack-mamba` | Stack + Mamba hybrid; experimental |
| `hdc-from-scratch` | Empty HDC playground for hand-building |

`graphnet.gallery` Python API; `gn.gallery.load("stack-tiny")` returns ready-
to-explore model. Same in GUI: File → New → choose template.

### 21.4 Benchmark library

Built-in standard tasks for evaluating any model end-to-end. No need for
external dataset wrangling for routine smoke-tests.

| Benchmark | What it measures | Implementation |
|---|---|---|
| `associative-recall` | HDC associative memory baseline | Synthetic; native |
| `parity-check` | Long-range information binding | Synthetic; native |
| `mnist-classification` | Image classification basic | Downloaded; cached |
| `wikitext-perplexity` | Language modeling | Downloaded; cached |
| `glue-cola` | Linguistic acceptability | Downloaded; cached |
| `arc-easy` | Reasoning | Downloaded; cached |
| `needle-in-haystack` | Long-context retrieval | Synthetic; native |
| `ruler-subtasks` | Long-context reasoning (multi-needle, var-tracking) | Synthetic; native |
| `hdc-cleanup-noise` | HDC robustness | Synthetic; native |

`model.benchmark("needle-in-haystack")` runs + reports metric + log entry.

### 21.5 Experiment tracker integrations

For users who already use these tools.

- **Weights & Biases** (`wandb`): `model.connect_wandb(project="my-exp")` —
  metrics, visualisations, and config sync to wandb cloud
- **MLflow** (`mlflow`): local or remote tracking server
- **TensorBoard**: legacy but widely understood; SummaryWriter integration
- **Neptune.ai**: alternative tracker

All optional; off by default. Configurable.

### 21.6 REST API for external control

Some users want to drive GraphNet from a script outside the REPL.

```
GET  /api/v1/sessions                       # list sessions
POST /api/v1/sessions                       # create session
GET  /api/v1/sessions/{id}/models           # list loaded models
POST /api/v1/sessions/{id}/models           # load model
POST /api/v1/sessions/{id}/forward          # run forward pass
POST /api/v1/sessions/{id}/intervene        # apply intervention
GET  /api/v1/sessions/{id}/probes/{name}    # current probe value
WS   /api/v1/sessions/{id}/stream           # live event stream
```

Off by default; opt-in via `gn.serve(port=8765)`. Local-only by default;
bind-to-LAN requires explicit `--lan` flag (per Forge convention).

### 21.7 AI-assisted notebook (LLM-judge)

Owner is building an AI; GraphNet builds AIs for them. Closing the loop:

GraphNet's notebook can host an embedded AI assistant (configurable
endpoint — Anthropic API, OpenRouter, local llama.cpp, etc.) that:

- Reads the current notebook context + model state
- Suggests architectural improvements ("this attention head looks redundant")
- Generates intervention code from natural-language instructions
- Explains what each visualisation means
- Walks through tutorials interactively

Off by default; user provides their own API key OR points at a local model.

### 21.8 Architecture provenance + lineage

Owner explicitly cares about supersociety — verifiable provenance is part of
that doctrine.

- Every architecture spec gets a content hash (blake3) at save time
- Loading a spec records its hash
- Modifying a loaded spec creates a child hash with parent-pointer
- Full lineage graph: this architecture descended from these architectures
- Citation tracking: each architectural choice can link to a paper / commit /
  prior architecture; visible in the GUI as a "References" hover
- Export includes lineage data; on import, parents can be downloaded if URLs
  are provided

### 21.9 Sharing via GitHub Gist / OpenReview / HF Hub

One-click upload to known sharing platforms:

- `model.share("gist")` — uploads architecture spec + optional weights to a
  GitHub Gist, returns the URL
- `model.share("huggingface")` — uploads to HF Hub under user's account
- `model.share("openreview")` — formats as supplementary material for paper
  submission

Authentication via existing tokens (env vars or `~/.graphnet/credentials`).

### 21.10 AVP-2 security model

Per AVP-2 doctrine, GraphNet's own security must be hardened.

| Layer | Mechanism |
|---|---|
| Session logs | Immutable append-only journal; blake3-hashed |
| Tamper detection | Sequential hash chain; any modification visible |
| Encryption at rest | AES-256-GCM via `aes-gcm` crate; key in OS keyring |
| Encryption in transit | TLS 1.3 only (REST API + WebSocket); no plaintext |
| Authentication | Local OS user by default; API key for REST |
| Authorization | RBAC for shared sessions (Phase 12+) |
| Secret scanning | gitleaks pre-commit hook on `examples/` |
| Audit log | Every intervention recorded with: who, when, what changed |
| Dependency audit | `cargo audit --deny warnings` on every push |
| Supply chain | `cargo vet` + reproducible builds |

### 21.11 Deployment / install story

How does an end-user get GraphNet?

| Channel | Method |
|---|---|
| Python users | `pip install plausiden-graphnet` (PyPI) |
| Rust users | `cargo add graphnet-engine` (crates.io) |
| Native GUI | `.app` / `.exe` / `AppImage` on GitHub Releases |
| Docker | `docker run plausiden/graphnet:latest` |
| Source | `git clone && pip install -e .` |
| Reproducible (Nix) | `nix run github:thepictishbeast/PlausiDen-GraphNet` |

Each channel ships from the same source; CI publishes to all on tagged
releases.

### 21.12 Update mechanism

- Pip / cargo auto-detect updates (out-of-the-box)
- Native GUI shows "Update available" notification non-intrusively
- All updates configurable: opt-in to pre-releases, opt-out of auto-update
- Changelog visible from within GraphNet (Help → What's New)

### 21.13 Help system

Every operation has hover-help.
Every node shows a `?` icon → full docs panel.
Every error message links to a docs page explaining the cause.
`graphnet help <topic>` opens topic docs in browser.
Tutorial notebooks accessible from File → Tutorials.

### 21.14 Keyboard shortcuts (power user mode)

Comprehensive shortcut map, all rebindable. Configurable per user.

| Action | Default shortcut |
|---|---|
| Save session | Ctrl/Cmd+S |
| Open session | Ctrl/Cmd+O |
| New session | Ctrl/Cmd+N |
| Undo intervention | Ctrl/Cmd+Z |
| Redo intervention | Ctrl/Cmd+Shift+Z |
| Run forward | Ctrl/Cmd+Enter |
| Continuous mode toggle | Space |
| Step (debugger) | F10 |
| Backstep | Shift+F10 |
| Add probe | Ctrl/Cmd+P |
| Open command palette | Ctrl/Cmd+K |
| Search anything | Ctrl/Cmd+/ |
| Toggle 3D view | Ctrl/Cmd+3 |
| Toggle dark mode | Ctrl/Cmd+D |
| Open Maths Panel | Ctrl/Cmd+M |
| Open Resources Panel | Ctrl/Cmd+R |

### 21.15 Power-user command palette

`Ctrl/Cmd+K` opens a fuzzy-search command palette like VS Code. Every GUI
action is reachable by keyword.

### 21.16 Performance regression CI gate

Beyond unit/integration tests:

- Every PR runs the benchmark suite
- Compares against `main` baseline
- Regression > 10% on any benchmark = CI failure
- Regression < 10% = warning posted on PR

Prevents silent slowdowns.

### 21.17 Memory leak + race condition detection

- `valgrind` runs in CI on a representative session
- `loom` (Rust crate) for testing concurrent code paths
- `thread sanitizer` builds available in CI
- Heap profiling via `dhat-rs` integrated; results in audit reports

### 21.18 Privacy mode

For users running GraphNet on sensitive data:

- Tensors marked `sensitive` are never logged at full precision (only shapes)
- PII patterns in text inputs are auto-redacted before logging (NER-based)
- Session export can scrub sensitive tensors on request
- "Forget session" command: shred all artefacts of a session, including
  logs (overwrites + unlinks, per AVP-2)

### 21.19 Sandboxed execution (untrusted models)

Loading a third-party model = running their code. GraphNet wraps untrusted
adapter loads in a sandbox:

- bwrap-style isolation (same primitives Loom's T46 uses)
- Resource ceilings (CPU/memory/network) per AVP-2
- Network egress allowlist (only model.safetensors + tokenizer.json hosts)
- Read-only filesystem outside the model directory

Off by default for trusted (PlausiDen-built) adapters; on by default for
HuggingFace community models.

### 21.20 Webcam + microphone + image / audio inputs

For interactive experimentation with vision and audio models:

- Webcam stream input (cross-platform via `nokhwa` Rust crate or Python
  `opencv-python`)
- Microphone stream input (`cpal` or `pyaudio`)
- Image upload (drag + drop into GUI)
- Audio file upload
- Live transcription view (when running an STT model)
- Live generation view (when running TTS, image synthesis, etc.)

### 21.21 MIDI / OSC for experimental music + live performance

Speculative but interesting: bind MIDI controllers to model parameters,
play the model like an instrument.

- `midir` (Rust) for MIDI in/out
- `rosc` (Rust) for OSC
- Map any controller knob to any model parameter
- Useful for live AV performance experiments (artists doing real-time
  AI manipulation)

Optional; ships in Phase 11+.

### 21.22 Notebook → script export

A Jupyter exploration as a standalone Python script that reproduces the
result, without GraphNet GUI dependency. For deploying experiments to
remote compute.

`graphnet notebook-to-script my-experiment.ipynb` → `my-experiment.py`

### 21.23 Documentation generator

Given a loaded model, generate a paper-appendix-quality markdown describing
the architecture:
- Architecture diagram (auto-rendered)
- Per-layer description with shape + parameter count
- Cited prior work (from architecture lineage)
- Performance numbers (from session benchmarks)
- LaTeX-rendered equations for novel operations

`model.generate_docs("my-arch-paper.md")`. Useful for paper appendices,
internal review, or just preserving understanding.

### 21.24 Hardware advisor

Given an architecture, recommend:
- Minimum hardware required (CPU class, RAM, GPU class + VRAM)
- Recommended hardware for fluid experimentation
- Cloud instance equivalents (AWS, GCP, Azure, RunPod)
- Estimated training time on each
- Estimated training cost on each

`model.hardware_advice()` → tabular output.

### 21.25 Energy + carbon dashboard

Per-session energy estimate via NVIDIA-SMI power draw integrated over time.
Multiply by grid carbon intensity (configurable per region; default uses
gridintensity.org-style data) → CO₂-equivalent.

Visible in the resource panel. Per-architecture comparison shows which
designs are more energy-efficient for the same task.

### 21.26 Failure-mode catalog

Built-in catalog of known neural-network failure modes; the audit tool
checks against them:
- Gradient explosion / vanishing
- Dead neurons (all zero or all max activation)
- Saturation
- Mode collapse (for generative models)
- Attention sink (one head dominates)
- Token forgetting (long-context degradation)
- Repetition loops
- Hallucination patterns (when LLM)

Audit produces a "health report" highlighting which failure modes are present
or imminent based on current state.

### 21.27 Multi-language localization

The GUI ships English-first; translation infrastructure ready from v1.
Initial languages: English, Spanish, French, German, Japanese, Chinese
(simplified), Hindi. Community translations welcomed (gettext-style PO
files).

LaTeX rendering handles Unicode mathematical symbols across all locales.
Locale-aware number formatting (1,234.56 vs 1.234,56) without breaking
internal numerics (locale only affects display).

### 21.28 Offline-first

GraphNet works without internet:
- All dependencies vendorable (cargo vendor + pip download)
- Adapter model weights downloaded once + cached
- Update mechanism opt-in (default off)
- Telemetry off by default (and even when on, batched + opt-out)
- No phone-home in core code paths

### 21.29 Snapshot to GitHub PR / Gist

`graphnet share` opens an interactive prompt: "share via GitHub Gist?
GraphNet Cloud (when available)? HF Hub?" — picks the right channel,
authenticates via existing tokens, returns a URL.

For reviews + collaboration without setting up shared infrastructure.

### 21.30 Collaborative real-time editing (Phase 13+, speculative)

Multiple users editing the same session in real time:
- CRDT-based state synchronisation (`automerge` or `yrs` Rust crates)
- WebSocket connection between editors
- See other users' cursors + selections
- Conflict-free intervention merging

Speculative; not in v1. Mentioned for future-proofing the architecture
(API stability with eventual collaboration in mind).

## 22. Triple-check log

Re-read of the entire plan (§1–§21) on 2026-05-17. Findings + fixes:

| Section | Issue caught | Fix |
|---|---|---|
| §2.2 | "Stacking different AI types" — example used image input but didn't specify how tensor shapes get translated between adjacent components | Added shape-translation as adapter responsibility |
| §2.4.2 | "Click to select" mentioned but no spec for the panel that opens | §15 made this explicit (GUI counterparts §16) |
| §2.7 | Diff format unspecified | §9 export format spec covers it |
| §3 | Architecture diagram missing the maths panel | Implicitly covered by §17 — maths panel reuses viz layer infrastructure |
| §4 | "FOSS picks" listed but not mapped to features | §13.3 explicit per-feature map added |
| §6 | Repo structure missing tests/ for visual regression specifically | Already present as `tests/visual/`, doc clarified |
| §7 Phase 4 | 3D viz mentioned but rotation/interactive controls not enumerated | §21 (time-travel section + GUI shortcuts) covers UX bits |
| §7 Phase 7 | "Architectural mutation" — what about UNDO of mutations | §21.1 time-travel covers undo |
| §7 Phase 11 | "Anything else" was vague | §21 makes it concrete |
| §8 | Cross-cutting reqs included light+dark but not mobile-specific Jupyter viz testing | §21.14+ keyboard shortcuts implicitly desktop; mobile viz constraints mentioned in §8 |
| §9.7 | AVP-2 audit reports — but doc says "appended to docs/AVP2_AUDIT.md" — should be per-phase as well | Doc says "Each Phase ships with an audit report appended"; clarified that means new entry per phase |
| §10 | Open questions — most could be answered by Phase outputs; left as is | Confirmed deliberate |
| §11 | Sequencing was tight; ok | — |
| §12 | First action sequence: did not mention Python package config explicitly | Added pyproject.toml + setup.py to Phase 0 task |
| §13 | Dependencies — `cargo audit` mentioned but exact version not pinned | Versions chosen are minimum-supported; flexible upward |
| §14 | Logging — log structure good but no log query CLI mentioned | Added: `graphnet logs query --session ... --filter ...` (will be implemented Phase 8) |
| §15 | UX philosophy good; "Pure GUI" mode lacked an example | Tutorial system §21.13 covers this |
| §16 | HDC GUI map — sparse for advanced ops (cleanup, hierarchical encoding) | Added: sequence encoding, set encoding, pair encoding, position encoding (§16) |
| §17 | Maths — wavelets + symbolic + statistics solid; what about category theory + topology? | Added persistent homology + manifold embedding to §17.1 |
| §18 | Future-proofing — quantum readiness was speculative; should not block design | Confirmed; quantum is "if it becomes practical", no design tax now |
| §19 | Loop directive — stop conditions explicit | Confirmed |
| §20 | Doc location — both copies needed? | One canonical; second is symlink in repo |
| §21 | New section additions; checked for overlap with §1-§20 | No duplicates |

Result: plan is internally consistent. Where ambiguity remained, the
"resolve at Phase X" pointers are explicit.

## 23. Final scope summary (one-liner per feature)

For the loop to refer back to:

```
[Phase 0]  Repo scaffold, CI, license, README                       Bounded
[Phase 1]  graphnet-engine Rust core (Model trait + Stack impl)     Bounded
[Phase 2]  PyO3 Python bindings                                     Bounded
[Phase 3]  Jupyter + 2D viz (matplotlib, plotly, graphviz)          Bounded
[Phase 4]  3D rotatable viz (plotly 3D + k3d-jupyter)                Bounded
[Phase 5]  Multi-arch adapters (Transformer, Mamba)                  Bounded
[Phase 6]  Continuous-execution mode                                 Bounded
[Phase 7]  Architectural mutation (live add/remove ops, undo/redo)   Bounded
[Phase 8]  Resource accounting + cost estimation                     Bounded
[Phase 9]  Export/import + session recording/playback                Bounded
[Phase 10] Test/audit/debug suite (proptest, cargo-mutants, viz reg) Bounded
[Phase 11] Anything-else features (time-travel, AI assistant, etc.)  Rolling
[Phase 12] Native GUI shell (egui + wgpu + tauri)                    Bounded

[Cross]    HDC GUI counterpart map (every op gets a node)            Embedded in Phase 3-4
[Cross]    Maths Panel (advanced linear alg, spectral, stats, optim) Embedded in Phase 4
[Cross]    Logging (tracing-based, file-rotating, replayable)        Embedded in Phase 1+
[Cross]    Future-proof abstractions (Backend, Adapter, Plugin)      Embedded in Phase 1+
[Cross]    AVP-2 security (encryption, audit log, supply chain)      Embedded throughout
```

Done. Implementation begins.
