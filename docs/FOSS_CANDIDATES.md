# FOSS to absorb into GraphNet

Owner direction: "find FOSS projects that do what we need" + "think of more
FOSS projects to use and integrate in various parts of the App also."

Already absorbed (working in shipped iters):
- `egui` / `eframe` / `egui_plot` / `egui-phosphor` — UI + plotting + icons
- `rfd` — native file dialogs
- `image` — PNG/JPG/BMP decoding for CV input
- `blake3` — hypervector fingerprints
- `bytemuck` — safe byte casts
- `safetensors` — HF model weight metadata reader (iter 138)
- `plausiden-hdc` (sibling) — HDC primitives
- `sysinfo` — CPU + RAM telemetry
- `serde` + `serde_yaml` — state persistence

Pending absorption (in priority order):

## 1. Real neural-net runtime
- **`candle-core` + `candle-nn` + `candle-transformers`** — Hugging
  Face's Rust ML stack. Loads + runs GPT-2, BERT, Llama, Stable Diffusion.
  Heavy (~200MB build) but the only complete Rust ML stack.
- **`burn`** — alternative Rust DL framework, more ergonomic API. Lighter
  than candle. Backend-flexible (ndarray / wgpu / candle).
- **`tract-onnx`** — ONNX model inference. Simpler than candle for
  static models; can't do training.
- **`luminal`** — newer Rust ML lib with auto-fusion + JIT.

## 2. Graph data + algorithms
- **`petgraph`** — battle-tested directed-graph library. Replaces my
  hand-rolled `has_cycle()` + opens up topo-sort, SCC, shortest-path, etc.
- **`graph_dot`** — for exporting NeuralGraph to GraphViz DOT format
  (useful for sharing architecture diagrams).

## 3. 2D node-graph UI
- **`egui_node_graph`** — full node-graph editor in egui. Drop-in for
  the Phase 2 2D viz from GENERAL_AI_VISUALIZATION.md.
- **`egui_dock`** — Blender/JetBrains-style dockable panel system.
  Replaces my hand-rolled SidePanel + minimize/float/close chrome with
  proper tabbed/split docking.

## 4. Tensor ops + math
- **`ndarray`** — n-dim array library, NumPy equivalent.
- **`nalgebra`** — linear algebra (matrix decompositions, etc.).
- **`statrs`** — statistics distributions for sampling / Bayesian work.
- **`rustfft`** — FFT (useful for spectral methods, Fourier neural ops).

## 5. Symbolic math + formula parsing
- **`evalexpr`** — runtime formula evaluator. For LayerKind::SymbolicFormula
  to actually compute. Lightweight (~20KB).
- **`meval`** — alternative formula parser.
- **`sympy-rs` (experimental)** — symbolic differentiation, useful for
  auto-diff novel architectures.

## 6. Physics simulation (for novel-AI physics-based architectures)
- **`rapier3d`** — 3D physics engine. Could simulate physical neural-net
  substrates (oscillator coupling, Hamiltonian dynamics).
- **`nalgebra-glm`** — graphics math (matrices/quaternions/projections).

## 7. Audio (sonification)
- **`rodio`** — already deferred; needs libasound2-dev on Linux. Brings
  sound to forward events.
- **`cpal`** — lower-level audio (rodio uses it underneath).
- **`fundsp`** — synthesis library; could generate per-op tones with
  proper envelopes.

## 8. Reproducible training + experiment tracking
- **`mlflow-rust`** — experiment tracking compatible with MLflow.
- **`wandb`-style local logging via `tracing`** — distributed tracing
  for training runs.

## 9. Random number + sampling
- **`rand`** + **`rand_pcg`** + **`rand_distr`** — RNG with PCG fast
  generator + distributions. Replaces my hand-rolled `wrapping_mul` PRNG.
- **`statrs::distribution::*`** — for Gaussian / Bernoulli / etc. sampling.

## 10. Serialization + interchange
- **`serde_json`** — already transitively used. JSON export of NeuralGraph
  for compatibility with PyTorch / TensorFlow consumers.
- **`bincode`** — binary serialization for fast disk I/O.
- **`zstd`** — compression for large state files (state.yaml is 130KB
  currently; could be 10KB compressed).

## 11. Reactive / observable patterns
- **`egui_extras::syntax_highlighting`** — for the console REPL to
  highlight commands.
- **`egui_commonmark`** — render markdown in help / docs panels.

## 12. Profiling + telemetry
- **`tracing`** + **`tracing-subscriber`** — structured logging.
- **`profiling`** — frame-level profiling for the 60fps live loop.
- **`puffin_egui`** — embeddable profiler UI inside the app.

## Adoption order

Imminent (next 5 iters):
1. `evalexpr` — makes LayerKind::SymbolicFormula actually compute. Tiny dep.
2. `egui_dock` — proper docking. Replaces hand-rolled min/float/close.
3. `petgraph` — graph algorithms for NeuralGraph traversal.

Medium-term (next 15 iters):
4. `egui_node_graph` — 2D node editor for Phase 2 of GENERAL_AI_VISUALIZATION.
5. `egui_commonmark` — markdown for help/about/walkthrough.
6. `rand_pcg` + `rand_distr` — replace hand-rolled PRNG.

Long-term (research direction):
7. `candle-core` — real neural-net runtime.
8. `burn` — alternative if candle too heavy.
9. `rapier3d` — physical-substrate simulations.

## Code-dedup pass (per "merge any duplicates")

Identified duplicates in current codebase:
- 3 sites toggle `self.live = !self.live` (keyboard L, hero button,
  console "live" command) — could extract to `fn toggle_live()`.
- 4 sites push to `slots[i]` (Ctrl+1-4 keyboard, Compare panel buttons,
  Tools menu, console "save" — currently keyboard + Compare + menu).
  Extract `fn save_to_slot(i)` + `fn recall_slot(i)`.
- `cosine_similarity_bar()` is called 2× (Input/Output card + Compare).
  Already shared.
- The hand-painted `sparkline()` / `latency_sparkline()` / `loss_sparkline()`
  / `contribution_bars()` are all dead code post egui_plot — delete.
