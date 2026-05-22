# GraphNet glossary — terms explained

Owner: "what are dims anyway? explain things more. have docs, have
interactive explanations."

This is the searchable reference for every AI / HDC / math term you'll
encounter in the GraphNet UI. Each entry has: (1) what it is in one
sentence, (2) why it matters, (3) where it shows up in the app.

---

## Dim (dimensionality)
**What**: the number of components in a hypervector. Default is 10000.
**Why it matters**: bigger dim = more capacity (more independent items
the network can hold) + more noise-tolerance, but more memory + slower
math. Theorem 1 in `docs/PROOFS.md`: with `D` components, two random
hypervectors have cos-similarity variance `1/D`, so D=10000 makes
random vectors very nearly orthogonal.
**Where in the app**: status bar "dim N", left panel "DIM" slider,
the central Input/Output cards show "dim = 10000".

## Hypervector
**What**: a vector of `±1` values with `D` components. In code:
`Vec<i8>` of length `dim`, each element constrained to `-1` or `+1`.
**Why it matters**: this is the substrate of HDC — every input,
output, op-key, and bundle is one hypervector. They behave like
"distributed symbols" in a vast space.
**Where in the app**: the binary-prefix text (`…01101110…`) shows the
first 24 components of a hypervector.

## cos_sim (cosine similarity)
**What**: a number in `[-1, +1]` measuring how similar two hypervectors
are. `+1` = identical, `0` = unrelated (orthogonal), `-1` = opposite.
**Why it matters**: the primary readout for "did the network preserve
the input?" or "how similar is this op's output to the bundled output?"
**Where in the app**: Output card shows `cos_sim ±X.XXX`, the bar
visualizes it, the Per-op contribution bars use cos_sim per op.

## Bundle
**What**: element-wise majority sign of N hypervectors. Output[i] = sign(
sum_j v_j[i]).
**Why it matters**: bundling is HDC's "OR" — combines multiple vectors
into one that's similar to all of them. Capacity is roughly `D/4`
items before recall degrades (Theorem 3, PROOFS.md).
**Where in the app**: the central BUNDLE node in the 3D viewport is the
result of bundling all op outputs.

## Op / Operation
**What**: a single transformation applied to a hypervector. GraphNet
has 5 kinds: identity, dense, hrr_bind, permute, negate.
**Why it matters**: each op contributes one output to the bundle. The
Stack is a parallel composition: input → [op_0, op_1, …, op_n] →
bundle(op_outputs).
**Where in the app**: op chips in left panel, op nodes in 3D viewport,
the per-op contribution chart.

## Identity (op kind)
**What**: returns the input unchanged. `identity(v) = v`.
**Why it matters**: when bundled with itself, it returns the input
(Theorem 7). Used to preserve original input alongside other ops.

## Dense (op kind)
**What**: element-wise multiply by a random bipolar key. `dense(v, k) =
v ⊙ k`.
**Why it matters**: produces a vector that looks random vs `v`. Used to
"label" or "transform" the input distinctly.

## HrrBind (op kind)
**What**: same as Dense for bipolar HDC — element-wise multiply by a
key. Named after Plate's Holographic Reduced Representation.
**Why it matters**: pseudo-invertible: `unbind(bind(v, k), k) = v`
(Theorem 4). Used for role-filler binding.

## Permute (op kind)
**What**: cyclic shift of the hypervector by `k` positions.
`permute(v, k)[i] = v[(i - k) mod D]`.
**Why it matters**: encodes position. Iterated permute is cyclic, so
sequences can be encoded by `permute(v, 1)` then bundled (Theorem 5).

## Negate (op kind)
**What**: flips all signs. `negate(v) = -v`.
**Why it matters**: an involution (`¬¬v = v`, Theorem 6). Encodes
"opposite of" or "not".

## Fingerprint (fp)
**What**: a short hash of a hypervector via blake3, displayed as a hex
prefix (e.g. `a4b91f2c`).
**Why it matters**: lets you visually identify whether two hypervectors
are bit-identical without printing all `D` components.

## Stack
**What**: an ordered list of ops that share an input and produce a
bundled output. The fundamental architectural unit in GraphNet.
**Why it matters**: heterogeneous parallel compute — different op
kinds in one Stack lets the same input get transformed in multiple
ways simultaneously.

## Forward
**What**: a single execution of the Stack: takes the current input,
applies every op in parallel, bundles the outputs. Latency ranges
from 0.1 ms (D=1000, 1 op) to 5 ms (D=16000, 10 ops).
**Why it matters**: this is the "compute step" — every Forward is
analogous to one inference pass in a regular neural net.

## Live mode
**What**: re-runs Forward continuously at ~60 fps. Toggled with `L`.
**Why it matters**: lets you see how output changes as you mutate the
stack in real time. Useful for "what does this op DO" experiments.

## Bipolar
**What**: a value constrained to `{-1, +1}`. As opposed to real-valued
floats.
**Why it matters**: bipolar hypervectors have efficient hardware
implementations + tight theoretical bounds. The substrate for HDC.

## Slot (A/B/C/D)
**What**: a saved copy of the current stack. Up to 4 slots.
**Why it matters**: lets you compare two stack designs side-by-side
in the Compare workspace. Save A, mutate, save B, switch between.

## Workspace (Edit / Live / Compare / Train)
**What**: a UI mode that pre-configures panels for a specific task.
**Why it matters**: same stack, different controls visible:
  Edit    — manual mutation
  Live    — continuous re-forward at 60fps
  Compare — A/B/C/D slot comparison + composition chain
  Train   — target-driven training with loss curve

## Loss
**What**: `1 − cos_sim(actual_output, target_output)`. Range `[0, 2]`.
**Why it matters**: in the Train workspace, you set a target then run
hill-climb / simulated-anneal to minimize loss. Smaller = better.

## NeuralGraph
**What**: a generic DAG of typed neural-net nodes (not just HDC).
Supports Dense, Conv2d, Attention, LSTM, Embedding, plus novel kinds
(EnergyBased, NeuralOde, Hamiltonian, Oscillator, Spiking,
SymbolicFormula).
**Why it matters**: the Architecture Inspector renders any
NeuralGraph; HDC stacks are wrapped as `LayerKind::Hdc`.

## Attention (LLM term)
**What**: a layer that computes a weighted sum over input positions,
where weights come from query-key dot products. Multiple "heads"
attend to different aspects.
**Why it matters**: the dominant operation in transformer/LLM
architectures. Visualized as a 12×12 heatmap in the inspector.

## Embedding (LLM term)
**What**: a learned lookup table mapping discrete tokens (e.g. word
IDs) to fixed-dim vectors.
**Why it matters**: GPT-2-small has a 50257 × 768 token embedding
(38M params just for words).

## Permute shift (parameter)
**What**: how many positions to cyclically shift in a Permute op.
**Why it matters**: different shifts produce different "tagged"
versions of the input. Cycle length = D / gcd(D, shift).

## FLOPs
**What**: Floating Point Operations per second. The compute cost of
one Forward through the stack.
**Why it matters**: tracks how expensive a stack is. Shown in status
bar as `X flops/fwd`.

## ResourceSample
**What**: a snapshot of CPU% + app RAM + host RAM (used/total/free)
sampled every 750 ms via `sysinfo`.
**Why it matters**: lets you see whether GraphNet is hogging your
machine. Status bar shows the latest sample.

## Hamming distance
**What**: count of bits where two vectors differ, normalized by D.
**Why it matters**: alternative similarity measure to cos_sim. Both
range `[0, 1]` (Hamming) or `[-1, +1]` (cos_sim).

## Achievement
**What**: an unlockable badge for completing a feature-tour milestone
(first forward, 100 forwards, used all 5 op kinds, etc.). 12 total.
**Why it matters**: gamification to nudge feature discovery.

## Objective
**What**: a numbered "do this next" goal in the Right Panel. 10 in
order; the next-undone shows as "🎯 OBJECTIVE N/10."
**Why it matters**: structured walkthrough — finishing all 10 means
you've touched every feature.

## Creature name
**What**: an auto-generated name from the current stack's op
composition (e.g. "Quad-Crystal Hydra" = 4 dense ops).
**Why it matters**: makes saved stacks memorable. Per `docs/GAME_DESIGN.md`.

---

## See also
- `docs/PROOFS.md` — formal mathematical statements + theorems
- `docs/GAME_DESIGN.md` — gamification + creature-collection vision
- `docs/GENERAL_AI_VISUALIZATION.md` — Phase 1-6 roadmap to LLMs
- `docs/FOSS_CANDIDATES.md` — libraries to absorb for more capability
