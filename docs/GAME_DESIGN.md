# Neural Garden — GraphNet as a game

A vision doc for evolving GraphNet (an interactive HDC neural-network
REPL) into a game-feel "build, breed, and battle" sandbox for real AI.

## Premise

Real neural networks are abstract. People feel disconnected from them
because the math is hidden behind opaque tensors. GraphNet flips that:
every operation is something you can SEE, MUTATE, and FEEL the effects
of, in real time, at 60 fps. The game layer turns that into a creature-
collection sandbox.

## Modes

### 1. Sandbox (current default)
Free experimentation. Templates, ops, dim slider, run forward. No
goals, no scoring — just exploration. This is the "tutorial-with-no-
rails" mode for grown-ups.

### 2. Challenge mode (#720 objectives — partial)
10 hand-written objectives walk the player through every feature once.
Future challenges: timed forwards (most forwards in 60s), cos-sim
targets ("get within 0.02 of 0.5"), resource bound ("≤1ms latency at
≥D=10k"). Each unlock teaches a concept.

### 3. Collection mode (future)
Every Stack you SAVE is a "creature" with:
- a name (auto-generated from op-kind composition)
- a portrait (the bundled-output heatmap, rendered as a sigil)
- statistics (avg cos_sim, latency, FLOPs)
- a generation number (lineage tracked via blake3 parent hash from
  #21 provenance — already shipped engine-side)

A Pokedex-style grid lets you browse, compare, and re-load.

### 4. Story mode (future)
Scripted narrative that introduces concepts via challenges. Each level
gates the next behind learning a specific HDC primitive: binding,
permutation, hyperdimensional cleanup, etc.

### 5. Live arena / battle mode (far future)
Two saved Stacks compete on shared inputs. The one whose output is
closer to a target wins points. Tournaments rank Stacks Elo-style.

## Creature design

Each op kind has a "personality":

| Op       | Color | Creature feel | Idle animation |
|----------|-------|---------------|----------------|
| identity | blue  | calm jellyfish | gentle bob     |
| dense    | green | crystal       | slow rotate    |
| hrr_bind | red   | spiral / vortex | spin           |
| permute  | amber | gears         | tick           |
| negate   | purple | inverter (pulse) | flicker      |

When ops are in a Stack, they're nearby like organisms in a tank.
Bound pairs (selected + inspector chip) glow with shared color. Dragging
an op feels like picking up a creature.

## Visual reward feedback

- **Forward pass**: glowing particle of data flows INPUT → ops → BUNDLE
  → OUT over ~0.8s. *(already shipped iter 17.)*
- **High cos_sim (>0.85)**: output card grows a green halo for 0.5s.
  *(future)*
- **Negative cos_sim**: red flicker on the BUNDLE node. *(future)*
- **Op added**: chip pulses with golden ring 1s. *(iter 46.)*
- **Achievement unlocked**: gold burst around the help button + chime.
  *(visual already, audio pending #723.)*

## AI-as-pet interactions

- **Pet (hover for ≥1s on op)** → reveals key heatmap as a popup.
  *(shipped iter 35 — #712.)*
- **Feed (regenerate input)** → input heatmap pulses. *(partial.)*
- **Stretch (right-click → reseed)** → key rotates. *(shipped #708.)*
- **Walk (live mode)** → continuous forward at 60fps shows the
  creature "thinking". *(shipped iter 3.)*
- **Sleep (pause demo)** → demo stops, creature settles. *(shipped.)*

## Networks-as-named-creatures

When the user saves a Stack to YAML, prompt for a name. Auto-suggest
based on op composition: "id-id-id-dense" → "Echo Beetle".
"dense-dense-dense-dense" → "Quad-Crystal Hydra". The saved file's
header includes name + portrait (PNG of bundled output).

## Sound design (#723)

- **Forward**: short blip (≤50ms, ~440Hz, square wave). Pitch reflects
  cos_sim (higher = more similar).
- **Achievement**: descending arpeggio chord (C major, 200ms).
- **Op add**: subtle "tick" (~880Hz, 20ms).
- **Op remove**: subtle "tock" (~440Hz, 20ms).
- **Live mode**: ambient pad (low-volume drone), fades in/out.
- **Demo step**: brief "ding" between templates.

All sound off by default; enabled via Settings → Audio toggle.

## Progression / leveling

Reuse forwards as XP. Levels unlock cosmetics (theme palettes, op-chip
shapes, particle trails). Every 100 forwards = level up. Hit Level 10
to unlock "Sandbox+" (more templates, custom ops). Level 25: arena.

## Camera & viewport

- Free-orbit 3D camera (yaw/pitch/roll — shipped iter 42).
- Wheel zoom (toolbar buttons → wave 2: scroll handler).
- Middle-mouse pan (future).
- Right-click rotate around selected op (future).
- Press F to "focus" on selected op (cam zooms in).
- Press numpad keys for axis-aligned views (front/top/side).

## What's already shipped

Reference iters 1-48 of this session. Roughly:
- core engine + HDC primitives (Phase 1-12 prior to this session)
- themed UI + dark/light modes
- 3D rotatable arch graph with shaded nodes + particle flow
- 10 templates with explanations
- 5 op kinds (id, dense, hrr_bind, permute, negate)
- live mode + sparklines
- 12 achievements + 10 objectives
- right-click context menu + drag-reorder
- console/REPL
- adaptive tutorial banner
- info tooltips on AI terms
- human-readable I/O summaries (fingerprint + binary prefix)
- save/load YAML + drag-drop + PNG export
- CPU/RAM/FLOPs monitor

## What's NOT shipped (open tasks)

- #711 A/B compare mode (clone stack, run both on same input)
- #721 Heatmap scroll/pinch zoom
- #722 Multi-stack workspace (4 slots)
- #723 Output sonification (audio)
- #726 Spotlight walkthrough (Shepherd-style)
- #728 Demo enhancement with UI highlights
- #729 Icon font wave 2 (Phosphor)
- #733 More 2D/3D animations
- #734 Real estate optimization
- #741 Blender-style floating windows
- #743 Multi-page workspaces (Edit/Live/Compare/Train)
- #746 Stack composition (graphical edges between stacks)
- #751 Game polish wave 1

This game-design doc is the north star for prioritizing these.

— iter 49, ad0efa5+
