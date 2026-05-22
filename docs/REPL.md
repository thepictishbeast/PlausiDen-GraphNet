# GraphNet REPL — interactive command reference

The REPL (Read-Eval-Print Loop) is GraphNet's text-driven control
surface. Same operations as the buttons, but typeable + scriptable.

## Opening + closing

| Action | Key |
|---|---|
| Toggle console | `` ` `` (backtick) |
| Toggle console | hero button `⌨ Console` |
| Close | `Esc` (when console focused) or click ✕ |
| Clear history | 🗑 button (right of header) |

## Editing keys

| Key | Effect |
|---|---|
| `Enter` | submit current line |
| `↑` | recall previous command (cycles back) |
| `↓` | move forward through history |
| `Tab` | complete command from prefix (`tem` → `template `) |
| `Esc` | close console (returns focus to viewport) |

## Output color code

| Color | Meaning |
|---|---|
| **bright cyan ›** | prompt marker |
| **white bold mono** | the command you typed |
| **green** | success / `loaded` / `saved` / `ran` / `added` |
| **red** | error / `invalid` / `failed` |
| **muted gray** | informational output |

## Commands

### Help + introspection

```
help               # list all commands
stat               # print stack summary (template, dim, op count, last latency)
clear              # wipe console history (also via 🗑 button)
```

### Forward execution

```
fwd                # run a forward pass (alias: forward)
forward            # same as fwd
live               # toggle live mode (continuous forward at 60 fps)
```

### Stack mutation

```
add identity       # append an identity op
add dense          # append a dense (random key) op
add hrr_bind       # append an HRR binding op
add permute        # append a permute (cyclic shift) op
add negate         # append a negate op

rm N               # remove op at index N (alias: remove)
remove N           # same as rm
reseed N           # regenerate the key of op N (Dense / HrrBind / Permute only)
```

### Templates

```
template standard          # load named template (matches the 10 in popup)
template minimal           # also: anti-correlation, echo-state, sparse-dense,
                           # noise-resilience, role-binding, identity-stack,
                           # sequence-permute, polarity-flip, deep-mixed
```

### State

```
regen              # regenerate input from a new seed
reset              # clear stack to 0 ops
dim N              # change dim (N clamped to 256..16384) — RESETS stack
undo               # undo last mutation
redo               # redo undone mutation
```

### I/O

```
save               # save stack as YAML (opens native file dialog)
load               # load stack from YAML (opens native file dialog)
png                # export 3D viewport as PNG
```

## Scripting tips

The console writes every action to `~/.config/graphnet/graphnet.log`
with timestamp + severity, so:

```bash
# watch live as you type commands
tail -f ~/.config/graphnet/graphnet.log

# replay yesterday's session
grep '2026-05-21' ~/.config/graphnet/graphnet.log

# extract every forward latency
grep '⏩ forward' ~/.config/graphnet/graphnet.log
```

For programmatic stacks, you can construct one in YAML directly:

```yaml
# my-stack.yaml
dim: 10000
operations:
  - !Identity
  - !Dense
    key:
      data: [1, -1, 1, 1, -1, ...]  # length must equal dim
  - !HrrBind
    key:
      data: [-1, 1, -1, ...]
  - !Permute
    shift: 7
  - !Negate
```

Then `load` it via console or `⌘O`.

## Worked example

```
›  template standard            # load 3-op identity+dense+hrr_bind
›  stat                         # confirm
›  fwd                          # run forward — cos_sim ≈ 0.58
›  add dense                    # 4-op now
›  fwd                          # cos_sim drops (more decorrelation)
›  reseed 3                     # new key on the new dense
›  fwd                          # cos_sim changes
›  dim 5000                     # halve dimensionality — clears stack
›  template noise-resilience    # rebuild
›  live                         # 60fps continuous
›  ...                          # mutate freely, watch cos_sim_history
›  live                         # stop
›  save                         # write to YAML
```

## Command-vs-button parity

Every command in this doc has a button or keyboard shortcut equivalent
elsewhere in the UI:

| Command | UI equivalent |
|---|---|
| `add KIND` | `A` / `D` / `F` / `P` / `N` keyboard shortcut, or `➕ Add op` palette |
| `fwd` | `Space` keyboard or `▶ Run forward` hero button |
| `live` | `L` keyboard, hero `▶ Live` button, or Live workspace tab |
| `template NAME` | `1`-`9`/`0` keys or `+ New…` templates popup |
| `dim N` | DIM slider in left panel (Settings tool mode) |
| `regen` | `R` keyboard or 🎲 button next to seed |
| `reset` | left-panel `🗑 Reset stack` button |
| `undo` | `⌘Z` / `Ctrl+Z` |
| `redo` | `⌘⇧Z` / `Ctrl+Shift+Z` |
| `save` | `⌘S` / File → Save YAML |
| `load` | `⌘O` / File → Open YAML |

REPL is parity, not replacement — pick whichever is faster for your
workflow.

## See also

- `docs/GLOSSARY.md` — what every term means
- `docs/PROOFS.md` — math + theorems behind operations
- `docs/GAME_DESIGN.md` — gamification vision
- `docs/GENERAL_AI_VISUALIZATION.md` — roadmap to general AI graphs
