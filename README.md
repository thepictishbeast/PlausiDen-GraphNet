> # ⚠️ DO NOT USE — UNVERIFIED — UNSAFE ⚠️
>
> This software is **unverified and unsafe for any production use**.
> It is published publicly only for transparency, third-party audit,
> and reproducibility. Treat every commit as guilty until proven
> innocent.
>
> By using this code you accept:
> - **No warranty** of any kind, express or implied.
> - **No fitness** for any particular purpose.
> - **No guarantee** of correctness, safety, or freedom from defects.
> - **Zero liability** on the maintainer for any damages — data loss,
>   security compromise, financial loss, or any consequential damages.
>
> The code is under active engineering development per the
> [Adversarial Validation Protocol v2](https://github.com/thepictishbeast/PlausiDen-AVP-Doctrine/blob/main/AVP2_PROTOCOL.md).
> Every commit's default verdict is **STILL BROKEN**. AVP-2 requires
> a minimum of 36 verification passes before a `SHIP-DECISION:`
> annotation may be considered. **No commit in this repository has
> reached `SHIP-DECISION:` status.**

# PlausiDen-GraphNet

Live REPL + interactive graphical environment for neural-network
architecture work. Tagline: *"a graphing calculator for AI."*

Primary focus is the PlausiDen Stack architecture (HDC / VSA + heterogeneous
operation modes — see [PlausiDen-Stack](https://github.com/thepictishbeast/PlausiDen-Stack)).
Secondary first-class support for transformer LLMs, state-space models
(Mamba), graph neural networks, and any framework with a thin adapter.

## What you can do with it

- **Load any model** — PlausiDen Stack, HuggingFace transformers, Mamba,
  custom PyTorch / JAX — via uniform adapters.
- **Compose them** — pipeline, parallel, routed, hierarchical.
- **See them** — 2D architecture graphs, 3D rotatable architecture views,
  live activation heatmaps, hypervector point clouds via PCA / UMAP.
- **Edit them live** — change a weight, add a Stack operation, remove an
  attention head; effects propagate immediately.
- **Watch them work** — continuous-execution mode at ~30 FPS visualisation;
  intervene mid-stream and see the behaviour shift.
- **Measure them** — live RAM / GPU / CPU / energy / $ cost.
- **Time-travel debug them** — step, backstep, watchpoints, breakpoints,
  rollback.
- **Export them** — architecture spec (YAML), weights (safetensors), full
  session (bincode + zstd).

## Status — pre-1.0, AVP-2 in flight, NOT production-ready

The repo currently houses:

- `crates/graphnet-engine` — Rust core (Model trait + Stack execution +
  intervention API + monitoring + recording / playback)
- `crates/graphnet-bindings` — PyO3 Python bindings
- `python/graphnet/` — Python package surface (REPL + viz + adapters)
- `examples/` — Jupyter notebook tutorials

Test infrastructure ships alongside the code (no "we'll add tests later"):
proptest + cargo mutants + cargo fuzz + visual regression via PlausiDen-Crawler.

## Plan

The full design lives at [`docs/PLAN.md`](./docs/PLAN.md) — 12 implementation
phases + cross-cutting work (HDC GUI counterparts, Maths Panel, structured
logging, future-proofing abstractions, AVP-2 security).

## License

[FSL-1.1-MIT](./LICENSE). Source-available with a 2-year competitor-
restriction window, then converts automatically to MIT.
