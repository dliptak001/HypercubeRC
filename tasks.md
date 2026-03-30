# Multi-Timescale Reservoir — Task Tracker

## In Progress

- [ ] **Leaky integrator** — Add global leak rate parameter to `Reservoir`,
      modify `UpdateState()`, default to 1.0 (no behavior change).
      Prereq for everything else.

## Up Next

- [ ] **Reservoir partitioning** — Split reservoir into fast/slow zones
      with per-zone leak rates. Design discussed in
      `docs/MultiTimescaleReservoir.md`, details TBD.

- [ ] **Fast NARMA-K diagnostic** — Generalize NARMA-10 to configurable
      order K for sweeping longer memory horizons. Design in
      `docs/FastNarma.md`.

## Future

- [ ] **Per-DIM default management** — Refactor scattered `constexpr`
      functions into a config struct. Needed before adding more tunable
      parameters.

- [ ] **Leak rate + SR co-optimization sweeps** — Find optimal defaults
      using Fast NARMA-K across DIMs.
