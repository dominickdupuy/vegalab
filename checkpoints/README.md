# Committed checkpoints

Frozen agent checkpoints (`agent.save` payloads: networks + target + obs-normalizer +
config; ~1.3 MB each). Never commit resume snapshots (`state.pt` — they embed the replay
buffer, ~23 MB).

**Commit policy — a checkpoint may be committed only if it is:**

(a) **gate-passing** — part of a seed ensemble that passed its wave's GV/BV/FF gates
    (these are the warm-start sources for later waves and the frozen subjects for
    Phase 6 held-out evaluation and Phase 7 distillation); or
(b) **the pinned subject of committed analysis** — e.g. a diagnostics run whose numbers
    are in the gate report, so the analysis stays exactly reproducible.

Every checkpoint carries a sibling `.json` manifest: sha256, seed/config, training tool
and environment, status (gate-passing vs diagnostic), and pointers to the analysis it
anchors. Load with `IQNAgent.from_checkpoint(path)`.

If this folder ever approaches ~50 MB, migrate to Git LFS or GitHub Releases; until
then plain git keeps review simple.
