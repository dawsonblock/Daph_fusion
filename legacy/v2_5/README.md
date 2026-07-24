# Legacy v2.5 Archive — Sparse Merge Research Trunk

This directory preserves the v2.5 DARE/TIES/Fisher-centric merge research
trunk for scientific traceability. The v3 dense-merge program replaces it
as the mainline, but these artifacts remain so later analysis can
understand exactly why the old methods appeared to work (or not).

## Relabeled methods

The following misleading labels from v2.5 are renamed for clarity:

| v2.5 label    | v3 legacy name                      | Reason                          |
|---------------|--------------------------------------|---------------------------------|
| `fisher`      | `legacy_delta_squared_weighting`     | Was delta², not real Fisher     |
| `exfusion`    | `legacy_dare_delta2_trim`            | DARE→TIES→delta², not dense     |
| `agx`         | `legacy_heuristic_agx`               | Heuristic proxy, not validated  |

## Status

```json
{
  "paper_ready": false,
  "fisher_verified": false,
  "agx_verified": false
}
```

The 138-test v2.5 baseline is preserved. These tests continue to pass
under the v3 package because the legacy operators are retained in
`daph_exfusion/merge/legacy/` and the old pipeline API remains importable.

## Contents

- `merge/` — copies of the v2.5 sparse operator implementations
- `experiments/` — v2.5 experiment scripts
- `results/` — v2.5 result JSONs (relabeled)
- `manifests/` — v2.5 provenance manifests
