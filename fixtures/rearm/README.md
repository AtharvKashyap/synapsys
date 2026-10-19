# Re-arm decision-sequence fixtures

`sequences.csv` (regenerate: `.venv/bin/python scripts/make_rearm_fixtures.py`) feeds the
C++ port of `synapsys_decoder.rearm` (plan Gate 4). Columns:
`sequence_id,confirm_count,refractory_samples,release_count,step,window_end_sample,decision,expected_emit,expected_state`.

- One row per decision point; a new `sequence_id` starts at `step` 0 with a freshly reset machine
  built from that row's (K, R, N). `sequence_id` is `<config>/<name>`.
- `window_end_sample = 625 + 125*step`. `decision` is YES/NO/HELP/REJECT.
- `expected_emit` is empty when the machine returns None; `expected_state` is the state
  after the update (ARMED / CANDIDATE / REFRACTORY).
- Configs: default (2,625,2), k1 (1,625,2), k3 (3,1250,3), n0 (2,625,0). Random sequences use
  `default_rng(20261018)`. Semantics: see the `rearm.py` module docstring.
