# Changelog

## 0.2.0

First release as **`pe-nbeats`** (import `pe_nbeats`). Previously published as `lightningnbeats` 0.1.x.

### Added

- `NBeatsNet.slim(backcast_length, forecast_length, trend_branch=False, rectify=False, width=128, ...)`: a right-sized preset with 10 stacks, a 128-wide trunk and a frozen DCT basis at full forecast rank.
- `NBeatsNet.from_recommendation(rec, backcast_length)` builds `slim()` from a pre-flight `Recommendation`.
- `pe_nbeats.recommend`, `pe_nbeats.analyze` and `pe_nbeats.Recommendation` are now exported at the top level.
- `pe_nbeats.__version__`.
- `UserWarning` when a wavelet family admits no decomposition at the target length, because the basis is then effectively the identity.
- `UserWarning` when `basis_offset > 0` is used with a legacy SVD-orthogonalised wavelet basis, because the offset does not select a frequency band there. Use `wavelet_type="ordered:<family>"` instead.
- Optional dependency groups: `viz`, `experiments`, `notebooks`, `xpu`, `dev`, `all`.

### Changed

- **Package renamed** to `pe-nbeats` / `pe_nbeats`. `import lightningnbeats` keeps working through a deprecated alias.
- **`sum_losses=True` no longer changes `val_loss` or `test_loss`.** Both now report the forecast loss alone, so early stopping and checkpointing behave the same with and without it. The combined objective is logged as `val_total_loss`. The training loss is unchanged.
- M4 data is no longer bundled in the wheel. It is downloaded from the official M4 repository on first use and cached in `~/.cache/pe_nbeats/M4/`. Source checkouts keep using their local copy. The wheel shrinks from 83 MB to under 1 MB.
- Traffic and Weather caches moved to `~/.cache/pe_nbeats/`. Existing `~/.cache/lightningnbeats/` downloads are reused.
- Runtime dependencies trimmed to `torch`, `lightning`, `numpy`, `pandas` and `PyWavelets`. Plotting, logging and notebook packages moved to extras.

### Unchanged

- `NBeatsNet` and `NHiTSNet` constructor defaults remain paper-faithful, so existing configurations reproduce.
