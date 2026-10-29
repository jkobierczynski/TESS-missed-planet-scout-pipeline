# scout

Find transiting exoplanet candidates in public TESS data that the official
pipelines (SPOC, QLP) missed, vet them, and package the survivors as ExoFOP
community TOIs. Design and roadmap: [DESIGN.md](DESIGN.md).

**Status:** M0 (skeleton) done, M1 (single star end to end) in progress.
Detections are *not vetted*: the §6 checklist arrives in M5, and the search
thresholds are provisional until the M2 injection-recovery maps exist.

## Install

Python ≥ 3.11.

```bash
pip install -e ".[mast,models,dev]"
```

- `mast` adds astroquery, needed only for `scout fetch`.
- `models` adds batman for limb-darkened synthetic transits (a trapezoid is used without it).

## Use

Every stage is a subcommand. Stages are checkpointed: rerunning skips anything
already up to date, and changing a config section reruns only that stage and
the ones after it.

```bash
# One star, all stages, using the CVZ config
scout run -c config/cvz-north.toml <TIC_ID>

# Or stage by stage
scout fetch   -c config/cvz-north.toml <TIC_ID>
scout stitch  -c config/cvz-north.toml <TIC_ID>
scout detrend -c config/cvz-north.toml <TIC_ID>
scout search  -c config/cvz-north.toml <TIC_ID>
scout report  -c config/cvz-north.toml <TIC_ID>
```

The report lands in `results/<run>/reports/TIC<id>.html`.

Without network access, try it on a synthetic star:

```bash
scout synth --run demo --preset sub-neptune --tic 999000001 --sectors 14-19
scout run   --run demo 999000001
```

Presets: `quiet`, `hot-jupiter`, `sub-neptune`, `m-dwarf`.

`scout config show -c config/cvz-north.toml` prints the effective configuration;
`scout config check` validates it.

## Layout

```
config/            run configurations (TOML)
src/scout/         the package: one module per stage
tests/             unit tests on synthetic light curves (no network)
cache/             downloaded MAST files, immutable      (gitignored)
results/<run>/     per-run outputs, see below           (gitignored)
```

```
results/<run>/
  run.json                    config, code version, every invocation
  stars/TIC<id>/fetch.json    chosen MAST files + TIC stellar parameters
  stars/TIC<id>/stitched.parquet, flat.parquet, search.json
  reports/TIC<id>.html
```

`results/` and `cache/` stay out of git on purpose: unsubmitted candidates are
private until they go to ExoFOP.

## Development

```bash
ruff check . && ruff format --check .
pytest                  # fast unit tests
pytest -m slow          # TLS on synthetic multi-sector data
```

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
