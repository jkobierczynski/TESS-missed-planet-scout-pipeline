# scout

Find transiting exoplanet candidates in public TESS data that the official
pipelines (SPOC, QLP) missed, vet them, and package the survivors as ExoFOP
community TOIs. Design and roadmap: [DESIGN.md](DESIGN.md).

**Status:** M0–M2 done (skeleton, single star end to end, injection-recovery
calibration: SDE 17 / SNR 7). M3 (batch CVZ run and triage) in progress. Detections are
*not vetted*: the §6 checklist arrives in M5.

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

### Batch run and triage (M3)

```bash
scout targets -c config/cvz-north.toml               # TIC cone around the ecliptic pole -> targets.csv
scout batch   -c config/cvz-north.toml --workers 4   # every target to its report; rerun to resume
scout triage  -c config/cvz-north.toml               # known? -> candidates.parquet, summary.html
```

`targets` applies the `[targets]` cuts and sorts brightest first
(`--max N` keeps the first N). `batch` skips finished stars and retries failed
ones, logging each star to `batch.jsonl`; with `fetch.prune_after_stitch` the
FITS (~20 MB per star) are deleted once stitched. `triage` cross-matches every
detection above threshold with the ExoFOP TOI and CTOI lists and the NASA
Exoplanet Archive, by TIC and period (including ½, 2, ⅓, 3 × aliases) and by
position for blends from neighbours within 120″. Local catalogues such as the
TESS EB catalogue can be added through `triage.extra_catalogs`.

### Calibrating the search (M2)

Thresholds are measured, not guessed (design §3.2, §5). On host stars with no
known planet or TOI, `scout inject` multiplies synthetic transits into the
stitched light curve, reruns detrend and search, and records which ones come
back. It also runs null searches on per-sector time-scrambled and inverted
copies of the flattened light curve. `scout calibrate` pools all hosts in a
run and picks the SDE/SNR pair that recovers the most injections while keeping
the false-alarm rate per null search at or below `inject.false_alarm_target`.

```bash
scout inject    -c config/m2-calibration.toml <host TICs...>   # resumable
scout calibrate -c config/m2-calibration.toml
```

The report (`results/<run>/calibration/calibration.html`) shows the recovery
map over period × radius, the null distribution against recovered injections,
and per-host numbers. Thresholds only carry over to a science run with the same
stitch, detrend and search settings and a similar sector span, so
`config/m2-calibration.toml` mirrors `config/cvz-north.toml` apart from
`fetch.sectors` and `[inject]`.

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
  targets.csv, batch.jsonl     target list and per-star batch log
  candidates.parquet           every detection with its triage verdict
  summary.html                 sortable overview, links to the star reports
  inject/TIC<id>/             injections.jsonl, nulls.jsonl (scout inject)
  calibration/                calibration.html, calibration.json (scout calibrate)
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
