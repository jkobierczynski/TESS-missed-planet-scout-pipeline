# scout

Find transiting exoplanet candidates in public TESS data that the official
pipelines (SPOC, QLP) missed, vet them, and package the survivors as ExoFOP
community TOIs. Design and roadmap: [DESIGN.md](DESIGN.md).

**Status:** M0–M3 done (skeleton, single star end to end, injection-recovery
calibration: SDE 17 / SNR 7, batch CVZ run and triage). M5 vetting in progress:
`scout vet` runs the §6 checklist automatically and `scout fpp` adds TRICERATOPS;
TPF difference images are still to come.

## Install

Python ≥ 3.11.

```bash
pip install -e ".[mast,models,dev]"
```

- `mast` adds astroquery, needed only for `scout fetch`.
- `models` adds batman for limb-darkened synthetic transits (a trapezoid is used without it).

## Run a survey on your own computer

One command does the whole search: it selects the targets, downloads and
searches every star, cross-matches the detections with known planets and TOIs,
and vets what is new. It needs Python 3.12 (3.11 works too; 3.13 runs
everything except `scout fpp`, and 3.14 is not supported yet) and an internet
connection to MAST.

The simplest way to get Python 3.12, whatever your system Python is, is
[uv](https://docs.astral.sh/uv/). On Linux, macOS or WSL (Ubuntu):

```bash
sudo apt install build-essential     # WSL/Ubuntu: a C compiler for batman
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env              # or open a new terminal

git clone https://github.com/jkobierczynski/TESS-missed-planet-scout-pipeline.git
cd TESS-missed-planet-scout-pipeline
uv venv -p 3.12 .venv                # downloads Python 3.12 if needed
source .venv/bin/activate
uv pip install -e ".[mast,models,fpp]"

scout survey -c config/cvz-north-t10.toml
```

uv's Python includes the C headers, so batman (in `models`) builds without a
`python3-dev` package. With a system Python 3.12 instead, use
`python3.12 -m venv .venv` and `pip install -e ".[mast,models,fpp]"`, and
install `python3.12-dev` and `build-essential` first (on Ubuntu 24.04 these
come from the standard repositories). An error like `Python.h: No such file or
directory` means those headers are missing. On native Windows, batman needs the
Microsoft C++ Build Tools; WSL avoids that.

`config/cvz-north-t10.toml` covers the 1,726 northern-CVZ dwarfs with Tmag
8.8–10 (the 500 brighter ones were run as `cvz-north`). Stop it any time with
Ctrl-C, or let the machine sleep: the same command continues where it stopped,
and failed downloads are retried. By default it uses half your CPU cores
(`--workers N` to change). A 4-core machine did about 100 stars an hour, and
each star keeps about 11 MB in `results/`, so this run needs about 20 GB of
disk. `--limit 50` makes a quick trial run on the 50 brightest stars.

When it finishes, open `results/cvz-north-t10/vet_summary.html`. Signals with
the verdict `candidate` or `review` then get their false-positive probability:

```bash
scout fpp -c config/cvz-north-t10.toml
```

Results stay on your machine: `results/` and `cache/` are never committed.

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

### Vetting (M5)

```bash
scout vet -c config/cvz-north.toml --workers 4   # §6 checklist -> vet/, vetting.parquet, vet_summary.html
scout fpp -c config/cvz-north.toml               # TRICERATOPS on the survivors (Python ≤ 3.12, see below)
```

`vet` takes every signal triage left as `new` or `new_on_known_host` and runs
the design §6 checklist on it: a limb-darkened transit fit (shape, radius,
stellar density), brightenings and sinusoids at the same period (variability),
odd/even depths, a secondary eclipse at phase 0.5 and elsewhere, depth per
sector and per transit, SPOC centroid shifts in transit, Gaia DR3 neighbours
that could make the dip plus the host's RUWE, and spacecraft systematics
(transits on dates where other stars in the run dip too, next to data gaps,
or only half covered). Depth errors include red noise, measured as the scatter
of transit-length averages out of transit. Each signal gets a page in
`vet/` and a verdict: `fail` (any check failed), `review` (warnings) or
`candidate`. None of it replaces reading the page.

TRICERATOPS needs lightkurve, which does not build on Python 3.13, so `fpp`
needs a Python 3.11 or 3.12 environment with the `fpp` extra (the survey
setup above). Everything else in scout works there too.

It fills in checklist item 10 (FPP and NFPP) for signals whose verdict is in
`vet.fpp_verdicts`, using the SPOC apertures and Gaia DR3 from MAST.

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
  vet/TIC<id>-<n>.html, .json  vetting page and numbers per signal (scout vet, scout fpp)
  vetting.parquet              one row per vetted signal with every check's status
  vet_summary.html             vetting overview, survivors first
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
