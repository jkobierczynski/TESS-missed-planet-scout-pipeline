# TESS Missed-Planet Scout: Design Principles

> Working title. Rename freely; replace `scout` below with the final package name.

## 1. Purpose

Find transiting exoplanet candidates in public TESS data that the official pipelines (SPOC, QLP) missed, vet them rigorously, submit the survivors to ExoFOP as community TOIs (CTOIs), and follow up the best ones from my own observatory (Belgium, ~51° N).

## 2. Scope

**In scope**
- Northern TESS continuous viewing zone (CVZ, around the north ecliptic pole in Draco) first: many sectors per star, circumpolar from Belgium, so year-round ground follow-up.
- Signal classes the official searches handle poorly:
  - single transits and duo transits (long periods, sector gaps)
  - shallow transits around M dwarfs
  - transits hidden by stellar variability or poor default detrending
  - faint and crowded-field stars
- End-to-end: target selection → light curves → detrending → search → triage → vetting → report → CTOI package.

**Out of scope (for now)**
- Radial-velocity confirmation, planet validation papers, full statistical occurrence rates.
- Non-TESS missions (Kepler/K2 can come later via the same interfaces).
- Any Java-based tooling.

## 3. Core principles

1. **Vet before you believe.** Most dips are not planets. No signal is called a candidate until it has passed the full vetting checklist (§6). The pipeline's job is to kill false positives, not to produce exciting plots.
2. **Know your blind spots.** Injection-recovery testing is part of the pipeline from day one, not an afterthought. Every search configuration has a measured recovery map (depth × period × duration). "Nothing found" only means something if we know what we could have found.
3. **Reproducible and traceable.** Every result records its inputs: TIC ID, sectors, light-curve source and version, detrending parameters, search parameters, code version (git hash). Any candidate can be regenerated from its record.
4. **Code public, results private.** The code is open source. Candidate lists, vetting reports and cached data stay out of git (`results/`, `cache/` gitignored) until submitted as CTOIs. Publishing unsubmitted candidates invites being scooped.
5. **Modular stages with plain interfaces.** Each stage reads and writes well-defined files (FITS / Parquet / JSON), so stages can be rerun, swapped or ported independently. No stage depends on another's internals.
6. **Resumable batch processing.** Thousands of stars, many gigabytes. Work is idempotent and checkpointed: a crash or reboot resumes where it stopped; finished stars are skipped.
7. **Cheap first, expensive later.** Fast filters (known-object cross-match, basic SNR cuts) run before slow ones (pixel-level centroid analysis, false-positive probability).
8. **Configuration over code edits.** Search grids, detrending windows, thresholds and paths live in a versioned config file (TOML). Runs are named by config.
9. **CLI first, GUI never required.** Every stage is a command-line subcommand. Human review happens through generated HTML reports, not an interactive app.
10. **Prototype in Python, port the hot paths.** Use the existing ecosystem to find out what works. Once stable, heavy loops (batch detrending, search) may move into TransitScope's C++ core behind the same file interfaces.

## 4. Pipeline stages

| Stage | Input | Output | Main tools |
|---|---|---|---|
| `targets` | sky region, mag/Teff cuts | target list (TIC IDs + metadata) | TIC, Gaia DR3 |
| `fetch` | target list | cached light curves + TPFs | lightkurve / astroquery (MAST): QLP, TESS-SPOC, TGLC |
| `stitch` | per-sector light curves | one normalised multi-sector light curve | lightkurve |
| `detrend` | stitched light curve | flattened light curve + trend | wotan (biweight, window ≈ 3× max duration) |
| `search` | flattened light curve | periodic + single-event detections | TLS; dedicated single-transit search |
| `triage` | detections | detections not already known | ExoFOP TOI/CTOI list, NASA Exoplanet Archive, TESS EB catalogue, VSX |
| `vet` | surviving detections | vetting metrics + verdict | own checks, TPF difference imaging, TRICERATOPS |
| `report` | vetted candidates | per-candidate HTML report + summary table | matplotlib / static HTML |
| `ctoi` | accepted candidates | ExoFOP CTOI submission package | — |
| `ephem` | candidate ephemeris | transit windows observable from my site | astropy |

Data handling rules:
- Mask momentum dumps, scattered-light segments and bad-quality flags before detrending.
- Keep cadence per sector (30 min / 10 min / 200 s FFIs) and handle it explicitly; never resample silently.
- Keep the raw cached data immutable; derived products go elsewhere.

## 5. Search strategy

- **Periodic search:** TLS over a period grid up to half the total baseline, requiring ≥2 (preferably ≥3) transits.
- **Single-event search:** matched-filter box/trapezoid scan over durations of ~1–15 h. Events are grouped per star across sectors to find duo transits and list period aliases.
- **Detection threshold:** SDE/SNR thresholds set from injection-recovery and from the false-alarm rate on scrambled or inverted light curves, not guessed.
- **Inverted light-curve test:** run the same search on flux-inverted data. Systematics that look like dips also look like bumps.

## 6. Vetting checklist

A candidate must pass all of these, each recorded in the report:

1. Transit shape plausible (not V-shaped unless grazing is justified).
2. Odd/even depth consistent.
3. No secondary eclipse at phase 0.5 (or at other phases, for eccentric orbits).
4. Depth consistent across sectors and light-curve sources.
5. No centroid shift in TPF difference images; signal on target, not a neighbour.
6. Gaia DR3: no bright neighbour within the aperture that could produce the depth; RUWE not suggesting an unresolved binary.
7. Implied companion radius < ~2 R_Jup given the TIC stellar radius.
8. Duration consistent with the stellar density for the implied period.
9. Not coincident with spacecraft events (momentum dumps, orbit gaps, scattered light).
10. False-positive probability (TRICERATOPS) below the chosen threshold.
11. Not already known (re-checked right before submission).

## 7. Outputs

- `results/<run>/candidates.parquet`: all detections with metrics and verdicts.
- `results/<run>/reports/<TIC>.html`: one page per candidate: stitched and flattened light curve, phase fold, individual transits, odd/even, secondary, centroid images, neighbour map, metrics table, provenance block.
- `results/<run>/summary.html`: sortable overview for review.
- `results/<run>/run.json`: config, code version, timestamps.

## 8. Testing

- **Unit tests** per stage on small synthetic light curves.
- **Injection-recovery** (batman-style transit models injected into real, planet-free light curves) producing recovery maps per configuration.
- **Known-planet recovery:** known TOIs and confirmed planets in the CVZ must be recovered as a sanity check.
- CI on GitHub Actions runs unit tests and a small injection-recovery suite on synthetic data (no MAST access needed).

## 9. Repository layout

```
scout/
├── pyproject.toml
├── LICENSE                  # GPL-3.0-or-later
├── README.md
├── DESIGN.md
├── config/
│   └── cvz-north.toml
├── src/scout/
│   ├── targets.py
│   ├── fetch.py
│   ├── stitch.py
│   ├── detrend.py
│   ├── search/
│   │   ├── periodic.py
│   │   └── single.py
│   ├── triage.py
│   ├── vet/
│   ├── report/
│   ├── ctoi.py
│   ├── ephem.py
│   └── cli.py
├── tests/
├── cache/        # gitignored: downloaded data
└── results/      # gitignored: candidates and reports
```

## 10. Conventions

- Licence: GPL-3.0-or-later.
- Python ≥ 3.11, type hints throughout, `ruff` for lint/format, `pytest` for tests.
- Times in BTJD internally; convert to BJD_TDB at the reporting and ephemeris boundary.
- Flux normalised to median 1.0 per sector before stitching.
- All thresholds and grids in config; none hard-coded.

## 11. Roadmap

1. **M0, skeleton:** repo, CLI, config loading, CI, synthetic data generator.
2. **M1, single star end-to-end:** fetch → stitch → detrend → TLS → report for one known planet host.
3. **M2, injection-recovery:** recovery maps; set thresholds.
4. **M3, batch CVZ run:** resumable processing of the bright CVZ sample; triage against known objects.
5. **M4, single/duo-transit search** and period-alias listing.
6. **M5, full vetting:** centroids, Gaia neighbours, TRICERATOPS.
7. **M6, first CTOI submission** and ground follow-up from my own observatory.
8. **Later:** fainter stars via TGLC; port hot paths to TransitScope (C++).
