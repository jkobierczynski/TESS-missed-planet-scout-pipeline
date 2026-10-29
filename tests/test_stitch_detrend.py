import dataclasses

import numpy as np

from scout.detrend import detrend, window_days
from scout.stitch import quality_mask, stitch
from scout.synthetic import MOMENTUM_DUMP_BIT, transit_model


def test_normalised_per_sector(raw_lc, cfg):
    lc = stitch(raw_lc, cfg.stitch)
    for s in lc.sectors:
        assert abs(np.median(lc.flux[lc.sector == s]) - 1) < 1e-12
    assert lc.meta["stitch"]["sectors"] == [14, 15]
    assert lc.meta["stitch"]["cadences_s"] == [120]


def test_quality_and_momentum_dumps(raw_lc, cfg):
    lc = raw_lc.select(np.ones(len(raw_lc), bool))
    lc.quality[10] = 8  # coarse point: in the default mask
    lc.flux[20] = np.nan
    keep = quality_mask(lc, cfg.stitch)
    assert not keep[10] and not keep[20]
    dumps = lc.time[(lc.quality & MOMENTUM_DUMP_BIT) != 0]
    pad = cfg.stitch.momentum_dump_pad_hours / 24
    for td in dumps:
        near = np.abs(lc.time - td) < pad * 0.99
        assert not keep[near].any()
    no_pad = dataclasses.replace(cfg.stitch, momentum_dump_pad_hours=0.0)
    assert quality_mask(lc, no_pad).sum() > keep.sum()


def test_detrend_removes_variability_keeps_transit(raw_lc, cfg, transit):
    st = stitch(raw_lc, cfg.stitch)
    flat = detrend(st, cfg.detrend)
    assert flat.trend is not None
    oot = transit_model(flat.time, transit) == 1.0
    # Variability (0.3 %) gone: out-of-transit scatter close to the 300 ppm noise.
    assert np.std(flat.flux[oot]) < 400e-6
    assert np.std(st.flux) > 2e-3
    # Transit depth survives within 15 %.
    model = transit_model(flat.time, transit)
    intr = model < 1 - 0.8 * transit.depth
    measured = 1 - np.median(flat.flux[intr])
    expected = 1 - np.median(model[intr])
    assert abs(measured - expected) / expected < 0.15
    assert flat.meta["detrend"]["window_days"] == window_days(cfg.detrend)


def test_upper_clip_removes_flare_not_dips(raw_lc, cfg):
    st = stitch(raw_lc, cfg.stitch)
    st.flux[500:503] += 0.02  # flare
    st.flux[900:903] -= 0.02  # dip
    t_flare, t_dip = st.time[501], st.time[901]
    flat = detrend(st, cfg.detrend)
    assert not np.any(flat.time == t_flare)
    assert np.any(flat.time == t_dip)
    assert flat.meta["detrend"]["n_clipped_upper"] >= 3
