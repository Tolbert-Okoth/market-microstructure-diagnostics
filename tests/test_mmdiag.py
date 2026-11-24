import unittest

import numpy as np
import pandas as pd

import mmdiag as md


def feed(days=40, seed=0, noise_bp=0.0, ar=0.0):
    idx = pd.date_range("2024-01-01", periods=days * 1440, freq="1min"); idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(seed); e = rng.normal(0, 2e-4, len(idx)); r = e.copy()
    for i in range(1, len(r)):
        r[i] += ar * r[i - 1]
    logp = np.cumsum(r) + rng.normal(0, noise_bp * 1e-4, len(idx))
    return pd.Series(100 * np.exp(logp), index=idx)


class Signature(unittest.TestCase):
    def test_flat_for_random_walk(self):
        sp = md.signature_plot(feed())
        self.assertTrue(np.allclose(sp.ratio_to_longest, 1, atol=0.12))

    def test_noise_inflates_short_intervals(self):
        sp = md.signature_plot(feed(noise_bp=3.0)); self.assertGreater(sp.ratio_to_longest.iloc[0], 1.5)


class VarianceRatio(unittest.TestCase):
    def test_random_walk_near_one(self):
        v = md.variance_ratio(feed(), "5min", 6); self.assertLess(abs(v["z"]), 3); self.assertAlmostEqual(v["vr"], 1, delta=0.1)

    def test_trending_above_one(self):
        v = md.variance_ratio(feed(ar=0.3), "1min", 6); self.assertGreater(v["vr"], 1.3); self.assertGreater(v["z"], 5)


class Spread(unittest.TestCase):
    def test_elasticity_recovers_proportional_spread(self):
        idx = pd.date_range("2024-01-01", periods=30 * 1440, freq="1min"); idx = idx[idx.dayofweek < 5]
        rng = np.random.default_rng(2); vol = 1e-4 * np.exp(rng.normal(0, .5, len(idx) // 60 + 1)).repeat(60)[: len(idx)]
        p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1, len(idx)) * vol)), index=idx)
        s = pd.Series(vol * 1e3 * np.exp(rng.normal(0, .1, len(idx))), index=idx)
        e = md.spread_vol_elasticity(p, s); self.assertAlmostEqual(e["elasticity"], 1.0, delta=0.15)

    def test_profile_columns(self):
        p = feed(5); prof = md.intraday_profile(p, spread=pd.Series(0.1, index=p.index))
        self.assertEqual(len(prof), 24); self.assertIn("median_spread", prof)


if __name__ == "__main__":
    unittest.main()
