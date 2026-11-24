"""Reproduce the README figures (synthetic feeds only). Run: python examples/headline.py  (needs matplotlib)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mmdiag as md

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"); os.makedirs(DOCS, exist_ok=True)
rng = np.random.default_rng(5)
idx = pd.date_range("2024-01-01", periods=60 * 1440, freq="1min"); idx = idx[idx.dayofweek < 5]
hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
season = 1 + 1.2 * np.exp(-((hour - 8) ** 2) / 3) + 1.8 * np.exp(-((hour - 14.5) ** 2) / 3)   # two session opens
daily = np.exp(rng.normal(0, .35, idx.normalize().nunique()))[pd.factorize(idx.normalize())[0]]
vol = 1.2e-4 * season * daily
eff = np.cumsum(rng.normal(0, 1, len(idx)) * vol)                                             # efficient log price

feeds = {
    "clean random walk": eff,
    "noisy quotes (bid-ask bounce)": eff + rng.normal(0, 1.5e-4, len(idx)),
    "smoothed / stale feed": pd.Series(eff).ewm(alpha=0.35).mean().to_numpy(),
}
feeds = {k: pd.Series(100 * np.exp(v), index=idx) for k, v in feeds.items()}

# 1) signature plots
fig, ax = plt.subplots(figsize=(7, 4))
for (lab, p), c in zip(feeds.items(), ("#2b8a3e", "#d9480f", "#1c7ed6")):
    sp = md.signature_plot(p); vr = md.variance_ratio(p, "1min", 5)
    ax.plot(sp.minutes, sp.ratio_to_longest, marker="o", color=c, label=f"{lab}  (VR(5) {vr['vr']:.2f}, z {vr['z']:+.0f})")
    print(f"{lab:32s} RV(1m)/RV(60m) {sp.ratio_to_longest.iloc[0]:.2f} | VR(5) {vr['vr']:.2f} z {vr['z']:+.1f}")
ax.axhline(1, color="#333", lw=.8); ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 15, 30, 60], ["1m", "2m", "5m", "10m", "15m", "30m", "1h"])
ax.set_xlabel("sampling interval"); ax.set_ylabel("realised variance / RV at 1 h"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("Signature plot: is the short-horizon data telling the truth?")
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "signature_plot.png"), dpi=130); plt.close(fig)

# 2) intraday profile + spread (spread = fixed part + part proportional to volatility)
spread = pd.Series(0.004 + 120 * vol * np.exp(rng.normal(0, .15, len(idx))), index=idx)
prof = md.intraday_profile(feeds["clean random walk"], spread, freq="5min")
fig, ax = plt.subplots(figsize=(8, 3.8)); ax2 = ax.twinx()
ax.bar(prof.index, prof.abs_return_bp, color="#ced4da", label="mean |5-min return| (bp)")
ax2.plot(prof.index, prof.median_spread, color="#d9480f", marker="o", ms=3, label="median spread")
ax.set_xlabel("hour of day (feed clock)"); ax.set_ylabel("mean |5-min return|, bp"); ax2.set_ylabel("median spread")
ax.set_title("Intraday profile: when the market moves, and what it charges to trade"); ax.grid(alpha=.3)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "intraday_profile.png"), dpi=130); plt.close(fig)

# 3) spread-volatility elasticity
e = md.spread_vol_elasticity(feeds["clean random walk"], spread, "1h")
print(f"spread-vol elasticity {e['elasticity']:.2f} ± {e['se']:.2f} (R² {e['r2']:.2f}, {e['buckets']} hourly buckets)")
d = e["data"]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(d.vol * 1e4, d.spread, s=6, alpha=.35, color="#1c7ed6")
xs = np.geomspace(d.vol.min(), d.vol.max(), 50); b, a = np.polyfit(np.log(d.vol), np.log(d.spread), 1)
ax.plot(xs * 1e4, np.exp(a) * xs ** b, color="#d9480f", lw=2, label=f"log-log slope {b:.2f} ± {e['se']:.2f}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("hourly realised volatility (bp)"); ax.set_ylabel("mean quoted spread")
ax.set_title("Is the spread a fixed fee or a volatility price?"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "spread_vol.png"), dpi=130); plt.close(fig)
