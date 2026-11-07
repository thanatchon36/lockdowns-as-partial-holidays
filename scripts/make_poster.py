#!/usr/bin/env python3
"""LinkedIn poster -> docs/assets/linkedin_poster.png (1440 x 1800 px, 4:5).

Every number is computed here: the HESI series and its peak from the real Google Mobility data in data/raw/ and the
observed Thai holiday calendar, the headline results from results/ through the same macros as the paper. Fonts:
Fira Sans (SIL OFL, docs/assets/fonts/), falling back to the matplotlib default. Run from the repository root
(`make poster`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from hesi import forecast as H  # noqa: E402
from hesi import paper_build as PB  # noqa: E402

OUT = Path("docs/assets/linkedin_poster.png")
FONTS = Path("docs/assets/fonts")
W, HT = 1440, 1800
BG, PANEL, RULE = "#03060d", "#0a1222", "#1c2a44"
AMBER, CREAM, INK, MUTED = "#f5a623", "#ffe8a3", "#eef0f4", "#8f9bb3"
DOI = "10.5281/zenodo.24036954"

for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
FAMILY = "Fira Sans" if any("Fira" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = FAMILY


def text(fig, x, y, s, size, color=INK, weight="regular", **kw):
    """x, y in pixels from the top-left corner; y is the top of the text unless va says otherwise."""
    kw.setdefault("va", "top")
    return fig.text(x / W, 1 - y / HT, s, fontsize=size * 0.72, color=color, fontweight=weight, **kw)


def main() -> Path:
    mob = H.load_mobility_th()
    hol = H.thai_holidays()
    hesi, ref = H.mobility_hesi(mob, hol)
    hesi = hesi.loc["2020-02-15":"2022-08-19"]
    peak_day, peak = hesi.idxmax(), float(hesi.max())
    assert peak_day not in hol, "the peak should be a working day"
    monthly = hesi.resample("MS").mean()
    wave = monthly.loc["2021"].idxmax()
    PB.theory_numbers()  # fills the theory macros (bM.closed) into PB.R
    R = PB.R

    fig = plt.figure(figsize=(W / 100, HT / 100), dpi=100, facecolor=BG)
    glow = fig.add_axes([0, 0, 1, 1], zorder=-10)
    yy, xx = np.mgrid[0:1:300j, 0:1:240j]
    glow.imshow(np.exp(-(((xx - 0.72) / 0.45) ** 2 + ((yy - 0.42) / 0.30) ** 2)), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "g", [BG, "#0b1a33"]), aspect="auto", extent=[0, 1, 0, 1], origin="upper")
    glow.axis("off")

    # header
    text(fig, 94, 66, "LOCKDOWNS AS PARTIAL HOLIDAYS", 27, AMBER, "bold")
    text(fig, 640, 68, "open research on forecasting Thai exports", 25, MUTED)

    # hero
    text(fig, 80, 128, f"{peak:.2f}", 250, AMBER, "bold")
    text(fig, 104, 392, "of a holiday", 64, CREAM)
    text(fig, 104, 488, f"{peak_day.day} April {peak_day.year} should have been Songkran.\n"
         "Instead it was a lockdown workday, and Thai\n"
         f"workplaces emptied {peak:.0%} as much as on a holiday.", 31, INK, linespacing=1.5)
    text(fig, 104, 672, "So treat a lockdown as a partial holiday.", 34, AMBER, "semibold")

    # the idea, in three steps
    text(fig, 930, 196, "THE IDEA", 20, MUTED, "medium")
    steps = [("Learn the work-stoppage effect\nfrom ten years of holidays:\nSongkran and Chinese New Year", 236),
             ("Measure every lockdown day\nin holiday units: the HESI", 386),
             ("Forecast with the same effect:\nno new parameter for the shock", 506)]
    for i, (s_, y) in enumerate(steps, 1):
        text(fig, 930, y - 6, str(i), 48, AMBER, "bold")
        text(fig, 985, y, s_, 24, INK, linespacing=1.4)
    fig.add_artist(plt.Line2D([905 / W] * 2, [1 - 190 / HT, 1 - 610 / HT], color=RULE, lw=2))
    text(fig, 930, 630, "x = H + (1 \u2212 H) \u00b7 HESI", 26, CREAM, "medium")

    # the HESI chart
    ax = fig.add_axes([94 / W, 1 - 1150 / HT, 1252 / W, 380 / HT], facecolor="none")
    days = [d for d in hol if hesi.index[0] <= d <= hesi.index[-1]]
    ax.vlines(days, 0, 1, color=INK, alpha=0.20, lw=1.3)
    ax.fill_between(hesi.index, 0, hesi.values, color=AMBER, alpha=0.92, lw=0)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(hesi.index[0], hesi.index[-1])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=MUTED, lw=1, alpha=0.6)
    ax.axhline(1, color=MUTED, lw=1, alpha=0.35, ls=(0, (2, 4)))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(colors=MUTED, length=0, labelsize=17, pad=10)
    ax.text(hesi.index[-1], 1.0, "a full public holiday  ", color=MUTED, fontsize=17, ha="right", va="bottom")
    ax.annotate(f"{peak_day:%d %b %Y}: {peak:.2f}\nthe postponed Songkran", xy=(peak_day, peak),
                xytext=(peak_day + pd.Timedelta(days=60), 0.86), color=CREAM, fontsize=18, fontweight="semibold",
                va="center", arrowprops=dict(arrowstyle="-", color=CREAM, lw=1, alpha=0.8))
    ax.annotate(f"{wave:%B %Y}: {monthly[wave]:.2f} on average\nthe 2021 wave", xy=(wave + pd.Timedelta(days=15), monthly[wave] + 0.05),
                xytext=(wave + pd.Timedelta(days=95), 0.80), color=CREAM, fontsize=18, fontweight="semibold",
                va="center", arrowprops=dict(arrowstyle="-", color=CREAM, lw=1, alpha=0.8))
    text(fig, 94, 1168 + 34, "Holiday-Equivalent Stoppage Index, Thailand: the drop in workplace mobility divided by its "
         f"drop on a public holiday ({ref:.0%}). Grey ticks: public holidays.", 18, MUTED)

    # stats row
    fig.add_artist(plt.Line2D([94 / W, 1346 / W], [1 - 1262 / HT] * 2, color=RULE, lw=1.5))
    stats = [
        (f"{R['e1.s1.pd.off.w24']}%→{R['e1.s1.pd.hesi.w24']}%", "forecast error (MAPE) 2020-21,\nno dose → HESI dose"),
        ("0", "new parameters needed\nfor the lockdown"),
        (R["bD.pm"].replace("-", "−"), f"monthly elasticity; theory\npredicted {R['bM.closed'].replace('-', chr(8722))}"),
        (R["nseeds"], "Monte Carlo\nreplications"),
    ]
    for i, (big, small) in enumerate(stats):
        x = [94, 560, 860, 1170][i]
        text(fig, x, 1292, big, 58 if i == 0 else 64, AMBER, "bold")
        text(fig, x, 1384, small, 21, MUTED, linespacing=1.45)

    # links box
    fig.add_artist(FancyBboxPatch((86 / W, 1 - 1740 / HT), 1268 / W, 258 / HT, boxstyle="round,pad=0,rounding_size=0.012",
                                  fc=PANEL, ec=RULE, lw=1.5, transform=fig.transFigure))
    text(fig, 130, 1512, "Code + paper", 20, MUTED)
    text(fig, 130, 1544, "github.com/thanatchon36/lockdowns-as-partial-holidays", 26, INK)
    text(fig, 130, 1594, "Read online", 20, MUTED)
    text(fig, 130, 1626, "thanatchon36.github.io/lockdowns-as-partial-holidays", 26, INK)
    text(fig, 960, 1512, "Dataset", 20, MUTED)
    text(fig, 960, 1544, "hf.co/datasets/thanatchon36/", 22, INK)
    text(fig, 960, 1574, "lockdowns-as-partial-holidays", 22, INK)
    text(fig, 960, 1626, "Thai version: paper_th.pdf", 20, MUTED)
    text(fig, 130, 1694, "Export values in the study are simulated; the holiday calendar, OxCGRT and Google Mobility "
         "data are real.", 18, MUTED)

    # footer
    text(fig, 94, 1772, f"Thanatchon Chongmankhong  ·  views are my own  ·  DOI {DOI}", 20, MUTED, va="center")
    text(fig, 1346, 1772, "Mobility: Google LLC, COVID-19 Community Mobility Reports", 16, MUTED, va="center", ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)  # exactly 1440 x 1800
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
