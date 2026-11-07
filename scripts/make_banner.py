#!/usr/bin/env python3
"""README banner -> docs/assets/banner.png (1680 x 624 px).

Left: the Thai Holiday-Equivalent Stoppage Index, 2020-2022, computed from the real Google Community Mobility
data in data/raw/ and the observed Thai holiday calendar (the same `mobility_hesi` the paper uses). Public holidays
are drawn as full-height ticks: a lockdown day is a fraction of one of them. Run from the repository root
(`make banner`).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hesi import forecast as H  # noqa: E402

BG, AMBER, INK, MUTED = "#03060d", "#f5a623", "#e8e9ec", "#8f9bb3"
OUT = "docs/assets/banner.png"


def main() -> str:
    mob = H.load_mobility_th()
    hol = H.thai_holidays()
    hesi, _ = H.mobility_hesi(mob, hol)
    hesi = hesi.loc["2020-02-15":"2022-08-19"]
    days = [d for d in hol if hesi.index[0] <= d <= hesi.index[-1]]

    fig = plt.figure(figsize=(16.8, 6.24), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.075, 0.17, 0.37, 0.66], facecolor=BG)
    ax.vlines(days, 0, 1, color=INK, alpha=0.22, lw=1.0)
    ax.fill_between(hesi.index, 0, hesi.values, color=AMBER, alpha=0.9, lw=0)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(hesi.index[0], hesi.index[-1])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=11)
    ax.set_yticks([0, 1], ["0", "1 holiday"])
    ticks = pd.to_datetime(["2020-04-01", "2021-01-01", "2022-01-01"])
    ax.set_xticks(ticks, ["Apr 2020", "2021", "2022"])
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)

    fig.text(0.50, 0.60, "Lockdowns as\nPartial Holidays", color=AMBER, fontsize=46, fontweight="bold",
             va="bottom", linespacing=1.05)
    fig.text(0.502, 0.40, "Work-stoppage elasticities learned from Songkran and\n"
             "Chinese New Year, transferred to forecast Thailand's\nmonthly exports through the COVID-19 shock",
             color=INK, fontsize=17, va="bottom", linespacing=1.45)
    fig.text(0.502, 0.27, "Prophet · HESI dose · temporal aggregation · Monte Carlo",
             color=MUTED, fontsize=14, va="bottom")
    fig.text(0.985, 0.035, "Thai Holiday-Equivalent Stoppage Index, Feb 2020 - Aug 2022 (amber) and public holidays "
             "(ticks) · Google LLC, COVID-19 Community Mobility Reports", color=MUTED, fontsize=9, ha="right")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
