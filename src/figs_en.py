#!/usr/bin/env python3
"""Publication figures for the English HES paper.

Design rules (enforced by `check_layout`):
  * every legend lives in its own empty axes row (never on top of data);
  * no text annotations inside data areas -- meanings are given by legends;
  * no text bounding box may intersect another text box or another panel's data area.
Colours follow the colour-blind-safe Okabe-Ito palette.
"""
from __future__ import annotations

import os
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import hesi_forecast as H  # noqa: E402

W = 6.3  # inches = LaTeX text width (A4, 2.5 cm margins) -> figures are never rescaled
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8C8C8C", lgrey="#C8C8C8",
          brown="#8C510A", tan="#F3E5CC", lilac="#EBD7E6", mint="#DDF1E6")
MODEL = {"Prophet-D HE": (OI["blue"], "Prophet, daily (P-D)"),
         "Prophet-M HE": (OI["sky"], "Prophet, monthly (P-M)"),
         "Prophet-M HE+AC": (OI["green"], "Prophet, monthly + aggregation correction (P-M+AC)"),
         "SARIMAX HE": (OI["orange"], "SARIMAX airline, monthly"),
         "LightGBM-D HE": (OI["verm"], "LightGBM hybrid, daily"),
         "LightGBM-M HE": (OI["purple"], "LightGBM hybrid, monthly")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.8, "axes.titlesize": 8.2,
                     "axes.labelsize": 7.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "legend.fontsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "axes.titlelocation": "left"})


# ----------------------------------------------------------------------------- layout checker
def _overlap(a, b, pad=0.5):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > pad) and (min(a.y1, b.y1) - max(a.y0, b.y0) > pad)


def _texts(fig):
    out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_visible() and t.get_text().strip():
                out.append(("title", t, ax))
        if ax.axison:
            for t in (ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
                if t.get_visible() and t.get_text().strip():
                    out.append(("label", t, ax))
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                    loc = tick.get_loc()
                    if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                        continue
                    for lab in (tick.label1, tick.label2):
                        if lab.get_visible() and lab.get_text().strip():
                            out.append(("tick", lab, ax))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append(("in-axes text", t, ax))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                out.append(("legend", t, ax))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(("fig text", t, None))
    return out


def check_layout(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(k, t, ax, t.get_window_extent(r)) for k, t, ax in _texts(fig)]
    probs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _overlap(items[i][3], items[j][3]):
                probs.append(f"text/text: '{items[i][1].get_text()}' x '{items[j][1].get_text()}'")
    data_axes = [ax for ax in fig.axes if ax.axison and ax.get_visible()]
    for k, t, own, bb in items:
        if k == "in-axes text":
            probs.append(f"in-axes text present: '{t.get_text()}'")
        for ax in data_axes:
            if ax is own:
                continue
            if _overlap(bb, ax.get_window_extent(r)):
                probs.append(f"{k} '{t.get_text()}' intrudes another panel")
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(r)
        for dax in data_axes:
            if _overlap(lb, dax.get_window_extent(r)):
                probs.append(f"legend covers data area of panel '{dax.get_title()}'")
    print(f"[layout] {name}: " + ("OK, no overlaps" if not probs else f"{len(probs)} problem(s)"), flush=True)
    for p in probs[:12]:
        print("    -", p)
    return probs


def _legend_row(fig, spec, handles, ncol):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=ncol, handlelength=2.2, columnspacing=1.4,
              borderaxespad=0.0)
    return ax


def _save(fig, out, name):
    probs = check_layout(fig, name)
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return probs


def _dates(ax, step=6, fmt="%b\n%Y"):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# ----------------------------------------------------------------------------- figures
def fig_data(out, paths, setup):
    cal = setup["cal"]
    fig = plt.figure(figsize=(W, 5.3))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 0.24, 1.0, 0.2], hspace=0.62, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(pd.Timestamp("2020-01-20"), pd.Timestamp("2022-09-30"), color=OI["tan"], lw=0)
    for S, col, lw in (("S2", OI["verm"], 1.1), ("S1", OI["blue"], 0.9)):
        s = paths["monthly"][S] / 1000
        ax.plot(s.index, s.values, color=col, lw=lw)
    for o in ("2017-12-31", "2019-12-31", "2022-12-31"):
        ax.axvline(pd.Timestamp(o), color=OI["black"], ls=":", lw=1.0)
    ax.set_ylabel("USD bn per month")
    ax.set_title("(a) Simulated monthly exports, replication 0")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _legend_row(fig, gs[1, :], [
        Line2D([], [], color=OI["blue"], lw=1.4, label="Scenario S1 (ideal transfer; equals S2 before 2020)"),
        Line2D([], [], color=OI["verm"], lw=1.4, label="Scenario S2 (realistic)"),
        Patch(color=OI["tan"], label="COVID window (20 Jan 2020 - 30 Sep 2022)"),
        Line2D([], [], color=OI["black"], ls=":", lw=1.2, label="forecast origins E0, E1, E2")], ncol=2)
    y = paths["daily_S1"]
    for k, (a, b, title) in enumerate((("2019-03-25", "2019-05-05", "(b) Daily exports around Songkran 2019"),
                                       ("2019-01-22", "2019-02-25", "(c) Daily exports around Chinese New Year 2019"))):
        ax = fig.add_subplot(gs[2, k])
        seg, c = y.loc[a:b], cal.loc[a:b]
        col = np.where(c.H.values == 1, OI["verm"], np.where(c.C.values == 1, OI["orange"],
                       np.where(c.weekend.values == 1, OI["lgrey"], OI["sky"])))
        ax.bar(seg.index, seg.values, color=col, width=0.85)
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if k == 0:
            ax.set_ylabel("USD mn per day")
    _legend_row(fig, gs[3, :], [Patch(color=OI["sky"], label="ordinary weekday"),
                                Patch(color=OI["lgrey"], label="Saturday / Sunday"),
                                Patch(color=OI["verm"], label="Thai public holiday"),
                                Patch(color=OI["orange"], label="Chinese-New-Year window day")], ncol=4)
    return _save(fig, out, "fig_data.png")


def fig_drivers(out, setup):
    drv, cal = setup["drv"], setup["cal"]
    mob = H.load_mobility_th()
    drop = (-mob / 100).clip(lower=0)
    hol = cal.index[cal.H.values == 1]
    m = drv.loc["2020-01-01":"2022-12-31"]
    xlim = (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))
    fig = plt.figure(figsize=(W, 7.6))
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 0.26, 1, 0.34, 1, 0.26], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    ax.plot(drop.index, drop.values, color=OI["grey"], lw=0.7)
    hd = drop[drop.index.isin(hol)]
    ax.scatter(hd.index, hd.values, s=8, color=OI["verm"], zorder=3)
    ax.axhline(setup["ref"], color=OI["verm"], ls="--", lw=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("drop vs. baseline")
    ax.set_title("(a) Workplace mobility in Thailand: holidays define the unit of HESI")
    _dates(ax)
    _legend_row(fig, gs[1], [
        Line2D([], [], color=OI["grey"], lw=1.2, label="daily drop in workplace visits (Google)"),
        Line2D([], [], color=OI["verm"], marker="o", ls="", ms=4, label="drop on Thai public holidays"),
        Line2D([], [], color=OI["verm"], ls="--", lw=1.2,
               label=f"reference: median weekday-holiday drop = {setup['ref']:.2f}")], ncol=2)
    ax = fig.add_subplot(gs[2])
    for a, b in _runs(m.bin_th):
        ax.axvspan(a, b + pd.Timedelta(days=1), color=OI["lilac"], lw=0)
    ax.plot(m.index, m.s_th, color=OI["orange"], lw=1.1)
    ax.plot(m.index, m.hesi_th, color=OI["blue"], lw=1.1)
    ax.plot(m.index, 0.5 * m.hesi_th, color=OI["green"], lw=1.1, ls=":")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("dose (0 - 1)")
    ax.set_title("(b) Domestic channel (Thailand)")
    _dates(ax)
    _legend_row(fig, gs[3], [
        Line2D([], [], color=OI["orange"], lw=1.4, label="OxCGRT stringency / 100 (real)"),
        Line2D([], [], color=OI["blue"], lw=1.4, label="HESI = mobility drop / holiday reference (real)"),
        Line2D([], [], color=OI["green"], lw=1.4, ls=":", label="true export dose in S2 (0.5 x HESI)"),
        Patch(color=OI["lilac"], label="months switched fully on by 'Binary' (stringency >= 0.70)")], ncol=2)
    ax = fig.add_subplot(gs[4])
    for a, b in _runs(m.bin_cn):
        ax.axvspan(a, b + pd.Timedelta(days=1), color=OI["lilac"], lw=0)
    ax.plot(m.index, m.s_cn, color=OI["orange"], lw=1.1)
    ax.plot(m.index, m.e_cn, color=OI["blue"], lw=1.1)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("dose (0 - 1)")
    ax.set_title("(c) Partner channel (China)")
    _dates(ax)
    _legend_row(fig, gs[5], [
        Line2D([], [], color=OI["orange"], lw=1.4, label="OxCGRT China stringency / 100 (real)"),
        Line2D([], [], color=OI["blue"], lw=1.4, label="industrial-stoppage episodes (stylised, CNY units)"),
        Patch(color=OI["lilac"], label="'Binary' months (episode >= 0.5)")], ncol=2)
    return _save(fig, out, "fig_drivers.png")


def _runs(flag):
    f, idx, out, start = flag.values.astype(bool), flag.index, [], None
    for i, v in enumerate(f):
        if v and start is None:
            start = idx[i]
        if (not v or i == len(f) - 1) and start is not None:
            out.append((start, idx[i - 1] if not v else idx[i]))
            start = None
    return out


def fig_e1_paths(out, paths):
    fig = plt.figure(figsize=(W, 3.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.42, wspace=0.10)
    spec = [(("Prophet-D HE", "off"), OI["grey"], "--", "P-D, Off (no COVID dose)"),
            (("Prophet-D HE", "binary"), OI["purple"], "-", "P-D, Binary whole-month switch"),
            (("Prophet-D HE", "stringency"), OI["orange"], "-", "P-D, raw stringency"),
            (("Prophet-D HE", "hesi"), OI["blue"], "-", "P-D, HESI dose ($\\lambda$ = 1)"),
            (("LightGBM-D HE", "hesi"), OI["verm"], "-.", "LightGBM daily, HESI dose")]
    axes = []
    for k, S in enumerate(("S1", "S2")):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        p = paths[("E1", S)]
        mo, P = p["months"], p["P"]
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-30"), color=OI["mint"], lw=0)
        ax.plot(mo, p["actual"] / 1000, color=OI["black"], lw=1.9)
        for key, col, ls, _ in spec:
            ax.plot(mo, P[key] / 1000, color=col, ls=ls, lw=1.1)
        w = mo >= H.W20_START
        ax.plot(mo[w], p["cal"]["Prophet-D HE"][w] / 1000, color=OI["green"], lw=1.5, ls=(0, (4, 1.5)))
        ax.set_title({"S1": "(a) S1: ideal transfer", "S2": "(b) S2: realistic"}[S])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        if k == 0:
            ax.set_ylabel("USD bn per month")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    handles = [Line2D([], [], color=OI["black"], lw=1.9, label="actual (simulated truth)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.3, label=lab) for _, c, ls, lab in spec]
    handles += [Line2D([], [], color=OI["green"], lw=1.5, ls=(0, (4, 1.5)), label="P-D, HESI-cal (dose calibrated)"),
                Patch(color=OI["mint"], label="calibration window Mar-Apr 2020")]
    _legend_row(fig, gs[1, :], handles, ncol=3)
    return _save(fig, out, "fig_e1_paths.png")


def fig_e1_dist(out, met):
    order = [("off", "Off (no dose)", OI["grey"]), ("binary", "Binary switch", OI["purple"]),
             ("stringency", "Raw stringency", OI["orange"]), ("hesi", "HESI ($\\lambda$ = 1)", OI["blue"]),
             ("hesi-cal", "HESI-cal", OI["green"]), ("oracle", "Oracle (true $\\kappa$)", OI["black"])]
    d = met[(met.exp == "E1") & (met.model == "Prophet-D HE") & (met.window == "W20")]
    fig = plt.figure(figsize=(W, 3.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.3], hspace=0.55, wspace=0.12)
    axes = []
    for k, S in enumerate(("S1", "S2")):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        data = [d[(d.scen == S) & (d.variant == v)].mape.values for v, _, _ in order][::-1]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=2.5, mfc="none"))
        for patch, (_, _, col) in zip(bp["boxes"], order[::-1]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title({"S1": "(a) S1: ideal transfer", "S2": "(b) S2: realistic"}[S])
        ax.set_xlabel("MAPE (%), May 2020 - Dec 2021")
    xmax = max(a.get_xlim()[1] for a in axes)
    for k, ax in enumerate(axes):
        ax.set_xlim(0, xmax)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels([lab for _, lab, _ in order][::-1] if k == 0 else [])
    _legend_row(fig, gs[1, :], [Patch(color=c, alpha=0.55, label=lab) for _, lab, c in order], ncol=3)
    return _save(fig, out, "fig_e1_dist.png")


def fig_lambda(out, met):
    d = met[(met.exp == "E1") & (met.variant == "lam") & (met.window == "W24")]
    g = d.groupby(["scen", "model", "lam"]).mape.mean().reset_index()
    ref = met[(met.exp == "E1") & (met.window == "W24") & (met.model == "Prophet-D HE")]
    fig = plt.figure(figsize=(W, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.42], hspace=0.42, wspace=0.10)
    axes = []
    for k, S in enumerate(("S1", "S2")):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        for mdl, (col, _) in MODEL.items():
            s = g[(g.scen == S) & (g.model == mdl)]
            ax.plot(s.lam, s.mape, color=col, lw=1.6 if mdl == "Prophet-D HE" else 1.1,
                    ls="--" if "LightGBM" in mdl else "-")
        ax.axvline(H.SCENARIOS[S]["kappa"], color=OI["black"], ls=":", lw=1.1)
        ax.axhline(ref[(ref.scen == S) & (ref.variant == "binary")].mape.mean(), color="#555555", ls="-.", lw=0.9)
        ax.axhline(ref[(ref.scen == S) & (ref.variant == "stringency")].mape.mean(), color="#555555",
                   ls=(0, (5, 2)), lw=0.9)
        ax.set_ylim(0, 50)
        ax.set_xlim(0, 3)
        ax.set_xlabel("dose multiplier $\\lambda$ on mobility-HESI")
        ax.set_title({"S1": "(a) S1: true $\\kappa$ = 1.0", "S2": "(b) S2: true $\\kappa$ = 0.5"}[S])
        if k == 0:
            ax.set_ylabel("mean MAPE (%), Jan 2020 - Dec 2021")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    handles = [Line2D([], [], color=c, lw=1.5, ls="--" if "LightGBM" in m else "-", label=lab)
               for m, (c, lab) in MODEL.items()]
    handles += [Line2D([], [], color=OI["black"], ls=":", lw=1.2, label="true pass-through $\\kappa$"),
                Line2D([], [], color="#555555", ls="-.", lw=1.0, label="reference: P-D with Binary switch"),
                Line2D([], [], color="#555555", ls=(0, (5, 2)), lw=1.0, label="reference: P-D with raw stringency")]
    _legend_row(fig, gs[1, :], handles, ncol=2)
    return _save(fig, out, "fig_lambda.png")


def implied_monthly_beta(setup, beta, col):
    x = setup["X"]["off"][col].loc[:"2019-12-31"]
    Mx = H.month_matrix(x)
    xbar = np.nanmean(Mx, axis=1)
    return float(np.sum(xbar * H.log_mean_exp(beta, Mx)) / np.sum(xbar ** 2))


def fig_beta(out, bet, setup):
    b = bet[bet.exp == "E1"]
    order = ["Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE"]
    short = ["P-D", "P-M", "P-M+AC", "SARIMAX"]
    fig = plt.figure(figsize=(W, 3.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.45, wspace=0.28)
    for k, (col, true, name, xc) in enumerate((("beta_D", H.TRUE["beta_D"], "(a) domestic elasticity $\\beta_D$", "xD"),
                                               ("beta_P", H.TRUE["beta_P"], "(b) partner elasticity $\\beta_P$", "xP"))):
        ax = fig.add_subplot(gs[0, k])
        bp = ax.boxplot([b[b.model == m][col].values for m in order], widths=0.55, patch_artist=True,
                        flierprops=dict(marker="o", ms=2.5, mfc="none"))
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(MODEL[m][0])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(true, color=OI["black"], ls="--", lw=1.0)
        ax.axhline(implied_monthly_beta(setup, true, xc), color="#555555", ls=":", lw=1.2)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(short)
        ax.set_title(name)
        ax.set_ylabel("estimate (log points per stoppage day)")
    handles = [Patch(color=MODEL[m][0], alpha=0.6, label=MODEL[m][1]) for m in order]
    handles += [Line2D([], [], color=OI["black"], ls="--", lw=1.2, label="true daily value"),
                Line2D([], [], color="#555555", ls=":", lw=1.4, label="aggregation-implied monthly value (Prop. 2)")]
    _legend_row(fig, gs[1, :], handles, ncol=2)
    return _save(fig, out, "fig_beta.png")


def fig_jensen(out, setup):
    beta = H.TRUE["beta_D"]
    X, drv = setup["X"]["off"], setup["drv"]
    cal = setup["cal"]
    tr = X.xD.loc[:"2019-12-31"]
    Mtr = H.month_matrix(tr)
    xb_tr, E_tr = np.nanmean(Mtr, axis=1), H.log_mean_exp(beta, Mtr)
    Hh = cal.H.values
    lock = pd.Series(Hh + (1 - Hh) * np.clip(drv.hesi_th.values, 0, 1), index=cal.index).loc["2020-03-01":"2021-12-31"]
    Ml = H.month_matrix(lock)
    xb_l, E_l = np.nanmean(Ml, axis=1), H.log_mean_exp(beta, Ml)
    slope = float(np.sum(xb_tr * E_tr) / np.sum(xb_tr ** 2))
    fig = plt.figure(figsize=(W, 3.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.48, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    lo = min((beta * xb_l).min(), (beta * xb_tr).min()) * 1.05
    ax.plot([lo, 0], [lo, 0], color=OI["black"], ls="--", lw=1.0)
    ax.plot([lo, 0], [slope / beta * lo, 0], color=OI["verm"], ls=":", lw=1.3)
    ax.scatter(beta * xb_tr, E_tr, s=10, color=OI["verm"], zorder=3)
    ax.scatter(beta * xb_l, E_l, s=14, marker="s", color=OI["blue"], zorder=3)
    ax.set_xlabel("linear monthly prediction  $\\beta\\,\\bar{x}_m$")
    ax.set_ylabel("exact monthly log effect  $E_m(\\beta)$")
    ax.set_title("(a) Concentrated vs. diffuse stoppage months")
    ax = fig.add_subplot(gs[0, 1])
    bg = np.linspace(-2.0, 0.0, 200)
    ax.plot(bg, bg, color=OI["black"], ls="--", lw=1.0)
    ax.plot(bg, np.exp(bg) - 1, color=OI["blue"], lw=1.5)
    ax.scatter([beta], [np.exp(beta) - 1], color=OI["verm"], s=22, zorder=3)
    ax.scatter([H.TRUE["beta_P"]], [np.exp(H.TRUE["beta_P"]) - 1], color=OI["orange"], s=22, zorder=3)
    ax.set_xlabel("daily elasticity $\\beta$")
    ax.set_ylabel("monthly-equivalent coefficient")
    ax.set_title("(b) Attenuation: $\\beta_M = e^{\\beta} - 1$")
    _legend_row(fig, gs[1, :], [
        Line2D([], [], color=OI["verm"], marker="o", ls="", ms=4, label="holiday months 2010-2019 (few days at $x = 1$)"),
        Line2D([], [], color=OI["blue"], marker="s", ls="", ms=4, label="lockdown months 2020-2021 (dose spread over month)"),
        Line2D([], [], color=OI["black"], ls="--", lw=1.2, label="45-degree line (no aggregation bias)"),
        Line2D([], [], color=OI["verm"], ls=":", lw=1.4, label=f"slope fitted on holiday months ({slope:.2f})"),
        Line2D([], [], color=OI["blue"], lw=1.5, label="$e^{\\beta} - 1$ (monthly-equivalent coefficient)"),
        Line2D([], [], color=OI["verm"], marker="o", ls="", ms=5, label="$\\beta_D = -0.80$ in the DGP"),
        Line2D([], [], color=OI["orange"], marker="o", ls="", ms=5, label="$\\beta_P = -0.22$ in the DGP")], ncol=2)
    return _save(fig, out, "fig_jensen.png")


def fig_e2(out, met):
    d = met[met.exp == "E2"]
    g = d.groupby(["scen", "model", "variant"]).mape.agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    fam = {"Prophet-D": (OI["blue"], "Prophet, daily (P-D)"), "Prophet-M": (OI["sky"], "Prophet, monthly (P-M)"),
           "SARIMAX": (OI["orange"], "SARIMAX airline, monthly"), "LightGBM-D": (OI["verm"], "LightGBM hybrid, daily"),
           "LightGBM-M": (OI["purple"], "LightGBM hybrid, monthly"), "SNaive": (OI["grey"], "seasonal naive")}
    vlab = {"vanilla": "no calendar", "cal": "COVID ignored", "mask": "mask COVID window",
            "sep": "separate COVID regressors", "binary": "Binary (shared)", "stringency": "raw stringency (shared)",
            "hesi": "HESI (shared)", "-": "seasonal naive"}
    keys = [("SNaive", "-"), ("LightGBM-M", "hesi"), ("LightGBM-D", "hesi"), ("SARIMAX", "cal"), ("SARIMAX", "hesi")]
    keys += [(m, v) for m in ("Prophet-M", "Prophet-D")
             for v in ("vanilla", "cal", "stringency", "binary", "sep", "mask", "hesi")]
    short = {"Prophet-D": "P-D", "Prophet-M": "P-M", "SARIMAX": "SARIMAX", "LightGBM-D": "LGBM-D",
             "LightGBM-M": "LGBM-M", "SNaive": "SNaive"}
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.13], hspace=0.28, wspace=0.06)
    axes = []
    for k, S in enumerate(("S1", "S2")):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        for i, (m, v) in enumerate(keys):
            r = g[(g.scen == S) & (g.model == m) & (g.variant == v)]
            ax.errorbar(r["mean"], i, xerr=r["ci"], fmt="o", color=fam[m][0], ms=3.6, lw=1.1, capsize=0)
        ax.set_xscale("log")
        ax.set_xlim(3.5, 45)
        ax.set_xticks([4, 6, 10, 15, 25, 40])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)
        ax.set_xlabel("MAPE 2023-2024 (%), mean and 95% CI")
        ax.set_title({"S1": "(a) S1: ideal transfer", "S2": "(b) S2: realistic"}[S])
        ax.set_yticks(range(len(keys)))
        if k == 0:
            ax.set_yticklabels([f"{short[m]}: {vlab[v]}" if m != "SNaive" else "Seasonal naive" for m, v in keys])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    _legend_row(fig, gs[1, :], [Line2D([], [], color=c, marker="o", ls="", ms=5, label=lab)
                                for c, lab in fam.values()], ncol=3)
    return _save(fig, out, "fig_e2.png")


def make_all(res="results", out="figures"):
    os.makedirs(out, exist_ok=True)
    met = pd.read_csv(os.path.join(res, "metrics.csv"))
    bet = pd.read_csv(os.path.join(res, "betas.csv"))
    paths = pickle.load(open(os.path.join(res, "paths_seed0.pkl"), "rb"))
    setup = pickle.load(open(os.path.join(res, "setup.pkl"), "rb"))
    probs = {}
    probs["data"] = fig_data(out, paths, setup)
    probs["drivers"] = fig_drivers(out, setup)
    probs["paths"] = fig_e1_paths(out, paths)
    probs["dist"] = fig_e1_dist(out, met)
    probs["lambda"] = fig_lambda(out, met)
    probs["beta"] = fig_beta(out, bet, setup)
    probs["jensen"] = fig_jensen(out, setup)
    probs["e2"] = fig_e2(out, met)
    return probs


if __name__ == "__main__":
    p = make_all()
    print("TOTAL problems:", sum(len(v) for v in p.values()))
