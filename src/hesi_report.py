#!/usr/bin/env python3
"""Figures, LaTeX tables and result macros for the HES paper (reads results_sim/)."""
from __future__ import annotations

import os
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import hesi_forecast as H  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "axes.titlesize": 9,
                     "legend.fontsize": 7.2, "legend.frameon": False})
C = dict(actual="#111111", off="#8c8c8c", binary="#7b3294", stringency="#a6611a", hesi="#2166ac",
         cal="#1b9e77", oracle="#4dac26", lgbm="#d7301f", sarimax="#f28e2b", pm="#74a9cf", pmac="#08306b")
MODEL_COL = {"Prophet-D HE": C["hesi"], "Prophet-M HE": C["pm"], "Prophet-M HE+AC": C["pmac"],
             "SARIMAX HE": C["sarimax"], "LightGBM-D HE": C["lgbm"], "LightGBM-M HE": "#fc9272"}
VAR_TH = {"off": "Off (ไม่ใส่โควิด)", "binary": "Binary ทั้งเดือน", "stringency": "Stringency ดิบ",
          "hesi": "HESI ($\\lambda{=}1$)", "hesi-cal": "HESI-cal", "oracle": "Oracle $\\kappa$",
          "vanilla": "Vanilla (ไม่มีปฏิทิน)", "cal": "Calendar only", "mask": "Mask ช่วงโควิด",
          "sep": "Separate COVID reg.", "-": "--"}


def _covid_shade(ax, a="2020-01-20", b="2022-09-30"):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f2e6d9", alpha=0.6, lw=0)


# ----------------------------------------------------------------------------- figures
def fig_data(out, paths, setup):
    cal = setup["cal"]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    _covid_shade(ax)
    for S, col, lab in (("S1", C["hesi"], "S1: ideal transfer ($\\kappa$=1)"),
                        ("S2", C["lgbm"], "S2: realistic ($\\kappa$=0.5 + demand shock)")):
        s = paths["monthly"][S] / 1000
        ax.plot(s.index, s.values, color=col, lw=1.1, label=lab)
    for e, (o, _, _) in H.EXPS.items():
        ax.axvline(pd.Timestamp(o), color="k", ls=":", lw=0.8)
        ax.text(pd.Timestamp(o), ax.get_ylim()[1] * 0.995, f" {e} origin", fontsize=7, va="top")
    ax.set_ylabel("Monthly exports (USD bn)")
    ax.set_title("(a) Simulated monthly exports, seed 0 (shaded: COVID window)", loc="left")
    ax.legend(loc="lower right")
    y = paths["daily_S1"]
    for k, (a, b, title, col) in enumerate(((("2019-03-25", "2019-05-05", "(b) Daily, Songkran 2019", "H")),
                                             (("2019-01-22", "2019-02-25", "(c) Daily, Chinese New Year 2019", "C")))):
        ax = fig.add_subplot(gs[1, k])
        seg = y.loc[a:b]
        flag = cal.loc[a:b, col]
        ax.bar(seg.index, seg.values, color=np.where(flag.values == 1, C["lgbm"], "#9ecae1"), width=0.85)
        ax.set_title(title, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1 if k else 2))
        ax.tick_params(axis="x", labelsize=6.8, rotation=0)
        ax.set_ylabel("USD mn / day")
    fig.text(0.5, 0.005, "red bars = stoppage days (Thai public holidays / CNY window); low bars in between = weekends",
             ha="center", fontsize=7, color="#555")
    fig.savefig(os.path.join(out, "fig_data.png"), bbox_inches="tight")
    plt.close(fig)


def fig_drivers(out, setup):
    drv, cal = setup["drv"], setup["cal"]
    mob = H.load_mobility_th()
    drop = (-mob / 100).clip(lower=0)
    hol = cal.index[cal.H.values == 1]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    ax = axs[0]
    ax.plot(drop.index, drop.values, color="#9e9e9e", lw=0.7, label="workplace mobility drop (Google)")
    hd = drop[drop.index.isin(hol)]
    ax.scatter(hd.index, hd.values, s=9, color=C["lgbm"], zorder=3, label="Thai public holidays")
    ax.axhline(setup["ref"], color=C["lgbm"], ls="--", lw=0.9)
    ax.text(pd.Timestamp("2020-02-10"), setup["ref"] + 0.025, f"median weekday-holiday drop = {setup['ref']:.2f}",
            fontsize=7, color=C["lgbm"])
    ax.set_ylim(0, 0.98)
    ax.set_title("(a) Reference: how much does a holiday stop work?  (real data, Thailand)", loc="left")
    ax.set_ylabel("drop vs baseline")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax = axs[1]
    m = drv.loc["2020-01-01":"2022-12-31"]
    for a, b in _runs(m.bin_th):
        ax.axvspan(a, b, color="#e7d4e8", lw=0)
    ax.plot(m.index, m.s_th, color=C["stringency"], lw=1, label="OxCGRT stringency/100 (real)")
    ax.plot(m.index, m.hesi_th, color=C["hesi"], lw=1, label="HESI = mobility drop / holiday drop (real)")
    ax.plot(m.index, 0.5 * m.hesi_th, color=C["oracle"], lw=0.9, ls=":", label="S2 true export dose (0.5 x HESI)")
    ax.set_title("(b) Domestic channel (Thailand); purple = 'Binary' whole-month-on months", loc="left")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=6.8)
    ax = axs[2]
    for a, b in _runs(m.bin_cn):
        ax.axvspan(a, b, color="#e7d4e8", lw=0)
    ax.plot(m.index, m.s_cn, color=C["stringency"], lw=1, label="OxCGRT China stringency/100 (real)")
    ax.plot(m.index, m.e_cn, color=C["hesi"], lw=1, label="industrial-stoppage episodes (stylised, CNY units)")
    ax.set_title("(c) Partner channel (China)", loc="left")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_drivers.png"), bbox_inches="tight")
    plt.close(fig)


def _runs(flag: pd.Series):
    f = flag.values.astype(bool)
    idx = flag.index
    out, start = [], None
    for i, v in enumerate(f):
        if v and start is None:
            start = idx[i]
        if (not v or i == len(f) - 1) and start is not None:
            out.append((start, idx[i]))
            start = None
    return out


def fig_e1_paths(out, paths):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, S in zip(axs, ("S1", "S2")):
        p = paths[("E1", S)]
        mo = p["months"]
        P = p["P"]
        ax.plot(mo, p["actual"] / 1000, color=C["actual"], lw=1.8, label="actual")
        lines = [(("Prophet-D HE", "off"), C["off"], "--", "Off"),
                 (("Prophet-D HE", "binary"), C["binary"], "-", "Binary (whole month)"),
                 (("Prophet-D HE", "stringency"), C["stringency"], "-", "Stringency"),
                 (("Prophet-D HE", "hesi"), C["hesi"], "-", "HESI ($\\lambda$=1)"),
                 (("LightGBM-D HE", "hesi"), C["lgbm"], "-.", "LightGBM-D + HESI")]
        for key, col, ls, lab in lines:
            ax.plot(mo, P[key] / 1000, color=col, ls=ls, lw=1.1, label=lab)
        cal = p["cal"]["Prophet-D HE"]
        w = mo >= H.W20_START
        ax.plot(mo[w], cal[w] / 1000, color=C["cal"], lw=1.3, label="HESI-cal (from May 2020)")
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-30"), color="#e5f5e0", lw=0)
        ax.set_title({"S1": "S1 (ideal transfer)", "S2": "S2 (realistic)"}[S] + ": forecasts made in Dec 2019", loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axs[0].set_ylabel("USD bn / month")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e1_paths.png"), bbox_inches="tight")
    plt.close(fig)


def fig_lambda(out, met):
    d = met[(met.exp == "E1") & (met.variant == "lam") & (met.window == "W24")]
    g = d.groupby(["scen", "model", "lam"]).mape.mean().reset_index()
    ref = met[(met.exp == "E1") & (met.window == "W24") & (met.model == "Prophet-D HE")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, S in zip(axs, ("S1", "S2")):
        for mdl, col in MODEL_COL.items():
            s = g[(g.scen == S) & (g.model == mdl)]
            ax.plot(s.lam, s.mape, color=col, lw=1.4 if "Prophet-D" in mdl else 1.0,
                    ls="-" if "LightGBM" not in mdl else "--", label=mdl)
        kappa = H.SCENARIOS[S]["kappa"]
        ax.axvline(kappa, color="k", ls=":", lw=0.9)
        ax.text(kappa + 0.03, 5.2, f"true $\\kappa$={kappa}", fontsize=7)
        for v, col in (("binary", C["binary"]), ("stringency", C["stringency"])):
            val = ref[(ref.scen == S) & (ref.variant == v)].mape.mean()
            ax.axhline(val, color=col, lw=0.8, ls="-.")
            ax.text(2.98, val, f"Prophet-D {v}", fontsize=6.5, color=col, ha="right", va="bottom")
        ax.set_xlabel("dose multiplier $\\lambda$ on mobility-HESI")
        ax.set_title(f"{S}: mean MAPE over 2020-2021 ({met.seed.nunique()} runs)", loc="left")
    axs[0].set_ylabel("MAPE (%)")
    axs[0].legend(loc="upper left", fontsize=6.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_lambda.png"), bbox_inches="tight")
    plt.close(fig)


def implied_monthly_beta(setup, beta, col):
    x = setup["X"]["off"][col].loc[: "2019-12-31"]
    Mx = H.month_matrix(x)
    xbar = np.nanmean(Mx, axis=1)
    eff = H.log_mean_exp(beta, Mx)
    return float(np.sum(xbar * eff) / np.sum(xbar ** 2))


def fig_beta(out, bet, setup):
    b = bet[bet.exp == "E1"]
    order = ["Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE"]
    lab = ["Prophet\ndaily", "Prophet\nmonthly", "Prophet monthly\n+ agg. correction", "SARIMAX\nmonthly"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, col, true, name in ((axs[0], "beta_D", H.TRUE["beta_D"], "domestic $\\beta_D$"),
                                (axs[1], "beta_P", H.TRUE["beta_P"], "partner $\\beta_P$")):
        data = [b[b.model == m][col].values for m in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True)
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(MODEL_COL[m])
            patch.set_alpha(0.55)
        ax.axhline(true, color="k", ls="--", lw=1)
        ax.text(4.45, true, "true daily", fontsize=7, va="bottom", ha="right")
        imp = implied_monthly_beta(setup, true, "xD" if col == "beta_D" else "xP")
        ax.axhline(imp, color="#777", ls=":", lw=1)
        ax.text(4.45, imp, "aggregation-implied\nmonthly", fontsize=6.5, va="bottom", ha="right", color="#555")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(lab, fontsize=6.8)
        ax.set_title(f"Estimated {name} (E1 fits, pre-COVID data, 30 runs)", loc="left", fontsize=8.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_beta.png"), bbox_inches="tight")
    plt.close(fig)


def fig_e2(out, met):
    d = met[(met.exp == "E2")]
    g = d.groupby(["scen", "model", "variant"]).mape.agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    keys = [("SNaive", "-"), ("SARIMAX", "cal"), ("SARIMAX", "hesi"), ("LightGBM-M", "hesi"),
            ("LightGBM-D", "hesi")] + [(m, v) for m in ("Prophet-M", "Prophet-D")
                                       for v in ("vanilla", "cal", "stringency", "binary", "sep", "mask", "hesi")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    for ax, S in zip(axs, ("S1", "S2")):
        for i, (m, v) in enumerate(keys):
            r = g[(g.scen == S) & (g.model == m) & (g.variant == v)]
            if r.empty:
                continue
            col = C["hesi"] if v == "hesi" and m.startswith("Prophet") else (C["stringency"] if v == "stringency" else "#555")
            ax.errorbar(r["mean"], i, xerr=r["ci"], fmt="o", color=col, ms=3.5, lw=1)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f"{m} · {v}" for m, v in keys], fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks([4, 6, 8, 10, 15, 20, 30])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("MAPE 2023-2024 (%), mean ± 95% CI, log scale")
        ax.set_title(f"{S}: E2 post-shock forecasts", loc="left")
        ax.grid(axis="x", color="#eee")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_e2.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def _cell(series, best=False):
    m, se = series.mean(), series.std() / np.sqrt(len(series))
    s = f"{m:.1f} ({se:.1f})"
    return f"\\textbf{{{s}}}" if best else s


def table_e1(met):
    d = met[(met.exp == "E1") & (met.variant != "lam")]
    rows = [("SNaive", "-"), ("Prophet-D", "vanilla")]
    for m in ("Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE", "LightGBM-D HE", "LightGBM-M HE"):
        for v in ("off", "binary", "stringency", "hesi", "hesi-cal", "oracle"):
            if m == "Prophet-M HE+AC" and v in ("off",):
                continue
            rows.append((m, v))
    cols = [("S1", "W24"), ("S1", "W20"), ("S2", "W24"), ("S2", "W20")]
    means = {c: {} for c in cols}
    for c in cols:
        for r in rows:
            s = d[(d.scen == c[0]) & (d.window == c[1]) & (d.model == r[0]) & (d.variant == r[1])].mape
            if len(s):
                means[c][r] = s
    best = {c: min((v.mean(), r) for r, v in means[c].items() if r[1] != "oracle")[1] for c in cols}
    lines, last = [], None
    for r in rows:
        name = r[0] if r[0] != last else ""
        if r[0] != last and last is not None:
            lines.append("\\midrule")
        last = r[0]
        cells = [(_cell(means[c][r], best[c] == r) if r in means[c] else "--") for c in cols]
        lines.append(f"{name} & {VAR_TH.get(r[1], r[1])} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_e2(met):
    d = met[met.exp == "E2"]
    rows = [("SNaive", "-")] + [(m, v) for m in ("Prophet-D", "Prophet-M")
                                for v in ("vanilla", "cal", "mask", "sep", "binary", "stringency", "hesi")]
    rows += [("SARIMAX", "cal"), ("SARIMAX", "hesi"), ("LightGBM-D", "hesi"), ("LightGBM-M", "hesi")]
    best = {}
    for S in ("S1", "S2"):
        best[S] = min((d[(d.scen == S) & (d.model == m) & (d.variant == v)].mape.mean(), (m, v)) for m, v in rows)[1]
    lines, last = [], None
    for r in rows:
        name = r[0] if r[0] != last else ""
        if r[0] != last and last is not None:
            lines.append("\\midrule")
        last = r[0]
        cells = []
        for S in ("S1", "S2"):
            s = d[(d.scen == S) & (d.model == r[0]) & (d.variant == r[1])]
            cells += [_cell(s.mape, best[S] == r), f"{s.mase.mean():.2f}"]
        lines.append(f"{name} & {VAR_TH.get(r[1], r[1])} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def table_e0(met):
    d = met[met.exp == "E0"]
    rows = [("SNaive", "-"), ("Prophet-D", "vanilla"), ("Prophet-D HE", "off"), ("Prophet-M", "vanilla"),
            ("Prophet-M HE", "off"), ("Prophet-M HE+AC", "off"), ("SARIMAX HE", "off"),
            ("LightGBM-D HE", "off"), ("LightGBM-M HE", "off")]
    best = min((d[(d.model == m) & (d.variant == v)].mape.mean(), (m, v)) for m, v in rows)[1]
    out = []
    for m, v in rows:
        s = d[(d.model == m) & (d.variant == v)]
        lab = "ปฏิทิน (holiday regressors)" if v == "off" else VAR_TH[v]
        out.append(f"{m} & {lab} & {_cell(s.mape, best == (m, v))} & {s.mase.mean():.2f} \\\\")
    return "\n".join(out)


def table_beta(bet, setup):
    b = bet[bet.exp == "E1"]
    out = []
    for m, lab in (("Prophet-D HE", "Prophet รายวัน"), ("Prophet-M HE", "Prophet รายเดือน"),
                   ("Prophet-M HE+AC", "Prophet รายเดือน + AC"), ("SARIMAX HE", "SARIMAX รายเดือน")):
        s = b[b.model == m]
        cells = []
        for col, true in (("beta_D", H.TRUE["beta_D"]), ("beta_P", H.TRUE["beta_P"])):
            v = s[col]
            rmse = np.sqrt(np.mean((v - true) ** 2))
            cells.append(f"{v.mean():.3f} & {v.std():.3f} & {rmse:.3f}")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    iD = implied_monthly_beta(setup, H.TRUE["beta_D"], "xD")
    iP = implied_monthly_beta(setup, H.TRUE["beta_P"], "xP")
    out.append("\\midrule")
    out.append(f"ค่าจริงรายวัน & {H.TRUE['beta_D']:.3f} & & & {H.TRUE['beta_P']:.3f} & & \\\\")
    out.append(f"ค่าที่คาดจากการรวมเป็นรายเดือน & {iD:.3f} & & & {iP:.3f} & & \\\\")
    return "\n".join(out), iD, iP


def table_lamhat(lam):
    out = []
    for m in ("Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE", "LightGBM-D HE", "LightGBM-M HE"):
        cells = []
        for S in ("S1", "S2"):
            v = lam[(lam.model == m) & (lam.scen == S)].lam_hat
            cells.append(f"{v.mean():.2f} & {v.std():.2f}")
        out.append(f"{m} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


# ----------------------------------------------------------------------------- macros
def macros(met, bet, lam, iD, iP):
    R = {}

    def mm(exp, S, model, var, win="W24", stat="mape"):
        s = met[(met.exp == exp) & (met.scen == S) & (met.model == model) & (met.variant == var) & (met.window == win)]
        return s[stat].mean()

    for S in ("S1", "S2"):
        s = S.lower()
        for v in ("off", "binary", "stringency", "hesi", "oracle"):
            for mdl, k in (("Prophet-D HE", "pd"), ("Prophet-M HE", "pm"), ("Prophet-M HE+AC", "pmac"),
                           ("SARIMAX HE", "sx"), ("LightGBM-D HE", "ld"), ("LightGBM-M HE", "lm")):
                for w in ("W24", "W20"):
                    val = mm("E1", S, mdl, v, w)
                    if np.isfinite(val):
                        R[f"e1.{s}.{k}.{v}.{w.lower()}"] = f"{val:.1f}"
        for mdl, k in (("Prophet-D HE", "pd"), ("Prophet-M HE", "pm"), ("Prophet-M HE+AC", "pmac"),
                       ("SARIMAX HE", "sx"), ("LightGBM-D HE", "ld")):
            R[f"e1.{s}.{k}.cal.w20"] = f"{mm('E1', S, mdl, 'hesi-cal', 'W20'):.1f}"
            v = lam[(lam.model == mdl) & (lam.scen == S)].lam_hat
            R[f"lam.{s}.{k}"] = f"{v.mean():.2f}"
            R[f"lamsd.{s}.{k}"] = f"{v.std():.2f}"
        R[f"e1.{s}.snaive.w24"] = f"{mm('E1', S, 'SNaive', '-'):.1f}"
        # wins
        a = met[(met.exp == "E1") & (met.scen == S) & (met.model == "Prophet-D HE") & (met.window == "W24")]
        piv = a.pivot_table(index="seed", columns="variant", values="mape")
        R[f"win.{s}.hesi.off"] = f"{100 * np.mean(piv['hesi'] < piv['off']):.0f}"
        R[f"win.{s}.hesi.binary"] = f"{100 * np.mean(piv['hesi'] < piv['binary']):.0f}"
        b = met[(met.exp == "E1") & (met.scen == S) & (met.window == "W24") & (met.variant == "hesi")]
        pv = b.pivot_table(index="seed", columns="model", values="mape")
        R[f"win.{s}.pd.pm"] = f"{100 * np.mean(pv['Prophet-D HE'] < pv['Prophet-M HE']):.0f}"
        R[f"win.{s}.pd.sx"] = f"{100 * np.mean(pv['Prophet-D HE'] < pv['SARIMAX HE']):.0f}"
        # lambda-curve minima
        g = met[(met.exp == "E1") & (met.scen == S) & (met.variant == "lam")].groupby(["model", "lam"]).mape.mean()
        for mdl, k in (("Prophet-D HE", "pd"), ("Prophet-M HE", "pm"), ("Prophet-M HE+AC", "pmac"), ("SARIMAX HE", "sx")):
            gg = g.loc[mdl]
            R[f"lamopt.{s}.{k}"] = f"{gg.idxmin():.2f}"
            R[f"lamoptmape.{s}.{k}"] = f"{gg.min():.1f}"
        for mdl, k in (("Prophet-D", "pd"), ("Prophet-M", "pm"), ("SARIMAX", "sx"), ("SNaive", "sn"),
                       ("LightGBM-D", "ld"), ("LightGBM-M", "lm")):
            for v in ("vanilla", "cal", "mask", "sep", "binary", "stringency", "hesi", "-"):
                val = mm("E2", S, mdl, v)
                if np.isfinite(val):
                    R[f"e2.{s}.{k}.{v if v != '-' else 'x'}"] = f"{val:.1f}"
    for mdl, k in (("SNaive", "sn"), ("Prophet-D", "pdv"), ("Prophet-D HE", "pd"), ("Prophet-M", "pmv"),
                   ("Prophet-M HE", "pm"), ("Prophet-M HE+AC", "pmac"), ("SARIMAX HE", "sx"),
                   ("LightGBM-D HE", "ld"), ("LightGBM-M HE", "lm")):
        v = "-" if mdl == "SNaive" else ("vanilla" if mdl in ("Prophet-D", "Prophet-M") else "off")
        R[f"e0.{k}"] = f"{mm('E0', 'S1', mdl, v):.1f}"
    b = bet[bet.exp == "E1"]
    for mdl, k in (("Prophet-D HE", "pd"), ("Prophet-M HE", "pm"), ("Prophet-M HE+AC", "pmac"), ("SARIMAX HE", "sx")):
        R[f"bD.{k}"] = f"{b[b.model == mdl].beta_D.mean():.2f}"
        R[f"bDsd.{k}"] = f"{b[b.model == mdl].beta_D.std():.2f}"
        R[f"bP.{k}"] = f"{b[b.model == mdl].beta_P.mean():.2f}"
        R[f"bPsd.{k}"] = f"{b[b.model == mdl].beta_P.std():.2f}"
    R["bD.true"] = f"{H.TRUE['beta_D']:.2f}"
    R["bP.true"] = f"{H.TRUE['beta_P']:.2f}"
    R["bD.implied"] = f"{iD:.2f}"
    R["bP.implied"] = f"{iP:.2f}"
    R["nseeds"] = str(met.seed.nunique())
    e2 = bet[(bet.exp == "E2") & (bet.model == "Prophet-D")]
    for S in ("S1", "S2"):
        for v in ("cal", "binary", "stringency", "hesi", "sep", "mask"):
            R[f"e2b.{S.lower()}.{v}"] = f"{e2[(e2.scen == S) & (e2.variant == v)].beta_D.mean():.2f}"
    return R


def make_all(out="results_sim", fig_dir=None):
    fig_dir = fig_dir or out
    met = pd.read_csv(os.path.join(out, "metrics.csv"))
    bet = pd.read_csv(os.path.join(out, "betas.csv"))
    lam = pd.read_csv(os.path.join(out, "lamhat.csv"))
    with open(os.path.join(out, "paths_seed0.pkl"), "rb") as fh:
        paths = pickle.load(fh)
    with open(os.path.join(out, "setup.pkl"), "rb") as fh:
        setup = pickle.load(fh)
    fig_data(fig_dir, paths, setup)
    fig_drivers(fig_dir, setup)
    fig_e1_paths(fig_dir, paths)
    fig_lambda(fig_dir, met)
    fig_beta(fig_dir, bet, setup)
    fig_e2(fig_dir, met)
    tb, iD, iP = table_beta(bet, setup)
    tabs = {"tab_e1.tex": table_e1(met), "tab_e2.tex": table_e2(met), "tab_e0.tex": table_e0(met),
            "tab_beta.tex": tb, "tab_lamhat.tex": table_lamhat(lam)}
    for k, v in tabs.items():
        with open(os.path.join(fig_dir, k), "w") as fh:
            fh.write(v + "\n")
    R = macros(met, bet, lam, iD, iP)
    with open(os.path.join(fig_dir, "results_macros.tex"), "w") as fh:
        for k, v in R.items():
            fh.write(f"\\expandafter\\def\\csname R@{k}\\endcsname{{{v}}}\n")
    return R


if __name__ == "__main__":
    import sys
    R = make_all(*(sys.argv[1:] or ["results_sim"]))
    for k, v in R.items():
        print(k, v)
