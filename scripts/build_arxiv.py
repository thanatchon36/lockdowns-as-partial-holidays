#!/usr/bin/env python3
"""Build paper/arxiv/main.pdf: tables, paired tests, theory checks, macros, pdflatex.
Run from the repository root with PYTHONPATH=src (the Makefile does this)."""
import os
import pickle
import re
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

import hesi_forecast as H
import hesi_report as HR

RES, OUT = "results", "paper/arxiv"
met = pd.read_csv(f"{RES}/metrics.csv")
bet = pd.read_csv(f"{RES}/betas.csv")
lam = pd.read_csv(f"{RES}/lamhat.csv")
rob = pd.read_csv(f"{RES}/robust_metrics.csv")
rbet = pd.read_csv(f"{RES}/robust_betas.csv")
setup = pickle.load(open(f"{RES}/setup.pkl", "rb"))

_, iD, iP = HR.table_beta(bet, setup)
R = HR.macros(met, bet, lam, iD, iP)
R["mobref"] = f"{setup['ref']:.2f}"

MODEL_EN = {"SNaive": "Seasonal naive", "Prophet-D": "Prophet daily", "Prophet-D HE": "Prophet daily (P-D)",
            "Prophet-M HE": "Prophet monthly (P-M)", "Prophet-M HE+AC": "P-M + AC", "SARIMAX HE": "SARIMAX",
            "LightGBM-D HE": "LightGBM daily", "LightGBM-M HE": "LightGBM monthly", "Prophet-M": "Prophet monthly",
            "SARIMAX": "SARIMAX", "LightGBM-D": "LightGBM daily", "LightGBM-M": "LightGBM monthly"}
VAR_EN = {"off": "Off (no dose)", "binary": "Binary whole-month switch", "stringency": "Raw stringency",
          "hesi": r"HESI ($\lambda=1$)", "hesi-cal": r"HESI-cal ($\hat\lambda$)", "oracle": r"Oracle ($\kappa$ known)",
          "vanilla": "No calendar regressors", "cal": "COVID ignored", "mask": "Mask COVID window",
          "sep": "Separate COVID regressors", "-": "--"}
ABBR = {"Prophet-D HE": "P-D", "Prophet-M HE": "P-M", "Prophet-M HE+AC": "P-M+AC", "SARIMAX HE": "SARIMAX",
        "LightGBM-D HE": "LightGBM-D", "LightGBM-M HE": "LightGBM-M", "Prophet-D": "P-D", "SARIMAX": "SARIMAX"}


def cell(s, best=False, d=1):
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
    txt = f"{m:.{d}f} ({se:.{d}f})"
    return rf"\textbf{{{txt}}}" if best else txt


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def table_e1(stat="mape"):
    d = met[(met.exp == "E1") & (met.variant != "lam")]
    rows = [("SNaive", "-"), ("Prophet-D", "vanilla")]
    for m in ("Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE", "LightGBM-D HE", "LightGBM-M HE"):
        for v in ("off", "binary", "stringency", "hesi", "hesi-cal", "oracle"):
            if not (m == "Prophet-M HE+AC" and v == "off"):
                rows.append((m, v))
    cols = [("S1", "W24"), ("S1", "W20"), ("S2", "W24"), ("S2", "W20")]
    vals = {c: {r: sel(d, scen=c[0], window=c[1], model=r[0], variant=r[1])[stat] for r in rows} for c in cols}
    best = {c: min((v.mean(), r) for r, v in vals[c].items() if len(v) and r[1] != "oracle")[1] for c in cols}
    out, last = [], None
    for r in rows:
        if last is not None and r[0] != last:
            out.append(r"\addlinespace")
        name = MODEL_EN[r[0]] if r[0] != last else ""
        last = r[0]
        cells = [cell(vals[c][r], best[c] == r, 2 if stat == "mase" else 1) if len(vals[c][r]) else "--" for c in cols]
        out.append(f"{name} & {VAR_EN[r[1]]} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def table_e2():
    d = met[met.exp == "E2"]
    rows = [("SNaive", "-")] + [(m, v) for m in ("Prophet-D", "Prophet-M")
                                for v in ("vanilla", "cal", "mask", "sep", "binary", "stringency", "hesi")]
    rows += [("SARIMAX", "cal"), ("SARIMAX", "hesi"), ("LightGBM-D", "hesi"), ("LightGBM-M", "hesi")]
    best = {S: min((sel(d, scen=S, model=m, variant=v).mape.mean(), (m, v)) for m, v in rows)[1] for S in ("S1", "S2")}
    lab = dict(VAR_EN, hesi="HESI (shared)", binary="Binary (shared)", stringency="Raw stringency (shared)")
    out, last = [], None
    for r in rows:
        if last is not None and r[0] != last:
            out.append(r"\addlinespace")
        name = MODEL_EN[r[0]] if r[0] != last else ""
        last = r[0]
        cells = []
        for S in ("S1", "S2"):
            s = sel(d, scen=S, model=r[0], variant=r[1])
            cells += [cell(s.mape, best[S] == r), f"{s.mase.mean():.2f}"]
        out.append(f"{name} & {lab[r[1]]} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def table_e0():
    d = met[met.exp == "E0"]
    rows = [("SNaive", "-", "--"), ("Prophet-D", "vanilla", "no calendar regressors"),
            ("Prophet-D HE", "off", "holiday regressors"), ("Prophet-M", "vanilla", "no calendar regressors"),
            ("Prophet-M HE", "off", "holiday regressors"), ("Prophet-M HE+AC", "off", "holiday regressors + AC"),
            ("SARIMAX HE", "off", "holiday regressors"), ("LightGBM-D HE", "off", "holiday features"),
            ("LightGBM-M HE", "off", "holiday features")]
    best = min((sel(d, model=m, variant=v).mape.mean(), (m, v)) for m, v, _ in rows)[1]
    out = []
    for m, v, lab in rows:
        s = sel(d, model=m, variant=v)
        out.append(f"{MODEL_EN[m]} & {lab} & {cell(s.mape, best == (m, v))} & {s.mase.mean():.2f} \\\\")
    return "\n".join(out)


def table_lamhat(lamstar):
    out = []
    for m in ("Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE", "LightGBM-D HE", "LightGBM-M HE"):
        cells = []
        for S in ("S1", "S2"):
            v = sel(lam, model=m, scen=S).lam_hat
            cells.append(f"{v.mean():.2f} & {v.std():.2f}")
        out.append(f"{MODEL_EN[m]} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    out.append(f"Prop.~\\ref{{prop:transfer}}(b) optimum for P-D & 1.00 & -- & {lamstar:.2f} & -- \\\\")
    return "\n".join(out)


def table_beta():
    b = bet[bet.exp == "E1"]
    out = []
    for m in ("Prophet-D HE", "Prophet-M HE", "Prophet-M HE+AC", "SARIMAX HE"):
        s = b[b.model == m]
        cells = []
        for col, true in (("beta_D", H.TRUE["beta_D"]), ("beta_P", H.TRUE["beta_P"])):
            v = s[col]
            cells.append(f"${v.mean():.3f}$ & {v.std():.3f} & {np.sqrt(np.mean((v - true) ** 2)):.3f}")
        out.append(f"{MODEL_EN[m]} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    bD, bP = H.TRUE["beta_D"], H.TRUE["beta_P"]
    out.append(f"True daily value & ${bD:.3f}$ & & & ${bP:.3f}$ & & \\\\")
    out.append(f"Exact aggregation (Prop.~\\ref{{prop:jensen}}) & ${iD:.3f}$ & & & ${iP:.3f}$ & & \\\\")
    out.append(f"Closed form $e^{{\\beta}}-1$ & ${np.exp(bD) - 1:.3f}$ & & & ${np.exp(bP) - 1:.3f}$ & & \\\\")
    pm = b[b.model == "Prophet-M HE"]
    acD, acP = np.log1p(pm.beta_D.mean()), np.log1p(pm.beta_P.mean())
    sdD, sdP = pm.beta_D.std() / (1 + pm.beta_D.mean()), pm.beta_P.std() / (1 + pm.beta_P.mean())
    out.append(f"AC closed form, Cor.~\\ref{{cor:ac}} (delta s.d.) & ${acD:.3f}$ & {sdD:.3f} & & ${acP:.3f}$ & {sdP:.3f} & \\\\")
    R["ac.closed"], R["ac.deltasd"] = f"{acD:.2f}", f"{sdD:.2f}"
    return "\n".join(out)


def table_robust():
    out = [r"\multicolumn{4}{l}{\textit{Panel A. R1, heterogeneous holidays: MAPE (\%)}} \\"]
    for mdl, lab in (("one-tier", "One-tier (pooled holidays)"), ("two-tier", "Two-tier (festival indicator)")):
        for k, (v, vl) in enumerate((("off", VAR_EN["off"]), ("hesi", VAR_EN["hesi"]), ("oracle", VAR_EN["oracle"]))):
            cells = [cell(sel(rob, exp="R1", model=mdl, variant=v, scen=S).mape) for S in ("S1", "S2")]
            out.append(f"{lab if k == 0 else ''} & {vl} & " + " & ".join(cells) + r" \\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{4}{l}{\textit{Panel B. R1 elasticities: mean (s.d.) across replications}} \\")
    one, two = rbet[rbet.model == "one-tier"], rbet[rbet.model == "two-tier"]
    out.append(f"One-tier pooled $\\hat\\beta$ & & \\multicolumn{{2}}{{c}}{{${one.beta_D.mean():.3f}$ ({one.beta_D.std():.3f})}} \\\\")
    out.append(f"Two-tier ordinary $\\hat\\beta_O$ (true ${np.log(0.55):.3f}$) & & \\multicolumn{{2}}{{c}}{{${two.beta_D.mean():.3f}$ ({two.beta_D.std():.3f})}} \\\\")
    out.append(f"Two-tier festival $\\hat\\beta_F$ (true ${np.log(0.35):.3f}$) & & \\multicolumn{{2}}{{c}}{{${two.beta_F.mean():.3f}$ ({two.beta_F.std():.3f})}} \\\\")
    out.append(r"\addlinespace")
    out.append(r"\multicolumn{4}{l}{\textit{Panel C. R2, real-time dose: MAPE (\%)}} \\")
    for v, lab, vl in (("hesi-centred", "Centred smoothing", VAR_EN["hesi"]), ("hesi-trailing", "Trailing smoothing", VAR_EN["hesi"]),
                       ("oracle-centred", "Centred smoothing", VAR_EN["oracle"]), ("oracle-trailing", "Trailing smoothing", VAR_EN["oracle"])):
        cells = [cell(sel(rob, exp="R2", variant=v, scen=S).mape) for S in ("S1", "S2")]
        out.append(f"{lab} & {vl} & " + " & ".join(cells) + r" \\")
    for S in ("S1", "S2"):
        s = S.lower()
        for mdl, k in (("one-tier", "one"), ("two-tier", "two")):
            for v in ("off", "hesi", "oracle"):
                R[f"r1.{s}.{k}.{v}"] = f"{sel(rob, exp='R1', model=mdl, variant=v, scen=S).mape.mean():.1f}"
        for v, k in (("hesi-centred", "hc"), ("hesi-trailing", "ht"), ("oracle-centred", "oc"), ("oracle-trailing", "ot")):
            R[f"r2.{s}.{k}"] = f"{sel(rob, exp='R2', variant=v, scen=S).mape.mean():.1f}"
    R["r1.b1"], R["r1.bO"], R["r1.bF"] = (f"{one.beta_D.mean():.2f}", f"{two.beta_D.mean():.2f}", f"{two.beta_F.mean():.2f}")
    R["r1.trueO"], R["r1.trueF"] = f"{np.log(0.55):.2f}", f"{np.log(0.35):.2f}"
    R["r1.mult"] = f"{one.beta_D.mean() / np.log(0.55):.2f}"
    return "\n".join(out)


PAIRS = [("E1", "W24", ("Prophet-D HE", "hesi"), ("Prophet-D HE", "off")),
         ("E1", "W24", ("Prophet-D HE", "hesi"), ("Prophet-D HE", "binary")),
         ("E1", "W24", ("Prophet-D HE", "hesi"), ("Prophet-D HE", "stringency")),
         ("E1", "W20", ("Prophet-D HE", "hesi-cal"), ("Prophet-D HE", "off")),
         ("E1", "W20", ("Prophet-D HE", "hesi-cal"), ("Prophet-D HE", "hesi")),
         ("E1", "W24", ("Prophet-D HE", "hesi"), ("Prophet-M HE", "hesi")),
         ("E1", "W24", ("Prophet-D HE", "hesi"), ("SARIMAX HE", "hesi")),
         ("E1", "W24", ("Prophet-D HE", "hesi"), ("LightGBM-D HE", "hesi")),
         ("E1", "W24", ("Prophet-M HE+AC", "hesi"), ("Prophet-M HE", "hesi")),
         ("E2", "W24", ("Prophet-D", "mask"), ("Prophet-D", "cal")),
         ("E2", "W24", ("Prophet-D", "hesi"), ("Prophet-D", "cal")),
         ("E2", "W24", ("SARIMAX", "hesi"), ("Prophet-D", "mask")),
         ("E0", "W24", ("LightGBM-D HE", "off"), ("Prophet-D HE", "off"))]
WIN = {("E1", "W24"): "E1, Jan 20--Dec 21", ("E1", "W20"): "E1, May 20--Dec 21",
       ("E2", "W24"): "E2, 2023--24", ("E0", "W24"): "E0, 2018--19"}
VSHORT = {"hesi": "HESI", "off": "Off", "binary": "Binary", "stringency": "stringency", "hesi-cal": "HESI-cal",
          "mask": "mask", "cal": "ignored", "-": ""}


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def table_paired():
    recs = []
    for ci, (exp, win, A, B) in enumerate(PAIRS, 1):
        for S in (("S1",) if exp == "E0" else ("S1", "S2")):
            a = sel(met, exp=exp, scen=S, model=A[0], variant=A[1], window=win).set_index("seed").mape
            b = sel(met, exp=exp, scen=S, model=B[0], variant=B[1], window=win).set_index("seed").mape
            d = (a - b).dropna()
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            recs.append(dict(exp=exp, win=win, A=A, B=B, S=S, mean=d.mean(), lo=d.mean() - t * se,
                             hi=d.mean() + t * se, win_rate=100 * np.mean(d < 0), p=p, ci=ci))
    ps = np.array([r["p"] for r in recs])
    order = np.argsort(ps)
    m = len(ps)
    holm = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        holm[i] = running
    out, last = [], None
    for r, hp in zip(recs, holm):
        R[f"ph.c{r['ci']}.{r['S'].lower()}"] = fmt_p(hp)
        R[f"pw.c{r['ci']}.{r['S'].lower()}"] = f"{r['win_rate']:.0f}"
        key = (r["exp"], r["win"], r["A"], r["B"])
        if last is not None and key != last:
            out.append(r"\addlinespace[1pt]")
        first = key != last
        last = key
        comp = f"{ABBR[r['A'][0]]} {VSHORT[r['A'][1]]} vs.\\ {ABBR[r['B'][0]]} {VSHORT[r['B'][1]]}"
        out.append(f"{WIN[(r['exp'], r['win'])] if first else ''} & {comp if first else ''} & {r['S']} & "
                   f"${r['mean']:.1f}$ [${r['lo']:.1f}$, ${r['hi']:.1f}$] & {r['win_rate']:.0f}\\% & "
                   f"{fmt_p(r['p'])} & {fmt_p(hp)} \\\\")
    return "\n".join(out)


def theory_numbers():
    cal, drv = setup["cal"], setup["drv"]
    R["bM.closed"] = f"{np.exp(H.TRUE['beta_D']) - 1:.2f}"
    R["bPM.closed"] = f"{np.exp(H.TRUE['beta_P']) - 1:.2f}"
    idx = cal.index
    mask = (idx >= "2020-03-01") & (idx <= "2020-04-30") & (cal.H.values == 0)
    h = np.clip(drv.hesi_th.values[mask], 0, 1)
    D = H.piecewise(idx[mask], H.DEMAND)
    lamstar = 0.5 + np.sum(h * D) / (H.TRUE["beta_D"] * np.sum(h * h))
    R["lamstar.cal"] = f"{lamstar:.2f}"
    x = setup["X"]["off"].xD.loc[:"2019-12-31"]
    xbar = x.groupby(x.index.to_period("M")).mean()
    mon = xbar.index.month
    resid = xbar - xbar.groupby(mon).transform("mean")
    t = np.arange(len(resid), dtype=float)
    resid = resid - np.polyval(np.polyfit(t, resid.values, 1), t)
    R["xraw.sd"], R["xres.sd"] = f"{xbar.std():.3f}", f"{resid.std():.3f}"
    R["xres.days"] = f"{resid.std() * 30.4:.1f}"
    key = pd.MultiIndex.from_arrays([x.index.month, x.index.dayofweek])
    xd_res = x.values - x.groupby(key).transform("mean").values
    R["xdres.sd"] = f"{np.std(xd_res):.3f}"
    return lamstar


def build():
    lamstar = theory_numbers()
    _paired = table_paired()
    tabs = {"TAB_E1": table_e1("mape"), "TAB_E1MASE": table_e1("mase"), "TAB_E2": table_e2(),
            "TAB_E0": table_e0(), "TAB_LAMHAT": table_lamhat(lamstar), "TAB_BETA": table_beta(),
            "TAB_ROBUST": table_robust(), "TAB_PAIRED": _paired}
    tpl = open(f"{OUT}/main_template.tex", encoding="utf-8").read()
    tpl = tpl.replace(r"statsmodels 0.15.0 \citep{box2015time}", r"statsmodels 0.15.0 \citep{seabold2010statsmodels}")
    tpl = tpl.replace(r"\bibitem[Silvestrini and Veredas(2008)]",
                      r"\bibitem[Seabold and Perktold(2010)]{seabold2010statsmodels} Seabold, S., \& Perktold, J. (2010). "
                      r"Statsmodels: Econometric and statistical modeling with Python. In \emph{Proceedings of the 9th Python "
                      r"in Science Conference} (pp.\ 92--96)." "\n" r"\bibitem[Silvestrini and Veredas(2008)]")
    for k, v in tabs.items():
        tpl = tpl.replace(f"%%{k}%%", v)
    missing = sorted(set(k for k in re.findall(r"<<([^<>]+)>>", tpl) if k not in R))
    assert not missing, missing
    tpl = re.sub(r"<<([^<>]+)>>", lambda m: R[m.group(1)], tpl)
    open(f"{OUT}/main.tex", "w", encoding="utf-8").write(tpl)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT,
                           capture_output=True, text=True)
    log = open(f"{OUT}/main.log", encoding="utf-8", errors="ignore").read()
    print("pdflatex rc", r.returncode)
    for line in log.splitlines():
        if line.startswith("!") or "undefined" in line.lower() or "Overfull \\hbox" in line:
            print("  ", line[:160])


if __name__ == "__main__":
    build()
    print({k: R[k] for k in ("lamstar.cal", "ac.closed", "ac.deltasd", "xraw.sd", "xres.sd", "xres.days",
                             "xdres.sd", "r1.mult", "bM.closed", "bPM.closed")})
