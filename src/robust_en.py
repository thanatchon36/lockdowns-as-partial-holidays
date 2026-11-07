#!/usr/bin/env python3
"""Robustness experiments for the HES paper (E1 shock period, Prophet daily).

R1  Heterogeneous holidays: festival days (Songkran, New Year) have a larger effect than
    ordinary public holidays; the COVID dose is measured in ordinary-holiday units.
    Compare a one-tier shared regressor (all holidays) with a two-tier specification
    (separate festival indicator + shared ordinary-holiday regressor carrying the dose).
R2  Real-time HESI: trailing (one-sided) 7-day smoothing instead of the centred window.

Usage: python robust_en.py SEED0 NREPS OUTDIR
"""
import os
import sys
import time

import holidays
import numpy as np
import pandas as pd

import hesi_forecast as H

BETA_F, BETA_O = float(np.log(0.35)), float(np.log(0.55))
FEST_KEYS = ("Songkran", "New Year")


def festival_flags(cal):
    th = holidays.Thailand(years=range(H.START.year, H.END.year + 1), language="en_US")
    days = {pd.Timestamp(d) for d, n in th.items() if any(k in n for k in FEST_KEYS)}
    return (cal.index.isin(pd.DatetimeIndex(sorted(days))) & (cal.H.values == 1)).astype(float)


def trailing_hesi(cal):
    mob = H.load_mobility_th()
    hol = cal.index[cal.H.values == 1]
    drop = (-mob / 100.0).clip(lower=0)
    is_hol = drop.index.isin(hol)
    ref = float(drop[is_hol & (drop.index.dayofweek < 5)].median())
    sm = drop.where(~is_hol).rolling(7, center=False, min_periods=3).mean()
    return (sm / ref).clip(0, 1).fillna(0.0).reindex(cal.index).fillna(0.0).values


def simulate_r1(cal, drv, F, scenario, seed, P=H.TRUE):
    sc, idx, n = H.SCENARIOS[scenario], cal.index, len(cal)
    rng = np.random.default_rng(seed)
    a = H._ar1(rng, n, P["ar_rho"], P["ar_sd"])
    per = idx.to_period("M")
    up = per.unique()
    b = pd.Series(H._ar1(rng, len(up), P["m_rho"], P["m_sd"]), index=up).reindex(per).values
    eps = rng.normal(0, P["eps_sd"], n)
    nu_sd = P["nu_sd"] * np.sqrt(1 - P["nu_rho"] ** 2)
    nuD, nuP = H._ar1(rng, n, P["nu_rho"], nu_sd), H._ar1(rng, n, P["nu_rho"], nu_sd)
    zD = np.clip(sc["kappa"] * drv.hesi_th.values * (1 + nuD), 0, 1)
    zP = np.clip(drv.e_cn.values * (1 + nuP), 0, 1)
    Hh, C = cal.H.values, cal.C.values
    O = Hh * (1 - F)
    xO, xP = O + (1 - Hh) * zD, C + (1 - C) * zP
    dem = H.piecewise(idx, H.DEMAND) if sc["demand"] else 0.0
    logy = (H.log_trend(idx, P["level0"]) + H.yearly(idx) + P["dow"][cal.dow.values]
            + BETA_F * F + BETA_O * xO + P["beta_P"] * xP + dem + a + b + eps)
    return pd.Series(np.exp(logy), index=idx)


def main(seed0, reps, out):
    os.makedirs(out, exist_ok=True)
    cal = H.build_calendar()
    drv = H.build_drivers(cal)
    F = festival_flags(cal)
    Hh, C = cal.H.values, cal.C.values
    O = Hh * (1 - F)
    h_c, h_t, e_cn = drv.hesi_th.values, trailing_hesi(cal), drv.e_cn.values
    o, a, b = (pd.Timestamp(x) for x in H.EXPS["E1"])
    d = cal.index
    dtr, dte = d[d <= o], d[(d >= a) & (d <= b)]
    tr, te = d <= o, (d >= a) & (d <= b)
    xP = C + (1 - C) * e_cn

    def dose(z):
        return np.clip(z, 0, 1)

    t0 = time.time()
    for seed in range(seed0, seed0 + reps):
        rows, bets = [], []
        # ---- R1: heterogeneous holidays
        ys = {S: simulate_r1(cal, drv, F, S, seed) for S in H.SCENARIOS}
        ly = np.log(ys["S1"].values[tr])
        X1 = pd.DataFrame({"xD": Hh, "xP": xP * 0 + C}, index=d)                 # training: holidays only
        X2 = pd.DataFrame({"xO": O, "xF": F, "xP": C}, index=d)
        m1 = H.ProphetLog("D", ["xD", "xP"]).fit(dtr, ly, X1.loc[dtr])
        m2 = H.ProphetLog("D", ["xO", "xF", "xP"]).fit(dtr, ly, X2.loc[dtr])
        b1, b2 = m1.base0(dte), m2.base0(dte)
        bets.append(dict(seed=seed, exp="R1", model="one-tier", beta_D=m1.coef["xD"], beta_F=np.nan,
                         beta_P=m1.coef["xP"]))
        bets.append(dict(seed=seed, exp="R1", model="two-tier", beta_D=m2.coef["xO"], beta_F=m2.coef["xF"],
                         beta_P=m2.coef["xP"]))
        for S, sc in H.SCENARIOS.items():
            act = ys[S].loc[dte].groupby(dte.to_period("M")).sum().values
            for v, z in (("off", 0 * h_c), ("hesi", h_c), ("oracle", sc["kappa"] * h_c)):
                zz = dose(z)
                f1 = H._d2m(m1.predict_log(b1, pd.DataFrame({"xD": Hh + (1 - Hh) * zz, "xP": xP}, index=d).loc[dte]),
                            m1.smear, dte)
                f2 = H._d2m(m2.predict_log(b2, pd.DataFrame({"xO": O + (1 - Hh) * zz, "xF": F, "xP": xP},
                                                            index=d).loc[dte]), m2.smear, dte)
                for mdl, f in (("one-tier", f1), ("two-tier", f2)):
                    rows.append(dict(seed=seed, exp="R1", scen=S, model=mdl, variant=v,
                                     mape=100 * np.mean(np.abs(act - f) / act)))
        # ---- R2: real-time (trailing) HESI on the baseline DGP
        ys = {S: H.simulate(cal, drv, S, seed) for S in H.SCENARIOS}
        ly = np.log(ys["S1"].values[tr])
        Xb = pd.DataFrame({"xD": Hh, "xP": C}, index=d)
        m = H.ProphetLog("D", ["xD", "xP"]).fit(dtr, ly, Xb.loc[dtr])
        b0 = m.base0(dte)
        for S, sc in H.SCENARIOS.items():
            act = ys[S].loc[dte].groupby(dte.to_period("M")).sum().values
            for v, z in (("hesi-centred", h_c), ("hesi-trailing", h_t),
                         ("oracle-centred", sc["kappa"] * h_c), ("oracle-trailing", sc["kappa"] * h_t)):
                zz = dose(z)
                f = H._d2m(m.predict_log(b0, pd.DataFrame({"xD": Hh + (1 - Hh) * zz, "xP": xP}, index=d).loc[dte]),
                           m.smear, dte)
                rows.append(dict(seed=seed, exp="R2", scen=S, model="Prophet-D HE", variant=v,
                                 mape=100 * np.mean(np.abs(act - f) / act)))
        for name, rec in (("robust_metrics", rows), ("robust_betas", bets)):
            fn = os.path.join(out, f"{name}.csv")
            pd.DataFrame(rec).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
        print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
