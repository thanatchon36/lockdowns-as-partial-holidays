#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hesi.forecast — Holiday-Equivalent Stoppage (HES) forecasting toolkit
=====================================================================
Research code for the paper
  "Lockdowns as Partial Holidays: Transferring Work-Stoppage Effects Learned from
   Songkran and Chinese New Year to Forecast Thailand's Monthly Exports through COVID-19"

Core idea
---------
One *shared* stoppage regressor per channel is learned from recurring holidays and is
re-used for pandemic periods, scaled by a Holiday-Equivalent Stoppage Index (HESI, 0..1):

    xD_d = H_d + (1 - H_d) * lambda * HESI_d     domestic channel (Thai public holidays)
    xP_d = C_d + (1 - C_d) * PHESI_d             partner channel  (Chinese-New-Year window)

    log y_d = trend + seasonality + beta_D * xD_d + beta_P * xP_d + noise

Holidays recur every year, so beta_D and beta_P are identified *before* COVID; the shock
period only needs a dose (HESI), not new parameters.

Commands
--------
  hesi simulate --reps 30 --out results                  # Monte-Carlo study
  hesi figures                                          # publication figures -> paper/figures/
  hesi real --exports th_exports.csv --out results_real  # official export data
(`hesi` is installed by `make setup`; `python -m hesi.cli` is equivalent.)

Real-data CSV: first column = month (YYYY-MM), second column = value (e.g. USD million).
External drivers are downloaded automatically (OxCGRT, Google Mobility mirror) into data/raw/.

Requires: numpy pandas prophet lightgbm statsmodels holidays matplotlib
"""
from __future__ import annotations

import logging
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _name in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(_name).disabled = True

START, END = pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")
DATA_DIR = os.environ.get("HESI_DATA", os.path.join("data", "raw"))
OXCGRT_URL = ("https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/"
              "data/timeseries_indices/OxCGRT_timeseries_StringencyIndex_v1.csv")
GMOB_URL = ("https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/"
            "google_reports/mobility_report_countries.csv")

# =============================================================================
# 1. CALENDAR  (observed Thai holidays incl. in-lieu/special days; CNY window)
# =============================================================================
NOT_DAY_OFF = {"National Children's Day"}          # observance, not a day off


def thai_holidays(start=START, end=END) -> pd.DatetimeIndex:
    """Observed Thai public holidays. Uses the *observed* calendar, e.g. Songkran 2020
    was postponed and replaced by 27 Jul, 4 Sep and 7 Sep 2020."""
    import holidays
    th = holidays.Thailand(years=range(start.year, end.year + 1), language="en_US")
    idx = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d, n in th.items() if n not in NOT_DAY_OFF}))
    return idx[(idx >= start) & (idx <= end)]


def cny_window(start=START, end=END, lower=-1, upper=6) -> pd.DatetimeIndex:
    """Chinese-New-Year window [day1+lower, day1+upper] (partner-side closures)."""
    import holidays
    cn = holidays.China(years=range(start.year, end.year + 1), language="en_US")
    first = {}
    for d, n in sorted(cn.items()):
        if n.startswith("Chinese New Year (Spring Festival)") and d.year not in first:
            first[d.year] = pd.Timestamp(d)
    days = {d + pd.Timedelta(days=k) for d in first.values() for k in range(lower, upper + 1)}
    idx = pd.DatetimeIndex(sorted(days))
    return idx[(idx >= start) & (idx <= end)]


def build_calendar(start=START, end=END) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame(index=idx)
    cal["H"] = idx.isin(thai_holidays(start, end)).astype(float)
    cal["C"] = idx.isin(cny_window(start, end)).astype(float)
    cal["dow"] = idx.dayofweek
    cal["doy"] = idx.dayofyear
    cal["weekend"] = (idx.dayofweek >= 5).astype(float)
    return cal


# =============================================================================
# 2. COVID DRIVERS  (real OxCGRT + Google Mobility; stylised China episodes)
# =============================================================================
def _fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import urllib.request
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_oxcgrt(country: str) -> pd.Series:
    """Daily national OxCGRT Stringency Index rescaled to 0..1."""
    df = pd.read_csv(_fetch(OXCGRT_URL, os.path.join(DATA_DIR, "oxcgrt_si.csv")))
    row = df[(df.CountryCode == country) & (df.RegionCode.isna())].iloc[0, 7:]
    s = pd.Series(row.values.astype(float), index=pd.to_datetime(row.index, format="%d%b%Y"))
    s = s.loc[: s.last_valid_index()].ffill()
    return (s / 100.0).rename(f"s_{country}")


def load_mobility_th() -> pd.Series:
    """Google COVID-19 Community Mobility Reports, Thailand, 'workplaces' (% vs baseline)."""
    path = os.path.join(DATA_DIR, "gmob_thailand.csv")
    if not os.path.exists(path):
        raw = pd.read_csv(_fetch(GMOB_URL, os.path.join(DATA_DIR, "gmob_countries.csv")), low_memory=False)
        raw[(raw.country == "Thailand") & (raw.region == "Total")].to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["workplaces"].astype(float)


def mobility_hesi(mob: pd.Series, hol: pd.DatetimeIndex, window: int = 7):
    """HESI_d = workplace-mobility drop on day d / median drop on weekday public holidays.
    Holiday days are removed before smoothing so holiday dips do not leak into neighbours."""
    drop = (-mob / 100.0).clip(lower=0)
    is_hol = drop.index.isin(hol)
    ref = float(drop[is_hol & (drop.index.dayofweek < 5)].median())
    smooth = drop.where(~is_hol).rolling(window, center=True, min_periods=3).mean()
    return (smooth / ref).clip(0, 1).fillna(0.0).rename("hesi_th"), ref


# Stylised Chinese industrial-stoppage episodes in CNY-equivalent units. OxCGRT's China index
# stays ~0.6-0.8 for three years under zero-COVID while factories mostly ran, so it is NOT a
# stoppage measure; we use episodes instead (Wuhan/national lockdown 2020, Shanghai 2022).
CN_EPISODES = [("2010-01-01", 0.0), ("2020-01-22", 0.0), ("2020-01-25", 0.8), ("2020-02-20", 0.8),
               ("2020-03-20", 0.3), ("2020-04-20", 0.0), ("2022-03-25", 0.0), ("2022-04-01", 0.5),
               ("2022-05-25", 0.5), ("2022-06-10", 0.0), ("2030-12-31", 0.0)]
BIN_TH, BIN_CN = 0.70, 0.50        # 'whole month on' thresholds (monthly mean of driver)


def piecewise(idx: pd.DatetimeIndex, knots) -> np.ndarray:
    kx = pd.to_datetime([k[0] for k in knots]).values.astype("datetime64[D]").astype("int64")
    ky = np.array([k[1] for k in knots], float)
    return np.interp(idx.values.astype("datetime64[D]").astype("int64"), kx, ky)


def build_drivers(cal: pd.DataFrame) -> pd.DataFrame:
    idx = cal.index
    per = idx.to_period("M")
    s_th = load_oxcgrt("THA").reindex(idx).fillna(0.0)
    s_cn = load_oxcgrt("CHN").reindex(idx).fillna(0.0)
    hesi, ref = mobility_hesi(load_mobility_th(), idx[cal.H.values == 1])
    drv = pd.DataFrame({"s_th": s_th, "s_cn": s_cn, "hesi_th": hesi.reindex(idx).fillna(0.0),
                        "e_cn": piecewise(idx, CN_EPISODES)}, index=idx)
    drv["bin_th"] = (drv.s_th.groupby(per).transform("mean") >= BIN_TH).astype(float)
    drv["bin_cn"] = (drv.e_cn.groupby(per).transform("mean") >= BIN_CN).astype(float)
    drv.attrs["mobility_ref_drop"] = ref
    return drv


def stoppage_X(cal, drv, variant: str, lam: float = 1.0, kappa: float = 1.0) -> pd.DataFrame:
    """Shared-channel regressors. variant in {off, binary, stringency, hesi, lam, oracle}."""
    H, C = cal.H.values, cal.C.values
    zD = np.zeros(len(cal))
    zP = np.zeros(len(cal))
    if variant == "binary":
        zD, zP = drv.bin_th.values, drv.bin_cn.values
    elif variant == "stringency":
        zD, zP = drv.s_th.values, drv.s_cn.values
    elif variant in ("hesi", "lam"):
        zD, zP = lam * drv.hesi_th.values, drv.e_cn.values
    elif variant == "oracle":
        zD, zP = kappa * drv.hesi_th.values, drv.e_cn.values
    elif variant != "off":
        raise ValueError(variant)
    zD, zP = np.clip(zD, 0, 1), np.clip(zP, 0, 1)
    return pd.DataFrame({"xD": H + (1 - H) * zD, "xP": C + (1 - C) * zP}, index=cal.index)


def sep_X(cal, drv) -> pd.DataFrame:
    """Standard approach: holidays + COVID drivers as *separate* regressors (own coefficients)."""
    return pd.DataFrame({"xD": cal.H.values, "xP": cal.C.values,
                         "sD": drv.hesi_th.values, "sP": drv.e_cn.values}, index=cal.index)


# =============================================================================
# 3. DATA-GENERATING PROCESS (daily, calibrated to the scale of Thai exports)
# =============================================================================
TRUE = dict(level0=np.log(560.0), beta_D=float(np.log(0.45)), beta_P=float(np.log(0.80)),
            dow=np.array([0.03, 0.04, 0.04, 0.03, 0.02, -0.30, -0.55]),
            ar_rho=0.985, ar_sd=0.008, m_rho=0.85, m_sd=0.015, eps_sd=0.12,
            nu_rho=0.97, nu_sd=0.15)
SCENARIOS = {"S1": dict(kappa=1.0, demand=False),   # ideal transfer: mobility-HESI = export HESI
             "S2": dict(kappa=0.5, demand=True)}    # realistic: half pass-through + demand shock
TREND = [("2010-01-01", 0.10), ("2012-01-01", -0.01), ("2016-01-01", 0.025),
         ("2020-01-01", 0.03), ("2023-01-01", 0.02)]
DEMAND = [("2010-01-01", 0.0), ("2020-02-15", 0.0), ("2020-04-15", -0.08), ("2020-06-15", -0.08),
          ("2020-09-30", -0.02), ("2020-12-31", 0.0), ("2021-03-31", 0.06), ("2021-10-31", 0.07),
          ("2022-03-31", 0.03), ("2022-08-31", 0.0), ("2030-12-31", 0.0)]


def log_trend(idx, level0):
    t = (idx - START).days.values / 365.25
    bps = [((pd.Timestamp(d) - START).days / 365.25, g) for d, g in TREND]
    out = np.full(len(idx), level0, float)
    for i, (t0, g) in enumerate(bps):
        t1 = bps[i + 1][0] if i + 1 < len(bps) else np.inf
        out = out + g * np.clip(t - t0, 0, t1 - t0)
    return out


def yearly(idx):
    f = 2 * np.pi * idx.dayofyear.values / 365.25
    return (-0.045 * np.cos(f) + 0.010 * np.sin(f) + 0.020 * np.cos(2 * f)
            + 0.015 * np.sin(2 * f) - 0.010 * np.cos(3 * f))


def _ar1(rng, n, rho, sd):
    e = rng.normal(0, sd, n)
    a = np.empty(n)
    a[0] = e[0] / np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        a[i] = rho * a[i - 1] + e[i]
    return a


def simulate(cal, drv, scenario: str, seed: int, P=TRUE) -> pd.Series:
    """Daily exports (USD mn). Same seed => identical noise in S1 and S2."""
    sc, idx, n = SCENARIOS[scenario], cal.index, len(cal)
    rng = np.random.default_rng(seed)
    a = _ar1(rng, n, P["ar_rho"], P["ar_sd"])
    per = idx.to_period("M")
    up = per.unique()
    b = pd.Series(_ar1(rng, len(up), P["m_rho"], P["m_sd"]), index=up).reindex(per).values
    eps = rng.normal(0, P["eps_sd"], n)
    nu_sd = P["nu_sd"] * np.sqrt(1 - P["nu_rho"] ** 2)
    nuD, nuP = _ar1(rng, n, P["nu_rho"], nu_sd), _ar1(rng, n, P["nu_rho"], nu_sd)
    zD = np.clip(sc["kappa"] * drv.hesi_th.values * (1 + nuD), 0, 1)   # true export stoppage
    zP = np.clip(drv.e_cn.values * (1 + nuP), 0, 1)
    H, C = cal.H.values, cal.C.values
    xD, xP = H + (1 - H) * zD, C + (1 - C) * zP
    dem = piecewise(idx, DEMAND) if sc["demand"] else 0.0
    logy = (log_trend(idx, P["level0"]) + yearly(idx) + P["dow"][cal.dow.values]
            + P["beta_D"] * xD + P["beta_P"] * xP + dem + a + b + eps)
    return pd.Series(np.exp(logy), index=idx, name="y")


def monthly(cal, y=None, X=None) -> pd.DataFrame:
    """Monthly frame: D (days), WK (weekend share), Y (total), M<col> (monthly mean of daily X)."""
    per = cal.index.to_period("M")
    m = pd.DataFrame({"D": cal.groupby(per).size(), "WK": cal.weekend.groupby(per).mean()})
    if y is not None:
        m["Y"] = y.groupby(per).sum(min_count=1)
    if X is not None:
        for c in X.columns:
            m["M" + c] = X[c].groupby(per).mean()
    m.index = m.index.to_timestamp()
    return m


# =============================================================================
# 4. MODELS  (all on log scale; linear models expose base0 + sum(coef * x))
# =============================================================================
def _prophet(freq):
    from prophet import Prophet
    if freq == "D":
        return Prophet(yearly_seasonality=10, weekly_seasonality=3, daily_seasonality=False,
                       uncertainty_samples=0)
    return Prophet(yearly_seasonality=5, weekly_seasonality=False, daily_seasonality=False,
                   n_changepoints=15, uncertainty_samples=0)


class ProphetLog:
    """Prophet on log(y); regressors additive & unstandardised, so coef is in log units and
    prediction(x) = base0 + sum_r coef_r * x_r exactly (used for cheap dose sweeps)."""

    def __init__(self, freq: str, regs=()):
        self.freq, self.regs, self.coef = freq, list(regs), {}

    def fit(self, ds, ly, X=None):
        df = pd.DataFrame({"ds": pd.DatetimeIndex(ds), "y": np.asarray(ly, float)})
        for r in self.regs:
            df[r] = X[r].values
        df = df[np.isfinite(df.y.values)].reset_index(drop=True)
        self.m = _prophet(self.freq)
        for r in self.regs:
            self.m.add_regressor(r, prior_scale=10.0, standardize=False)
        self.m.fit(df)
        if self.regs:
            from prophet.utilities import regressor_coefficients
            rc = regressor_coefficients(self.m)
            self.coef = dict(zip(rc.regressor, rc.coef.astype(float)))
        fitted = self.predict_log(self.base0(df.ds), df)
        self.smear = float(np.mean(np.exp(df.y.values - fitted)))     # Duan smearing
        return self

    def base0(self, ds):
        fut = pd.DataFrame({"ds": pd.DatetimeIndex(ds)})
        for r in self.regs:
            fut[r] = 0.0
        return self.m.predict(fut).yhat.values

    def predict_log(self, base0, X):
        out = np.array(base0, float)
        for r in self.regs:
            out = out + self.coef[r] * np.asarray(X[r], float)
        return out


class SarimaxLog:
    """Airline model (0,1,1)(0,1,1)_12 on log(avg daily exports) with exogenous regressors
    (regression with SARIMA errors => forecasts are linear in future exog)."""

    def __init__(self, regs):
        self.regs = list(regs)

    def fit(self, ly, X):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.res = SARIMAX(np.asarray(ly, float), exog=X[self.regs].values, order=(0, 1, 1),
                           seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=300)
        self.coef = dict(zip(self.regs, self.res.params[: len(self.regs)]))
        self.smear = float(np.exp(self.res.params[-1] / 2))
        return self

    def base0(self, h):
        return self.res.forecast(steps=h, exog=np.zeros((h, len(self.regs))))

    def predict_log(self, base0, X):
        out = np.array(base0, float)
        for r in self.regs:
            out = out + self.coef[r] * np.asarray(X[r], float)
        return out


class LGBMHybrid:
    """Linear log-trend (OLS) + LightGBM on the detrended residual — a fair tree baseline."""

    def __init__(self, freq: str, feats):
        self.freq, self.feats = freq, list(feats)

    def fit(self, t, ly, F):
        import lightgbm as lgb
        ok = np.isfinite(ly)
        self.tc = np.polyfit(t[ok], ly[ok], 1)
        r = ly[ok] - np.polyval(self.tc, t[ok])
        p = (dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1) if self.freq == "D" else
             dict(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=5))
        self.model = lgb.LGBMRegressor(random_state=0, verbose=-1, **p)
        self.model.fit(F[self.feats].values[ok], r)
        self.smear = float(np.mean(np.exp(r - self.model.predict(F[self.feats].values[ok]))))
        return self

    def predict_log(self, t, F):
        return np.polyval(self.tc, t) + self.model.predict(F[self.feats].values)


def month_matrix(x: pd.Series) -> np.ndarray:
    """Months x days matrix (NaN-padded) of a daily series."""
    groups = [g.values for _, g in x.groupby(x.index.to_period("M"))]
    out = np.full((len(groups), max(len(g) for g in groups)), np.nan)
    for i, g in enumerate(groups):
        out[i, : len(g)] = g
    return out


def log_mean_exp(beta: float, Mx: np.ndarray) -> np.ndarray:
    """Exact monthly log-effect of a daily log-linear effect beta*x_d."""
    return np.log(np.nanmean(np.exp(beta * Mx), axis=1))


def deaggregate_beta(beta_m: float, x_train: pd.Series, grid=np.linspace(-3.0, 0.0, 601)) -> float:
    """Aggregation (Jensen) correction for monthly-only data: find the daily coefficient whose
    monthly log-mean-exp reproduces the fitted monthly linear effects beta_m * mean(x)."""
    Mx = month_matrix(x_train)
    target = beta_m * np.nanmean(Mx, axis=1)
    E = np.log(np.nanmean(np.exp(grid[:, None, None] * Mx[None]), axis=2))
    return float(grid[np.argmin(((E - target[None]) ** 2).sum(axis=1))])


def snaive(train_Y: pd.Series, months: pd.DatetimeIndex) -> np.ndarray:
    hist = train_Y.dropna()
    return np.array([hist[hist.index.month == d.month].iloc[-1] for d in months])


def score(actual, pred, scale) -> dict:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    e = actual - pred
    return dict(mape=100 * np.mean(np.abs(e) / actual), rmse=float(np.sqrt(np.mean(e ** 2))),
                mase=float(np.mean(np.abs(e)) / scale))


def mase_scale(train_Y: pd.Series) -> float:
    v = train_Y.dropna().values
    return float(np.mean(np.abs(v[12:] - v[:-12])))


def years(idx) -> np.ndarray:
    return (pd.DatetimeIndex(idx) - START).days.values / 365.25


# =============================================================================
# 5. SIMULATION EXPERIMENTS
# =============================================================================
EXPS = {"E0": ("2017-12-31", "2018-01-01", "2019-12-31"),    # normal period
        "E1": ("2019-12-31", "2020-01-01", "2021-12-31"),    # shock period, conditional on drivers
        "E2": ("2022-12-31", "2023-01-01", "2024-12-31")}    # post-shock
MASK = ("2020-01-20", "2022-09-30")
CAL_MONTHS = pd.DatetimeIndex(["2020-03-01", "2020-04-01"])  # first two lockdown months
W20_START = pd.Timestamp("2020-05-01")
LAM_GRID = np.round(np.arange(0.0, 3.0001, 0.05), 2)
MREGS = ["MxD", "MxP", "WK"]


class Setup:
    """Seed-independent inputs: calendar, drivers, regressors and their monthly means."""

    def __init__(self):
        self.cal = build_calendar()
        self.drv = build_drivers(self.cal)
        self.X = {v: stoppage_X(self.cal, self.drv, v) for v in ("off", "binary", "stringency", "hesi")}
        self.X["sep"] = sep_X(self.cal, self.drv)
        for S, sc in SCENARIOS.items():
            self.X["oracle_" + S] = stoppage_X(self.cal, self.drv, "oracle", kappa=sc["kappa"])
        self.M = {v: monthly(self.cal, X=x) for v, x in self.X.items()}
        self.base_m = monthly(self.cal)
        self.F = self.cal[["dow", "doy"]]


def _split(setup, exp):
    o, a, b = (pd.Timestamp(x) for x in EXPS[exp])
    d = setup.cal.index
    mi = setup.base_m.index
    return d[d <= o], d[(d >= a) & (d <= b)], mi[mi <= o], mi[(mi >= a) & (mi <= b)]


def _d2m(logp, smear, days):
    return pd.Series(np.exp(logp) * smear, index=days).groupby(days.to_period("M")).sum().values


def _lgbm_frame(setup, days, X):
    return pd.concat([setup.F.loc[days], X.loc[days, ["xD", "xP"]]], axis=1)


def _monthly_frame(setup, months, v):
    return setup.M[v].loc[months].assign(month=months.month)


def run_seed(setup: Setup, seed: int, keep_paths=False):
    rows, betas, lamhat, paths = [], [], [], {}
    cal, drv = setup.cal, setup.drv
    ys = {S: simulate(cal, drv, S, seed) for S in SCENARIOS}

    def add(scen, exp, model, variant, window, act, pred, scale, lam=np.nan):
        rows.append(dict(seed=seed, scen=scen, exp=exp, model=model, variant=variant,
                         window=window, lam=lam, **score(act, pred, scale)))

    # ---------------- E0 (normal period) and E1 (shock period): pre-2020 training identical
    for exp in ("E0", "E1"):
        y = ys["S1"]
        dtr, dte, mtr, mte = _split(setup, exp)
        Ym = {S: monthly(cal, y=ys[S]).Y for S in SCENARIOS}
        ly, lym = np.log(y.loc[dtr].values), np.log((Ym["S1"] / setup.base_m.D).loc[mtr].values)
        scale = mase_scale(Ym["S1"].loc[mtr])
        Dte = setup.base_m.D.loc[mte].values
        P = {}                                   # (model, variant) -> monthly forecast
        LAM = {}                                 # model -> {lam: forecast}
        # Prophet daily
        pdv = ProphetLog("D").fit(dtr, ly)
        P[("Prophet-D", "vanilla")] = _d2m(pdv.base0(dte), pdv.smear, dte)
        pdh = ProphetLog("D", ["xD", "xP"]).fit(dtr, ly, setup.X["off"].loc[dtr])
        b0 = pdh.base0(dte)
        # Prophet monthly
        pmv = ProphetLog("M").fit(mtr, lym)
        P[("Prophet-M", "vanilla")] = np.exp(pmv.base0(mte)) * pmv.smear * Dte
        pmh = ProphetLog("M", MREGS).fit(mtr, lym, setup.M["off"].loc[mtr])
        m0 = pmh.base0(mte)
        bdD = deaggregate_beta(pmh.coef["MxD"], setup.X["off"].xD.loc[dtr])
        bdP = deaggregate_beta(pmh.coef["MxP"], setup.X["off"].xP.loc[dtr])
        mac0 = m0 + pmh.coef["WK"] * setup.base_m.WK.loc[mte].values
        # SARIMAX monthly
        sx = SarimaxLog(MREGS).fit(lym, setup.M["off"].loc[mtr])
        s0 = sx.base0(len(mte))
        # LightGBM daily / monthly
        ld = LGBMHybrid("D", ["dow", "doy", "xD", "xP"]).fit(
            years(dtr), ly, _lgbm_frame(setup, dtr, setup.X["off"]))
        lm = LGBMHybrid("M", ["month", "MxD", "MxP", "WK"]).fit(
            years(mtr), lym, _monthly_frame(setup, mtr, "off"))
        P[("SNaive", "-")] = snaive(Ym["S1"].loc[mtr], mte)

        def predict_all(X, M):
            return {
                "Prophet-D HE": _d2m(pdh.predict_log(b0, X.loc[dte]), pdh.smear, dte),
                "Prophet-M HE": np.exp(pmh.predict_log(m0, M.loc[mte])) * pmh.smear * Dte,
                "Prophet-M HE+AC": np.exp(mac0 + log_mean_exp(bdD, month_matrix(X.xD.loc[dte]))
                                          + log_mean_exp(bdP, month_matrix(X.xP.loc[dte]))) * pmh.smear * Dte,
                "SARIMAX HE": np.exp(sx.predict_log(s0, M.loc[mte])) * sx.smear * Dte,
                "LightGBM-D HE": _d2m(ld.predict_log(years(dte), _lgbm_frame(setup, dte, X)), ld.smear, dte),
                "LightGBM-M HE": np.exp(lm.predict_log(years(mte), M.loc[mte].assign(month=mte.month))) * lm.smear * Dte,
            }

        variants = ["off"] if exp == "E0" else ["off", "binary", "stringency", "hesi"]
        for v in variants:
            for mdl, f in predict_all(setup.X[v], setup.M[v]).items():
                P[(mdl, v)] = f
        if exp == "E1":
            for S in SCENARIOS:
                for mdl, f in predict_all(setup.X["oracle_" + S], setup.M["oracle_" + S]).items():
                    P[(mdl, "oracle_" + S)] = f
            for lam in LAM_GRID:
                Xl = stoppage_X(cal, drv, "lam", lam=lam)
                Ml = monthly(cal, X=Xl)
                for mdl, f in predict_all(Xl, Ml).items():
                    LAM.setdefault(mdl, {})[lam] = f
            for mdl, (obj, keys) in {"Prophet-D HE": (pdh, ["xD", "xP"]),
                                     "Prophet-M HE": (pmh, ["MxD", "MxP"]),
                                     "SARIMAX HE": (sx, ["MxD", "MxP"])}.items():
                betas.append(dict(seed=seed, exp=exp, scen="pre", model=mdl, variant="off",
                                  beta_D=obj.coef[keys[0]], beta_P=obj.coef[keys[1]]))
            betas.append(dict(seed=seed, exp=exp, scen="pre", model="Prophet-M HE+AC", variant="off",
                              beta_D=bdD, beta_P=bdP))

        scens = ["S1"] if exp == "E0" else list(SCENARIOS)
        w20 = mte >= W20_START
        cal_mask = mte.isin(CAL_MONTHS)
        for S in scens:
            act = Ym[S].loc[mte].values
            for (mdl, v), f in P.items():
                if v.startswith("oracle_") and v != "oracle_" + S:
                    continue
                vv = "oracle" if v.startswith("oracle_") else v
                add(S, exp, mdl, vv, "W24", act, f, scale)
                if exp == "E1":
                    add(S, exp, mdl, vv, "W20", act[w20], f[w20], scale)
            if exp == "E1":
                for mdl, grid in LAM.items():
                    lams = np.array(sorted(grid))
                    sse = [np.sum((np.log(act[cal_mask]) - np.log(grid[l][cal_mask])) ** 2) for l in lams]
                    lh = float(lams[int(np.argmin(sse))])
                    lamhat.append(dict(seed=seed, scen=S, model=mdl, lam_hat=lh))
                    add(S, exp, mdl, "hesi-cal", "W20", act[w20], grid[lh][w20], scale, lam=lh)
                    for l in lams:
                        add(S, exp, mdl, "lam", "W24", act, grid[l], scale, lam=float(l))
                if keep_paths:
                    paths[("E1", S)] = dict(months=mte, actual=act, P={k: v for k, v in P.items()},
                                            LAM={m: {0.5: g[0.5], 1.0: g[1.0]} for m, g in LAM.items()},
                                            cal={m: g[[r for r in lamhat if r["seed"] == seed and r["scen"] == S and r["model"] == m][0]["lam_hat"]]
                                                 for m, g in LAM.items()})
        if keep_paths and exp == "E0":
            paths[("E0", "S1")] = dict(months=mte, actual=Ym["S1"].loc[mte].values, P=P)

    # ---------------- E2 (post-shock): training includes COVID, separate per scenario
    for S in SCENARIOS:
        y = ys[S]
        dtr, dte, mtr, mte = _split(setup, "E2")
        Ym = monthly(cal, y=y).Y
        ly = np.log(y.loc[dtr].values)
        lym = np.log((Ym / setup.base_m.D).loc[mtr].values)
        scale = mase_scale(Ym.loc[mtr])
        Dte = setup.base_m.D.loc[mte].values
        act = Ym.loc[mte].values
        msk_d = (dtr >= MASK[0]) & (dtr <= MASK[1])
        msk_m = (mtr >= MASK[0]) & (mtr <= MASK[1])
        P = {("SNaive", "-"): snaive(Ym.loc[mtr], mte)}
        Xf, Mf = setup.X["off"], setup.M["off"]           # future: holidays only
        Xf_sep = setup.X["sep"].loc[dte].assign(sD=0.0, sP=0.0)
        Mf_sep = setup.M["sep"].loc[mte].assign(MsD=0.0, MsP=0.0)
        specs = {"vanilla": ([], None, False), "cal": (["xD", "xP"], "off", False),
                 "mask": (["xD", "xP"], "off", True), "sep": (["xD", "xP", "sD", "sP"], "sep", False),
                 "binary": (["xD", "xP"], "binary", False), "stringency": (["xD", "xP"], "stringency", False),
                 "hesi": (["xD", "xP"], "hesi", False)}
        for v, (regs, xv, msk) in specs.items():
            lyv = ly.copy()
            lymv = lym.copy()
            if msk:
                lyv[msk_d] = np.nan
                lymv[msk_m] = np.nan
            # daily
            pd_ = ProphetLog("D", regs).fit(dtr, lyv, setup.X[xv].loc[dtr] if xv else None)
            XF = Xf_sep if v == "sep" else Xf.loc[dte]
            P[("Prophet-D", v)] = _d2m(pd_.predict_log(pd_.base0(dte), XF), pd_.smear, dte)
            # monthly
            mregs = ["M" + r for r in regs] + (["WK"] if regs else [])
            pm_ = ProphetLog("M", mregs).fit(mtr, lymv, setup.M[xv].loc[mtr] if xv else None)
            MF = Mf_sep if v == "sep" else Mf.loc[mte]
            P[("Prophet-M", v)] = np.exp(pm_.predict_log(pm_.base0(mte), MF)) * pm_.smear * Dte
            if regs:
                betas.append(dict(seed=seed, exp="E2", scen=S, model="Prophet-D", variant=v,
                                  beta_D=pd_.coef["xD"], beta_P=pd_.coef["xP"]))
                betas.append(dict(seed=seed, exp="E2", scen=S, model="Prophet-M", variant=v,
                                  beta_D=pm_.coef["MxD"], beta_P=pm_.coef["MxP"]))
        for v in ("cal", "hesi"):
            xv = "off" if v == "cal" else v
            sx = SarimaxLog(MREGS).fit(lym, setup.M[xv].loc[mtr])
            P[("SARIMAX", v)] = np.exp(sx.predict_log(sx.base0(len(mte)), Mf.loc[mte])) * sx.smear * Dte
        ld = LGBMHybrid("D", ["dow", "doy", "xD", "xP"]).fit(
            years(dtr), ly, _lgbm_frame(setup, dtr, setup.X["hesi"]))
        P[("LightGBM-D", "hesi")] = _d2m(ld.predict_log(years(dte), _lgbm_frame(setup, dte, Xf)), ld.smear, dte)
        lm = LGBMHybrid("M", ["month", "MxD", "MxP", "WK"]).fit(
            years(mtr), lym, _monthly_frame(setup, mtr, "hesi"))
        P[("LightGBM-M", "hesi")] = np.exp(lm.predict_log(years(mte), _monthly_frame(setup, mte, "off"))) * lm.smear * Dte
        for (mdl, v), f in P.items():
            add(S, "E2", mdl, v, "W24", act, f, scale)
        if keep_paths:
            paths[("E2", S)] = dict(months=mte, actual=act, P=P)
    if keep_paths:
        paths["daily_S1"] = ys["S1"]
        paths["monthly"] = {S: monthly(cal, y=ys[S]).Y for S in SCENARIOS}
    return rows, betas, lamhat, paths


def cmd_simulate(args):
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    setup = Setup()
    with open(os.path.join(args.out, "setup.pkl"), "wb") as fh:
        pickle.dump(dict(cal=setup.cal, drv=setup.drv, X=setup.X, M=setup.M,
                         ref=setup.drv.attrs.get("mobility_ref_drop")), fh)
    for i, seed in enumerate(range(args.seed0, args.seed0 + args.reps)):
        rows, betas, lamhat, paths = run_seed(setup, seed, keep_paths=(seed == 0))
        for name, rec in (("metrics", rows), ("betas", betas), ("lamhat", lamhat)):
            f = os.path.join(args.out, f"{name}.csv")
            pd.DataFrame(rec).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        if paths:
            with open(os.path.join(args.out, "paths_seed0.pkl"), "wb") as fh:
                pickle.dump(paths, fh)
        print(f"seed {seed} done ({i + 1}/{args.reps}) elapsed {time.time() - t0:.0f}s", flush=True)


# =============================================================================
# 6. REAL-DATA RUNNER  (monthly public data; rolling origins; scenario-conditional)
# =============================================================================
def cmd_real(args):
    """Monthly Thai exports (public, e.g. MOC/BOT) -> backtests of all monthly variants.
    For origins before/inside COVID the driver paths after the origin are *observed*
    (scenario-conditional / nowcasting). Report this explicitly in any paper."""
    os.makedirs(args.out, exist_ok=True)
    ex = pd.read_csv(args.exports)
    months = pd.to_datetime(ex.iloc[:, 0].astype(str)).dt.to_period("M").dt.to_timestamp()
    Y = pd.Series(ex.iloc[:, 1].astype(float).values, index=months).sort_index()
    end = Y.index.max() + pd.offsets.MonthEnd(args.horizon + 1)
    cal = build_calendar(Y.index.min(), end)
    drv = build_drivers(cal)
    X = {v: stoppage_X(cal, drv, v, lam=args.lam) for v in ("off", "binary", "stringency", "hesi")}
    X["sep"] = sep_X(cal, drv)
    M = {v: monthly(cal, X=x) for v, x in X.items()}
    base = monthly(cal)
    Yfull = Y.reindex(base.index)
    lyall = np.log(Yfull / base.D)
    rows, fc_rows, coef_rows = [], [], []
    specs = {"vanilla": ([], None, False), "cal": (["MxD", "MxP", "WK"], "off", False),
             "mask": (["MxD", "MxP", "WK"], "off", True), "sep": (["MxD", "MxP", "MsD", "MsP", "WK"], "sep", False),
             "binary": (["MxD", "MxP", "WK"], "binary", False), "stringency": (["MxD", "MxP", "WK"], "stringency", False),
             "hesi": (["MxD", "MxP", "WK"], "hesi", False)}
    for origin in [pd.Timestamp(o) for o in args.origins] + [Y.index.max()]:
        mtr = base.index[(base.index <= origin) & Yfull.notna().values]
        mte = base.index[(base.index > origin)][: args.horizon]
        D = base.D.loc[mte].values
        conditional = origin < pd.Timestamp("2022-10-01")
        preds = {("SNaive", "-"): snaive(Yfull.loc[mtr], mte)}
        for v, (regs, xv, msk) in specs.items():
            ly = lyall.loc[mtr].values.copy()
            if msk:
                ly[(mtr >= MASK[0]) & (mtr <= MASK[1])] = np.nan
            m = ProphetLog("M", regs).fit(mtr, ly, M[xv].loc[mtr] if xv else None)
            fut = (M[xv] if xv else base).loc[mte].copy()
            if not conditional:                       # genuine ex-ante: no future COVID
                fut = (M["sep"].loc[mte].assign(MsD=0.0, MsP=0.0) if v == "sep" else M["off"].loc[mte]) if xv else fut
            preds[("Prophet-M", v)] = np.exp(m.predict_log(m.base0(mte), fut)) * m.smear * D
            for r, c in m.coef.items():
                coef_rows.append(dict(origin=origin.date(), model="Prophet-M", variant=v, regressor=r, coef=c))
        sx = SarimaxLog(MREGS).fit(lyall.loc[mtr].values, M["hesi"].loc[mtr])
        futm = M["hesi"].loc[mte] if conditional else M["off"].loc[mte]
        preds[("SARIMAX", "hesi")] = np.exp(sx.predict_log(sx.base0(len(mte)), futm)) * sx.smear * D
        lm = LGBMHybrid("M", ["month", "MxD", "MxP", "WK"]).fit(
            years(mtr), lyall.loc[mtr].values, M["hesi"].loc[mtr].assign(month=mtr.month))
        preds[("LightGBM-M", "hesi")] = np.exp(lm.predict_log(years(mte), futm.assign(month=mte.month))) * lm.smear * D
        act = Yfull.loc[mte].values
        ok = np.isfinite(act)
        sc = mase_scale(Yfull.loc[mtr])
        for (mdl, v), f in preds.items():
            for d, a, p in zip(mte, act, f):
                fc_rows.append(dict(origin=origin.date(), month=d.date(), model=mdl, variant=v, actual=a, forecast=p))
            if ok.sum() >= 3:
                rows.append(dict(origin=origin.date(), conditional=conditional, model=mdl, variant=v,
                                 n=int(ok.sum()), **score(act[ok], f[ok], sc)))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "real_metrics.csv"), index=False)
    pd.DataFrame(fc_rows).to_csv(os.path.join(args.out, "real_forecasts.csv"), index=False)
    pd.DataFrame(coef_rows).to_csv(os.path.join(args.out, "real_coefficients.csv"), index=False)
    print(pd.DataFrame(rows).pivot_table(index=["model", "variant"], columns="origin", values="mape").round(2))
