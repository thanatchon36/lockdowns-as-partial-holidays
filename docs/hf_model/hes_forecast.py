"""Load the HES models in this repository and forecast monthly exports from daily stoppage regressors.

Standalone: needs numpy, pandas, prophet, lightgbm and statsmodels, not the `hesi` package. The models were
trained on simulated daily exports, 2010-01-01 to 2019-12-31 (replication 0 of the paper's Monte Carlo study), so
they forecast that simulated series, not official Thai exports.

    import pandas as pd, hes_forecast as hf
    models = hf.load(".")                                   # the directory holding meta.json
    X = regressors.query("variant == 'hesi'").set_index("date")[["xD", "xP"]]   # daily, from the dataset
    fc = hf.forecast(models, X, months=pd.date_range("2020-01-01", "2021-12-01", freq="MS"))

`fc` has one column per model: forecast monthly export value (the simulated series' units).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2010-01-01")      # origin of the LightGBM trend clock (years since START)


def load(model_dir: str | Path = ".") -> dict:
    """meta.json plus the fitted Prophet, LightGBM and SARIMAX objects."""
    import lightgbm as lgb
    from prophet.serialize import model_from_json
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text())
    out = {"meta": meta}
    for name, m in meta["models"].items():
        if m["kind"] == "prophet":
            out[name] = model_from_json((d / m["file"]).read_text())
        elif m["kind"] == "lightgbm":
            out[name] = lgb.Booster(model_file=str(d / m["file"]))
        elif m["kind"] == "sarimax":
            s = json.loads((d / m["file"]).read_text())
            tr = s["train"]
            mod = SARIMAX(np.asarray(tr["y"], float), exog=np.asarray(tr["exog"], float),
                          order=tuple(s["order"]), seasonal_order=tuple(s["seasonal_order"]))
            out[name] = mod.filter(np.asarray(s["params"], float))
    return out


def _monthly(X: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly means of the daily regressors (MxD, MxP), weekend share WK and days D, as in the paper."""
    days = pd.date_range(months[0], months[-1] + pd.offsets.MonthEnd(0), freq="D")
    Xd = X.reindex(days)
    if Xd[["xD", "xP"]].isna().any().any():
        raise ValueError("X must cover every day of the forecast months")
    per = days.to_period("M")
    m = pd.DataFrame({"MxD": Xd.xD.groupby(per).mean(), "MxP": Xd.xP.groupby(per).mean(),
                      "WK": pd.Series((days.dayofweek >= 5).astype(float), index=days).groupby(per).mean(),
                      "D": pd.Series(1, index=days).groupby(per).sum()})
    m.index = m.index.to_timestamp()
    return m.loc[months]


def _log_mean_exp(beta: float, x: pd.Series) -> np.ndarray:
    """Exact monthly log-effect of a daily effect beta * x_d (the aggregation correction)."""
    return np.log(np.exp(beta * x).groupby(x.index.to_period("M")).mean().values)


def _sum_daily(logp: np.ndarray, smear: float, days: pd.DatetimeIndex) -> np.ndarray:
    return pd.Series(np.exp(logp) * smear, index=days).groupby(days.to_period("M")).sum().values


def forecast(models: dict, X: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly forecasts from each model, given daily regressors X (columns xD, xP) over those months."""
    meta = models["meta"]["models"]
    months = pd.DatetimeIndex(months)
    X = X.copy()
    X.index = pd.DatetimeIndex(X.index)
    days = pd.date_range(months[0], months[-1] + pd.offsets.MonthEnd(0), freq="D")
    Xd = X.reindex(days)[["xD", "xP"]]
    M = _monthly(X, months)
    out = {}

    # Prophet daily: base (regressors at zero) + coef * x, exp, smearing, summed by month
    m = meta["Prophet-D HE"]
    base = models["Prophet-D HE"].predict(pd.DataFrame({"ds": days, "xD": 0.0, "xP": 0.0})).yhat.values
    logp = base + m["coef"]["xD"] * Xd.xD.values + m["coef"]["xP"] * Xd.xP.values
    out["Prophet-D HE"] = _sum_daily(logp, m["smear"], days)

    # Prophet monthly on log average daily exports, times days in the month
    m = meta["Prophet-M HE"]
    regs = m["regressors"]
    base = models["Prophet-M HE"].predict(pd.DataFrame({"ds": months, **{r: 0.0 for r in regs}})).yhat.values
    logm = base + sum(m["coef"][r] * M[r].values for r in regs)
    out["Prophet-M HE"] = np.exp(logm) * m["smear"] * M.D.values

    # ... with the aggregation correction: daily-equivalent elasticities through log-mean-exp
    a = meta["Prophet-M HE+AC"]
    logac = (base + m["coef"]["WK"] * M.WK.values + _log_mean_exp(a["beta_D"], Xd.xD)
             + _log_mean_exp(a["beta_P"], Xd.xP))
    out["Prophet-M HE+AC"] = np.exp(logac) * m["smear"] * M.D.values

    # SARIMAX: regression with SARIMA errors, forecast from the end of training (January 2020)
    m = meta["SARIMAX HE"]
    first = pd.Timestamp(m["first_forecast_month"])
    h = (months[-1].year - first.year) * 12 + months[-1].month - first.month + 1
    allm = pd.date_range(first, periods=h, freq="MS")
    base = pd.Series(models["SARIMAX HE"].forecast(steps=h, exog=np.zeros((h, len(m["regressors"])))), index=allm)
    logm = base.loc[months].values + sum(m["coef"][r] * M[r].values for r in m["regressors"])
    out["SARIMAX HE"] = np.exp(logm) * m["smear"] * M.D.values

    # LightGBM: linear log-trend in years since 2010 + trees on the detrended residual
    m = meta["LightGBM-D HE"]
    F = pd.DataFrame({"dow": days.dayofweek, "doy": days.dayofyear, "xD": Xd.xD.values, "xP": Xd.xP.values})
    t = (days - START).days.values / 365.25
    logp = np.polyval(m["trend"], t) + models["LightGBM-D HE"].predict(F[m["features"]].values)
    out["LightGBM-D HE"] = _sum_daily(logp, m["smear"], days)

    m = meta["LightGBM-M HE"]
    F = M.assign(month=months.month)
    t = (months - START).days.values / 365.25
    logm = np.polyval(m["trend"], t) + models["LightGBM-M HE"].predict(F[m["features"]].values)
    out["LightGBM-M HE"] = np.exp(logm) * m["smear"] * M.D.values

    return pd.DataFrame(out, index=months).rename_axis("month")
