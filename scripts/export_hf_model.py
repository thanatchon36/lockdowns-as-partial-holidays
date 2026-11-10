#!/usr/bin/env python3
"""Stage the Hugging Face model repository thanatchon36/lockdowns-as-partial-holidays in dist/hf_model/.

Refits the six HES models of the shock experiment (E1) for replication 0 exactly as `hesi.forecast.run_seed` does:
trained on the simulated daily exports of scenario S1 from 2010-01-01 to 2019-12-31, with the shared stoppage
regressors (holiday indicators only, since every dose is zero before 2020). Saves them without pickle: Prophet as
JSON (prophet.serialize), LightGBM as its text model, SARIMAX as parameters plus its training data. Then reloads them
with the standalone docs/hf_model/hes_forecast.py, forecasts 2020-2021 for each dose variant, and fails unless the
reloaded forecasts equal those of the in-memory models. It also reports how far they are from the paper's stored
replication-0 forecasts (results/paths_seed0.pkl): the optimizers of Prophet and SARIMAX land on slightly different
solutions on different platforms, so a refit is close to, not identical with, the paper's run. Run from the repository root
(`make model`); upload with `hf upload thanatchon36/lockdowns-as-partial-holidays dist/hf_model . --repo-type model`.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import importlib.util
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hesi import forecast as H

CARD, OUT, SEED = Path("docs/hf_model"), Path("dist/hf_model"), 0
VARIANTS = ["off", "binary", "stringency", "hesi"]
RTOL = 1e-9          # reloaded vs in-memory models
PAPER_RTOL = 0.05   # refit vs the paper's run: only a guard against gross errors


def main() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    from prophet.serialize import model_to_json

    setup = H.Setup()
    cal = setup.cal
    y = H.simulate(cal, setup.drv, "S1", SEED)
    dtr, dte, mtr, mte = H._split(setup, "E1")
    ly = np.log(y.loc[dtr].values)
    lym = np.log((H.monthly(cal, y=y).Y / setup.base_m.D).loc[mtr].values)

    pdh = H.ProphetLog("D", ["xD", "xP"]).fit(dtr, ly, setup.X["off"].loc[dtr])
    pmh = H.ProphetLog("M", H.MREGS).fit(mtr, lym, setup.M["off"].loc[mtr])
    bdD = H.deaggregate_beta(pmh.coef["MxD"], setup.X["off"].xD.loc[dtr])
    bdP = H.deaggregate_beta(pmh.coef["MxP"], setup.X["off"].xP.loc[dtr])
    sx = H.SarimaxLog(H.MREGS).fit(lym, setup.M["off"].loc[mtr])
    ld = H.LGBMHybrid("D", ["dow", "doy", "xD", "xP"]).fit(H.years(dtr), ly, H._lgbm_frame(setup, dtr, setup.X["off"]))
    lm = H.LGBMHybrid("M", ["month", "MxD", "MxP", "WK"]).fit(H.years(mtr), lym, H._monthly_frame(setup, mtr, "off"))

    (OUT / "prophet_daily_he.json").write_text(model_to_json(pdh.m))
    (OUT / "prophet_monthly_he.json").write_text(model_to_json(pmh.m))
    ld.model.booster_.save_model(str(OUT / "lightgbm_daily_he.txt"))
    lm.model.booster_.save_model(str(OUT / "lightgbm_monthly_he.txt"))
    (OUT / "sarimax_he.json").write_text(json.dumps({
        "order": [0, 1, 1], "seasonal_order": [0, 1, 1, 12],
        "param_names": list(sx.res.model.param_names), "params": [float(p) for p in sx.res.params],
        "train": {"month": [m.strftime("%Y-%m") for m in mtr], "y": [float(v) for v in lym],
                  "exog_names": H.MREGS, "exog": setup.M["off"].loc[mtr, H.MREGS].values.round(12).tolist()},
    }, indent=1))

    fl = lambda d: {k: float(v) for k, v in d.items()}  # noqa: E731
    meta = {
        "description": "HES models of the shock experiment (E1), replication 0, trained on simulated daily exports "
                       "of scenario S1, 2010-01-01 to 2019-12-31. Forecasts are for that simulated series.",
        "paper": "https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/",
        "code": "https://github.com/thanatchon36/lockdowns-as-partial-holidays",
        "dataset": "https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays",
        "doi": "10.5281/zenodo.24036954",
        "seed": SEED, "scenario": "S1", "train_start": "2010-01-01", "train_end": "2019-12-31",
        "true_beta_D": H.TRUE["beta_D"], "true_beta_P": H.TRUE["beta_P"],
        "target": "daily models: log daily exports; monthly models: log average daily exports in the month",
        "versions": {p: md.version(p) for p in ("prophet", "lightgbm", "statsmodels", "numpy", "pandas")},
        "models": {
            "Prophet-D HE": {"kind": "prophet", "file": "prophet_daily_he.json", "regressors": ["xD", "xP"],
                             "coef": fl(pdh.coef), "smear": pdh.smear},
            "Prophet-M HE": {"kind": "prophet", "file": "prophet_monthly_he.json", "regressors": H.MREGS,
                             "coef": fl(pmh.coef), "smear": pmh.smear},
            "Prophet-M HE+AC": {"kind": "correction", "of": "Prophet-M HE", "beta_D": bdD, "beta_P": bdP},
            "SARIMAX HE": {"kind": "sarimax", "file": "sarimax_he.json", "regressors": H.MREGS,
                           "coef": fl(sx.coef), "smear": sx.smear, "first_forecast_month": "2020-01"},
            "LightGBM-D HE": {"kind": "lightgbm", "file": "lightgbm_daily_he.txt", "features": ld.feats,
                              "trend": [float(v) for v in ld.tc], "smear": ld.smear},
            "LightGBM-M HE": {"kind": "lightgbm", "file": "lightgbm_monthly_he.txt", "features": lm.feats,
                              "trend": [float(v) for v in lm.tc], "smear": lm.smear},
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in ("README.md", "LICENSE", "hes_forecast.py"):
        shutil.copyfile(CARD / f, OUT / f)

    # in-memory forecasts, assembled exactly as hesi.forecast.run_seed does
    b0, m0, s0 = pdh.base0(dte), pmh.base0(mte), sx.base0(len(mte))
    mac0 = m0 + pmh.coef["WK"] * setup.base_m.WK.loc[mte].values
    Dte = setup.base_m.D.loc[mte].values

    def in_memory(X, M):
        return {
            "Prophet-D HE": H._d2m(pdh.predict_log(b0, X.loc[dte]), pdh.smear, dte),
            "Prophet-M HE": np.exp(pmh.predict_log(m0, M.loc[mte])) * pmh.smear * Dte,
            "Prophet-M HE+AC": np.exp(mac0 + H.log_mean_exp(bdD, H.month_matrix(X.xD.loc[dte]))
                                      + H.log_mean_exp(bdP, H.month_matrix(X.xP.loc[dte]))) * pmh.smear * Dte,
            "SARIMAX HE": np.exp(sx.predict_log(s0, M.loc[mte])) * sx.smear * Dte,
            "LightGBM-D HE": H._d2m(ld.predict_log(H.years(dte), H._lgbm_frame(setup, dte, X)), ld.smear, dte),
            "LightGBM-M HE": np.exp(lm.predict_log(H.years(mte), M.loc[mte].assign(month=mte.month))) * lm.smear * Dte,
        }

    # reload from disk with the shipped loader; compare with the in-memory models and the paper's run
    spec = importlib.util.spec_from_file_location("hes_forecast", OUT / "hes_forecast.py")
    hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf)
    models = hf.load(OUT)
    shutil.rmtree(OUT / "__pycache__", ignore_errors=True)
    paths = pickle.load(open("results/paths_seed0.pkl", "rb"))[("E1", "S1")]
    rows = []
    for v in VARIANTS:
        fc = hf.forecast(models, setup.X[v], mte)
        mem = in_memory(setup.X[v], setup.M[v])
        for mdl in fc.columns:
            ref = paths["P"][(mdl, v)]
            for month, f, g, r, a in zip(mte, fc[mdl].values, mem[mdl], ref, paths["actual"]):
                rows.append(dict(model=mdl, variant=v, month=month.strftime("%Y-%m"), forecast=f,
                                 reload_rel_diff=abs(f - g) / abs(g), paper_forecast=r,
                                 paper_rel_diff=abs(f - r) / abs(r), actual_S1=a))
    check = pd.DataFrame(rows)
    worst = check.groupby("model")[["reload_rel_diff", "paper_rel_diff"]].max()
    print("max relative difference, reloaded vs in-memory and vs the paper's run:\n" + worst.to_string())
    if (worst.reload_rel_diff > RTOL).any():
        raise SystemExit(f"reloaded models differ from the in-memory models by more than {RTOL}")
    if (worst.paper_rel_diff > PAPER_RTOL).any():
        raise SystemExit(f"refit differs from the paper's run by more than {PAPER_RTOL}")
    meta["max_rel_diff_to_paper_run"] = {k: float(v) for k, v in worst.paper_rel_diff.items()}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    check.drop(columns="reload_rel_diff").round({"forecast": 4, "paper_forecast": 4, "actual_S1": 4,
                                                 "paper_rel_diff": 6}).to_csv(OUT / "forecast_check.csv", index=False)
    ape = check.assign(ape=100 * (check.forecast - check.actual_S1).abs() / check.actual_S1,
                       ape_paper=100 * (check.paper_forecast - check.actual_S1).abs() / check.actual_S1)
    for col, label in (("ape", "these models"), ("ape_paper", "the paper's run")):
        tab = ape.pivot_table(index="model", columns="variant", values=col, aggfunc="mean")[VARIANTS].round(1)
        print(f"\nMAPE 2020-2021, scenario S1, replication 0, {label}:\n" + tab.to_string())

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    (OUT / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n"
                                            for p in files))
    for p in files:
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size // 1024} kB")
    return OUT


if __name__ == "__main__":
    print(main())
