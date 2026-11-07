#!/usr/bin/env python3
"""Stage the Hugging Face dataset thanatchon36/lockdowns-as-partial-holidays in dist/hf_dataset/.

Everything is CSV (the result pickles are flattened, since the Hub flags pickles as unsafe): the daily drivers
and shared regressors, the real OxCGRT and Google Mobility extracts, the simulated export series of replication 0,
its forecast paths, and the Monte Carlo results. The dataset card (README.md) and LICENSE are copied from
docs/hf_dataset/, and SHA256SUMS covers every file. Run from the repository root (`make dataset`); upload with
`hf upload thanatchon36/lockdowns-as-partial-holidays dist/hf_dataset . --repo-type dataset`.
"""
from __future__ import annotations

import hashlib
import pickle
import shutil
from pathlib import Path

import pandas as pd

RES, RAW, CARD = Path("results"), Path("data/raw"), Path("docs/hf_dataset")
OUT = Path("dist/hf_dataset")


def main() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("data/raw", "simulated", "results"):
        (OUT / sub).mkdir(parents=True)
    setup = pickle.load(open(RES / "setup.pkl", "rb"))
    paths = pickle.load(open(RES / "paths_seed0.pkl", "rb"))

    # daily drivers: calendar + COVID drivers + the raw workplace-mobility series behind the HESI
    mob = pd.read_csv(RAW / "gmob_thailand.csv", parse_dates=["date"]).set_index("date")["workplaces"]
    cal = setup["cal"][["H", "C", "weekend"]].astype(int)
    drv = setup["drv"].round(6) + 0.0  # + 0.0 turns -0.0 into 0.0
    daily = cal.join(mob.rename("workplaces_pct")).join(drv)
    daily.index.name = "date"
    daily.to_csv(OUT / "data/drivers_daily.csv", date_format="%Y-%m-%d")

    # shared stoppage regressors per dose variant, long format
    reg = (pd.concat(setup["X"], names=["variant", "date"]).round(6) + 0.0).reset_index()
    reg = reg[["date", "variant"] + [c for c in ("xD", "xP", "sD", "sP") if c in reg.columns]]
    reg.to_csv(OUT / "data/regressors_daily.csv", index=False, date_format="%Y-%m-%d")

    for f in ("oxcgrt_si.csv", "gmob_thailand.csv"):
        shutil.copyfile(RAW / f, OUT / "data/raw" / f)

    # simulated exports, replication 0
    y = paths["daily_S1"].rename("exports_S1").round(4)
    y.index.name = "date"
    y.to_csv(OUT / "simulated/exports_daily_S1_seed0.csv", date_format="%Y-%m-%d")
    monthly = pd.DataFrame({f"exports_{s}": v for s, v in paths["monthly"].items()}).round(4)
    monthly.index.name = "month"
    monthly.to_csv(OUT / "simulated/exports_monthly_seed0.csv", date_format="%Y-%m")

    # forecast paths of replication 0: one row per experiment, scenario, model, variant and month
    rows = []
    for key, d in paths.items():
        if not (isinstance(key, tuple) and isinstance(d, dict) and "P" in d):
            continue
        exp, scen = key
        for (model, variant), fc in d["P"].items():
            for month, actual, f in zip(d["months"], d["actual"], fc):
                rows.append(dict(exp=exp, scen=scen, model=model, variant=variant, month=month.strftime("%Y-%m"),
                                 actual=round(float(actual), 4), forecast=round(float(f), 4)))
    pd.DataFrame(rows).to_csv(OUT / "results/forecast_paths_seed0.csv", index=False)

    for f in ("metrics", "betas", "lamhat", "robust_metrics", "robust_betas"):
        shutil.copyfile(RES / f"{f}.csv", OUT / "results" / f"{f}.csv")
    for f in ("README.md", "LICENSE"):
        shutil.copyfile(CARD / f, OUT / f)

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files)
    (OUT / "SHA256SUMS").write_text(sums)
    for p in files:
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size // 1024} kB")
    return OUT


if __name__ == "__main__":
    print(main())
