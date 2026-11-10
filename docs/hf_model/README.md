---
license: cc-by-4.0
library_name: prophet
pipeline_tag: time-series-forecasting
datasets:
  - thanatchon36/lockdowns-as-partial-holidays
language:
  - en
tags:
  - time-series-forecasting
  - forecasting
  - prophet
  - lightgbm
  - sarimax
  - statsmodels
  - thailand
  - exports
  - covid-19
  - holidays
  - calendar-effects
  - simulated
---

# Lockdowns as Partial Holidays: the fitted HES models

> **Trained on simulated exports.** These models were fitted to the simulated daily export series of replication 0
> of the paper's Monte Carlo study (scenario S1, 2010-2019). They reproduce the paper's shock experiment and show
> the method at work; they are **not** a forecaster of official Thai exports. To apply the method to official data,
> run `hesi real` from the [code repository](https://github.com/thanatchon36/lockdowns-as-partial-holidays),
> which refits every model on your series.

Models for the preprint *Lockdowns as Partial Holidays: Identifying Work-Stoppage Elasticities from Recurring
Holidays and Transferring Them to Forecast Thailand's Monthly Exports through the COVID-19 Shock* by Thanatchon
Chongmankhong ([read online](https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/) ·
[PDF](https://github.com/thanatchon36/lockdowns-as-partial-holidays/blob/main/paper/paper.pdf) ·
[ฉบับภาษาไทย](https://github.com/thanatchon36/lockdowns-as-partial-holidays/blob/main/paper/thai/paper_th.pdf)).
Data: [`thanatchon36/lockdowns-as-partial-holidays`](https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays).
Archived on Zenodo: [10.5281/zenodo.24036954](https://doi.org/10.5281/zenodo.24036954).

*Independent personal research. Does not represent the views of any employer or institution.*

## The idea

A lockdown is treated as a partial holiday. Each model carries one shared stoppage regressor per channel:

```
x^D_d = H_d + (1 - H_d) * dose_d      domestic: H_d = 1 on Thai public holidays (Songkran included)
x^P_d = C_d + (1 - C_d) * dose_d      partner:  C_d = 1 in the Chinese New Year window
```

Before 2020 every dose is zero, so the elasticities `beta_D`, `beta_P` are learned from ten years of holidays
alone. To forecast a lockdown, only the dose changes; no new parameter is estimated. The dose of the paper is the
Holiday-Equivalent Stoppage Index (HESI): the drop in workplace mobility on a day divided by its median drop on a
weekday public holiday.

## Files

| file | model |
|---|---|
| `prophet_daily_he.json` | **Prophet-D HE**: daily Prophet on log exports with regressors `xD`, `xP` (prophet.serialize JSON) |
| `prophet_monthly_he.json` | **Prophet-M HE**: monthly Prophet on log average daily exports with `MxD`, `MxP` (monthly means of the daily regressors) and `WK` (weekend share). **Prophet-M HE+AC** is the same model with the closed-form aggregation correction: daily-equivalent elasticities in `meta.json` |
| `sarimax_he.json` | **SARIMAX HE**: airline model (0,1,1)(0,1,1)12 on log average daily exports with `MxD`, `MxP`, `WK`; parameters and the 120 training months, rebuilt with statsmodels `SARIMAX(...).filter(params)` |
| `lightgbm_daily_he.txt`, `lightgbm_monthly_he.txt` | **LightGBM-D HE**, **LightGBM-M HE**: linear log-trend plus LightGBM on the detrended residual (native text models); trend coefficients in `meta.json` |
| `meta.json` | regressors and coefficients (log units), Duan smearing factors, trend coefficients, the aggregation-corrected elasticities, training window, seed, library versions |
| `hes_forecast.py` | standalone loader and forecaster (numpy, pandas, prophet, lightgbm, statsmodels; no pickle) |
| `forecast_check.csv` | 2020-2021 forecasts of every model and dose variant, the paper's forecasts for the same replication, and the simulated actuals |
| `SHA256SUMS` | checksums of every file above |

Nothing is pickled: Prophet loads from JSON, LightGBM from its text format and SARIMAX from its parameters.

## Elasticities (replication 0)

The true daily elasticities of the simulation are `beta_D = log(0.45) = -0.799` and `beta_P = log(0.80) = -0.223`.

| model | `beta_D` | `beta_P` |
|---|---:|---:|
| Prophet-D HE (daily) | -0.800 | -0.217 |
| Prophet-M HE (monthly, attenuated) | -0.613 | -0.219 |
| Prophet-M HE+AC (aggregation-corrected) | -0.880 | -0.240 |
| SARIMAX HE (monthly) | -0.752 | -0.256 |

Monthly data attenuate the daily elasticity towards `exp(beta) - 1` (Jensen); the paper derives the attenuation and
the correction. Over the 30 replications the means are -0.80 (daily), -0.54 (monthly) and -0.76 (corrected).

## Use

```python
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

d = snapshot_download("thanatchon36/lockdowns-as-partial-holidays")
import sys; sys.path.insert(0, d)
import hes_forecast as hf

models = hf.load(d)
reg = pd.read_csv(hf_hub_download("thanatchon36/lockdowns-as-partial-holidays", "data/regressors_daily.csv",
                                  repo_type="dataset"), parse_dates=["date"])
X = reg[reg.variant == "hesi"].set_index("date")[["xD", "xP"]]      # the HESI dose
months = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
fc = hf.forecast(models, X, months)                                  # one column per model
```

Other doses: `variant` `off` (no dose), `binary` (whole-month switch), `stringency` (raw OxCGRT), `oracle_S1`,
`oracle_S2`, or any daily `xD`, `xP` you build yourself (in 0..1, equal to 1 on holidays).

## How well they forecast the shock (replication 0)

MAPE of monthly exports, January 2020 - December 2021, scenario S1, forecasts made at the end of 2019:

| model | no dose | binary switch | raw stringency | HESI dose |
|---|---:|---:|---:|---:|
| Prophet-D HE | 30.3% | 26.3% | 24.9% | **8.5%** |
| Prophet-M HE | 29.5% | 24.0% | 17.9% | 11.2% |
| Prophet-M HE+AC | 29.6% | 26.3% | 29.1% | **7.4%** |
| SARIMAX HE | 26.4% | 22.1% | 26.4% | **7.1%** |
| LightGBM-D HE | 16.3% | 15.3% | 43.8% | 42.7% |
| LightGBM-M HE | 14.9% | 14.2% | 13.6% | 13.7% |

One replication only; the paper's tables average 30 (daily Prophet: 26.0% without a dose, 7.8% with the HESI dose).
LightGBM cannot transfer a dose: its trees split the stoppage feature at 1e-35, so any positive dose becomes a full
holiday.

## Reproducibility

`make model` in the code repository refits these models from the simulated series (identical to the paper's
replication 0 to machine precision), saves them, reloads them with `hes_forecast.py` and checks that the reloaded
forecasts equal the in-memory ones. They were fitted on macOS arm64 with prophet 1.5.0, lightgbm 4.7.0,
statsmodels 0.15.0, numpy 2.5.4 and pandas 3.1.1. LightGBM reproduces the paper's run exactly. The Prophet and
SARIMAX optimizers land on slightly different solutions on a different platform: the largest monthly difference from
the paper's forecasts is 0.1-0.5% for Prophet and 1.4% for SARIMAX, and no MAPE above moves by more than 0.7
points. `forecast_check.csv` lists every month.

## Licence

CC BY 4.0 (© 2026 Thanatchon Chongmankhong). The models hold only simulated exports and the Thai and Chinese
holiday calendars (training doses are zero). The HESI dose in the dataset is derived from Google COVID-19 Community
Mobility Reports and remains under Google's terms; see the dataset's licence.

## Citation

```bibtex
@misc{chongmankhong2026lockdowns,
  author = {Chongmankhong, Thanatchon},
  title  = {Lockdowns as Partial Holidays: Identifying Work-Stoppage Elasticities from Recurring Holidays
            and Transferring Them to Forecast {Thailand}'s Monthly Exports through the {COVID-19} Shock},
  year   = {2026},
  doi    = {10.5281/zenodo.24036954},
  note   = {Preprint; export values are simulated},
  url    = {https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/}
}
```
