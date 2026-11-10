---
license: other
license_name: cc-by-4.0-with-third-party-terms
license_link: LICENSE
pretty_name: "Lockdowns as Partial Holidays: holiday-equivalent stoppage data and Monte Carlo results for Thai exports"
language:
  - en
  - th
size_categories:
  - 10K<n<100K
task_categories:
  - time-series-forecasting
tags:
  - forecasting
  - thailand
  - exports
  - covid-19
  - holidays
  - calendar-effects
  - mobility
  - prophet
  - monte-carlo
  - simulated
configs:
  - config_name: drivers_daily
    data_files: data/drivers_daily.csv
    default: true
  - config_name: regressors_daily
    data_files: data/regressors_daily.csv
  - config_name: exports_monthly_seed0
    data_files: simulated/exports_monthly_seed0.csv
  - config_name: forecast_paths_seed0
    data_files: results/forecast_paths_seed0.csv
  - config_name: metrics
    data_files: results/metrics.csv
  - config_name: betas
    data_files: results/betas.csv
  - config_name: robust_metrics
    data_files: results/robust_metrics.csv
---

# Lockdowns as Partial Holidays: data and Monte Carlo results

> **The export values are simulated.** Every series under `simulated/` and every forecast and score under
> `results/` come from a Monte Carlo study whose export values are generated. The drivers in `data/` are real: the
> observed Thai holiday calendar, the OxCGRT stringency indices and the Google Community Mobility data. The one
> exception in `data/` is `e_cn`, a stylised path of Chinese industrial stoppage (see below). Nothing here is a
> result on official Thai export statistics.

Data for the preprint *Lockdowns as Partial Holidays: Identifying Work-Stoppage Elasticities from Recurring
Holidays and Transferring Them to Forecast Thailand's Monthly Exports through the COVID-19 Shock* by Thanatchon
Chongmankhong ([read online](https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/) ·
[PDF](https://github.com/thanatchon36/lockdowns-as-partial-holidays/blob/main/paper/paper.pdf) ·
[ฉบับภาษาไทย](https://github.com/thanatchon36/lockdowns-as-partial-holidays/blob/main/paper/thai/paper_th.pdf)).
The paper treats a lockdown as a partial holiday. Recurring holidays (Thai public holidays including Songkran, and
the Chinese New Year window) identify a work-stoppage elasticity through one shared regressor per channel in
Prophet. During the pandemic the same regressor carries a dose, the Holiday-Equivalent Stoppage Index (HESI): the
drop in workplace mobility on a day divided by the median drop on a weekday public holiday. Code:
<https://github.com/thanatchon36/lockdowns-as-partial-holidays>; fitted models:
[`thanatchon36/lockdowns-as-partial-holidays`](https://huggingface.co/thanatchon36/lockdowns-as-partial-holidays); archived on Zenodo:
[10.5281/zenodo.24036954](https://doi.org/10.5281/zenodo.24036954).

*Independent personal research. Does not represent the views of any employer or institution.*

## Files

| path | content |
|---|---|
| `data/drivers_daily.csv` | **real, except `e_cn`.** One row per day, 2010-01-01 to 2024-12-31 (5,479 rows): the calendar and every COVID driver the models use. Columns below |
| `data/regressors_daily.csv` | the shared stoppage regressors `xD` (domestic) and `xP` (partner) for each dose variant, long format (38,353 rows); variant `sep` adds the separate COVID regressors `sD` and `sP` |
| `data/raw/oxcgrt_si.csv` | **real.** OxCGRT national Stringency Index, Thailand and China, January 2020 to February 2023, in the source's wide layout (one column per day) |
| `data/raw/gmob_thailand.csv` | **real.** Google COVID-19 Community Mobility Reports, Thailand, national level, 15 February 2020 to 19 August 2022, % change against the baseline for six place categories |
| `simulated/exports_daily_S1_seed0.csv` | **simulated.** Daily export values of replication 0, scenario S1 |
| `simulated/exports_monthly_seed0.csv` | **simulated.** Monthly export values of replication 0, scenarios S1 and S2, 2010-01 to 2024-12 |
| `results/forecast_paths_seed0.csv` | **simulated.** Replication 0: monthly forecast and actual by experiment, scenario, model and dose variant (3,000 rows) |
| `results/metrics.csv` | **simulated.** All 30 replications: MAPE, RMSE and MASE by seed, scenario, experiment, model, variant and window, plus the dose sweep (27,690 rows) |
| `results/betas.csv` | **simulated.** Estimated domestic and partner elasticities `beta_D`, `beta_P` by seed, experiment, scenario, model and variant |
| `results/lamhat.csv` | **simulated.** Dose multiplier calibrated on March-April 2020, by seed, scenario and model |
| `results/robust_metrics.csv`, `results/robust_betas.csv` | **simulated.** Robustness experiments R1 (heterogeneous holidays: one-tier against two-tier festival specification) and R2 (centred against trailing, real-time HESI) |
| `SHA256SUMS` | checksums of every file above |

### `data/drivers_daily.csv`

| column | meaning |
|---|---|
| `H` | 1 on an observed Thai public holiday (day off), from the `holidays` package. Observed dates: Songkran 2020 was postponed and replaced by 27 July, 4 and 7 September 2020 |
| `C` | 1 in the Chinese New Year window, one day before to six days after (eight days) |
| `weekend` | 1 on Saturday and Sunday |
| `workplaces_pct` | Google Mobility, workplaces, % change against the baseline (empty outside 15 Feb 2020 - 19 Aug 2022) |
| `s_th`, `s_cn` | OxCGRT Stringency Index / 100, Thailand and China |
| `hesi_th` | the Thai HESI in 0..1: the workplace-mobility drop divided by its median drop on weekday public holidays (0.47), smoothed over 7 days with holiday days left out |
| `e_cn` | **stylised, not measured.** Chinese industrial stoppage in Chinese-New-Year units: the 2020 national lockdown and the 2022 Shanghai lockdown. OxCGRT's China index stays high for three years under zero-COVID while factories mostly ran, so it is not used as a stoppage measure |
| `bin_th`, `bin_cn` | whole-month binary switches: 1 for every day of a month whose mean driver exceeds 0.70 (Thailand) or 0.50 (China) |

### Codes

- **Scenarios**: `S1` ideal transfer, the HESI is the true export stoppage; `S2` realistic, mobility overstates the
  true stoppage two-fold and a demand shock is present.
- **Experiments**: `E0` normal times (forecasts of 2018-2019), `E1` the shock (2020-2021, made in December 2019),
  `E2` after the shock (2023-2024). `R1`, `R2` robustness.
- **Windows**: `W24` January 2020 - December 2021 (all 24 months), `W20` May 2020 - December 2021 (after the
  March-April 2020 calibration months).
- **Models**: `SNaive` seasonal naive; `Prophet-D`, `Prophet-M` daily and monthly Prophet; `HE` with the shared
  stoppage regressors; `+AC` with the closed-form aggregation correction; `SARIMAX`; `LightGBM-D`, `LightGBM-M`.
- **Variants**: `off` no COVID dose; `binary` whole-month switch; `stringency` raw OxCGRT; `hesi` HESI dose with
  lambda = 1; `hesi-cal` dose calibrated on March-April 2020 (lambda in `lam`); `oracle` the true pass-through;
  `lam` the dose sweep, lambda in `lam` from 0 to 3; `sep` separate COVID regressors; `mask` the COVID window
  masked; `cal` holiday regressors only, COVID ignored; `vanilla` no calendar regressors; `-` not applicable.

## Headline numbers

Daily Prophet, MAPE over 2020-2021 (`exp == "E1"`, `window == "W24"`, mean over 30 seeds): no dose 26.0% (S1) and
14.7% (S2); HESI dose 7.8% and 10.9%; oracle 7.8% and 8.3%. To reproduce:

```python
from datasets import load_dataset

m = load_dataset("thanatchon36/lockdowns-as-partial-holidays", "metrics", split="train").to_pandas()
sel = m[(m.exp == "E1") & (m.window == "W24") & (m.model == "Prophet-D HE")]
print(sel.groupby(["scen", "variant"]).mape.mean().round(1).unstack(0))
```

`make simulate robust` in the code repository regenerates `results/`, and `make dataset` rebuilds these files.

## Licence

`LICENSE` gives the terms. The simulated series, the derived regressors and all results are CC BY 4.0
(© 2026 Thanatchon Chongmankhong). `data/raw/oxcgrt_si.csv` and the `s_th`, `s_cn` columns come from the Oxford
COVID-19 Government Response Tracker, CC BY 4.0; cite Hale et al. (2021). `data/raw/gmob_thailand.csv` and the
`workplaces_pct` and `hesi_th` columns come from the Google COVID-19 Community Mobility Reports and remain under
Google's terms; attribute "Google LLC, Google COVID-19 Community Mobility Reports".

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

Hale, T., Angrist, N., Goldszmidt, R., et al. (2021). A global panel database of pandemic policies (Oxford
COVID-19 Government Response Tracker). *Nature Human Behaviour*, 5, 529-538.
