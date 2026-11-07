# Data

| File | Content | Source | Terms |
|---|---|---|---|
| `oxcgrt_si.csv` | Daily national Stringency Index for Thailand and China (subset of the OxCGRT time series, Jan 2020 - Feb 2023) | Oxford COVID-19 Government Response Tracker, <https://github.com/OxCGRT/covid-policy-dataset>, file `data/timeseries_indices/OxCGRT_timeseries_StringencyIndex_v1.csv` | Check the licence in the OxCGRT repository; cite Hale et al. (2021) |
| `gmob_thailand.csv` | Google COVID-19 Community Mobility Reports, Thailand, national level, 15 Feb 2020 - 19 Aug 2022 | Google LLC, via the mirror <https://github.com/ActiveConclusion/COVID19_mobility> | Attribute "Google LLC, Google COVID-19 Community Mobility Reports"; see Google's terms |

Holiday calendars are generated at run time with the `holidays` package.
If a file is removed, the code downloads the full source file again (the mobility mirror is about 105 MB; only the Thailand rows are kept).
Official export statistics are **not** included; see the main README for the CSV format expected by the `real` command.
