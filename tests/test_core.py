"""Unit tests for the core HES building blocks."""
import numpy as np
import pandas as pd

from hesi import forecast as H


def test_songkran_2020_postponed_to_substitute_days():
    hol = H.thai_holidays(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    assert not any((d.month == 4) and (13 <= d.day <= 15) for d in hol)
    for d in ("2020-07-27", "2020-09-04", "2020-09-07"):
        assert pd.Timestamp(d) in hol


def test_cny_window_has_eight_days():
    w = H.cny_window(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"))
    assert len(w) == 8 and w.min() == pd.Timestamp("2019-02-04")


def test_shared_regressors_are_bounded_and_one_on_holidays():
    cal = H.build_calendar(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31"))
    drv = pd.DataFrame({"s_th": 0.5, "s_cn": 0.7, "hesi_th": np.linspace(0, 1.5, len(cal)), "e_cn": 0.3,
                        "bin_th": 0.0, "bin_cn": 1.0}, index=cal.index)
    for v in ("off", "binary", "stringency", "hesi"):
        X = H.stoppage_X(cal, drv, v, lam=2.0)
        assert X.xD.between(0, 1).all() and X.xP.between(0, 1).all()
        assert (X.xD[cal.H == 1] == 1).all()


def test_jensen_concentrated_vs_diffuse_months():
    Mx = np.array([[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]], float)
    E = H.log_mean_exp(-0.8, Mx)
    assert E[0] > -0.8 * 0.25              # concentrated stoppage is attenuated
    assert np.isclose(E[1], -0.8 * 0.25)   # diffuse stoppage aggregates exactly


def test_aggregation_correction_inverts_closed_form():
    idx = pd.date_range("2010-01-01", "2019-12-31")
    x = pd.Series(((idx.month == 4) & idx.day.isin([13, 14, 15])).astype(float), index=idx)
    beta = -0.8
    Mx = H.month_matrix(x)
    xbar = np.nanmean(Mx, axis=1)
    beta_m = np.sum(xbar * H.log_mean_exp(beta, Mx)) / np.sum(xbar ** 2)
    assert abs(beta_m - (np.exp(beta) - 1)) < 0.03
    assert abs(H.deaggregate_beta(beta_m, x) - beta) < 0.02


def test_prophet_forecast_is_linear_in_regressors():
    idx = pd.date_range("2012-01-01", periods=96, freq="MS")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"MxD": rng.random(96) * 0.2}, index=idx)
    ly = 6 + 0.01 * np.arange(96) - 0.6 * X.MxD.values + rng.normal(0, 0.02, 96)
    m = H.ProphetLog("M", ["MxD"]).fit(idx, ly, X)
    fut = pd.date_range("2020-01-01", periods=12, freq="MS")
    Xf = pd.DataFrame({"MxD": np.linspace(0, 0.6, 12)}, index=fut)
    direct = m.m.predict(pd.DataFrame({"ds": fut, "MxD": Xf.MxD.values})).yhat.values
    assert np.allclose(direct, m.predict_log(m.base0(fut), Xf), atol=1e-8)
