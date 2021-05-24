import numpy as np
import pandas as pd
import pytest

from pstatmodel import utils

ecData = pd.read_csv("tests/data/ec_ersstv5.txt", sep=",", parse_dates=[0])
expected = pd.read_csv("tests/data/e_reformat.txt", sep=";")


def test_shiftPredictor():
    _dates = pd.date_range("2000-01-02", "2016-12-31", freq="MS") + pd.Timedelta("14D")

    df = pd.DataFrame(
        {
            "time": _dates,
            "col1": np.random.randn(_dates.size),
            "col2": np.random.randn(_dates.size),
        }
    )

    result = utils.shift_predictor(df, "col1", "08").loc[2006]
    expected = df.query("(time>'2005-08-01')&(time<'2006-08-01')")["col1"].copy()
    expected.index = [
        "col1_Agosto",
        "col1_Setiembre",
        "col1_Octubre",
        "col1_Noviembre",
        "col1_Diciembre",
        "col1_Enero",
        "col1_Febrero",
        "col1_Marzo",
        "col1_Abril",
        "col1_Mayo",
        "col1_Junio",
        "col1_Julio",
    ]
    expected.name = 2006
    pd.testing.assert_series_equal(result, expected)

    result = utils.shift_predictor(df[df.time.dt.month != 12], "col1", "08").loc[2006]
    expected.iloc[4] = np.nan
    pd.testing.assert_series_equal(result, expected)

    result = utils.shift_predictor(
        df[df.time.dt.month != 12], "col1", "08", use_seasons=True
    ).loc[2006]
    expected.index = [
        "col1_JAS",
        "col1_ASO",
        "col1_SON",
        "col1_OND",
        "col1_NDJ",
        "col1_DJF",
        "col1_JFM",
        "col1_FMA",
        "col1_MAM",
        "col1_AMJ",
        "col1_MJJ",
        "col1_JJA",
    ]
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "source_data", utils.DATA_CONTAINTER.values(), ids=utils.DATA_CONTAINTER.keys()
)
def test_VariableFetcher(source_data):
    if source_data["format"] == "long":
        raw_data = utils.parse_fwf(**source_data)
    elif source_data["format"] == "wide":
        raw_data = utils.wide_to_long(**source_data)
    else:
        raw_data = None
    assert raw_data is not None


def test_decadeResampler():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2010-01-02", "2010-12-31", freq="1D"),
            "col1": np.random.randn(364),
            "col2": np.random.randn(364),
        }
    )
    result = utils.decadeResampler(df).set_index("time").index
    expected = pd.date_range("2010-01-01", "2010-12-31", freq="1MS", name="time")
    expected = expected.union(expected + pd.Timedelta("10D")).union(
        expected + pd.Timedelta("20D")
    )
    pd.testing.assert_index_equal(result, expected)


def test_monthResampler():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2010-01-02", "2010-12-31", freq="1D"),
            "col1": np.random.randn(364),
            "col2": np.random.randn(364),
        }
    )
    result = utils.monthResampler(df).set_index("time").index
    expected = pd.date_range(
        "2010-01-01", "2010-12-31", freq="1MS", name="time"
    ) + pd.Timedelta("14D")
    pd.testing.assert_index_equal(result, expected)


def test_splitByDay():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2010-01-01", "2010-12-31", freq="1D"),
            "col1": np.random.randn(365),
            "col2": np.random.randn(365),
        }
    )

    DAY_MAP = {date: f"{date}d" for date in range(1, 32)}

    result = utils.splitByDay(df, DAY_MAP=DAY_MAP)[14]
    expected = df[df["time"].dt.day == 15].reset_index(drop=True)
    expected.columns = ["time", "col1_15d", "col2_15d"]
    pd.testing.assert_frame_equal(result, expected)
