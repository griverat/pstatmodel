import numpy as np
import pandas as pd
import pytest

from pstatmodel import utils

ecData = pd.read_csv("tests/data/ec_ersstv5.txt", sep=",", parse_dates=[0])
expected = pd.read_csv("tests/data/e_reformat.txt", sep=";")


def test_ShiftPredictor():
    result = utils.shift_predictor(ecData, "E", "08")
    np.array_equal(result, expected)


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
