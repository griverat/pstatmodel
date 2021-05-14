import numpy as np
import pandas as pd
import pytest

from pstatmodel import utils

ecData = pd.read_csv("tests/data/ec_ersstv5.txt", sep=",")
expected = pd.read_csv("tests/data/e_reformat.txt", sep=";")


def testShiftPredictor():
    result = utils.shift_predictor(ecData, "E", "08")
    np.array_equal(result, expected)


@pytest.mark.parametrize(
    "source_data", utils.DATA_CONTAINTER.values(), ids=utils.DATA_CONTAINTER.keys()
)
def testVariableFetcher(source_data):
    if source_data["format"] == "long":
        raw_data = utils.parse_fwf(**source_data)
    elif source_data["format"] == "wide":
        raw_data = utils.wide_to_long(**source_data)
    else:
        raw_data = None
    assert raw_data is not None


# taken from https://stackoverflow.com/a/38778401
def assert_series_not_equal(*args, **kwargs):
    try:
        pd.testing.assert_series_equal(*args, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError


def test_decadeResampler():
    df = utils.parse_fwf(**utils.DATA_CONTAINTER["RMM"])
    resampled = utils.resampleToDecade(df)
    assert_series_not_equal(resampled["time"], df["time"])


def test_splitByDay():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2010-01-01", "2010-12-31", freq="1D"),
            "col1": np.random.randn(365),
            "col2": np.random.randn(365),
        }
    )
    result = utils.splitByDay(df)[0]
    expected = df[df["time"].dt.day == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)
