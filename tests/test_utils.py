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
