import numpy as np
import pandas as pd

from pstatmodel import utils

ecData = pd.read_csv("tests/data/ec_ersstv5.txt", sep=",")
expected = pd.read_csv("tests/data/e_reformat.txt", sep=";")


def testShiftPredictor():
    result = utils.shift_predictor(ecData, "E", "08")
    np.array_equal(result, expected)


def testVariableFetcher():
    for pargs in utils.DATA_CONTAINTER.values():
        if pargs["format"] == "long":
            raw_data = utils.parse_fwf(**pargs)
        elif pargs["format"] == "wide":
            raw_data = utils.wide_to_long(**pargs)
        else:
            raw_data = None
        assert raw_data is not None
