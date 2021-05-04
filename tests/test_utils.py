import numpy as np
import pandas as pd

from pstatmodel import utils

ecData = pd.read_csv("tests/data/ec_ersstv5.txt", sep=",")
expected = pd.read_csv("tests/data/e_reformat.txt", sep=";")


def testShiftPredictor():
    result = utils.shift_predictor(ecData, "E", "08")
    np.array_equal(result, expected)
