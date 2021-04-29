import json
from functools import wraps
from time import time

import pandas as pd
import pytest

from pstatmodel.stepwise import base, base_old

predictors = pd.read_excel(
    "tests/data/Predcitores_IniJul_Gerardo_ec.xlsx", engine="openpyxl"
)

testSetPisco = pd.read_excel(
    "tests/data/testSet.xlsx", skiprows=1, nrows=37, engine="openpyxl"
)

with open("tests/data/expectedResult.json", "r") as JSON:
    expectedDict = json.load(JSON)


def shiftData(case, predictors=predictors, testset=testSetPisco, expected=expectedDict):
    testsetCase = testset[f"TestSet{case}"]
    expectedCase = expected[f"expectedResult{case}"]
    month = testsetCase.iloc[0]
    if month in [10, 11, 12]:
        return (
            testsetCase.iloc[1:-1].reset_index(drop=True),
            predictors.iloc[5:40].reset_index(drop=True),
            expectedCase,
        )
    return (
        testsetCase.iloc[1:].reset_index(drop=True),
        predictors.iloc[4:40].reset_index(drop=True),
        expectedCase,
    )


# https://stackoverflow.com/a/51503837
def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it


# @measure
@pytest.mark.parametrize(
    "testSet_old,predictors_old,expected_old", [shiftData(x) for x in range(1, 9)]
)
def test_base_old(testSet_old, predictors_old, expected_old):
    model = (
        base_old.stepwise_selection(
            predictors_old,
            testSet_old,
            threshold_in=0.05,
            threshold_out=0.1,
            max_vars=12,
            min_vars=4,
            verbose=True,
        ),
    )
    assert model[0][0] == expected_old["vars"]
    assert model[0][2] == expected_old["pvalue"]


@pytest.mark.parametrize(
    "testSet,predictors,expected", [shiftData(x) for x in range(1, 9)]
)
def test_base(testSet, predictors, expected):
    model = (
        base.stepwise_selection(
            predictors,
            testSet,
            threshold_in=0.05,
            threshold_out=0.1,
            max_vars=12,
            min_vars=4,
            verbose=True,
        ),
    )
    assert model[0][0] == expected["vars"]
    assert model[0][2] == expected["pvalue"]
