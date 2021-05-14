import json

import numpy as np
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

iTEST = 1
fTEST = 13


def shiftData(case, predictors=predictors, testset=testSetPisco, expected=expectedDict):
    testsetCase = testset[f"TestSet{case}"]
    expectedCase = expected[f"expectedResult{case}"]
    if expectedCase["pvalue"] == "nan":
        expectedCase["pvalue"] = np.nan
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


@pytest.mark.parametrize(
    "testSetOld,predictorsOld,expectedOld",
    [shiftData(x) for x in range(iTEST, fTEST)],
)
def test_base_old(testSetOld, predictorsOld, expectedOld):
    model = (
        base_old.stepwise_selection(
            predictorsOld,
            testSetOld,
            threshold_in=0.05,
            threshold_out=0.1,
            max_vars=12,
            min_vars=4,
            verbose=True,
        ),
    )
    assert model[0][0] == expectedOld["vars"]
    np.testing.assert_equal(model[0][2], expectedOld["pvalue"])


@pytest.mark.parametrize(
    "testSet,predictors,expected", [shiftData(x) for x in range(iTEST, fTEST)]
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
    np.testing.assert_equal(model[0][2], expected["pvalue"])
