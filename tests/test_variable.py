import pandas as pd
import pytest

from pstatmodel import ModelVariables, PredictorVariable
from pstatmodel.utils import DATA_CONTAINTER


@pytest.mark.parametrize(
    "predictor,source_data", DATA_CONTAINTER.items(), ids=DATA_CONTAINTER.keys()
)
def test_Variables(predictor, source_data):
    Variable = PredictorVariable(predictor, **source_data)
    assert Variable.raw_data is not None


@pytest.mark.parametrize(
    "predictor,source_data",
    DATA_CONTAINTER.items(),
    ids=[f"{k}_df" for k in DATA_CONTAINTER.keys()],
)
def test_from_dataframe(predictor, source_data):
    expected = PredictorVariable(predictor, **source_data)

    _input_data = (
        pd.concat(expected.raw_data, join="outer", axis=1)
        if isinstance(expected.raw_data, list)
        else expected.raw_data
    )
    _input_data = _input_data.loc[:, ~_input_data.columns.duplicated()]

    if predictor == "RMM":
        source_data["variable"] = [x for x in _input_data.columns if x != "time"]
    result = PredictorVariable.from_dataframe(
        predictor,
        source_data["variable"],
        _input_data,
    )

    if isinstance(result.raw_data, list):
        for res, exp in zip(result.raw_data, expected.raw_data):
            pd.testing.assert_frame_equal(res, exp)
    else:
        pd.testing.assert_frame_equal(result.raw_data, expected.raw_data)
    assert result.format == "custom"
    assert result.source == "user-generated"


def test_ModelVariables():
    Model = ModelVariables()
    assert Model.variables.keys() == DATA_CONTAINTER.keys()

    ecData = pd.read_csv("tests/data/ec_ersstv5.txt", parse_dates=[0])
    Model.register_variable("EC_index", ["E", "C"], ecData)
    assert "EC_index" in Model.variables.keys()
