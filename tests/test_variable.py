import pandas as pd
import pytest

from pstatmodel import ModelVariables, PredictorVariable
from pstatmodel.utils import DATA_CONTAINTER, monthResampler


@pytest.mark.parametrize(
    "predictor,source_data", DATA_CONTAINTER.items(), ids=DATA_CONTAINTER.keys()
)
def test_Variables(predictor, source_data):
    Variable = PredictorVariable(predictor, **source_data)
    assert Variable.raw_data is not None


def month_names(name, seasons=False):
    if seasons:
        return [
            f"{name}_JAS",
            f"{name}_ASO",
            f"{name}_SON",
            f"{name}_OND",
            f"{name}_NDJ",
            f"{name}_DJF",
            f"{name}_JFM",
            f"{name}_FMA",
            f"{name}_MAM",
            f"{name}_AMJ",
            f"{name}_MJJ",
            f"{name}_JJA",
        ]

    else:
        return [
            f"{name}_Agosto",
            f"{name}_Setiembre",
            f"{name}_Octubre",
            f"{name}_Noviembre",
            f"{name}_Diciembre",
            f"{name}_Enero",
            f"{name}_Febrero",
            f"{name}_Marzo",
            f"{name}_Abril",
            f"{name}_Mayo",
            f"{name}_Junio",
            f"{name}_Julio",
        ]


@pytest.mark.parametrize(
    "predictor,source_data", DATA_CONTAINTER.items(), ids=DATA_CONTAINTER.keys()
)
def test_ShiftVariables(predictor, source_data):
    Variable = PredictorVariable(predictor, **source_data)
    Variable.shiftData(init_month="08", fyear=2022)
    if isinstance(Variable.shifted_data, list):
        for df, df_shifted in zip(Variable.shifted_data, Variable.raw_data):
            df_shifted = monthResampler(df_shifted)
            expected = (
                df_shifted.query("(time>'2005-08-01')&(time<'2006-08-01')")
                .copy()
                .iloc[:, 1]
            )
            expected.index = month_names(expected.name, Variable.use_seasons)
            expected.name = 2006
            pd.testing.assert_series_equal(df.loc[2006], expected.loc[df.columns])
    else:
        expected = (
            monthResampler(Variable.raw_data)
            .query("(time>'2005-08-01')&(time<'2006-08-01')")
            .copy()
            .iloc[:, 1]
        )
        expected.index = month_names(expected.name, Variable.use_seasons)
        expected.name = 2006
        pd.testing.assert_series_equal(
            Variable.shifted_data.loc[2006], expected.loc[Variable.shifted_data.columns]
        )


@pytest.mark.parametrize(
    "predictor,source_data",
    DATA_CONTAINTER.items(),
    ids=DATA_CONTAINTER.keys(),
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
