from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd

from pstatmodel.utils import (
    DATA_CONTAINTER,
    decadeResampler,
    monthResampler,
    parse_fwf,
    splitByDay,
    wide_to_long,
)

DATA_PARSER = dict(wide=wide_to_long, long=parse_fwf, custom=None)
DATA_RESAMPLER = dict(months=monthResampler, decades=decadeResampler)


def default_variables():
    return {
        name: PredictorVariable(name, **var_args)
        for name, var_args in DATA_CONTAINTER.items()
    }


@dataclass
class PredictorVariable:
    predictor: str
    source: str
    variable: Union[str, List[str]]
    format: str
    parse_kwargs: dict
    raw_data: Union[List[pd.DataFrame], pd.DataFrame, None] = field(
        default=None, repr=False
    )
    columns: dict[str, str] = None
    FILL_VALUE: float = None
    timefix: bool = True
    webscrap: bool = False
    resample: List[str] = field(default_factory=list)
    use_seasons: bool = False
    period: List[int] = field(default_factory=lambda: [-12, 12])

    def __post_init__(self) -> None:
        _parser = DATA_PARSER[self.format]
        if _parser is not None:
            raw_data = _parser(
                source=self.source,
                variable=self.variable,
                parse_kwargs=self.parse_kwargs,
                columns=self.columns,
                FILL_VALUE=self.FILL_VALUE,
                timefix=self.timefix,
                webscrap=self.webscrap,
            )
        else:
            raw_data = self.raw_data
        if len(self.resample) != 0:
            _resampled = []
            for method in self.resample:
                _result = DATA_RESAMPLER[method](raw_data)
                if method == "decades":
                    _result = splitByDay(_result)
                _resampled = (
                    _resampled + _result
                    if isinstance(_result, list)
                    else _resampled + [_result]
                )
        else:
            _resampled = raw_data

        self.raw_data = _resampled[0] if len(_resampled) == 1 else _resampled

    @classmethod
    def from_dataframe(cls, predictor, variable, dataframe, **kwargs):
        return cls(
            predictor=predictor,
            source="user-generated",
            variable=variable,
            format="long",
            raw_data=dataframe,
            **kwargs
        )


@dataclass
class ModelVariables:
    variables: dict[str, PredictorVariable] = field(default_factory=default_variables)

    def register_variable(self, var_name, variable, table, **kwargs):
        self.variables[var_name] = PredictorVariable.from_dataframe(
            var_name, variable, table, **kwargs
        )
