from functools import wraps
from io import StringIO
from time import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests

DATA_CONTAINTER = {
    "AAO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii",
        "parse_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AAO"},
        "variable": "AAO",
        "format": "long",
    },
    "AO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        "parse_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AO"},
        "variable": "AO",
        "format": "long",
    },
    "PMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/PMM.txt",
        "parse_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "PMM"},
        "variable": "PMM",
        "format": "long",
    },
    "AMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/AMM.txt",
        "parse_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "AMM"},
        "variable": "AMM",
        "format": "long",
    },
    "TNA": {
        "source": "https://psl.noaa.gov/data/correlation/tna.data",
        "parse_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "variable": "TNA",
        "format": "wide",
    },
    "TSA": {
        "source": "https://psl.noaa.gov/data/correlation/tsa.data",
        "parse_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "variable": "TSA",
        "format": "wide",
    },
    "NAO": {
        "source": "https://psl.noaa.gov/data/correlation/nao.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=3,
            header=None,
            widths=[
                5,
            ]
            + [7] * 12,
        ),
        "variable": "NAO",
        "format": "wide",
        "FILL_VALUE": -99.9,
    },
    "EP/NP": {
        "source": "https://psl.noaa.gov/data/correlation/epo.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=3,
            header=None,
            widths=[
                5,
            ]
            + [7] * 12,
        ),
        "variable": "EP/NP",
        "format": "wide",
        "FILL_VALUE": -99.9,
    },
    "WP": {
        "source": "https://psl.noaa.gov/data/correlation/wp.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=3,
            header=None,
            widths=[
                5,
            ]
            + [7] * 12,
        ),
        "variable": "WP",
        "format": "wide",
        "FILL_VALUE": -99.9,
    },
    "AMI": {
        "source": "https://psl.noaa.gov/data/correlation/amon.us.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=4,
            header=None,
            widths=[
                5,
            ]
            + [9] * 12,
        ),
        "variable": "AMI",
        "format": "wide",
        "FILL_VALUE": -99.99,
    },
    "SOI": {
        "source": "https://psl.noaa.gov/data/correlation/soi.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=3,
            header=None,
            widths=[
                5,
            ]
            + [7] * 12,
        ),
        "variable": "SOI",
        "format": "wide",
        "FILL_VALUE": -99.99,
    },
    "RMM": {
        "source": "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
        "parse_kwargs": dict(
            skiprows=2,
            parse_dates={"time": [0, 1, 2]},
            names=[
                "year",
                "month",
                "day",
                "RMM1",
                "RMM2",
                "phase",
                "amplitude",
                "Final_Value",
            ],
            widths=[12, 12, 12, 16, 16, 12, 16, 32],
        ),
        "timefix": False,
        "format": "long",
        "webscrap": True,
        "variable": ["RMM1", "RMM2", "amplitude"],
        "period": [-7, 12],
        "resample": ["months", "decades"],
        "FILL_VALUE": 9.99999962e35,
    },
    "ONI": {
        "source": "https://psl.noaa.gov/data/correlation/oni.data",
        "parse_kwargs": dict(
            skiprows=1,
            skipfooter=8,
            header=None,
            widths=[
                5,
            ]
            + [7] * 12,
        ),
        "variable": "ONI",
        "format": "wide",
        "use_seasons": True,
        "period": [-7, 11],
        "FILL_VALUE": -99.9,
    },
    "ICEN": {
        "source": "http://met.igp.gob.pe/datos/icen.txt",
        "parse_kwargs": dict(skiprows=12, parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "ICEN"},
        "variable": "ICEN",
        "format": "long",
        "use_seasons": True,
        "period": [-4, 11],
    },
}


def _scrap_data(source):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.37"
    data = requests.get(source, headers={"User-Agent": user_agent})
    return StringIO(data.text)


def parse_fwf(
    source: str,
    variable: Union[str, List[str]],
    parse_kwargs: dict = {},
    columns: Optional[dict[str, str]] = None,
    FILL_VALUE: Optional[float] = None,
    timefix: bool = True,
    webscrap: bool = False,
    **kwargs: dict,
) -> pd.DataFrame:

    if webscrap is True:
        source = _scrap_data(source)

    long_data = pd.read_fwf(source, **parse_kwargs)

    if columns is not None:
        long_data = long_data.rename(columns=columns)
    if timefix is True:
        long_data = _datefix(long_data)

    var = variable if isinstance(variable, list) else [variable]

    if FILL_VALUE is not None:
        long_data = long_data.replace(FILL_VALUE, np.nan)

    return long_data[["time"] + var]


def wide_to_long(
    source: str,
    variable: Union[str, List[str]],
    parse_kwargs: dict = {},
    FILL_VALUE: Optional[float] = None,
    **kwargs: dict,
) -> pd.DataFrame:

    wide_data = pd.read_fwf(source, **parse_kwargs)

    if FILL_VALUE is None:
        FILL_VALUE = wide_data.iloc[-1, 0]
        wide_data = wide_data.iloc[:-1, :]
    else:
        FILL_VALUE = FILL_VALUE

    long_data = pd.melt(wide_data, id_vars=[0], var_name="month", value_name=variable)
    long_data["time"] = long_data.apply(
        lambda x: pd.to_datetime(f"{x[0]:.0f}-{x['month']:.0f}-15"), axis=1
    )
    long_data = long_data.sort_values("time")[["time", variable]]
    long_data = long_data[long_data[variable] != FILL_VALUE]

    return long_data.reset_index(drop=True)


def _monthsAreComplete(table: pd.DataFrame) -> bool:
    months = table["time"].dt.month.unique()
    for mnum in range(1, 13):
        if mnum not in months:
            return False
    return True


def _datefix(table: pd.DataFrame):
    table["time"] = table["time"].apply(lambda dt: dt.replace(day=15))
    return table


def shift_predictor(
    table: pd.DataFrame,
    predictor: str,
    init_month: str,
    iyear: int = 1975,
    fyear: int = 2017,
    use_seasons: bool = False,
) -> pd.DataFrame:
    _collection = []
    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Setiembre",
        "Octubre",
        "Noviembre",
        "Diciembre",
    ]

    seasons = [
        "DJF",
        "JFM",
        "FMA",
        "MAM",
        "AMJ",
        "MJJ",
        "JJA",
        "JAS",
        "ASO",
        "SON",
        "OND",
        "NDJ",
    ]

    if use_seasons:
        months = seasons

    if table.columns.size > 2:
        table = table[["time", predictor]]
    if not _monthsAreComplete(table):
        table = monthResampler(table)
    for year in range(iyear, fyear):
        idate = f"{year-1}-{init_month}-15"
        fdate = f"{year}-{init_month}-01"
        _query = table.query(f"(time>='{idate}')&(time<'{fdate}')")
        if _query["time"].size != 12:
            _query = _query.set_index("time")
            _query = _query.reindex(
                pd.date_range(idate, fdate, freq=pd.DateOffset(months=1, day=15))
            )
        _collection.append(_query[predictor].reset_index(drop=True))
    result = pd.concat(_collection, axis=1).T.reset_index(drop=True)
    shiftIndex = int(init_month) - 1
    months_shifted = months[shiftIndex:] + months[:shiftIndex]
    column_names = [f"{predictor}_{month}" for month in months_shifted]
    result.columns = column_names
    result.index = range(iyear, fyear)
    result.index.name = "year"
    return result


def decadeResampler(table: pd.DataFrame) -> pd.DataFrame:
    data = table.copy()
    ST_DATES = [1, 11, 21]
    data["group"] = data["time"].dt.day.isin(ST_DATES).cumsum()
    cols = {colname: "mean" for colname in data.columns[1:-1].tolist()}
    data = data.groupby("group").agg({**{"time": "first"}, **cols})
    if data.iloc[0, 0].day not in ST_DATES:
        closest_day = ST_DATES[ST_DATES.index(data.iloc[1, 0].day) - 1]
        data.iloc[0, 0] = data.iloc[0, 0].replace(day=closest_day)
    return data.reset_index(drop=True)


def monthResampler(table: pd.DataFrame):
    table = table.resample("M", on="time").mean().reset_index()
    table = _datefix(table)
    return table


def splitByDay(
    table: pd.DataFrame,
    rename_cols: bool = True,
    datefix: bool = True,
    DAY_MAP: dict[int, str] = {1: "1D", 11: "2D", 21: "3D"},
) -> List[pd.DataFrame]:
    groups = table.groupby(table["time"].dt.day).groups
    data_list = [table.iloc[x].reset_index(drop=True).copy() for x in groups.values()]
    if rename_cols is True:
        for data in data_list:
            day = data["time"].iloc[0].day
            data.rename(
                columns={
                    colname: f"{colname}_{DAY_MAP[day]}"
                    for colname in data.columns
                    if colname != "time"
                },
                inplace=True,
            )
            if datefix is True:
                data = _datefix(data)
    return data_list


# https://stackoverflow.com/a/51503837
def measure(func):  # pragma: no cover
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it


if __name__ == "__main__":  # pragma: no cover
    for predictor, pargs in DATA_CONTAINTER.items():
        if predictor != "RMM":
            continue
        if pargs["format"] == "long":
            raw_data = parse_fwf(**pargs)
        else:
            raw_data = wide_to_long(**pargs)
        print(raw_data)
