from io import StringIO

import pandas as pd
import requests

DATA_CONTAINTER = {
    "AAO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii",
        "fwf_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AAO"},
        "variable": "AAO",
        "format": "long",
    },
    "AO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        "fwf_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AO"},
        "variable": "AO",
        "format": "long",
    },
    "PMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/PMM.txt",
        "fwf_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "PMM"},
        "variable": "PMM",
        "format": "long",
    },
    "AMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/AMM.txt",
        "fwf_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "AMM"},
        "variable": "AMM",
        "format": "long",
    },
    "TNA": {
        "source": "https://psl.noaa.gov/data/correlation/tna.data",
        "fwf_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "variable": "TNA",
        "format": "wide",
    },
    "TSA": {
        "source": "https://psl.noaa.gov/data/correlation/tsa.data",
        "fwf_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "variable": "TSA",
        "format": "wide",
    },
    "NAO": {
        "source": "https://psl.noaa.gov/data/correlation/nao.data",
        "fwf_kwargs": dict(
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
        "fwf_kwargs": dict(
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
        "fwf_kwargs": dict(
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
        "fwf_kwargs": dict(
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
        "fwf_kwargs": dict(
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
        "fwf_kwargs": dict(
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
        "format": "long",
        "webscrap": True,
        "variable": ["RMM1", "RMM2", "amplitude"],
    },
}


def scrap_data(source):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.37"
    data = requests.get(source, headers={"User-Agent": user_agent})
    return StringIO(data.text)


def parse_fwf(
    source: str,
    timefix: bool = True,
    fwf_kwargs: dict = {},
    **kwargs: dict,
) -> pd.DataFrame:
    if "webscrap" in kwargs.keys():
        source = scrap_data(source)
    variable = pd.read_fwf(source, **fwf_kwargs)
    if "columns" in kwargs.keys():
        variable = variable.rename(columns=kwargs["columns"])
    if timefix is True:
        variable["time"] = variable["time"] + pd.Timedelta("14D")
    var = (
        kwargs["variable"]
        if isinstance(kwargs["variable"], list)
        else [kwargs["variable"]]
    )
    return variable[["time"] + var]


def wide_to_long(source: str, fwf_kwargs: dict = {}, **kwargs: dict) -> pd.DataFrame:
    wide_data = pd.read_fwf(source, **fwf_kwargs)

    if "FILL_VALUE" not in kwargs.keys():
        FILL_VALUE = wide_data.iloc[-1, 0]
        wide_data = wide_data.iloc[:-1, :]
    else:
        FILL_VALUE = kwargs["FILL_VALUE"]

    long_data = pd.melt(
        wide_data, id_vars=[0], var_name="month", value_name=kwargs["variable"]
    )
    long_data["time"] = long_data.apply(
        lambda x: pd.to_datetime(f"{x[0]:.0f}-{x['month']:.0f}-15"), axis=1
    )
    long_data = long_data.sort_values("time")[["time", kwargs["variable"]]]
    long_data = long_data[long_data[kwargs["variable"]] != FILL_VALUE]

    return long_data.reset_index(drop=True)


def shift_predictor(
    table: pd.DataFrame,
    predictor: str,
    init_month: str,
    iyear: int = 1975,
    fyear: int = 2017,
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
    for year in range(iyear, fyear):
        _collection.append(
            table.query(
                f"(time>'{year -1 }-{init_month}-01')&(time<'{year}-{init_month}-01')"
            )[predictor].reset_index(drop=True)
        )
    result = pd.concat(_collection, axis=1).T.reset_index(drop=True)
    shiftIndex = int(init_month) - 1
    months_shifted = months[shiftIndex:] + months[:shiftIndex]
    column_names = [f"{predictor}_{month}" for month in months_shifted]
    result.columns = column_names
    return result


def compute_decade(data: pd.DataFrame):
    data = data.copy()
    ST_DATES = [1, 11, 21]
    data["groups"] = data["time"].dt.day.isin(ST_DATES).cumsum()
    cols = {colname: "mean" for colname in data.columns[1:-1].tolist()}
    data = data.groupby("group").agg({**{"time": "first"}, **cols})
    if data.iloc[0, 0].day not in ST_DATES:
        closest_day = ST_DATES[ST_DATES.index(data.iloc[1, 0].day) - 1]
        data.iloc[0, 0] = data.iloc[0, 0].replace(day=closest_day)
    return data.reset_index(drop=True)[data.columns[:-1]]


if __name__ == "__main__":  # pragma: no cover
    for predictor, pargs in DATA_CONTAINTER.items():
        if predictor != "RMM":
            continue
        if pargs["format"] == "long":
            raw_data = parse_fwf(**pargs)
        else:
            raw_data = wide_to_long(**pargs)
        print(raw_data)
