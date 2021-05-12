import pandas as pd

DATA_CONTAINTER = {
    "AAO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii",
        "fwf_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AAO"},
        "name": "AAO",
        "format": "long",
    },
    "AO": {
        "source": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        "fwf_kwargs": dict(parse_dates=[[0, 1]], header=None),
        "columns": {"0_1": "time", 2: "AO"},
        "name": "AO",
        "format": "long",
    },
    "PMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/PMM.txt",
        "fwf_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "PMM"},
        "name": "PMM",
        "format": "long",
    },
    "AMM": {
        "source": "https://www.aos.wisc.edu/~dvimont/MModes/RealTime/AMM.txt",
        "fwf_kwargs": dict(parse_dates=[["Year", "Mo"]]),
        "columns": {"Year_Mo": "time", "SST": "AMM"},
        "name": "AMM",
        "format": "long",
    },
    "TNA": {
        "source": "https://psl.noaa.gov/data/correlation/tna.data",
        "fwf_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "name": "TNA",
        "format": "wide",
    },
    "TSA": {
        "source": "https://psl.noaa.gov/data/correlation/tsa.data",
        "fwf_kwargs": dict(skiprows=1, skipfooter=1, header=None),
        "name": "TSA",
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
        "name": "NAO",
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
        "name": "EP/NP",
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
        "name": "WP",
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
        "name": "AMI",
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
        "name": "SOI",
        "format": "wide",
        "FILL_VALUE": -99.99,
    },
}


def parse_fwf(source, *args, timefix=True, fwf_kwargs={}, **kwargs):
    variable = pd.read_fwf(source, **fwf_kwargs)
    if "columns" in kwargs.keys():
        variable = variable.rename(columns=kwargs["columns"])
    if timefix is True:
        variable["time"] = variable["time"] + pd.Timedelta("14D")
    return variable[["time", kwargs["name"]]]


def wide_to_long(source, *args, fwf_kwargs={}, **kwargs):
    wide_data = pd.read_fwf(source, **fwf_kwargs)

    if "FILL_VALUE" not in kwargs.keys():
        FILL_VALUE = wide_data.iloc[-1, 0]
        wide_data = wide_data.iloc[:-1, :]
    else:
        FILL_VALUE = kwargs["FILL_VALUE"]

    long_data = pd.melt(
        wide_data, id_vars=[0], var_name="month", value_name=kwargs["name"]
    )
    long_data["time"] = long_data.apply(
        lambda x: pd.to_datetime(f"{x[0]:.0f}-{x['month']:.0f}-15"), axis=1
    )
    long_data = long_data.sort_values("time")[["time", kwargs["name"]]]
    long_data = long_data[long_data[kwargs["name"]] != FILL_VALUE]

    return long_data.reset_index(drop=True)


def shift_predictor(
    table: pd.DataFrame,
    predictor: str,
    init_month: str,
    iyear: int = 1975,
    fyear: int = 2017,
):
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


if __name__ == "__main__":
    for predictor, pargs in DATA_CONTAINTER.items():
        if pargs["format"] == "long":
            raw_data = parse_fwf(**pargs)
        else:
            raw_data = wide_to_long(**pargs)
        print(raw_data)
