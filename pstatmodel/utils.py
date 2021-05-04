import pandas as pd


def shift_predictor(
    table: pd.DataFrame,
    predictor: str,
    init_month: str,
    iyear: int = 1975,
    fyear: int = 2017,
):
    _collection = []
    for year in range(iyear, fyear):
        _collection.append(
            table.query(
                f"(time>'{year -1 }-{init_month}-01')&(time<'{year}-{init_month}-01')"
            )[predictor].reset_index(drop=True)
        )
    return pd.concat(_collection, axis=1).T.reset_index(drop=True)
