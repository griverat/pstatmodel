import pandas as pd


def shift_predictor(predictor: pd.DataFrame, init_month, iyear=1975, fyear=2017):
    _collection = []
    for year in range(iyear, fyear):
        _collection.append(
            predictor.query(
                f"(time>'{year -1 }-{init_month}-01')&(time<'{year}-{init_month}-01')"
            )
            .iloc[:, 0]
            .reset_index(drop=True)
        )
    return pd.concat(_collection, axis=1).T
