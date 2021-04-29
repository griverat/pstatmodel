import numpy as np
import pandas as pd
import statsmodels.api as sm

# import xarray as xr
# from dask import compute, delayed


def stepwise_selection(
    X,
    y,
    initial_list=[],
    threshold_in=0.05,
    threshold_out=0.1,
    verbose=True,
    max_vars=12,
    min_vars=4,
):
    """Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    included_pvals = []
    included_rvals = []
    # min_pvalue = 0.1
    lower = False
    rcond = False
    dropped = False
    #     over = False
    if np.isnan(y).any():
        return [], np.nan, np.nan
    if verbose:
        print(f"Initial threshold_in value: {threshold_in}")
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            included_pvals.append(best_pval)
            included_rvals.append(model.rsquared ** (0.5))
            changed = True
            if verbose:
                print("Add  {:30} with p-value {:.6}".format(best_feature, best_pval))
        # backward step
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        #         print(f"{best_pval=} // {worst_pval=}")
        if worst_pval > threshold_out:
            changed = True
            dropped = True
            worst_feature = pvalues.index[pvalues.argmax()]
            _idx = included.index(worst_feature)
            included_pvals.pop(_idx)
            included_rvals.pop(_idx)
            included.remove(worst_feature)
            if verbose:
                print("Drop {:30} with p-value {:.6}".format(worst_feature, worst_pval))

        if len(included) >= min_vars and len(included) <= max_vars and not changed:
            break
        elif len(included) > max_vars and not dropped:
            psize = 999
            included_pvals = np.array(included_pvals)
            idem = False
            while psize > max_vars or idem:
                threshold_in = np.round(max([0.01, threshold_in - 0.01]), decimals=2)
                threshold_in_next = np.round(
                    max([0.01, threshold_in - 0.01]), decimals=2
                )

                mask = included_pvals < threshold_in
                mask_next = included_pvals < threshold_in_next

                if False in mask:
                    psize = mask[: np.where(mask == False)[0][0]].sum()
                else:
                    psize = mask.sum()
                if False in mask_next:
                    psize_next = mask_next[: np.where(mask_next == False)[0][0]].sum()
                else:
                    psize_next = mask_next.sum()

                if psize == psize_next and threshold_in != 0.01 and psize != 0:
                    if verbose:
                        print(f"{psize = }, {threshold_in = }")
                        print(f"{psize_next = }, {threshold_in_next = }")
                    idem = True
                    continue
                else:
                    idem = False
                if psize >= min_vars and psize <= max_vars:
                    if verbose:
                        print(f"break: {psize = }, {threshold_in = }")
                    pass
                elif psize < min_vars and threshold_in != 0.1:
                    included_rvals = np.array(included_rvals)
                    mask = included_rvals < 0.9
                    threshold_in = np.round(min([0.1, threshold_in + 0.01]), decimals=2)
                    rcond = True
                    if verbose:
                        print("breaking on R condition")
                elif threshold_in == 0.01 or threshold_in == 0.1:
                    break
                else:
                    continue
                break
            mfalse = np.where(mask == False)[0]
            if rcond:
                mfalse = mfalse[mfalse > 3]
            if mfalse.size != 0:
                included = included[: mfalse[0]]
                model = sm.OLS(y, sm.add_constant(X[included])).fit()
            break
        elif dropped:
            if len(included) > max_vars and threshold_in != 0.01 and not lower:
                threshold_in = np.round(max([0.01, threshold_in - 0.01]), decimals=2)
                if verbose:
                    print(f"Upped threshold_in value to {threshold_in}")
                included = []
                changed = True

            if len(included) >= min_vars and lower:
                if model.rsquared ** (0.5) > 0.9:
                    changed = False
                    if verbose:
                        print("Breaking condition met: R value over 0.9")

        if not changed:
            #             break
            if len(included) < min_vars and threshold_in != 0.1:
                threshold_in = np.round(min([0.1, threshold_in + 0.01]), decimals=2)
                if verbose:
                    print(f"Dropped threshold_in value to {threshold_in}")
                included = []
                lower = True
            else:
                break
    return included, model, threshold_in
