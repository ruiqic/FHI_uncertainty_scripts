import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def fit_regressor(df):
    """
    Fit a linear regressor that recalibrates uncertainty measures.
    Outlined in https://arxiv.org/pdf/2105.13303.pdf
    
    df: pandas.DataFrame
        dataframe with training data
        columns expected: "residual" and "uncertainty"
        
    returns: sklearn.linear_model._base.LinearRegression
        trained regressor 
    """
    
    # ignore outliers for binning
    sorted_unc = np.sort(df["uncertainty"])
    upper_index = int(len(sorted_unc)*0.9)
    min_unc = sorted_unc[0]
    max_unc = sorted_unc[upper_index]
    
    # binning uncertainties
    ii = pd.interval_range(start=min_unc, end=max_unc, periods=15)
    cuts = pd.cut(df["uncertainty"], bins=ii)
    unc_dict = {}
    res_dict = {}
    for interval in ii:
        unc_dict[interval] = []
        res_dict[interval] = []
    for cut, unc, res in zip(cuts, df["uncertainty"], df["residual"]):
        if pd.isnull(cut):
            continue
        unc_dict[cut].append(unc)
        res_dict[cut].append(res)
        
    # build training data
    mean_unc = []
    std_res = []
    counts = []
    for interval in ii:
        count = len(unc_dict[interval])
        if count == 0:
            continue
        mean_unc.append(np.mean(unc_dict[interval]))
        std_res.append(np.std(res_dict[interval]))
        counts.append(count)
    
    # fit regressor
    X = np.array(mean_unc).reshape(-1,1)
    y = np.array(std_res)
    regressor = LinearRegression()
    regressor.fit(X, y, sample_weight=counts)
    
    return regressor
    
    