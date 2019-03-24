import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error


def checkna(X):
    n = len(X)
    nans = X.isna()
    nan_sums = nans.sum()
    nonzero_sums = nan_sums[nan_sums > 0]
    if nonzero_sums.empty:
        print("X contains no NaNs!")
    else:
        num_nans = len([i for _, i in nans.iterrows() if np.any(i)])
        percentage = 100.0*num_nans/n
        print("%d/%d (%d%%) rows contain NaNs, in %d columns:" % (num_nans, n, percentage, len(nonzero_sums)))
        print(nan_sums.to_string())


def plot_pairs(X):
    sns.pairplot(X.dropna())


def train_regressor(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=5)
    print("R2: {}".format(r2_score(y_pred, y)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(y_pred, y))))
    plt.figure()
    plt.scatter(y_pred, y)
    plt.title("CV Results")
