# Copyright (C) Microsoft Corporation.  All rights reserved.
"""Prepare the data for pairing duplicate and original questions."""

import pandas as pd
import math


def round_sample(X, frac=0.1, min=1):
    """Sample X ensuring at least min samples are selected."""
    n = max(min, math.floor(len(X) * frac))
    return X.sample(n)


def round_sample_strat(X, strat, **kwargs):
    """Sample X ensuring at least min samples are selected."""
    return X.groupby(strat).apply(round_sample, **kwargs)


def max_sample(X, max):
    n = min(X.shape[0], max)
    return X.sample(n)


def random_merge(A, B, N=20, on='AnswerId', key='key', n='n'):
    assert key not in A and key not in B
    X = A.copy()
    X[key] = A[on]
    Y = B.copy()
    Y[key] = B[on]
    match = X.merge(Y, on=key).drop(key, axis=1)
    match[n] = 0
    df_list = [match]
    for i in A.index:
        X = A.loc[[i]]
        Y = B[B[on] != X[on].iloc[0]].sample(N-1)
        X[key] = 1
        Y[key] = 1
        Z = X.merge(Y, how='outer', on=key).drop(key, axis=1)
        Z[n] = range(1, N)
        df_list.append(Z)
    df = pd.concat(df_list, ignore_index=True)
    return df
