#!/usr/bin/env python3
"""
Module to standardize tabular data using Scikit-learn.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using StandardScaler.

    Arguments:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The standardized version of the input data,
                       with the same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
