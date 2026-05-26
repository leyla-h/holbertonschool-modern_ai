#!/usr/bin/env python3
"""
Module to evaluate K-Means clustering quality.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality for a range of cluster numbers.

    Arguments:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        max_clusters (int): Maximum number of clusters to evaluate (>=2)
        random_state (int): Random seed for reproducibility

    Returns:
        list: Evaluated cluster numbers
        list: Inertia values corresponding to each cluster number
        list: Silhouette scores corresponding to each cluster number
    """
    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, n_clusters=k, random_state=random_state)
        labels = model.labels_

        ks.append(k)
        inertia_values.append(model.inertia_)
        silhouette_values.append(
            metrics.silhouette_score(X, labels, random_state=random_state)
        )

    return ks, inertia_values, silhouette_values
