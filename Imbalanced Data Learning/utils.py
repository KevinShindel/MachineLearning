"""
Date: 2026-05-23
Author: Kevin Shindel
Description:
    This module contains utility functions and classes for imbalanced data learning.
    - SMOTE (Synthetic Minority Over-sampling Technique) is a popular oversampling technique that generates synthetic samples for the minority class to balance the dataset.
    - ADASYN (Adaptive Synthetic Sampling) is another oversampling technique that focuses on generating synthetic samples for the minority class based on the density of the data points, giving more attention to those that are harder to learn.
    It might also contain functions for data preprocessing, feature engineering, and evaluation metrics specific to imbalanced data learning.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Smote:
    def __init__(self, k_neighbors=5, sampling_strategy="auto"):
        """
        Multi-class SMOTE implementation.

        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to use for synthetic sample generation.
        sampling_strategy : str or dict
            'auto' : balance all minority classes to the majority class count
            'minority' : balance minority class to next majority class
            dict : {class_label: target_count, ...}
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        """Resample X, y using SMOTE."""
        unique_classes, counts = np.unique(y, return_counts=True)

        # Determine target count for each class
        if self.sampling_strategy == "auto":
            target_count = np.max(counts)  # Balance to majority class
        else:
            target_count = self.sampling_strategy

        X_resampled = []
        y_resampled = []

        # Process each class independently
        for class_label in unique_classes:
            X_class = X[y == class_label]
            class_count = len(X_class)

            X_resampled.append(X_class)
            y_resampled.append(np.full(class_count, class_label))

            # Generate synthetic samples if needed
            if class_count < target_count:
                n_samples_to_generate = target_count - class_count
                synthetic_samples = self._generate_synthetic_samples(
                    X_class, n_samples_to_generate
                )
                X_resampled.append(synthetic_samples)
                y_resampled.append(np.full(n_samples_to_generate, class_label))

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _generate_synthetic_samples(self, X_minority, n_samples_to_generate):
        """Generate synthetic samples using SMOTE algorithm."""
        neighbors = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X_minority)
        synthetic_samples = []

        for _ in range(n_samples_to_generate):
            idx = np.random.randint(0, len(X_minority))
            x_i = X_minority[idx]
            neighbor_indices = neighbors.kneighbors([x_i], return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbor_indices)
            x_neighbor = X_minority[neighbor_idx]

            # Generate a synthetic sample
            synthetic_sample = x_i + np.random.rand() * (x_neighbor - x_i)
            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)
