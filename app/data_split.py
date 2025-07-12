# data_split.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_ratio, random_state):
    stratify_option = y if len(np.unique(y)) > 1 else None
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=stratify_option)

def split_data_non_iid_uneven(X, y, num_clients, min_samples=20, power_law_factor=1.6, seed=42):
    n_samples = len(X)
    proportions = np.random.default_rng(seed).power(power_law_factor, num_clients)
    proportions /= proportions.sum()
    client_sizes = (proportions * n_samples).astype(int)
    client_sizes = np.maximum(min_samples, client_sizes)
    client_sizes = (client_sizes / client_sizes.sum() * n_samples).astype(int)
    diff = n_samples - client_sizes.sum()
    client_sizes[np.random.choice(num_clients)] += diff

    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    current_idx = 0
    client_data = {}
    for i in range(num_clients):
        size = client_sizes[i]
        idx = indices[current_idx: current_idx + size]
        client_data[f'client_{i+1}'] = (torch.tensor(X[idx]), torch.tensor(y[idx]))
        current_idx += size
    return client_data
