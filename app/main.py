# main.py

import pandas as pd
from config import *
from data_loader import load_and_sample_data, preprocess_data
from data_split import split_train_test, split_data_non_iid_uneven
from federated_trainer import hierarchical_training
from plots import plot_full_history
import torch

def main():
    # Load and preprocess data
    df = load_and_sample_data()
    X, y, input_dim = preprocess_data(df, TARGET_COLUMN)

    # Split test data
    X_train_all, X_test, y_train_all, y_test = split_train_test(X, y, TEST_SPLIT_RATIO, RANDOM_STATE)

    # Split training data among clients
    client_data = split_data_non_iid_uneven(X_train_all, y_train_all, NUM_CLIENTS)

    X_train_split = [client_data[k][0] for k in sorted(client_data.keys())]
    y_train_split = [client_data[k][1] for k in sorted(client_data.keys())]

    # Convert test data to tensors
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    history, client_test_accs, client_test_f1s = hierarchical_training(client_data, X_test_tensor, y_test_tensor)

    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv("metrics_log.csv", index=False)

    plot_full_history(
        history=history,
        client_test_accuracy=client_test_accs,
        client_test_f1=client_test_f1s,
        num_clients=NUM_CLIENTS
    )

if __name__ == "__main__":
    main()

