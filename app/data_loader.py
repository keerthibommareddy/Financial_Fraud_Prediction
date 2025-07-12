# data_loader.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import DATASET_FILENAME, SAMPLE_SIZE, TEST_SPLIT_RATIO, RANDOM_STATE, TARGET_COLUMN

def load_and_sample_data():
    try:
        df_full = pd.read_csv(DATASET_FILENAME)
        print(f"Full dataset '{DATASET_FILENAME}' loaded. Shape: {df_full.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: Dataset file '{DATASET_FILENAME}' not found.")
    except Exception as e:
        raise RuntimeError(f"Error reading dataset file: {e}") from e

    if len(df_full) >= SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} entries...")
        df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        print(f"Dataset smaller than {SAMPLE_SIZE}, using full data.")
        df = df_full.reset_index(drop=True)

    print(f"Using dataset shape: {df.shape}")
    return df

def preprocess_data(df_input, target_col_name):
    print(f"\nPreprocessing DataFrame of shape: {df_input.shape}")
    df_processed = df_input.drop('LoanID', axis=1) if 'LoanID' in df_input.columns else df_input.copy()
    if target_col_name not in df_processed.columns:
        raise ValueError(f"Target '{target_col_name}' not found.")
    
    y = pd.to_numeric(df_processed[target_col_name], errors='coerce')
    if y.isnull().any():
        raise ValueError("Target has NaNs after coercion.")
    y = y.astype(int)

    print(f"Target counts:\n{y.value_counts(normalize=True)}")

    X_df = df_processed.drop(target_col_name, axis=1)
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_df.select_dtypes(include='object').columns.tolist()

    print(f"Processing {len(numerical_cols)} numerical, {len(categorical_cols)} categorical features.")

    if numerical_cols and X_df[numerical_cols].isnull().sum().sum() > 0:
        X_df[numerical_cols] = SimpleImputer(strategy='median').fit_transform(X_df[numerical_cols])
    
    if categorical_cols and X_df[categorical_cols].isnull().sum().sum() > 0:
        X_df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X_df[categorical_cols])

    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True, dtype=float)

    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)

    scaler = StandardScaler()
    X_encoded[X_encoded.columns] = scaler.fit_transform(X_encoded[X_encoded.columns])

    X_processed_np = X_encoded.values.astype(np.float32)
    y_processed_np = y.values

    return X_processed_np, y_processed_np, X_encoded.shape[1]

df = load_and_sample_data()
try:
    X_processed, y_processed, INPUT_DIM = preprocess_data(df, TARGET_COLUMN)
    print(f"Final Input Dimension: {INPUT_DIM}")
except Exception as e: print(f"Preprocessing error: {e}"); raise

def split_train_test(X_processed, y_processed):
    stratify_option = y_processed if np.unique(y_processed).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=stratify_option
    )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test_tensor, y_test_tensor
