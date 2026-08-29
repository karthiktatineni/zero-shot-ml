import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

DATA_DIR = Path("data/tabular_benchmarks")

def _load_csv_dataset(filename: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / filename)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

def get_iris() -> Tuple[pd.DataFrame, pd.Series]:
    return _load_csv_dataset("iris.csv", "target")

def get_wine() -> Tuple[pd.DataFrame, pd.Series]:
    return _load_csv_dataset("wine.csv", "target")

def get_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    return _load_csv_dataset("breast_cancer.csv", "target")

def get_digits() -> Tuple[pd.DataFrame, pd.Series]:
    return _load_csv_dataset("digits.csv", "target")

def get_titanic() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = _load_csv_dataset("titanic.csv", "survived")
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = X[features].copy()
    return X, y

def load_all_datasets():
    """
    Returns a dictionary of dataset names to (X, y) tuples.
    """
    return {
        "Iris": get_iris(),
        "Wine": get_wine(),
        "Breast Cancer": get_breast_cancer(),
        "Digits": get_digits(),
        "Titanic": get_titanic()
    }
