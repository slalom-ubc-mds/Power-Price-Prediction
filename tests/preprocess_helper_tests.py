import pandas as pd
import numpy as np
import os
import pytest
from unittest import mock
import warnings
import sys
from datetime import datetime

sys.path.append("notebooks/utils/")

from preprocess_helper import *

def test_preprocess_intertie_data():
    """Test preprocess_intertie_data function."""
    
    preprocess_intertie_data()
    path = "data/processed/intertie.csv"
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert df.shape[0] > 0

def test_process_supply_data():
    """Test process_supply_data function."""
    process_supply_data()
    path = "data/processed/supply_load_price.csv"
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert df.shape[0] > 0

def test_merge_data():
    """Test merge_data function."""
    merge_data()
    path = "data/processed/preprocessed_features.csv"
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert df.shape[0] > 0

def test_get_data():
    """Test get_data function."""
    start_date = "2023-01-01"
    end_date = "2023-01-07"

    # Call the function to get the data
    data = get_data(start_date, end_date)

    # Check if the returned value is a DataFrame
    assert isinstance(data, pd.DataFrame)

    # Check if the DataFrame has more than 0 rows
    assert data.shape[0] > 0

    # Check if the index of the DataFrame is a datetime index
    assert isinstance(data.index, pd.DatetimeIndex)

    # Check if the start and end dates of the retrieved data are within the specified range
    assert data.index.min().date() >= datetime.strptime(start_date, "%Y-%m-%d").date()
    assert data.index.max().date() <= datetime.strptime(end_date, "%Y-%m-%d").date()

def test_create_lagged_columns():
    """Test create_lagged_columns function."""
    # Create a sample DataFrame
    X = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [6, 7, 8, 9, 10],
        "C": [11, 12, 13, 14, 15]
    })

    # Call the function to create lagged columns
    lagged_columns = create_lagged_columns(X, lag_range=3)

    # Define the expected lagged column names
    expected_columns = [
        "A_lag3", "A_lag2", "A_lag1",
        "B_lag3", "B_lag2", "B_lag1",
        "C_lag3", "C_lag2", "C_lag1"
    ]

    # Check if the generated lagged column names match the expected column names
    assert lagged_columns == expected_columns

if __name__ == "__main__":
    pytest.main([__file__])
