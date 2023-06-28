import pytest
import pandas as pd
from datetime import datetime
import os

import sys

sys.path.append("src/data_preprocessing/")

from data_preprocessing import compute_weekly_profile, save_df_to_csv, preprocess_data


def test_compute_weekly_profile():
    # Mock a pd.Series that would resemble a row in the DataFrame
    test_series = pd.Series({"peak_or_not": 1}, name=pd.Timestamp("2023-01-01"))
    x = compute_weekly_profile(test_series)
    assert compute_weekly_profile(test_series) == 4

    test_series = pd.Series({"peak_or_not": 0}, name=pd.Timestamp("2023-01-02"))
    assert compute_weekly_profile(test_series) == 2


def test_save_df_to_csv(tmp_path):
    # Create a temporary directory using pytest's tmp_path
    dir_path = tmp_path / "subfolder"

    # Mock a simple DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # File name
    file_name = "test.csv"

    # Call the function to save the DataFrame
    save_df_to_csv(df, dir_path, file_name)

    # Check that the file was indeed created
    assert (dir_path / file_name).is_file()

    # Check that the contents are correct
    loaded_df = pd.read_csv(dir_path / file_name, index_col=0)
    pd.testing.assert_frame_equal(loaded_df, df)

def test_preprocess_data():

    # Run the preprocess_data function
    preprocess_data("temp_results/")

    # Check if the processed features and target data files are created
    features_file_path = os.path.join(os.getcwd(), "temp_results", "complete_data", "features.csv")
    target_file_path = os.path.join(os.getcwd(),  "temp_results", "complete_data", "target.csv")
    assert os.path.exists(features_file_path)
    assert os.path.exists(target_file_path)

    # Check if the train-test split data files are created
    train_features_file_path = os.path.join(os.getcwd(), "temp_results", "train", "X_train.csv")
    train_target_file_path = os.path.join(os.getcwd(), "temp_results", "train", "y_train.csv")
    test_features_file_path = os.path.join(os.getcwd(), "temp_results", "test", "X_test.csv")
    test_target_file_path = os.path.join(os.getcwd(), "temp_results", "test", "y_test.csv")

    assert os.path.exists(train_features_file_path) == True
    assert os.path.exists(train_target_file_path) == True
    assert os.path.exists(test_features_file_path) == True
    assert os.path.exists(test_target_file_path) == True

    import shutil

    folder_path = "temp_results/"
    shutil.rmtree(folder_path)


if __name__ == "__main__":
    pytest.main([__file__])
