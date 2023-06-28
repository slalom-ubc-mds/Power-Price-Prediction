"""
This script is used to get the cross-validation, AESO results, and test results.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle


def main():
    """Main function to run the script."""

    cv_result_files = [
        "cv_arima_results.pkl",
        "cv_elasticnet_results.pkl",
        "cv_xgb_results.pkl",
        "cv_lgbm_results.pkl",
    ]

    models = ["ARIMA", "Elastic Net", "XGBoost", "LightGBM"]

    # Load the data from cv_result_files
    rmse_cv_results = []
    rmse_cv_std = []
    rmse_cv_min = []
    rmse_cv_max = []
    for file in cv_result_files:
        root_path = "notebooks/cross_validation/cv_results/"
        with open(root_path + file, "rb") as f:
            results = pickle.load(f)
            rmse = results["test_MeanSquaredError"].mean()
            rmse_std = results["test_MeanSquaredError"].std()
            rmse_min = results["test_MeanSquaredError"].min()
            rmse_max = results["test_MeanSquaredError"].max()
            rmse_cv_results.append(rmse)
            rmse_cv_std.append(rmse_std)
            rmse_cv_min.append(rmse_min)
            rmse_cv_max.append(rmse_max)

    rmse_cv_results_df = pd.DataFrame(
        {
            "Model": models,
            "RMSE_CV": rmse_cv_results,
            "RMSE_CV_STD": rmse_cv_std,
            "RMSE_MIN": rmse_cv_min,
            "RMSE_MAX": rmse_cv_max,
        }
    ).sort_values(by=["RMSE_CV"])

    # create folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    rmse_cv_results_df.to_csv("results/cv_results.csv", index=False)

    print("Script complete...")


if __name__ == "__main__":
    main()
