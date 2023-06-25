# from databricks.sdk.runtime import *  # this line should be commented out if running locally
import pandas as pd
import os
import copy
import numpy as np
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from lightgbm import LGBMRegressor
from sktime.forecasting.compose import ForecastingPipeline
import requests
from datetime import timedelta
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

import shap


def generate_shap_dataframe(y_pred, lgbm_pipeline, X_feature):
    """
    Function to generate a DataFrame of SHAP values for a given LightGBM pipeline and prediction data.

    Parameters:
    y_pred (pd.Series): Series with prediction data.
    lgbm_pipeline (sklearn.Pipeline): The LightGBM pipeline model.
    X_feature: Series of features of training or testing set

    Returns:
    final_feature_df (pd.DataFrame): DataFrame with date, feature, value, and base for each feature's SHAP values.
    """

    # Define the features
    features = ["price"] + X_feature.columns.to_list()
    n_cols = len(features)

    # Define the window length
    window_length = 24

    # Extract the last window from the pipeline
    y_last, X_last = lgbm_pipeline.forecaster_.forecaster_._get_last_window()
    X_pred = np.zeros((1, n_cols, window_length))

    # Fill the array with last values
    X_pred[:, 0, :] = y_last
    X_pred[:, 1:, :] = X_last.T

    # Reshape the array
    X_pred = X_pred.reshape(1, -1)
    X_pred = X_pred.flatten()

    # Create a list of feature names with lag information
    reduced_feature_list = [
        f"{feature}_lag{i}" for feature in features for i in range(24, 0, -1)
    ]

    # Create a DataFrame from the reshaped array
    X_reduced_df = pd.DataFrame([X_pred], columns=reduced_feature_list).round(3)

    # Get the list of LightGBM models from the pipeline
    lgbm_models = lgbm_pipeline.forecaster_.forecaster_.estimators_

    # Initialize a DataFrame for storing SHAP values
    shap_df = pd.DataFrame([], columns=reduced_feature_list)
    # base_list = []
    base = []

    # Calculate and store the SHAP values for each model
    for model in lgbm_models:
        explainerModel = shap.TreeExplainer(model)
        shap_values = explainerModel.shap_values(X_reduced_df).flatten()
        shap_df.loc[len(shap_df)] = shap_values

        # Append base values
        # base_list.append(lgbm_pipeline.forecaster_._get_inverse_transform(lgbm_pipeline.forecaster_.transformers_pre_, np.array([explainerModel.expected_value]))[0])
        base.append(explainerModel.expected_value)

    # Rename columns for clarity and add additional columns
    shap_df = shap_df.rename(columns=lambda x: x.replace("_", " ").capitalize())

    shap_df["date"] = y_pred.index
    # shap_df['base_val'] = base_list
    shap_df["base"] = base

    # Initialize a DataFrame for storing top features
    top_features_df = pd.DataFrame(columns=["Date", "Feature", "Value", "Base"])

    # Calculate and store the top 4 features and the sum of the remaining features
    for index, row in shap_df.iterrows():
        # Get the date and base values
        date = row["date"]
        base = row["base"]

        # Drop the 'Date' column for sorting
        sorted_row = row.drop(["date", "base"]).sort_values(ascending=False)

        # Select top 4 features and values
        # Create a series with absolute values
        sorted_row = sorted_row.apply(pd.to_numeric, errors="coerce")
        abs_series = sorted_row.abs()

        # Get the indices of the top 4 rows with highest absolute values
        top_4_indices = abs_series.nlargest(4).index
        top_features = sorted_row.loc[top_4_indices]

        # Calculate the sum of the remaining features
        other_value = sorted_row.drop(top_4_indices).sum()

        # Create a DataFrame for top features and 'other' feature
        temp_df = pd.DataFrame(
            {
                "Date": [date] * 5,
                "Feature": list(top_features.index) + ["Other"],
                "Value": list(top_features.values) + [other_value],
                "Base": [base] * 5,
            }
        )

        # Append to the top features DataFrame
        top_features_df = pd.concat([top_features_df, temp_df], ignore_index=True)

    # Normalize the 'Value' column
    top_features_df["Value"] = top_features_df["Value"] / top_features_df["Base"]

    # Get the original date of the first row
    original_date = top_features_df.iloc[0]["Date"]

    # Calculate the start date for duplication (12 hours before the original date)
    start_date = original_date - pd.DateOffset(hours=12)

    # Create a list of dates for duplication
    dates = pd.date_range(start=start_date, periods=12, freq="H")

    # Duplicate the first 5 rows for each date
    duplicated_rows = pd.concat(
        [top_features_df.iloc[:5].assign(Date=date) for date in dates]
    )
    duplicated_rows.reset_index(drop=True, inplace=True)

    # Concatenate the top features DataFrame with the duplicated rows and sort by 'Date'
    final_feature_df = pd.concat([top_features_df, duplicated_rows])
    final_feature_df.reset_index(drop=True, inplace=True)
    final_feature_df.sort_values(by=["Date"], inplace=True)

    return final_feature_df


def generate_sentence_dataframe(df):
    """
    Function to generate a DataFrame of sentences summarizing the impact of top features on predictions.

    Parameters:
    df (pd.DataFrame): DataFrame with date, feature, value, and base for each feature's SHAP values.

    Returns:
    result_df (pd.DataFrame): DataFrame with date and generated sentence for each date.
    """

    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Group the DataFrame by date
    grouped = df.groupby("Date")

    # Create an empty DataFrame to store the generated sentences
    result_df = pd.DataFrame(columns=["Date", "Sentence"])

    # Iterate over each date and generate the sentence
    for date, group in grouped:
        # Get the base value for the date
        base_value = group["Base"].iloc[0]

        # Initialize the sentence with the average power price
        sentence = f"The average power price of the past month is ${base_value:.1f}. The top two variables that impact the prediction are:"

        # Get the top two features by absolute impact on the prediction
        sorted_group = (
            group[group["Feature"] != "Other"]
            .sort_values("Value", key=abs, ascending=False)
            .head(2)
        )

        # Add each feature's impact to the sentence
        for _, row in sorted_group.iterrows():
            feature = row["Feature"]
            value = row["Value"]

            # Determine whether the feature increases or decreases the prediction
            if value < 0:
                change_type = "decreases"
            else:
                change_type = "increases"

            # Calculate the percentage change and add to the sentence
            change_percentage = abs(value) * 100
            sentence += (
                f" {feature} {change_type} the prediction by {change_percentage:.3f}%,"
            )

        # Complete the sentence by removing the last comma and adding a period
        sentence = sentence.rstrip(",") + "."

        # Add the sentence to the DataFrame
        temp_df = pd.DataFrame({"Date": [date], "Sentence": [sentence]})
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # Sort the DataFrame by date
    result_df = result_df.sort_values("Date", ascending=True)

    return result_df


def generate_tableau_required_dataframe(y_pred):
    """
    Function to prepare DataFrame in required format for Tableau dashboard.

    Parameters:
    y_pred (pd.DataFrame): DataFrame containing the predicted values.

    Returns:
    predicted_price (pd.DataFrame): DataFrame in the format needed for Tableau dashboard.
    """

    # Renaming the DataFrame column and resetting the index
    price_prediction = (
        y_pred.rename(columns={y_pred.columns.tolist()[0]: "price"})
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Creating upper and lower bounds for the predicted prices
    price_prediction_upper = copy.deepcopy(price_prediction)
    price_prediction_upper["indicator"] = "past_predicted"
    price_prediction_upper["upper_bound"] = np.nan
    price_prediction_upper["lower_bound"] = np.nan

    price_prediction_lower = copy.deepcopy(price_prediction)
    price_prediction_lower["indicator"] = np.nan
    price_prediction_lower["price"] = np.nan

    price_prediction = pd.concat([price_prediction_upper, price_prediction_lower])
    price_prediction["date"] = pd.to_datetime(price_prediction["date"])

    first_row_date = price_prediction["date"].iloc[0]
    start_time = first_row_date - pd.Timedelta(hours=12)

    # Creating past prediction DataFrame
    price_past = pd.DataFrame(
        {
            "date": pd.date_range(start=start_time, periods=12, freq="H"),
            "price": [None] * 12,
            "upper_bound": [None] * 12,
            "lower_bound": [None] * 12,
            "indicator": ["past_predicted"] * 12,
        }
    )

    # Calculate start and end dates for fetching API data
    start_date = (first_row_date - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (first_row_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Request for pool price report
    url = "https://api.aeso.ca/report/v1.1/price/poolPrice"
    headers = {
        "accept": "application/json",
        "X-API-Key": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ6MHo4MnIiLCJpYXQiOjE2ODM1NzQyMTh9.Gbod9kjeDwP4SOJibSFof63X7GGZxbZdBmBVrgE409w",
    }
    params = {"startDate": start_date, "endDate": end_date}

    response = requests.get(url, headers=headers, params=params)
    api_data = pd.DataFrame(response.json()["return"]["Pool Price Report"])
    api_data = api_data.drop(
        ["begin_datetime_utc", "forecast_pool_price", "rolling_30day_avg"], axis=1
    )

    price_past["date"] = pd.to_datetime(price_past["date"])
    api_data["begin_datetime_mpt"] = pd.to_datetime(api_data["begin_datetime_mpt"])

    # Join the price_past and api_data on the date columns
    joined_df = pd.merge(
        price_past, api_data, left_on="date", right_on="begin_datetime_mpt"
    )

    # Fill null prices in joined_df with corresponding pool prices from api_data
    joined_df["price"] = joined_df["price"].fillna(joined_df["pool_price"])

    joined_df = joined_df.drop(["begin_datetime_mpt", "pool_price"], axis=1)

    # Import the feature data from csv and join it with joined_df
    input_features = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/filtered_features_medium.csv"
    )
    input_features = input_features[
        [
            "date",
            "wind_supply_mix",
            "wind_reserve_margin",
            "gas_supply_mix",
            "load_on_gas_reserve",
        ]
    ]
    input_features["date"] = pd.to_datetime(input_features["date"])

    joined_df = pd.merge(joined_df, input_features, left_on="date", right_on="date")

    # Combine all the data to form the final DataFrame
    predicted_price = pd.concat(
        [joined_df, price_prediction], axis=0, ignore_index=True
    )

    return predicted_price


def save_to_dbfs_and_disk(
    IS_LOCAL,
    final_feature_df,
    explain_feature_importance,
    predicted_price,
    lgbm_pipeline,
    lgbm_pipeline_low,
    lgbm_pipeline_high,
):
    """
    Function to save DataFrames to Databricks File System (DBFS) and models to disk.

    Parameters:
    IS_LOCAL (bool): Flag to check if the environment is local.
    final_feature_df (pd.DataFrame): DataFrame to be saved to DBFS.
    explain_feature_importance (pd.DataFrame): DataFrame to be saved to DBFS.
    predicted_price (pd.DataFrame): DataFrame to be saved to DBFS.
    lgbm_pipeline: Model to be saved to disk.
    lgbm_pipeline_low: Model to be saved to disk.
    lgbm_pipeline_high: Model to be saved to disk.
    """

    if not IS_LOCAL:
        spark_df = spark.createDataFrame(final_feature_df)
        spark_df = spark_df.withColumn("Base", spark_df["Base"].cast("long"))
        spark_df.write.format("delta").mode("overwrite").save(
            "dbfs:/user/hive/warehouse/shap"
        )

        df_shap_explain = spark.createDataFrame(explain_feature_importance)
        df_shap_explain.write.format("delta").mode("overwrite").save(
            "dbfs:/user/hive/warehouse/shap_explain"
        )

        df_predicted_price = spark.createDataFrame(predicted_price)
        df_predicted_price = df_predicted_price.withColumn(
            "price", df_predicted_price["price"].cast("Double")
        )
        df_predicted_price.write.format("delta").mode("overwrite").save(
            "dbfs:/user/hive/warehouse/predicted_price"
        )

        folder_path = "/dbfs/saved_models"
    else:
        folder_path = "../../saved_models"

    # Create the folder path if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the models to the disk
    joblib.dump(lgbm_pipeline, os.path.join(folder_path, "lgbm_pipeline.pkl"))
    joblib.dump(lgbm_pipeline_low, os.path.join(folder_path, "lgbm_pipeline_low.pkl"))
    joblib.dump(lgbm_pipeline_high, os.path.join(folder_path, "lgbm_pipeline_high.pkl"))

    # Save data to disk
    final_feature_df.to_csv(os.path.join(folder_path, "final_feature_df.csv"))
    explain_feature_importance.to_csv(os.path.join(folder_path, "explain_feature.csv"))
    predicted_price.to_csv(os.path.join(folder_path, "predicted_price.csv"))


def initialize_optimized_lgbm_forecaster(n_estimators=1000, device="gpu"):
    """
    Initializes and returns a LightGBM forecaster pipeline.

    Args:
        n_estimators (int, optional): The number of boosting stages to perform. Defaults to 1.

    Returns:
        ForecastingPipeline: A scikit-learn compatible pipeline for forecasting.
    """
    pipe = ForecastingPipeline(
        steps=[
            (
                "forecaster",
                TransformedTargetForecaster(
                    [
                        (
                            "forecast",
                            make_reduction(
                                LGBMRegressor(
                                    boosting_type="dart",
                                    device=device,
                                    learning_rate=0.01,
                                    max_depth=15,
                                    n_estimators=n_estimators,
                                    num_leaves=70,
                                    n_jobs=-1,
                                    reg_alpha=30,
                                    reg_lambda=20,
                                ),
                                window_length=24,
                                strategy="direct",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    return pipe


def initialize_lgbm_forecaster_quantile(
    alpha,
    n_estimators=1000,
    device="gpu",
):
    """
    Initializes and returns a LightGBM forecaster pipeline with quantile regression.

    Args:
        n_estimators (int, optional): The number of boosting stages to perform. Defaults to 1.
        alpha (float, optional): The quantile at which the prediction is made. Defaults to 0.025.

    Returns:
        ForecastingPipeline: A scikit-learn compatible pipeline for forecasting.
    """
    pipe = ForecastingPipeline(
        steps=[
            (
                "forecaster",
                TransformedTargetForecaster(
                    [
                        (
                            "forecast",
                            make_reduction(
                                LGBMRegressor(
                                    device=device,
                                    learning_rate=0.01,
                                    max_depth=15,
                                    n_estimators=n_estimators,
                                    num_leaves=70,
                                    n_jobs=-1,
                                    reg_alpha=30,
                                    reg_lambda=20,
                                    objective="quantile",
                                    alpha=alpha,
                                ),
                                window_length=24,
                                strategy="direct",
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    return pipe


def get_initial_artifacts(
    n_estimators=1000, alpha_low=0.025, alpha_high=0.975, device="gpu"
):
    """
    Initializes LightGBM forecasting pipelines, trains them on provided data, and makes predictions.

    Args:
        n_jobs (int, optional): The number of jobs to run in parallel. -1 means using all processors. Defaults to -1.
        n_estimators (int, optional): The number of boosting stages to perform. Defaults to 1.
        alpha_low (float, optional): The quantile at which the lower bound prediction is made. Defaults to 0.025.
        alpha_high (float, optional): The quantile at which the upper bound prediction is made. Defaults to 0.975.
        forecast_len (int, optional): The length of the forecast horizon. Defaults to 12

    Returns:
        Tuple: Predictions, Main forecasting pipeline, Training data, Lower bound forecasting pipeline, Upper bound forecasting pipeline.
    """

    X_train = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/train/X_train.csv",
        parse_dates=["date"],
        index_col="date",
    )

    y_train = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/train/y_train.csv",
        parse_dates=["date"],
        index_col="date",
    )

    X_train = X_train.sort_values(by="date")
    X_train = X_train.asfreq("H")
    y_train = y_train.sort_values(by="date")
    y_train = y_train.asfreq("H")

    forecast_len = 12

    lgbm_pipeline = initialize_optimized_lgbm_forecaster(
        n_estimators=n_estimators, device=device
    )

    lgbm_pipeline_low = initialize_lgbm_forecaster_quantile(
        alpha=alpha_low,
        n_estimators=n_estimators,
        device=device,
    )

    lgbm_pipeline_high = initialize_lgbm_forecaster_quantile(
        alpha=alpha_high,
        n_estimators=n_estimators,
        device=device,
    )

    fh = ForecastingHorizon(np.arange(1, forecast_len + 1))

    lgbm_pipeline.fit(y=y_train, X=X_train, fh=fh)
    lgbm_pipeline_low.fit(y=y_train, X=X_train, fh=fh)
    lgbm_pipeline_high.fit(y=y_train, X=X_train, fh=fh)

    y_pred = lgbm_pipeline.predict(fh, X=X_train.tail(1))
    y_pred_lower = lgbm_pipeline_low.predict(fh, X=X_train.tail(1))
    y_pred_higher = lgbm_pipeline_high.predict(fh, X=X_train.tail(1))

    y_pred.columns = ["predictions"]
    y_pred_lower.columns = ["lower_bound"]
    y_pred_higher.columns = ["upper_bound"]

    y_pred = pd.concat([y_pred, y_pred_lower, y_pred_higher], axis=1)

    return y_pred, X_train, lgbm_pipeline, lgbm_pipeline_low, lgbm_pipeline_high


def update_and_predict_next_steps(IS_LOCAL=False):
    """
    Function to update the existing LightGBM models with new observations and make predictions.

    Args:
        IS_LOCAL (bool, optional): A flag indicating whether the code is running in a local environment or not. Defaults to False.
        forecast_len (int, optional): The length of the forecast horizon. Defaults to 12.

    Returns:
        DataFrame: Returns a pandas DataFrame with predictions, lower bound and upper bound.
        X_test: Returns the test features DataFrame.
        lgbm_pipeline: Returns the updated LightGBM pipeline model.
        lgbm_pipeline_low: Returns the updated lower bound LightGBM pipeline model.
        lgbm_pipeline_high: Returns the updated upper bound LightGBM pipeline model.

    """
    # Read data

    forecast_len = 12

    X_test = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/test/X_test.csv",
        parse_dates=["date"],
        index_col="date",
    )

    y_test = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/test/y_test.csv",
        parse_dates=["date"],
        index_col="date",
    )

    # Prepare data
    X_test = X_test.sort_values(by="date").asfreq("H")
    y_test = y_test.sort_values(by="date").asfreq("H")
    fh = ForecastingHorizon(np.arange(1, forecast_len + 1))

    # Choose the folder path based on environment
    folder_path = "/dbfs/saved_models" if not IS_LOCAL else "../../saved_models"

    # Create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Read models from saved files
    lgbm_pipeline = joblib.load(os.path.join(folder_path, "lgbm_pipeline.pkl"))
    lgbm_pipeline_low = joblib.load(os.path.join(folder_path, "lgbm_pipeline_low.pkl"))
    lgbm_pipeline_high = joblib.load(
        os.path.join(folder_path, "lgbm_pipeline_high.pkl")
    )

    # Get the new observations
    cut_off = lgbm_pipeline.cutoff + timedelta(hours=1)
    new_observation_X = X_test.loc[cut_off].asfreq("H")
    new_observation_y = y_test.loc[cut_off].asfreq("H")

    # Update the pipelines with the new observations
    lgbm_pipeline.update(y=new_observation_y, X=new_observation_X, update_params=True)
    lgbm_pipeline_low.update(
        y=new_observation_y, X=new_observation_X, update_params=True
    )
    lgbm_pipeline_high.update(
        y=new_observation_y, X=new_observation_X, update_params=True
    )

    lgbm_pipeline.cutoff.freq = "H"

    # Make the predictions
    y_pred = lgbm_pipeline.predict(fh, X=new_observation_X)
    y_pred_lower = lgbm_pipeline_low.predict(fh, X=new_observation_X)
    y_pred_higher = lgbm_pipeline_high.predict(fh, X=new_observation_X)

    # Format the predictions
    y_pred.columns = ["predictions"]
    y_pred_lower.columns = ["lower_bound"]
    y_pred_higher.columns = ["upper_bound"]
    y_pred = pd.concat([y_pred, y_pred_lower, y_pred_higher], axis=1)

    return y_pred, X_test, lgbm_pipeline, lgbm_pipeline_low, lgbm_pipeline_high
