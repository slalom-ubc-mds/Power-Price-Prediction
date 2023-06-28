"""
preprocessing.py

This module provides functions for generating rolling predictions, step predictions, and plotting animations for the predictions.

"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import requests
from sktime_custom_reduce import make_reduction
from lightgbm import LGBMRegressor
from sktime_custom_pipeline import ForecastingPipeline, TransformedTargetForecaster


def get_fold_predictions(rolling_prediction_df, y_test_full, verbose=True):
    rmse_list = []
    fold_actuals = []
    fold_predictions_list = []

    for col in range(rolling_prediction_df.shape[1] - 1):
        fold_predictions = rolling_prediction_df.iloc[:, col].dropna()

        fold_indices = fold_predictions.index

        y_test_subset = y_test_full.loc[fold_indices]

        rmse = np.sqrt(mean_squared_error(y_test_subset, fold_predictions))

        rmse_list.append(rmse)

        fold_actuals.append(y_test_subset)

        fold_predictions_list.append(fold_predictions)

    if verbose:
        # Print Average, STD RMSE of all folds
        print(f"Average RMSE for each fold: {np.mean(rmse_list)}")
        print(f"STD RMSE for each fold: {np.std(rmse_list)}")

    return fold_actuals, fold_predictions_list, rmse_list


def get_plotting_df(fold_actuals, fold_predictions_list, y_hist):
    results_df = pd.DataFrame(columns=["Date", "Data", "RMSE"])

    ddf = pd.DataFrame(
        columns=["Historical Price", "Future Actual Price", "Predicted", "timestep"]
    )

    for i in range(len(fold_actuals)):
        df = y_hist[y_hist.index < fold_predictions_list[i].index[0]]

        df = df.iloc[-24:, :]

        predictions = np.array(fold_predictions_list[i])

        date_index = fold_actuals[i].index

        hist = pd.DataFrame(df.iloc[-12:, :]["price"]).rename(
            columns={"price": "Historical Price"}
        )

        fitu = pd.DataFrame(fold_actuals[i]).rename(
            columns={"price": "Future Actual Price"}
        )

        pred = pd.DataFrame(predictions, index=date_index).rename(
            columns={0: "Predicted"}
        )

        histfitu = pd.merge(hist, fitu, how="outer", left_index=True, right_index=True)

        hfp = pd.merge(histfitu, pred, how="outer", left_index=True, right_index=True)

        hfp["timestep"] = i

        hfp["periodstep"] = range(1, len(hfp) + 1)

        hfp = hfp.reset_index()

        results_df = results_df.append(
            {"Date": df.index[-1], "Data": hfp}, ignore_index=True
        )

        ddf = pd.concat([ddf, hfp], axis=0)

    return ddf


def get_upper_lower_plotting_df(
    fold_actuals,
    fold_predictions_list,
    fold_predictions_low_list,
    fold_predictions_high_list,
    y_hist,
):
    results_df = pd.DataFrame(columns=["Date", "Data", "RMSE"])

    ddf = pd.DataFrame(
        columns=["Historical Price", "Future Actual Price", "Predicted", "timestep"]
    )

    for i in range(len(fold_actuals)):
        df = y_hist[y_hist.index < fold_predictions_list[i].index[0]]

        df = df.iloc[-24:, :]

        predictions = np.array(fold_predictions_list[i])

        predictions_upper = np.array(fold_predictions_high_list[i])

        predictions_lower = np.array(fold_predictions_low_list[i])

        date_index = fold_actuals[i].index

        hist = pd.DataFrame(df.iloc[-12:, :]["price"]).rename(
            columns={"price": "Historical Price"}
        )

        fitu = pd.DataFrame(fold_actuals[i]).rename(
            columns={"price": "Future Actual Price"}
        )

        pred = pd.DataFrame(predictions, index=date_index).rename(
            columns={0: "Predicted"}
        )

        pred_upper = pd.DataFrame(predictions_upper, index=date_index).rename(
            columns={0: "Predicted Upper"}
        )

        pred_lower = pd.DataFrame(predictions_lower, index=date_index).rename(
            columns={0: "Predicted Lower"}
        )

        histfitu = pd.merge(hist, fitu, how="outer", left_index=True, right_index=True)

        hfp = pd.merge(histfitu, pred, how="outer", left_index=True, right_index=True)

        hfp = pd.merge(hfp, pred_upper, how="outer", left_index=True, right_index=True)

        hfp = pd.merge(hfp, pred_lower, how="outer", left_index=True, right_index=True)

        hfp["timestep"] = i

        hfp["periodstep"] = range(1, len(hfp) + 1)

        hfp = hfp.reset_index()

        results_df = results_df.append(
            {"Date": df.index[-1], "Data": hfp}, ignore_index=True
        )

        ddf = pd.concat([ddf, hfp], axis=0)

    return ddf


def get_aeso_predictions(start_date, end_date):
    url = "https://api.aeso.ca/report/v1.1/price/poolPrice"
    headers = {
        "accept": "application/json",
        "X-API-Key": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ6MHo4MnIiLCJpYXQiOjE2ODM1NzQyMTh9.Gbod9kjeDwP4SOJibSFof63X7GGZxbZdBmBVrgE409w",
    }
    params = {
        "startDate": start_date.date().strftime("%Y-%m-%d"),
        "endDate": end_date.date().strftime("%Y-%m-%d"),
    }

    response = requests.get(url, headers=headers, params=params)

    data = response.json()["return"]["Pool Price Report"]
    df = pd.DataFrame(data)
    df["actual"] = pd.to_numeric(df["pool_price"])
    df["forecast"] = pd.to_numeric(df["forecast_pool_price"])

    rmse_aeso_predictions = mean_squared_error(
        df["actual"], df["forecast"], squared=False
    )

    print(
        f"One step prediction errors for AESO forecasts: {round(rmse_aeso_predictions, 2)} CAD/MWh.\nAs these are one step predictions, the error should be lesser than ours since ours is 12 step prediction errors."
    )


def generate_step_predictions(rolling_prediction_df, y_test_full, num_steps):
    step_predictions = []

    for step in range(0, num_steps):
        diag_values = np.diag(rolling_prediction_df.values, -step)

        index_range = y_test_full.index[step : step + len(diag_values)]

        column_name = f"{step+1}_step_prediction"

        prediction_df = pd.DataFrame(
            diag_values, index=index_range, columns=[column_name]
        )

        if y_test_full[step : step + len(prediction_df)].index.equals(
            prediction_df.index
        ):
            step_predictions.append(prediction_df)
        else:
            print(f"Error: Index mismatch for {step}-step prediction.")

    return step_predictions


def generate_step_errors(
    predictions,
    y_test_full,
    forecast_len,
):
    step_sizes = np.arange(1, forecast_len + 1)
    actuals = []
    rmses = []

    for step, prediction_series in zip(step_sizes, predictions):
        filtered_test = y_test_full[step - 1 : step - 1 + len(prediction_series)]

        if filtered_test.index.equals(prediction_series.index):
            actual = y_test_full[step - 1 : step - 1 + len(prediction_series)]["price"]

            actuals.append(actual)

            rmse = mean_squared_error(actual, prediction_series, squared=False)

            rmses.append(rmse)
        else:
            print(f"Error: Index mismatch for {step}-step prediction.")

    for step, rmse in zip(range(1, forecast_len + 1), rmses):
        print(f"{step} Step RMSE for model: {rmse}")

    return actuals, rmses


def get_rolling_predictions(
    pipeline, X_train, X_test, y_test_full, fh, step_length, forecast_len, verbose=True
):
    y_test = y_test_full[:-forecast_len]

    rolling_prediction_df = pd.DataFrame(index=y_test_full.index)

    y_pred = pipeline.predict(fh, X=X_train.tail(1))
    y_pred.columns = [f"cutoff_hour_{pipeline.cutoff.hour[0]}"]
    rolling_prediction_df = pd.concat([rolling_prediction_df, y_pred], axis=1)

    # emulating the rolling prediction for the next hours

    for i in range(0, len(y_test), step_length):
        new_observation_y, new_observation_X = (
            y_test_full[i : i + step_length],
            X_test[i : i + step_length],
        )

        new_observation_y = new_observation_y.asfreq("H")
        new_observation_X = new_observation_X.asfreq("H")

        if verbose:
            print(f"Updating with actual values at {new_observation_y.index[0]}")
            print(f"Cut off before update: {pipeline.cutoff}")

        pipeline.update(y=new_observation_y, X=new_observation_X, update_params=True)

        if verbose:
            print(f"Cut off after update: {pipeline.cutoff}")

        pipeline.cutoff.freq = "H"

        cutoff_time = pipeline.cutoff
        prediction_for = cutoff_time + pd.DateOffset(hours=step_length)

        if verbose:
            print(f"Predicting for {prediction_for}")

        y_pred = pipeline.predict(fh, X=new_observation_X)

        y_pred.columns = [f"cutoff_hour_{pipeline.cutoff.hour[0]}"]

        rolling_prediction_df = pd.concat([rolling_prediction_df, y_pred], axis=1)

        if verbose:
            print(f"Update and prediction done for {new_observation_y.index[0]}")
            print(
                "----------------------------------------------------------------------------------"
            )

    return rolling_prediction_df


def get_rolling_predictions_with_bounds(
    pipeline,
    pipeline_low,
    pipeline_high,
    X_train,
    X_test,
    y_test_full,
    fh,
    step_length,
    forecast_len,
    verbose=True,
):
    y_test = y_test_full[:-forecast_len]

    rolling_prediction_df = pd.DataFrame(index=y_test_full.index)
    rolling_prediction_df_low = pd.DataFrame(index=y_test_full.index)
    rolling_prediction_df_high = pd.DataFrame(index=y_test_full.index)

    y_pred = pipeline.predict(fh, X=X_train.tail(1))
    y_pred.columns = [f"cutoff_hour_{pipeline.cutoff.hour[0]}"]
    rolling_prediction_df = pd.concat([rolling_prediction_df, y_pred], axis=1)

    y_pred_low = pipeline_low.predict(fh, X=X_train.tail(1))
    y_pred_low.columns = [f"cutoff_hour_{pipeline_low.cutoff.hour[0]}"]
    rolling_prediction_df_low = pd.concat(
        [rolling_prediction_df_low, y_pred_low], axis=1
    )

    y_pred_high = pipeline_high.predict(fh, X=X_train.tail(1))
    y_pred_high.columns = [f"cutoff_hour_{pipeline_high.cutoff.hour[0]}"]
    rolling_prediction_df_high = pd.concat(
        [rolling_prediction_df_high, y_pred_high], axis=1
    )

    # emulating the rolling prediction for the next hours

    for i in range(0, len(y_test), step_length):
        new_observation_y, new_observation_X = (
            y_test_full[i : i + step_length],
            X_test[i : i + step_length],
        )

        new_observation_y = new_observation_y.asfreq("H")
        new_observation_X = new_observation_X.asfreq("H")

        if verbose:
            print(f"Updating with actual values at {new_observation_y.index[0]}")
            print(f"Cut off before update: {pipeline.cutoff}")

        pipeline.update(y=new_observation_y, X=new_observation_X, update_params=True)
        pipeline_low.update(
            y=new_observation_y, X=new_observation_X, update_params=True
        )
        pipeline_high.update(
            y=new_observation_y, X=new_observation_X, update_params=True
        )

        if verbose:
            print(f"Cut off after update: {pipeline.cutoff}")

        pipeline.cutoff.freq = "H"

        cutoff_time = pipeline.cutoff
        prediction_for = cutoff_time + pd.DateOffset(hours=i)

        if verbose:
            print(f"Predicting for {prediction_for}")

        y_pred = pipeline.predict(fh, X=new_observation_X)
        y_pred_low = pipeline_low.predict(fh, X=new_observation_X)
        y_pred_high = pipeline_high.predict(fh, X=new_observation_X)

        y_pred.columns = [f"cutoff_hour_{pipeline.cutoff.hour[0]}"]
        y_pred_low.columns = [f"cutoff_hour_{pipeline_low.cutoff.hour[0]}"]
        y_pred_high.columns = [f"cutoff_hour_{pipeline_high.cutoff.hour[0]}"]

        rolling_prediction_df = pd.concat([rolling_prediction_df, y_pred], axis=1)
        rolling_prediction_df_low = pd.concat(
            [rolling_prediction_df_low, y_pred_low], axis=1
        )
        rolling_prediction_df_high = pd.concat(
            [rolling_prediction_df_high, y_pred_high], axis=1
        )

        if verbose:
            print(f"Update and prediction done for {new_observation_y.index[0]}")
            print(
                "----------------------------------------------------------------------------------"
            )

    return rolling_prediction_df, rolling_prediction_df_low, rolling_prediction_df_high


def initialize_default_lgbm_forecaster(num_threads=-1, n_estimators=1000, device="gpu"):
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
                                    n_jobs=num_threads,
                                    n_estimators=n_estimators,
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


def initialize_optimized_lgbm_forecaster(
    num_threads=-1, n_estimators=1000, device="gpu"
):
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
                                    n_jobs=num_threads,
                                    n_estimators=n_estimators,
                                    learning_rate=0.01,
                                    max_depth=15,
                                    num_leaves=70,
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


def initialize_optimized_lgbm_lower_bound():
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
                                    device="gpu",
                                    learning_rate=0.01,
                                    max_depth=15,
                                    min_data_in_leaf=20,
                                    n_estimators=1000,
                                    num_leaves=70,
                                    num_threads=12,
                                    reg_alpha=30,
                                    reg_lambda=20,
                                    objective="quantile",
                                    alpha=0.025,
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


def initialize_optimized_lgbm_upper_bound():
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
                                    device="gpu",
                                    learning_rate=0.01,
                                    max_depth=15,
                                    min_data_in_leaf=20,
                                    n_estimators=1000,
                                    num_leaves=70,
                                    num_threads=12,
                                    reg_alpha=30,
                                    reg_lambda=20,
                                    objective="quantile",
                                    alpha=0.975,
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
