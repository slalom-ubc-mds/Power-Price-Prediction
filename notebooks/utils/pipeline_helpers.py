from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import requests
from sktime_custom_reduce import make_reduction
from lightgbm import LGBMRegressor
from sktime_custom_pipeline import ForecastingPipeline, TransformedTargetForecaster


def get_fold_predictions(rolling_prediction_df, y_test_full):
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

    return fold_actuals, fold_predictions_list, rmse_list


def get_plotting_df(fold_actuals, fold_predictions_list, y_hist):
    results_df = pd.DataFrame(columns=["Date", "Data", "RMSE"])

    ddf = pd.DataFrame(
        columns=["HistoricalPrice", "FuturePrice", "Predicted", "timestep"]
    )

    for i in range(len(fold_actuals)):
        df = y_hist[y_hist.index < fold_predictions_list[i].index[0]]

        df = df.iloc[-24:, :]

        predictions = np.array(fold_predictions_list[i])

        date_index = fold_actuals[i].index

        hist = pd.DataFrame(df.iloc[-12:, :]["price"]).rename(
            columns={"price": "HistoricalPrice"}
        )

        fitu = pd.DataFrame(fold_actuals[i]).rename(columns={"price": "FuturePrice"})

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
    return df


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
            predictions.append(prediction_series)

            rmse = mean_squared_error(actual, prediction_series, squared=False)

            rmses.append(rmse)
        else:
            print(f"Error: Index mismatch for {step}-step prediction.")

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

    for i in range(0, len(y_test)):
        new_observation_y, new_observation_X = (
            y_test[i : i + step_length],
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
        prediction_for = cutoff_time + pd.DateOffset(hours=i)

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
