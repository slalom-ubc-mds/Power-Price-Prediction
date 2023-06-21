"""
Usage:
  script.py --model_train_start_date=<date> --predict_until=<date> [--n_estimators=<value>]

Options:
  --model_train_start_date=<date>    Start date of model training (format: YYYY-MM-DD)
  --predict_until=<date>             Prediction end date (format: YYYY-MM-DD)
  --n_estimators=<value>             Number of estimators (1-1000)
"""

from datetime import datetime
from docopt import docopt

import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_squared_error
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("../../notebooks/utils/")

import pipeline_helpers as ph


def validate_date(date_str):
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return date
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")


def get_train_test_split(model_train_start_date, predict_until):
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

    X_train = X_train[model_train_start_date:]
    y_train = y_train[model_train_start_date:]

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

    X_test = X_test.sort_values(by="date")
    X_test = X_test.asfreq("H")
    y_test = y_test.sort_values(by="date")
    y_test = y_test.asfreq("H")

    X_test = X_test[:predict_until]
    y_test = y_test[:predict_until]

    y_hist = pd.read_csv(
        "https://raw.githubusercontent.com/slalom-ubc-mds/Power-Price-Prediction/main/data/processed/filtered_target_medium.csv",
        parse_dates=["date"],
        index_col="date",
    )

    y_hist = y_hist.sort_values(by="date")
    y_hist = y_hist.asfreq("H")
    return X_train, y_train, X_test, y_test, y_hist


if __name__ == "__main__":
    args = docopt(__doc__)

    model_train_start_date = validate_date(args["--model_train_start_date"])
    predict_until = validate_date(args["--predict_until"])
    n_estimators = int(args["--n_estimators"])

    if model_train_start_date < datetime(
        2021, 1, 1
    ) or model_train_start_date > datetime(2022, 12, 31):
        raise ValueError(
            "model_train_start_date should be greater than 1st Jan 2021 and less than Dec 31st 2022."
        )

    if predict_until <= datetime(2023, 1, 31) or predict_until >= datetime(2023, 5, 30):
        raise ValueError(
            "predict_until should be greater than Jan 31st 2023 and less than May 30th 2023."
        )

    X_train, y_train, X_test, y_test, y_hist = get_train_test_split(
        model_train_start_date, predict_until
    )

    print(
        "Downloading data from {} to {} complete...".format(
            model_train_start_date, predict_until
        )
    )

    lgbm_pipeline = ph.initialize_optimized_lgbm_forecaster(n_estimators=n_estimators)
    lgbm_pipeline_low = ph.initialize_lgbm_forecaster_quantile(
        0.025, n_estimators=n_estimators
    )
    lgbm_pipeline_high = ph.initialize_lgbm_forecaster_quantile(
        0.975, n_estimators=n_estimators
    )

    fh = ForecastingHorizon(np.arange(1, 12 + 1))

    forecast_len = 12
    step_length = 1

    lgbm_pipeline.fit(y=y_train, X=X_train, fh=fh)
    lgbm_pipeline_low.fit(y=y_train, X=X_train, fh=fh)
    lgbm_pipeline_high.fit(y=y_train, X=X_train, fh=fh)

    print("Model training complete...")

    (
        rolling_prediction_df,
        rolling_prediction_df_low,
        rolling_prediction_df_high,
    ) = ph.get_rolling_predictions_with_bounds(
        lgbm_pipeline,
        lgbm_pipeline_low,
        lgbm_pipeline_high,
        X_train,
        X_test,
        y_test,
        fh,
        1,
        forecast_len,
        verbose=False,
    )

    print("Generating rolling predictions complete...")

    fold_actuals, fold_predictions_list, rmse_list = ph.get_fold_predictions(
        rolling_prediction_df, y_test
    )

    _, fold_predictions_low_list, _ = ph.get_fold_predictions(
        rolling_prediction_df_low, y_test, False
    )

    _, fold_predictions_high_list, _ = ph.get_fold_predictions(
        rolling_prediction_df_high, y_test, False
    )

    predictions = ph.generate_step_predictions(
        rolling_prediction_df, y_test, forecast_len
    )

    actuals, rmses = ph.generate_step_errors(predictions, y_test, forecast_len)

    ddf = ph.get_upper_lower_plotting_df(
        fold_actuals=fold_actuals,
        fold_predictions_list=fold_predictions_list,
        fold_predictions_low_list=fold_predictions_low_list,
        fold_predictions_high_list=fold_predictions_high_list,
        y_hist=y_hist,
    )

    fig = px.line(
        ddf,
        x="periodstep",
        y=[
            "HistoricalPrice",
            "FuturePrice",
            "Predicted",
            "Predicted Upper",
            "Predicted Lower",
        ],
        animation_frame="timestep",
    )
    fig.update_layout(
        height=700, title="Energy Price Forecast Animation", title_font=dict(size=22)
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

    # Set y-axis label with increased font size
    fig.update_yaxes(title_text="Price", title_font=dict(size=20))

    # Remove x-axis ticks
    fig.update_xaxes(ticks="", title_text="", showticklabels=False)

    # Add HTML text annotation for RMSE values with increased font size
    annotations = [
        dict(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Average RMSE of predictions: {round(np.mean(rmse_list), 2)} CAD",
            showarrow=False,
            font=dict(size=20),
        )
    ]

    fig.update_layout(annotations=annotations)

    fig.show()
    fig.write_html("predictions_plot.html")

    print("Plotting complete...")

    rolling_prediction_df.index.name = "date"
    # Save predictions to csv
    rolling_prediction_df.to_csv("rolling_predictions.csv")

    print("Saving rolling predictions complete...")

    print("Script complete...")
