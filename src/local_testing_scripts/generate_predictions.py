"""
Usage:
  script.py --model_train_start_date=<date> --predict_until=<date> [--n_estimators=<value>] [--device=<device_name>]

Options:
  --model_train_start_date=<date>    Start date of model training (format: YYYY-MM-DD)
  --predict_until=<date>             Prediction end date (format: YYYY-MM-DD)
  --n_estimators=<value>             Number of estimators (1-1000)
  --device=<device_name>             Device name for training (e.g., cpu, gpu)
"""

from datetime import datetime
from docopt import docopt

import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("notebooks/utils/")

import pipeline_helpers as ph

import os


def create_folder(folder_path):
    """
    Create a folder at the specified path if it does not already exist.

    Parameters
    ----------
    folder_path : str
        The path of the folder to be created.

    Returns
    -------
    None

    Notes
    -----
    This function uses the `os.makedirs` function to create the folder. If the folder
    already exists, a message indicating the existence of the folder is printed.

    Examples
    --------
    >>> create_folder('path/to/folder')
    Folder created: path/to/folder

    >>> create_folder('path/to/existing_folder')
    Folder already exists: path/to/existing_folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def validate_date(date_str):
    """
    Checks whether a string is a valid date in the format "YYYY-MM-DD".

    Parameters
    ----------
    date_str : str
        The string representing the date in the format "YYYY-MM-DD".

    Returns
    -------
    datetime.datetime
        The parsed date as a datetime object.

    Raises
    ------
    ValueError
        If the date_str is not in the valid format "YYYY-MM-DD".

    Examples
    --------
    >>> validate_date("2023-06-25")
    datetime.datetime(2023, 6, 25, 0, 0)

    >>> validate_date("2023/06/25")
    ValueError: Invalid date format. Please use YYYY-MM-DD.

    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return date
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")


def get_train_test_split(model_train_start_date, predict_until):
    """
    Fetches and prepares the training and testing data for a power price prediction model.

    Parameters:
    ----------
    model_train_start_date : str
        The start date for the training data in the format "YYYY-MM-DD".
    predict_until : int
        The number of records to include in the test data.

    Returns:
    -------
    X_train : pandas.DataFrame
        The training input features as a pandas DataFrame.
    y_train : pandas.DataFrame
        The training target variable as a pandas DataFrame.
    X_test : pandas.DataFrame
        The testing input features as a pandas DataFrame.
    y_test : pandas.DataFrame
        The testing target variable as a pandas DataFrame.
    y_hist : pandas.DataFrame
        The historical target variable used for reference as a pandas DataFrame.
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


def generate_plot(rmse_list, ddf):
    """
    Generate an animated plot of energy price forecast.

    Parameters:
    rmse_list (list): A list of root mean square error values.
    ddf (pandas.DataFrame): A DataFrame containing the data for plotting.

    Returns:
    plotly.graph_objects.Figure: An animated plot of energy price forecast.

    """
    ddf["date"] = ddf["index"]

    frames = []

    labels = [
        "Historical Price",
        "Future Actual Price",
        "Predicted",
        "Predicted Upper",
        "Predicted Lower",
    ]

    for timestep in ddf["timestep"].unique():
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=ddf.loc[ddf["timestep"] == timestep, "periodstep"],
                    y=ddf.loc[ddf["timestep"] == timestep, label],
                    mode="markers+lines",
                    name=label,  # Setting the name attribute for each trace
                    customdata=ddf.loc[ddf["timestep"] == timestep, ["date"]],
                    hovertemplate="<br>Label=%{fullData.name}<br>Price=%{y}<br>Date=%{customdata[0]}<extra></extra>",
                )
                for label in labels
            ]
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0]["data"],
        layout=go.Layout(
            title=dict(text="Energy Price Forecast Animation", font=dict(size=22)),
            xaxis=dict(ticks="", title_text="", showticklabels=False),
            yaxis=dict(title_text="Price", title_font=dict(size=20)),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 2000, "redraw": False}}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                    ],
                )
            ],
            annotations=[
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Average RMSE of predictions: {round(np.mean(rmse_list), 2)} CAD",
                    showarrow=False,
                    font=dict(size=20),
                )
            ],
            hovermode="x unified",
            height=700,
        ),
        frames=frames,
    )

    fig.show()
    return fig


def check_dates(model_train_start_date, predict_until):
    """
    Checks whether the provided dates for model training and prediction are within the valid range.

    Parameters
    ----------
    model_train_start_date : datetime.datetime
        The start date for model training.

    predict_until : datetime.datetime
        The end date for prediction.

    Raises
    ------
    ValueError
        If `model_train_start_date` is not within the valid range or if `predict_until` is not within the valid range.

    Notes
    -----
    The valid range for `model_train_start_date` is from January 1, 2021, to December 31, 2022 (inclusive).

    The valid range for `predict_until` is from February 1, 2023, to May 29, 2023 (inclusive).
    """
    if model_train_start_date < datetime(
        2021, 1, 1
    ) or model_train_start_date > datetime(2022, 12, 31):
        raise ValueError(
            "model_train_start_date should be greater than January 1, 2021, and less than December 31, 2022."
        )

    if predict_until <= datetime(2023, 1, 31) or predict_until >= datetime(2023, 5, 30):
        raise ValueError(
            "predict_until should be greater than January 31, 2023, and less than May 30, 2023."
        )


def save_results(fig, rolling_prediction_df, error_df, results_path="results/"):
    """
    Saves prediction results and error data to files.

    Parameters:
        fig (plotly.graph_objs._figure.Figure): The plotly figure object containing the predictions plot.
        rolling_prediction_df (pandas.DataFrame): The DataFrame containing rolling predictions.
        error_df (pandas.DataFrame): The DataFrame containing error data.
        results_path (str, optional): The path to the results directory. Defaults to "results/".

    Returns:
        None

    """
    create_folder(results_path)

    fig.write_html(
        results_path + "predictions_plot.html", auto_play=False, include_plotlyjs=True
    )

    print("Plotting complete...")

    rolling_prediction_df.index.name = "date"

    rolling_prediction_df.to_csv(results_path + "rolling_predictions.csv")

    print("Saving rolling predictions complete...")

    error_df.to_csv(results_path + "rolling_predictions_rmse.csv", index=False)


def main(args):
    """Main function to run the script."""
    model_train_start_date = validate_date(args["--model_train_start_date"])
    predict_until = validate_date(args["--predict_until"])
    n_estimators = int(args["--n_estimators"])
    device = args["--device"]

    check_dates(model_train_start_date, predict_until)

    X_train, y_train, X_test, y_test, y_hist = get_train_test_split(
        model_train_start_date, predict_until
    )

    print(
        "Downloading data from {} to {} complete...".format(
            model_train_start_date, predict_until
        )
    )

    lgbm_pipeline = ph.initialize_optimized_lgbm_forecaster(
        n_estimators=n_estimators, device=device
    )

    lgbm_pipeline_low = ph.initialize_lgbm_forecaster_quantile(
        0.025, n_estimators=n_estimators, device=device
    )

    lgbm_pipeline_high = ph.initialize_lgbm_forecaster_quantile(
        0.975, n_estimators=n_estimators, device=device
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
        step_length,
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

    fig = generate_plot(rmse_list, ddf)

    print("Plotting complete...")

    rolling_prediction_df.index.name = "date"

    data = {
        f"{step}_step_rmse": [rmse]
        for step, rmse in zip(range(1, forecast_len + 1), rmses)
    }

    error_df = pd.DataFrame(data)
    error_df["model_start_date"] = model_train_start_date
    error_df["prediction_end_date"] = predict_until
    error_df["avg_fold_rmse"] = round(np.mean(rmse_list), 2)

    save_results(fig, rolling_prediction_df, error_df)

    print("Script complete...")


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
