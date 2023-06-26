import pytest
import pandas as pd
from datetime import datetime
import os

import sys

sys.path.append("src/local_testing_scripts/")

from generate_predictions import (
    create_folder,
    validate_date,
    get_train_test_split,
    generate_plot,
    check_dates,
    save_results,
)


def test_create_folder():
    create_folder("./temp_folder")
    assert os.path.exists("./temp_folder") == True

    os.rmdir("./temp_folder")
    assert os.path.exists("./temp_folder") == False


def test_validate_date():
    assert validate_date("2021-01-01") == datetime(2021, 1, 1)

    with pytest.raises(ValueError):
        validate_date("20210101")

    with pytest.raises(ValueError):
        validate_date("2021-13-01")


def test_get_train_test_split():
    model_train_start_date = datetime(2021, 1, 1)
    predict_until = datetime(2023, 5, 30)

    X_train, y_train, X_test, y_test, y_hist = get_train_test_split(
        model_train_start_date, predict_until
    )

    assert len(X_train) != 0
    assert len(y_train) != 0
    assert len(X_test) != 0
    assert len(y_test) != 0
    assert len(y_hist) != 0


def test_generate_plot():
    rmse_list = [2, 3, 4]

    data = {
        "Historical Price": [100, 105, 110, 115, 120],
        "Future Actual Price": [105, 108, 112, 115, 118],
        "Predicted": [103, 107, 111, 114, 117],
        "Predicted Upper": [105, 110, 114, 116, 119],
        "Predicted Lower": [100, 104, 109, 112, 115],
        "timestep": [1, 2, 3, 4, 5],
        "periodstep": [1, 1, 1, 1, 1],
        "index": pd.date_range(start="1/1/2021", periods=5),
    }

    # Create DataFrame
    ddf = pd.DataFrame(data)

    fig = generate_plot(rmse_list, ddf)

    assert fig.layout.title.text == "Energy Price Forecast Animation"

    with pytest.raises(KeyError):
        ddf = pd.DataFrame({"index": pd.date_range(start="1/1/2021", periods=5)})
        generate_plot(rmse_list, ddf)

    with pytest.raises(KeyError):
        rmse_list = []
        generate_plot(rmse_list, ddf)


def test_check_dates():
    with pytest.raises(ValueError):
        check_dates(datetime(2020, 12, 31), datetime(2023, 6, 30))

    with pytest.raises(ValueError):
        check_dates(datetime(2021, 1, 1), datetime(2023, 6, 30))

    check_dates(datetime(2021, 1, 1), datetime(2023, 3, 30))


def test_save_results():
    rmse_list = [2, 3, 4]
    data = {
        "Historical Price": [100, 105, 110, 115, 120],
        "Future Actual Price": [105, 108, 112, 115, 118],
        "Predicted": [103, 107, 111, 114, 117],
        "Predicted Upper": [105, 110, 114, 116, 119],
        "Predicted Lower": [100, 104, 109, 112, 115],
        "timestep": [1, 2, 3, 4, 5],
        "periodstep": [1, 1, 1, 1, 1],
        "index": pd.date_range(start="1/1/2021", periods=5),
    }

    # Create DataFrame
    ddf = pd.DataFrame(data)

    fig = generate_plot(rmse_list, ddf)
    rolling_prediction_df = pd.DataFrame(
        {"date": pd.date_range(start="1/1/2021", periods=5), "value": [2] * 5}
    )
    error_df = pd.DataFrame({"value": [3] * 5})

    save_results(fig, rolling_prediction_df, error_df, "./temp_results/")

    assert os.path.exists("./temp_results/predictions_plot.html") == True
    assert os.path.exists("./temp_results/rolling_predictions.csv") == True
    assert os.path.exists("./temp_results/rolling_predictions_rmse.csv") == True

    os.remove("./temp_results/predictions_plot.html")
    os.remove("./temp_results/rolling_predictions.csv")
    os.remove("./temp_results/rolling_predictions_rmse.csv")
    os.rmdir("./temp_results")


if __name__ == "__main__":
    test_generate_plot()
