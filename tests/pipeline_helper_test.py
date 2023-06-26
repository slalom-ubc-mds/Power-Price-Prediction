import pandas as pd
import numpy as np
import pytest
import sys

sys.path.append("src/deployed_scripts/")

from pipeline_helper import *


def test_generate_sentence_dataframe():
    # Define the input DataFrame
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="1/1/2020", periods=10),
            "Feature": ["price"] * 10,
            "Value": np.random.rand(10),
            "Base": np.random.rand(10),
        }
    )

    # Test the function
    output = generate_sentence_dataframe(df)

    # Assert that the output is a DataFrame
    assert isinstance(output, pd.DataFrame)

    # Assert that the output has the expected number of columns
    assert len(output.columns) == 2

    # Assert that the output has the expected column names
    assert list(output.columns) == ["Date", "Sentence"]

    # Assert that the sentence is correctly formatted
    assert (
        "The average power price of the past month is $" in output["Sentence"].iloc[0]
    )


def test_generate_tableau_required_dataframe():
    # Prepare a sample DataFrame for testing
    data = {
        "predictions": [
            74.380361,
            62.697907,
            65.038264,
            57.072062,
            58.505975,
            73.002258,
            83.119340,
            107.230484,
            90.060522,
            79.584282,
            77.160666,
            72.709050,
        ],
        "lower_bound": [
            58.722277,
            50.048370,
            51.530106,
            46.288219,
            48.008368,
            45.724813,
            47.939297,
            47.148052,
            46.426701,
            44.567575,
            43.531427,
            42.791301,
        ],
        "upper_bound": [
            174.589122,
            203.500652,
            238.746433,
            181.062615,
            241.152583,
            271.126336,
            305.724652,
            379.961233,
            372.202719,
            348.386136,
            483.350045,
            384.818586,
        ],
    }

    df = pd.DataFrame(data)

    # Set the index to be the timestamp
    df.index = pd.date_range("2023-02-01", periods=len(df), freq="H")

    # Call the function with the sample DataFrame
    result = generate_tableau_required_dataframe(df)

    # Assert that the output is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that the output has the expected number of columns
    assert len(result.columns) == 9

    # Assert that the output has the expected column names
    assert list(result.columns) == [
        "date",
        "price",
        "upper_bound",
        "lower_bound",
        "indicator",
        "wind_supply_mix",
        "wind_reserve_margin",
        "gas_supply_mix",
        "load_on_gas_reserve",
    ]


def test_initialize_optimized_lgbm_forecaster():
    # Call the function with default arguments
    pipeline = initialize_optimized_lgbm_forecaster()

    # Assert that the returned object is an instance of Pipeline
    pipeline = initialize_optimized_lgbm_forecaster()
    assert isinstance(pipeline, ForecastingPipeline)

    forecaster = pipeline.steps[0][1]["forecast"]
    assert isinstance(forecaster, ForecastingPipeline)


if __name__ == "__main__":
    pytest.main()
