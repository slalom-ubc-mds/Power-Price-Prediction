import os
import re
import sys
import pandas as pd

sys.path.append("notebooks/utils")

from preprocess_helper import *

def compute_weekly_profile(row):
    day_of_week = row.name.dayofweek
    peak_or_not = row["peak_or_not"]

    if day_of_week in [1, 2] and peak_or_not == 1:
        return 6
    elif day_of_week in [0, 3] and peak_or_not == 1:
        return 5
    elif day_of_week in [4, 5, 6] and peak_or_not == 1:
        return 4
    elif day_of_week in [1, 2] and peak_or_not == 0:
        return 3
    elif day_of_week in [0, 3] and peak_or_not == 0:
        return 2
    elif day_of_week in [4, 5, 6] and peak_or_not == 0:
        return 1
    else:
        return 0

def save_df_to_csv(df, dir_path, file_name):
    # Ensure the directories exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Save dataframe to csv
    file_path = os.path.join(dir_path, file_name)
    df.to_csv(file_path)

# Preprocess intertie data
def preprocess_data():
    try:
        preprocess_intertie_data()
    except Exception as e:
        print(f"Error while running preprocess_intertie_data: {str(e)}")
        sys.exit(1)

    # Process supply data
    try:
        process_supply_data()
    except Exception as e:
        print(f"Error while running process_supply_data: {str(e)}")
        sys.exit(1)

    # Merge data
    try:
        merge_data()
    except Exception as e:
        print(f"Error while running merge_data: {str(e)}")
        sys.exit(1)

    print("Started the feature selection...")

    # Load the preprocessed features dataset

    price_old_df = pd.read_csv(
        "data/processed/preprocessed_features.csv",
        parse_dates=["date"],
        index_col="date",
    )

    # Resample the dataframe by hour and sort it by date
    price_old_df = price_old_df.asfreq("H").sort_values(by="date")

    # Rename the columns for clarity
    price_old_df = price_old_df.rename(columns={
        "calgary": "calgary_load",
        "central": "central_load",
        "edmonton": "edmonton_load",
        "northeast": "northeast_load",
        "northwest": "northwest_load",
        "south": "south_load",
    })

    # Multiply the specified columns by 100 to make it percentages
    selected_columns = [col for col in price_old_df.columns if col.endswith(("_reserve_margin", "_supply_mix", "_ratio"))]

    # Add other columns
    selected_columns.extend(["relative_gas_reserve", "load_on_gas_reserve"])

    # Apply the operation on the selected columns
    price_old_df.loc[:, selected_columns] *= 100

    # Apply the operation for the 'gas_cost' column
    price_old_df["gas_cost"] /= 100

    # Create X and y
    y = price_old_df[["price"]]
    y = y.asfreq("H")

    X = price_old_df.drop(columns=["price"])
    X = X.asfreq("H")

    # Calculate statistics as features
    window = 24
    rolling_y = y.rolling(window)

    # Calculate rolling statistics
    X[["rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_median"]] = rolling_y.agg(["mean", "std", "min", "max", "median"])

    # Calculate exponential moving average
    X["exp_moving_avg"] = y.ewm(span=window).mean()


    X.dropna(inplace=True)
    y = y.loc[X.index]

    # Feature engineer weekly profile based on peak hours and day of the week
    X["season"] = X["season"].replace({"WINTER": 1, "SUMMER": 0})
    X["peak_or_not"] = X["peak_or_not"].replace({"ON PEAK": 1, "OFF PEAK": 0})

    X["weekly_profile"] = X.apply(compute_weekly_profile, axis=1)

    # Specify your date ranges
    dates = [
        ("2021-01-01", "2021-06-31"),
        ("2021-07-01", "2021-12-25"),
        ("2021-12-26", "2021-12-31"),
        ("2022-01-01", "2022-06-31"),
        ("2022-07-01", "2022-12-25"),
        ("2022-12-26", "2023-05-31"),
    ]

    # Get the data for system marginal price and volume for each date range and compute averages and sums
    average_dfs = []
    sum_dfs = []
    for start_date, end_date in dates:
        data = get_data(start_date, end_date)

        average_df = data[["volume", "system_marginal_price"]].resample("H").mean()
        average_df.columns = ["volume_avg", "system_marginal_price_avg"]
        average_dfs.append(average_df)

        sum_df = data[["volume", "system_marginal_price"]].resample("H").sum()
        sum_df.columns = ["volume_sum", "system_marginal_price_sum"]
        sum_dfs.append(sum_df)

    # You now have lists of average and sum dataframes. 
    all_averages = pd.concat(average_dfs)
    all_sums = pd.concat(sum_dfs)

    # created dataframe for system marginal price (average and sum)
    smp_df = pd.merge(all_averages, all_sums, left_index=True, right_index=True)
    smp_df = smp_df.asfreq("H")

    # Impute missing values
    X = pd.merge(X, smp_df, left_index=True, right_index=True)
    X = X.asfreq("H")
    y = y.asfreq("H")
    X["volume_avg"].fillna(X["volume_avg"].mean(), inplace=True)
    X["system_marginal_price_avg"].fillna(
        X["system_marginal_price_avg"].mean(), inplace=True
    )
    float64_cols = X.select_dtypes(include=["float64"]).columns.tolist()
    X[float64_cols] = X[float64_cols].astype("float32")

    # Change weekly_profile, season, peak_or_not to int
    X["weekly_profile"] = X["weekly_profile"].astype("int32")
    X["season"] = X["season"].astype("int32")
    X["peak_or_not"] = X["peak_or_not"].astype("int32")

    # sorting the features based on useful features, this was selected based on the EDA and coefficients from Random forest regressor model and Elastic Net CV model
    sorted_useful_values = [
        "fossil_fuel_ratio",
        "renewable_energy_ratio",
        "gas_supply_mix",
        "wind_supply_mix",
        "system_marginal_price_avg",
        "system_marginal_price_sum",
        "gas_cost",
        "hydro_reserve_margin",
        "wind_reserve_margin",
        "other_reserve_margin",
        "gas_reserve_margin",
        "rolling_std",
        "rolling_median",
        "rolling_min",
        "rolling_max",
        "exp_moving_avg",
        "rolling_mean",
        "gas_tng",
        "relative_gas_reserve",
        "hydro_tng",
        "load_on_gas_reserve",
        "northwest_load",
        "calgary_load",
        "system_load",
        "demand_supply_ratio",
        "total_reserve_margin",
        "volume_sum",
        "volume_avg",
        "weekly_profile"
    ]

    #Replace prices less than zero as 5 to avoid problems with log
    y[y < 5] = 5

    X.index.name = "date"
    y.index.name = "date"
    X = X.asfreq("H")
    y = y.asfreq("H")

    # export again
    try:
        folder_path = "data/processed/complete_data"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the DataFrame to a file inside the folder
        file_path = os.path.join(folder_path, "features.csv")
        pd.DataFrame(X[sorted_useful_values]).to_csv(file_path)

        file_path = os.path.join(folder_path, "target.csv")
        pd.DataFrame(y).to_csv(file_path)
    except Exception as e:
        print(f"Error while saving features and target: {str(e)}")
        sys.exit(1)

    # train test split
    try:
                # Data split
        X_train = X[sorted_useful_values].loc["2021-01-01":"2023-01-31"]
        X_test = X[sorted_useful_values].loc["2023-02-01":]
        y_train = y.loc["2021-01-01":"2023-01-31"]
        y_test = y.loc["2023-02-01":]

        # Save to csv
        save_df_to_csv(X_train, "data/processed/train", "X_train.csv")
        save_df_to_csv(X_test, "data/processed/test", "X_test.csv")
        save_df_to_csv(y_train, "data/processed/train", "y_train.csv")
        save_df_to_csv(y_test, "data/processed/test", "y_test.csv")
    except Exception as e:
        print(f"Error while doing train-test split and saving: {str(e)}")
        sys.exit(1)

    print("Completed the feature selection...")

if __name__ == "__main__":
    preprocess_data()