import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import warnings
import os


def preprocess_intertie_data():
    warnings.filterwarnings("ignore")
    plt.style.use("ggplot")
    plt.rcParams.update(
        {"font.size": 14, "axes.labelweight": "bold", "lines.linewidth": 2}
    )

    try:
        print("Started the preprocessing of data...")
        # Read data from CSV file
        intertie_df = pd.read_csv(
            "data/raw/intertie.csv", parse_dates=["Date (MST)"], index_col="Date (MST)"
        )

        intertie_df["imported"] = intertie_df[["BC", "MT", "SK"]].apply(
            lambda row: row[row > 0].sum(), axis=1
        )
        intertie_df["exported"] = intertie_df[["BC", "MT", "SK"]].apply(
            lambda row: row[row < 0].sum(), axis=1
        )
        intertie_df["total_flow"] = (
            intertie_df["WECC"] + intertie_df["SK"]
        )  # net flow between alberta and the interties

        # Preprocess the data
        intertie_df = intertie_df.rename_axis("date")  # Set the index name to 'date'
        intertie_df = intertie_df.sort_values(by="date")  # Sort the DataFrame by date
        intertie_df = intertie_df.asfreq("H")  # Resample to hourly frequency
        intertie_df = intertie_df.drop(
            "Date - MST", axis=1
        )  # Drop the 'Date - MST' column

        # Column mapping for renaming columns
        column_mapping = {
            "ATC SK Export": "atc_sk_export",
            "ATC SK Import": "atc_sk_import",
            "ATC WECC Export": "atc_wecc_export",
            "ATC WECC Import": "atc_wecc_import",
            "TTC SK Export": "ttc_sk_export",
            "TTC SK Import": "ttc_sk_import",
            "TTC WECC Export": "ttc_wecc_export",
            "TTC WECC Import": "ttc_wecc_import",
            "BC": "bc",
            "MT": "mt",
            "SK": "sk",
            "WECC": "wecc",
        }

        # Rename the columns using the column mapping
        intertie_df = intertie_df.rename(columns=column_mapping)

        # Define the folder path for saving the processed data
        folder_path = "data/processed"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the DataFrame to a file inside the folder
        file_path = os.path.join(folder_path, "intertie.csv")
        intertie_df.to_csv(file_path)

        print("Intertie data preprocessing completed.")

    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")


def process_supply_data():
    print("Started the preprocessing of supply data...")

    warnings.filterwarnings("ignore")

    # Get the load and price data
    ail_df = pd.read_csv(
        "data/raw/ail_price.csv", parse_dates=["Date (MST)"], index_col="Date (MST)"
    )
    ail_df = ail_df.rename_axis("date")
    ail_df = ail_df.asfreq("H")
    ail_df = ail_df.sort_values(by="date")

    # get the supply data
    supply_df = pd.read_csv(
        "data/raw/gen.csv", parse_dates=["Date (MST)"], index_col="Date (MST)"
    )
    supply_df = supply_df.sort_values(by="Date (MST)")

    reset_df = supply_df.reset_index()

    # For each factor,  transform the tables
    gen_df = reset_df[["Date (MST)", "Fuel Type", "Total Generation"]]
    wide_tng_df = gen_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="Total Generation"
    )

    sys_df = reset_df[["Date (MST)", "Fuel Type", "System Generation"]]
    wide_sys_df = sys_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="System Generation"
    )

    sys_cap_df = reset_df[["Date (MST)", "Fuel Type", "System Capacity"]]
    wide_sys_cap_df = sys_cap_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="System Capacity"
    )

    sys_avail_df = reset_df[["Date (MST)", "Fuel Type", "System Available"]]
    wide_sys_avail_df = sys_avail_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="System Available"
    )

    max_cap_df = reset_df[["Date (MST)", "Fuel Type", "Maximum Capacity"]]
    wide_max_cap_df = max_cap_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="Maximum Capacity"
    )

    cap_fac_df = reset_df[["Date (MST)", "Fuel Type", "Capacity Factor"]]
    wide_cap_fac_df = cap_fac_df.pivot(
        index="Date (MST)", columns="Fuel Type", values="Capacity Factor"
    )

    avail_util = reset_df[["Date (MST)", "Fuel Type", "Availability Utilization"]]
    wide_avail_util = avail_util.pivot(
        index="Date (MST)", columns="Fuel Type", values="Availability Utilization"
    )

    avail_fact = reset_df[["Date (MST)", "Fuel Type", "Availability Factor"]]
    wide_avail_fact = avail_fact.pivot(
        index="Date (MST)", columns="Fuel Type", values="Availability Factor"
    )

    wide_tng_df = wide_tng_df.fillna(0)
    wide_sys_avail_df = wide_sys_avail_df.fillna(0)

    # Change the column names
    column_mapping = {
        "Gas Fired Steam": "gas_fired_steam_tng",
        "Combined Cycle": "combined_cycle_tng",
        "Simple Cycle": "simple_cycle_tng",
        "Solar": "solar_tng",
        "Storage": "storage_tng",
        "Wind": "wind_tng",
        "Hydro": "hydro_tng",
        "Cogeneration": "cogeneration_tng",
        "Coal": "coal_tng",
        "Dual Fuel": "dual_fuel_tng",
        "Other": "other_tng",
    }

    # Rename the columns using the mapping
    wide_tng_df = wide_tng_df.rename(columns=column_mapping)

    # Total energy generations using gas
    wide_tng_df["gas_tng"] = (
        wide_tng_df["cogeneration_tng"]
        + wide_tng_df["combined_cycle_tng"]
        + wide_tng_df["gas_fired_steam_tng"]
        + wide_tng_df["simple_cycle_tng"]
    )

    tng_df = wide_tng_df[
        [
            "gas_tng",
            "dual_fuel_tng",
            "coal_tng",
            "wind_tng",
            "solar_tng",
            "hydro_tng",
            "storage_tng",
            "other_tng",
        ]
    ]

    tng_df.columns = [
        "gas_tng",
        "dual_fuel_tng",
        "coal_tng",
        "wind_tng",
        "solar_tng",
        "hydro_tng",
        "storage_tng",
        "other_tng",
    ]

    column_mapping = {
        "Gas Fired Steam": "gas_fired_steam_avail",
        "Combined Cycle": "combined_cycle_avail",
        "Simple Cycle": "simple_cycle_avail",
        "Solar": "solar_avail",
        "Storage": "storage_avail",
        "Wind": "wind_avail",
        "Hydro": "hydro_avail",
        "Cogeneration": "cogeneration_avail",
        "Coal": "coal_avail",
        "Dual Fuel": "dual_fuel_avail",
        "Other": "other_avail",
    }

    wide_sys_avail_df = wide_sys_avail_df.rename(columns=column_mapping)
    
    # Total system available energy using gas
    wide_sys_avail_df["gas_avail"] = (
        wide_sys_avail_df["cogeneration_avail"]
        + wide_sys_avail_df["combined_cycle_avail"]
        + wide_sys_avail_df["gas_fired_steam_avail"]
        + wide_sys_avail_df["simple_cycle_avail"]
    )

    avail_df = wide_sys_avail_df[
        [
            "gas_avail",
            "dual_fuel_avail",
            "coal_avail",
            "wind_avail",
            "solar_avail",
            "hydro_avail",
            "storage_avail",
            "other_avail",
        ]
    ]

    avail_df.columns = [
        "gas_avail",
        "dual_fuel_avail",
        "coal_avail",
        "wind_avail",
        "solar_avail",
        "hydro_avail",
        "storage_avail",
        "other_avail",
    ]

    merged_df = pd.concat([tng_df, avail_df], axis=1)

    fuel_types = ["gas", "coal", "wind", "solar", "hydro", "storage", "other"]

    for fuel_type in fuel_types:
        tng_column = f"{fuel_type}_tng"
        avail_column = f"{fuel_type}_avail"
        reserve_margin_column = f"{fuel_type}_reserve_margin"
        merged_df[reserve_margin_column] = np.where(
            merged_df[avail_column] == 0,
            0,
            1 - (merged_df[tng_column] / merged_df[avail_column]),
        )

    column_mapping = {
        "Gas Price": "gas_price",
        "Price": "price",
        "Hourly Profile1": "peak_or_not",
        "Season1": "season",
        "AIL": "ail",
    }

    ail_df = ail_df.rename(columns=column_mapping)
    ail_df = ail_df[["ail", "gas_price", "price", "peak_or_not", "season"]]
    final_df = pd.merge(ail_df, merged_df, left_index=True, right_on="Date (MST)")

    final_df["total_tng"] = (
        final_df["gas_tng"]
        + final_df["dual_fuel_tng"]
        + final_df["coal_tng"]
        + final_df["wind_tng"]
        + final_df["solar_avail"]
        + final_df["hydro_tng"]
        + final_df["storage_tng"]
        + final_df["other_tng"]
    )

    final_df["total_avail"] = (
        final_df["gas_avail"]
        + final_df["dual_fuel_avail"]
        + final_df["coal_avail"]
        + final_df["wind_avail"]
        + final_df["solar_avail"]
        + final_df["hydro_avail"]
        + final_df["storage_avail"]
        + final_df["other_avail"]
    )

    fuel_types = [
        "gas",
        "dual_fuel",
        "coal",
        "wind",
        "solar",
        "hydro",
        "storage",
        "other",
    ]

    for fuel_type in fuel_types:
        tng_column = f"{fuel_type}_tng"
        supply_mix_column = f"{fuel_type}_supply_mix"
        final_df[supply_mix_column] = final_df[tng_column] / final_df["total_tng"]

    final_df["total_reserve_margin"] = 1 - (
        final_df["total_tng"] / final_df["total_avail"]
    )

    final_df["relative_gas_reserve"] = (
        final_df["gas_reserve_margin"] / final_df["total_reserve_margin"]
    )

    final_df["demand_supply_ratio"] = final_df["ail"] / final_df["total_avail"]
    final_df["avail_gen_ratio"] = final_df["total_avail"] / final_df["total_tng"]

    final_df["load_on_gas_reserve"] = (final_df["gas_avail"] - final_df["gas_tng"]) / (
        final_df["ail"] * final_df["gas_supply_mix"]
    )

    final_df["renewable_energy_ratio"] = (
        final_df["wind_tng"]
        + final_df["solar_tng"]
        + final_df["hydro_tng"]
        + final_df["storage_tng"]
    ) / final_df["total_tng"]

    final_df["fossil_fuel_ratio"] = (
        final_df["gas_tng"] + final_df["coal_tng"]
    ) / final_df["total_tng"]

    final_df["energy_reserve_margin"] = final_df["total_avail"] - final_df["total_tng"]

    final_df["gas_cost"] = (
        final_df["gas_supply_mix"] * final_df["ail"] * final_df["gas_price"]
    )

    final_df["gas_avail_ratio"] = final_df["gas_avail"] / final_df["total_avail"]
    final_df["coal_avail_ratio"] = final_df["coal_avail"] / final_df["total_avail"]

    final_df["gas_tng_ratio"] = final_df["gas_tng"] / final_df["total_tng"]
    final_df["coal_tng_ratio"] = final_df["coal_tng"] / final_df["total_tng"]

    final_df["renewable_energy_penetration"] = (
        final_df["wind_supply_mix"]
        + final_df["solar_supply_mix"]
        + final_df["hydro_supply_mix"]
        + final_df["storage_supply_mix"]
    ) / 100
    final_df = final_df.sort_index()
    final_df = final_df.asfreq("H")

    region_df = pd.read_csv(
        "data/raw/region_load.csv", parse_dates=["Date (MST)"], index_col="Date (MST)"
    )
    region_df = region_df.rename_axis("date")
    region_df = region_df.sort_values(by="date")
    region_df = region_df.asfreq("H")

    columns_to_drop = ["Date - MST", "Date", "Hourly Profile", "Season", "Date (MPT)"]
    region_df = region_df.drop(columns=columns_to_drop)

    region_df.columns = region_df.columns.str.lower().str.replace(" ", "_")
    final_region_df = pd.merge(region_df, final_df, left_index=True, right_index=True)
    final_region_df.index.name = "date"

    final_region_df.to_csv("data/processed/supply_load_price.csv")

    print("Supply and load data preprocessing completed.")


def merge_data():
    print("Started the merging of data...")
    supply_load_price = pd.read_csv(
        "data/processed/supply_load_price.csv", parse_dates=["date"], index_col="date"
    )
    supply_load_price = supply_load_price.asfreq("H")
    supply_load_price = supply_load_price.sort_values(by="date")

    intertie_df = pd.read_csv(
        "data/processed/intertie.csv", parse_dates=["date"], index_col="date"
    )
    intertie_df = intertie_df.asfreq("H")
    intertie_df = intertie_df.sort_values(by="date")

    merged_df = pd.merge(
        supply_load_price, intertie_df, left_index=True, right_index=True
    )
    merged_df.to_csv("data/processed/preprocessed_features.csv")

    print("Completed the merging of data...")


def get_data(start_date, end_date):
    url = "https://api.aeso.ca/report/v1.1/price/systemMarginalPrice"
    params = {"startDate": start_date, "endDate": end_date}
    headers = {
        "accept": "application/json",
        "X-API-Key": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ6MHo4MnIiLCJpYXQiOjE2ODM1NzQyMTh9.Gbod9kjeDwP4SOJibSFof63X7GGZxbZdBmBVrgE409w",
    }

    response = requests.get(url, params=params, headers=headers)

    # Access the response data
    data = response.json()
    data = pd.DataFrame(data["return"]["System Marginal Price Report"])

    data["begin_datetime_mpt"] = pd.to_datetime(data["begin_datetime_mpt"])
    data = data.set_index("begin_datetime_mpt")
    data = data.drop(
        columns=["begin_datetime_utc", "end_datetime_utc", "end_datetime_mpt"]
    )
    data = data.sort_index()

    # Convert to numeric
    data["system_marginal_price"] = pd.to_numeric(
        data["system_marginal_price"], errors="coerce"
    )
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")

    return data


def create_lagged_columns(X, lag_range=24):
    lagged_names = []
    for col in X:
        for lag in range(lag_range, 0, -1):
            lagged_names.append(f"{col}_lag{lag}")
    return lagged_names
