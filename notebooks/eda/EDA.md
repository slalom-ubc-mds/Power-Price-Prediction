# Exploratory Data Analysis (EDA) Report

## Introduction
This file provides an introduction to the "eda.py" script, which performs Explanatory Data Analysis (EDA) tasks on a given dataset. The script provides a set of tools and functions to perform exploratory data analysis on various datasets related to the Power-Price-Prediction project. The EDA script accompanying this report allows you to load data, visualize time series data, analyze correlations, and explore seasonality in your desired datasets.

## Setup Instructions

To use this script, follow the setup instructions below:

1. Clone the repository from GitHub by running the following command in your terminal:

    ```
    git clone https://github.com/slalom-ubc-mds/Power-Price-Prediction.git
    ```

2. Navigate to the "eda" folder using the command:

    ```
    cd Power-Price-Prediction/notebooks/eda
    ```

3. Install the conda environment with all the required dependencies by executing the following command:

    ```
    conda env create --file environment.yml
    ```

4. Activate the conda environment using the command:

    ```
    conda activate power_price_pred
    ```

By following these steps, you will have the necessary environment and dependencies set up to run the EDA script smoothly.

## Script Usage
To use the EDA script, you need to have Python installed on your machine along with the required libraries. The script offers the following functionalities:

### 1. Loading Data
You can load different datasets by specifying the appropriate argument. The supported datasets include:
- `price`: Loads price data
- `ail_price`: Loads AIL (Artificial Intelligence Lab) price data
- Supply-related datasets:
  - `Availability Factor`
  - `Availability Utilization`
  - `Capacity Factor`
  - `Maximum Capacity`
  - `System Available`
  - `System Capacity`
  - `System Generation`
  - `Total Generation`
- Fuel-related datasets:
  - `Dual Fuel`
  - `Other`
  - `Combined Cycle`
  - `Wind`
  - `Coal`
  - `Hydro`
  - `Storage`
  - `Simple Cycle`
  - `Cogeneration`
  - `Solar`
- `weather`: (Not yet available)
- `region_load`: Loads region load data

To load a specific dataset, use the following command:
```
python eda.py load_data <arg>
```
**Example:**

```python eda.py load_data Wind```

```python eda.py load_data price```

### 2. Plotting Time Series
You can plot time series data from a loaded dataset using the `plot_df_timeseries` command. This function plots each column of the DataFrame as a separate time series on a single page. It also provides interactive features like range selection and zooming.
```
python eda.py plot_df_timeseries <df>
```
**Example:**
```python eda.py plot_df_timeseries Wind```

### 3. Scatter Plot
The `plot_scatter` command allows you to create a scatter plot between two columns from different DataFrames. This helps in analyzing the relationship between two variables.
```
python eda.py plot_scatter <df1> <df1_column> <df2> <df2_column>
```
**Example:**
```python eda.py plot_scatter Wind 'Total Generation' price price```

### 4. Correlation Analysis
The `correlation` command calculates and displays the correlation matrix for a given dataset. It provides insights into the relationships between different variables.
```
python eda.py correlation <df>
```
**Example:**
```python eda.py correlation Wind```

### 5. Daily Seasonality Plot
The `plot_daily_seasonality` command generates subplots of the hourly variations within each day of the week for a specific column in a dataset.
```
python eda.py plot_daily_seasonality <df> <df_column>
```
**Example:**
```python eda.py plot_daily_seasonality Wind 'Total Generation' ```

### 6. Seasonal Decomposition
The `seasonal_decomposition` command performs seasonal decomposition of a time series and displays the observed, trend, seasonal, and residual components. It helps in identifying the underlying patterns and trends in the data.
```
python eda.py seasonal_decomposition <df> <df_column> <period>
```
**Example:**
```python eda.py seasonal_decomposition Wind 'Total Generation' 24  ```

### 7. Seasonality Plot
The `plot_seasonality` command allows you to visualize the average values of a specific column based on different cycles, such as `hour`, `day`, `week`, or `month`. It helps in understanding the cyclic patterns in the data.
```
python eda.py plot_seasonality <df> <df_column> <cycle>
```
**Example:**
```python eda.py plot_seasonality Wind 'Total Generation' day  ```


### 8. Autocorrelation and Partial Autocorrelation
The `acf_pacf` command calculates and plots the autocorrelation and partial autocorrelation functions for a specific column in a dataset. It provides insights into the lags and dependencies in the data.
```
python eda.py acf_pacf <df> <df_column> <nlags>
```
**Example:**
```python eda.py acf_pacf Wind 'Total Generation' 24  ```