import re
import sys
sys.path.append("../utils/")
from preprocess_helper import *

preprocess_intertie_data()

process_supply_data()

merge_data()

print("Started the feature selection...")

price_old_df = pd.read_csv('../../data/processed/preprocessed_features.csv', parse_dates=['date'], index_col='date')
price_old_df = price_old_df.asfreq('H').sort_values(by='date')

price_old_df = price_old_df.rename(columns={"calgary": "calgary_load"})
price_old_df = price_old_df.rename(columns={"central": "central_load"})
price_old_df = price_old_df.rename(columns={"edmonton": "edmonton_load"})
price_old_df = price_old_df.rename(columns={"northeast": "northeast_load"})
price_old_df = price_old_df.rename(columns={"northwest": "northwest_load"})
price_old_df = price_old_df.rename(columns={"south": "south_load"})

columns_to_multiply = []

for column in price_old_df.columns:
    if column.endswith(("_reserve_margin", "_supply_mix", "_ratio")):
        price_old_df[column] = price_old_df[column] * 100
        columns_to_multiply.append(column)


price_old_df["relative_gas_reserve"] = price_old_df["relative_gas_reserve"] * 100
price_old_df["load_on_gas_reserve"] = price_old_df["load_on_gas_reserve"] * 100
price_old_df["gas_cost"] = price_old_df["gas_cost"] / 100

y = price_old_df[['price']]
y = y.asfreq("H")

X = price_old_df.drop(columns=['price'])
X = X.asfreq("H")

window = 24
rolling_y = y.rolling(window)
X['rolling_mean'] = rolling_y.mean()
X['rolling_std'] = rolling_y.std()
X['rolling_min'] = rolling_y.min()
X['rolling_max'] = rolling_y.max()
X['rolling_median'] = rolling_y.median()
X['exp_moving_avg'] = y.ewm(span=24).mean()

X.dropna(inplace=True)
y = y.loc[X.index]

X['season'] = X['season'].replace({'WINTER': 1, 'SUMMER': 0})
X['peak_or_not'] = X['peak_or_not'].replace({'ON PEAK': 1, 'OFF PEAK': 0})

X['weekly_profile'] = 0
X.loc[((X.index.dayofweek == 1) | (X.index.dayofweek == 2)) & (X['peak_or_not'] == 1), 'weekly_profile'] = 6
X.loc[((X.index.dayofweek == 0) | (X.index.dayofweek == 3)) & (X['peak_or_not'] == 1), 'weekly_profile'] = 5
X.loc[((X.index.dayofweek == 4) | (X.index.dayofweek == 5) | (X.index.dayofweek == 6)) & (X['peak_or_not'] == 1), 'weekly_profile'] = 4
X.loc[((X.index.dayofweek == 1) | (X.index.dayofweek == 2)) & (X['peak_or_not'] == 0), 'weekly_profile'] = 3
X.loc[((X.index.dayofweek == 0) | (X.index.dayofweek == 3)) & (X['peak_or_not'] == 0), 'weekly_profile'] = 2
X.loc[((X.index.dayofweek == 4) | (X.index.dayofweek == 5) | (X.index.dayofweek == 6)) & (X['peak_or_not'] == 0), 'weekly_profile'] = 1

# Specify your date ranges
dates = [
    ('2021-01-01', '2021-06-31'),
    ('2021-07-01', '2021-12-25'),
    ('2021-12-26', '2021-12-31'),
    ('2022-01-01', '2022-06-31'),
    ('2022-07-01', '2022-12-25'),
    ('2022-12-26', '2023-05-31'),
]

# Get the data for each date range and compute averages and sums
average_dfs = []
sum_dfs = []
for start_date, end_date in dates:
    data = get_data(start_date, end_date)
    
    average_df = data[['volume', 'system_marginal_price']].resample('H').mean()
    average_df.columns = ['volume_avg', 'system_marginal_price_avg']
    average_dfs.append(average_df)
    
    sum_df = data[['volume', 'system_marginal_price']].resample('H').sum()
    sum_df.columns = ['volume_sum', 'system_marginal_price_sum']
    sum_dfs.append(sum_df)

# You now have lists of average and sum dataframes. You can concatenate them if needed.
# For example:
all_averages = pd.concat(average_dfs)
all_sums = pd.concat(sum_dfs)    

smp_df = pd.merge(all_averages, all_sums, left_index=True, right_index=True)
smp_df = smp_df.asfreq('H')
X = pd.merge(X, smp_df, left_index=True, right_index=True)
X = X.asfreq('H')
y = y.asfreq('H')
X['volume_avg'].fillna(X['volume_avg'].mean(), inplace=True)
X['system_marginal_price_avg'].fillna(X['system_marginal_price_avg'].mean(), inplace=True)
float64_cols = X.select_dtypes(include=['float64']).columns.tolist()
X[float64_cols] = X[float64_cols].astype('float32')

# Change weekly_profile, season, peak_or_not to int 
X['weekly_profile'] = X['weekly_profile'].astype('int32')
X['season'] = X['season'].astype('int32')
X['peak_or_not'] = X['peak_or_not'].astype('int32')


sorted_useful_values = [
    "renewable_energy_penetration",
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
    "gas_tng_ratio",
    "relative_gas_reserve",
    "hydro_tng",
    "wind_tng",
    "load_on_gas_reserve",
    "northwest_load",
    "calgary_load",
]

sorted_useful_values.remove('gas_tng_ratio')
sorted_useful_values.remove('wind_tng')
sorted_useful_values.remove('renewable_energy_penetration')
y[y<5] = 5

X.index.name = "date"
y.index.name = "date"
X = X.asfreq("H")
y = y.asfreq("H")

# Add demand_supply_ratio, weekly_profile, system_load total_reserve_margin volum_sum volumn_avg to useful_values
sorted_useful_values.append('system_load')
sorted_useful_values.append('demand_supply_ratio')
sorted_useful_values.append('total_reserve_margin')
sorted_useful_values.append('volume_sum')
sorted_useful_values.append('volume_avg')
sorted_useful_values.append('weekly_profile')

# export again

folder_path = '../../data/processed/complete_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the DataFrame to a file inside the folder
file_path = os.path.join(folder_path, 'features.csv')
pd.DataFrame(X[sorted_useful_values]).to_csv(file_path)

file_path = os.path.join(folder_path, 'target.csv')
pd.DataFrame(y).to_csv(file_path)

# train test split
X_train = X[sorted_useful_values].loc['2021-01-01':'2023-01-31']
X_test = X[sorted_useful_values].loc['2023-02-01':]

y_train = y.loc['2021-01-01':'2023-01-31']
y_test = y.loc['2023-02-01':]

train_path = '../../data/processed/train'
if not os.path.exists(train_path):
    os.makedirs(train_path)
train_path = os.path.join(train_path, 'X_train.csv')
pd.DataFrame(X_train).to_csv(train_path)

test_path = '../../data/processed/test'
if not os.path.exists(test_path):
    os.makedirs(test_path)
test_path = os.path.join(test_path, 'X_test.csv')
pd.DataFrame(X_test).to_csv(test_path)
pd.DataFrame(y_train).to_csv('../../data/processed/train/y_train.csv')
pd.DataFrame(y_test).to_csv('../../data/processed/test/y_test.csv')

print("Completed the feature selection...")

