#Explanatory Data Analysis
"""
Usage:
  eda.py load_data <arg>
  eda.py plot_df_timeseries <df> 
  eda.py plot_scatter <df1> <df1_column> <df2> <df2_column>
  eda.py correlation <df> 
  eda.py plot_daily_seasonality <df> <df_column>
  eda.py seasonal_decomposition <df> <df_column> <period>
  eda.py -h | --help

Options:
  -h, --help     Show help message
"""

import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from docopt import docopt
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


# Loading data
def load_data(arg):
    print ("Please be patient, it might take a few minutes to proceed!")
    supply = ['Availability Factor', 'Availability Utilization', 'Capacity Factor',	
              'Maximum Capacity',	'System Available',	'System Capacity',	
              'System Generation', 'Total Generation']

    Fuel = ['Dual Fuel', 'Other', 'Combined Cycle', 'Wind', 'Coal', 'Hydro',
            'Storage', 'Simple Cycle', 'Cogeneration', 'Solar']

    Zones = ['Calgary', 'Central', 'Edmonton', 'Losses', 'Northeast', 'Northwest', 'South', 'System Load']
    df = pd.DataFrame()
    if arg == 'price':
        df = pd.read_csv('../../data/processed/ail_price.csv', parse_dates=['date'], index_col='date')
        df = df.drop('Unnamed: 0', axis=1)
        df = df.sort_values(by='date')
        df = df['price']
    elif arg == 'ail_price':
        df = pd.read_csv('../../data/processed/ail_price.csv', parse_dates=['date'], index_col='date')
        df = df.drop('Unnamed: 0', axis=1)
        df = df.sort_values(by='date')
    elif arg in supply:
        df = pd.read_csv('../../data/raw/gen.csv', parse_dates=['Date (MST)'], index_col='Date (MST)')
        df = df.sort_values(by='Date (MST)')
        reset_df = df.reset_index()
        supply_df  = reset_df[['Date (MST)', 'Fuel Type', arg]]
        df = supply_df.pivot(index='Date (MST)', columns='Fuel Type', values=arg)
    elif arg in Fuel:
        df = pd.read_csv('../../data/raw/gen.csv', parse_dates=['Date (MST)'], index_col='Date (MST)')
        df = df.sort_values(by='Date (MST)')
        df = df[df['Fuel Type'] == arg]
        df = df.filter(supply)
    elif arg == 'weather':
        print('not yet available')
    elif arg == 'region_load':
        df = pd.read_csv('../../data/raw/region_load.csv', parse_dates=['Date - MST'], index_col='Date - MST')
        df = df.sort_values(by='Date - MST')
        df = df.filter(Zones)
    else: 
        print('Wrong Argument!')
    
    df.to_csv(f'{arg}.csv', index=True)
    return df.head(3)


# Plot all of the columns as timeseries separately
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_df_timeseries(df):
    """
    Plots separate time series data from a DataFrame in a single page.
    Args:
        df (pandas.DataFrame): The DataFrame containing the time series data.
    Example:
        plot_df_timeseries(ail_price)
    """
    print ("Please be patient, it might take a few minutes to proceed!")
    name = df
    df = pd.read_csv(f'{df}.csv', index_col=0)
    
    fig = make_subplots(rows=len(df.columns), cols=1, shared_xaxes=True)
    
    for i, item in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[item], name=item), row=i+1, col=1)
    
    # configure the range slider and selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="Daily", step="day", stepmode="backward"),  
                dict(count=1, label="1-Month", step="month", stepmode="backward"),
                dict(count=6, label="6-Month", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1-Year", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        row=len(df.columns), col=1
    )
    
    fig.update_layout(height=800, width=800, title=f"Time Series Plots of {name}")
    fig.show()

    return df.head(1)


# Scatter plot of any two timeseries
def plot_scatter(df1, df1_column, df2, df2_column):
    """
    Plots a scatter plot between two columns from different DataFrames.
    
    Args:
        df1 (str): The filename of the first DataFrame (without the extension).
        df1_column (str): The column name from the first DataFrame to be plotted on the x-axis.
        df2 (str): The filename of the second DataFrame (without the extension).
        df2_column (str): The column name from the second DataFrame to be plotted on the y-axis.
    
    Example:
        plot_scatter(Coal, 'Total Generation', ail_price, 'price')
    """
    name1 = df1
    name2 = df2
    df1 = pd.read_csv(f'{df1}.csv', index_col=0)
    df2 = pd.read_csv(f'{df2}.csv', index_col=0)

    df = pd.merge(df1[df1_column], df2[df2_column], left_index=True, right_index=True)
    df = df.dropna()
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], opacity=0.3, 
                 trendline="lowess", trendline_color_override="red", 
                 trendline_options=dict(frac=0.1))

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="Daily", step="day", stepmode="backward"),  
                dict(count=1, label="1-Month", step="month", stepmode="backward"),
                dict(count=6, label="6-Month", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1-Year", step="year", stepmode="backward"),
                dict(step="all")
                        ])
            )
        )
    fig.update_layout(title=f"{name1}[{df1_column}] vs. {name2}[{df2_column}]")
    fig.show()

def correlation(df):
    name = df
    df = pd.read_csv(f'{df}.csv', index_col=0)
    corr = df.corr()
    fig = px.imshow(corr,
                color_continuous_scale='RdBu_r',
                title=f"Correlation Matrix for {name} dataframe")
    fig.show()

def plot_daily_seasonality(df, df_column):
    name = df
    df = pd.read_csv(f'{df}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = pd.DataFrame(df[df_column])

    # Data for each day of the week (replace with your actual data)
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    df['day'] = df.index.day_name()
    df['hour'] = df.index.hour

    Monday = df[df['day'] == 'Monday']
    Tuesday = df[df['day'] == 'Tuesday']
    Wednesday = df[df['day'] == 'Wednesday']
    Thursday = df[df['day'] == 'Thursday']
    Friday = df[df['day'] == 'Friday']
    Saturday = df[df['day'] == 'Saturday']
    Sunday = df[df['day'] == 'Sunday']

    data = [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]

    # Create subplots with a grid layout of 2 rows and 4 columns
    fig = make_subplots(rows=4, cols=2, subplot_titles=days_of_week)

    # Iterate over each day and add a scatter plot to the corresponding subplot
    for i, day in enumerate(days_of_week):
        subplot_row = (i % 4) + 1
        subplot_col = (i // 4) + 1
        
        fig.add_trace(go.Scatter(x=data[i]['hour'], y=data[i][df_column], mode='markers',
                                marker=dict(size=5, opacity=0.2)),
                    row=subplot_row, col=subplot_col)

        # Add x-axis label and y-axis label for each subplot
        #fig.update_xaxes(title_text='Hour', row=subplot_row, col=subplot_col)
        fig.update_yaxes(title_text=df_column, row=subplot_row, col=subplot_col)

    # Drop the legend
    fig.update_layout(showlegend=False)

    # Add a title for the whole plot
    fig.update_layout(title_text=f"{name}[{df_column}]'s hourly variations within each day of the week")

    # Show the figure
    fig.show()

def seasonal_decomposition(df, df_column, period):
    def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
        x_values = dates if dates is not None else np.arange(len(result.observed))
        fig = (make_subplots(rows=4, cols=1, subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
            .add_trace(go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'), row=1, col=1)
            .add_trace(go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'), row=2, col=1)
            .add_trace(go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'), row=3, col=1)
            .add_trace(go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'), row=4, col=1)
            .update_layout(height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False)
            .update_xaxes(range=["2023-03-01", "2023-03-31"], row=1, col=1)
            .update_xaxes(range=["2023-03-01", "2023-03-31"], row=2, col=1)
            .update_xaxes(range=["2023-03-01", "2023-03-31"], row=3, col=1)
            .update_xaxes(range=["2023-03-01", "2023-03-31"], row=4, col=1)
            )

        return fig


    name = df
    df = pd.read_csv(f'{df}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = pd.DataFrame(df[df_column])

    df['hour'] = df.index.hour
    df['day'] = df.index.strftime('%A')
    df['week'] = df.index.week
    df['month'] = df.index.strftime('%B')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    decomposition = seasonal_decompose(df[df_column], model='additive', period=int(period))
    fig = plot_seasonal_decompose(decomposition, dates=df.index)
    fig.show()

def main():
    arguments = docopt(__doc__)
    operation = None

    if arguments['load_data']:
        operation = load_data
        arg = arguments['<arg>']
    elif arguments['plot_df_timeseries']:
        operation = plot_df_timeseries
        arg = arguments['<df>']
    elif arguments['plot_scatter']:
        operation = plot_scatter
        arg1 =  arguments['<df1>'] 
        arg2 = arguments['<df1_column>'] 
        arg3 = arguments['<df2>'] 
        arg4 = arguments['<df2_column>']
    elif arguments['correlation']:
        operation = correlation
        arg = arguments['<df>']
    elif arguments['plot_daily_seasonality']:
        operation = plot_daily_seasonality
        arg1 = arguments['<df>']
        arg2 = arguments['<df_column>']
    elif arguments['seasonal_decomposition']:
        operation = seasonal_decomposition
        arg1 = arguments['<df>']
        arg2 = arguments['<df_column>']
        arg3 = arguments['<period>']

    if operation in [load_data  , plot_df_timeseries, correlation]:
        result = operation(arg)
        print(result)
    elif operation in [plot_scatter]:
        result = operation(arg1, arg2, arg3, arg4)
    elif operation in [plot_daily_seasonality]:
        result = operation(arg1, arg2)
    elif operation in [seasonal_decomposition]:
        result = operation(arg1, arg2, arg3)
    else:
        print("Invalid operation!")

if __name__ == '__main__':
    main()
