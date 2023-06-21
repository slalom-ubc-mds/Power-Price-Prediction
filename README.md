[![pages-build-deployment](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment)

# Power-Price-Prediction

Welcome to our GitHub page!

We are dedicated to providing an innovative and comprehensive business solution for power price prediction in the Alberta Energy Market. Our goal is to develop an interpretable and explainable data science product that empowers organizations to make well-informed decisions regarding their energy purchases.

[Link to Sample Tableau Dashboard](https://public.tableau.com/app/profile/gaoxiang.wang8077/viz/AESOVersion6_0/PredictedDash?publish=yes)

To read more about our wonderful app, feel free to jump over to one of the sections below or continue scrolling down.

- [Meet the Team](#meet-the-team)
- [Motivation and Purpose](#motivation-and-purpose)
- [Dashboard Description](#dashboard-description)
- [Installation](#installation)
- [Usage](#usage)

## Meet the Team

The developers are students of the MDS program at the University of British Columbia and in partnership with [Slalom Consulting, LLC](https://www.slalom.com/).

- [Arjun Radhakrishnan](https://github.com/rkrishnan-arjun)
- [Sneha Sunil](https://github.com/snesunil)
- [Gaoxiang Wang](https://github.com/louiewang820)
- [Mehdi Naji](https://github.com/mehdi-naji)

## Motivation and Purpose

Over the past few decades, the electricity markets have transformed from regulated to competitive and deregulated. Alberta’s electricity market started deregulating in 1996, resulting in highly volatile and uncertain power prices. Many organizations purchase large quantities of energy on demand and rely on energy forecasts to determine their costs in advance. Power price prediction can also be critical for many power generation companies to make effective decisions toward maximizing their profit, determining pricing strategies in the market, and scheduling technical maintenance periods. The current energy forecasts only provide a short-term coverage of 6 hours, which is volatile and lacks interpretation or model visibility. To reduce their expenses, companies could plan and potentially explore alternative energy options if they have access to accurate forecasts which covers a longer window and is also interpretable and explainable. This project aims to help businesses by providing cost analysis and forecasting hourly energy prices 12 hours in advance. Our objective is to empower companies to plan for alternative energy solutions, such as sourcing energy from elsewhere, purchasing at different times, or even developing their own energy generation systems.

## Dashboard Description

Our Live Updates & Forward Predictions Dashboard delivers real-time and interpretable information, enabling users to stay informed about energy price fluctuations in the Alberta Energy Market. By leveraging the power of data science and visualization, we provide stakeholders with the tools they need to make well-informed decisions and optimize their energy purchasing strategies.

- Time Series Chart and Confidence Interval:
The top section of the dashboard showcases a 24-hour time series chart that summarizes energy prices over the last 12 hours and predicts prices for the next 12 hours. A notable feature of this chart is the inclusion of a 95% confidence interval for each hour's prediction. The green area within the chart represents this interval, offering users a clear understanding of the reliability of our forecasts.

- Interactive Features & Tool-Tips:
The dashboard is enhanced with interactive features, enabling users to explore the data further. As users move the cursor across the timeline, tool-tips appear, providing date and corresponding price information for past prices, as well as the confidence level for future predictions. This interactive element enhances the user experience and facilitates easy navigation through the dashboard.

- Interactive Bar Chart & Explanatory Text:
The lower half of the dashboard presents an interactive bar chart that adds depth to the understanding of our model's predictions. Alongside the chart, explanatory text provides valuable insights into the top four influences on our predictions for each hour. This feature empowers users to gain a comprehensive understanding of the factors driving our forecasts and make informed decisions based on this information.

- Prediction Influences & Contextual Information:
To provide additional context, the dashboard breaks down the influences that affect our predictions. We quantify the contribution of each feature in relation to the average power price for the previous year, which serves as the base value. Significant factors such as the gas supply mix, total reserve margin, and weekly profile are identified as key influencers. Furthermore, we have discovered four global key influencing factors strongly correlated with the price: the proportion of total energy generated by gas and wind, as well as the available reserves of each to meet surges in demand. Time series plots of these features are included in the bottom right section of the dashboard, offering users valuable insights into their impact on the predicted energy prices. Users can also choose to hide/unhide the contexutal explaintions for simplicity.

## Installation

Tableau: Visit the [Tableau website](https://www.tableau.com/) and download the appropriate version of Tableau for your operating system (Windows or macOS).

Databricks Tableau driver:   Visit the [ODBC Driver website](https://www.databricks.com/spark/odbc-drivers-download?_gl=1*wbycmt*_gcl_au*MTExNDA4MjAzOC4xNjg1Mzg0MjQw&_ga=2.190062569.311368728.1687321881-777036860.1685384240) and download the appropriate version of Tableau ODBC Driver for your operating system.

VS Code: Visit the [VS Code website](https://code.visualstudio.com/) and download the appropriate version of VS code for your operating system.

## Usage

- To run the prediction pipeline locally, follow the below steps:

clone the forked repo to your local machine in VS Code Command Line by running

```
git clone https://github.com/slalom-ubc-mds/Power-Price-Prediction.git
```

Navigate to the project repo:

```
cd Power-Price-Prediction
```

Install the conda environment for required dependencies:

```
conda env create --file environment.yml
```

Activate the conda environment:

```
conda activate power_price_pred
```

To generate the required training and testing data, run the following command from the root of the project directory:

```
cd src/data_preprocessing
```

```
python data_preprocessing.py 
```

To train the model and generate the predictions, run the following command from the root of the project directory:

```
cd src/local_testing_scripts
```

```
python generate_predictions.py --model_train_start_date=2021-01-02 --predict_until=2023-02-02 --n_estimators=1000
```

The original model is trained from Jan 1st, 2021 to Jan 31, 2023. Hence, `model_train_start_date` needs to be greater than January 1st, 2021, and less than December 31st, 2022 to allow at least one month of training data. Predictions will start from Feb 1st, 2023. Hence, `predict_until` should be greater than January 31st, 2023, and less than May 30th, 2023. 
The number of estimators can be adjusted to improve the model performance or reduce the training time. 

[Connect tableau to local files](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/Tableau_ReadME.md#connect-tableau-with-local-files)

- To run prediction on Databricks, follow the below steps:

Log in to [Databricks](https://univbritcol-slalom-capstone23.cloud.databricks.com/login.html?o=8254429304025469)

Click Workspace and navigate to `Shared/final_pipelines` folder

Choose `prediction_pipeline` and click `Run all`. The initial prediction will be triggered.

Choose `update_predict_pipeline` and click `Schedule`. The prediction will be updated based on preferable refresh time (hourly).

[Connect tableau to Databricks](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/Tableau_ReadME.md#connect-tableau-with-databricks)
