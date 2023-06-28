[![pages-build-deployment](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment)

# Power-Price-Prediction

Welcome to our Power Price Prediction project, a comprehensive and explainable solution for predicting energy prices in the Alberta Energy Market. We leverage data science to deliver forecasts, empowering organizations to optimize their energy procurement strategies.

Explore our [Sample Tableau Dashboard](https://public.tableau.com/app/profile/gaoxiang.wang8077/viz/AESOVersion6_0/PredictedDash?publish=yes) for a live demonstration. The report is available on [GitHub Pages](https://slalom-ubc-mds.github.io/Power-Price-Prediction/final_report.html).

Dive into the sections below to discover more about our project:

- [Team](#team)
- [Project Overview](#project-overview)
- [Dashboard Features](#dashboard-features)
- [Installation](#installation)
- [Usage](#usage)

## Team

Our team comprises Masters of Data Science students at the University of British Columbia, in collaboration with UBC Capstone Mentor [Quan Nguyen](https://github.com/quan3010) and [Slalom Consulting, LLC](https://www.slalom.com/).

- [Arjun Radhakrishnan](https://github.com/rkrishnan-arjun)
- [Sneha Sunil](https://github.com/snesunil)
- [Gaoxiang Wang](https://github.com/louiewang820)
- [Mehdi Naji](https://github.com/mehdi-naji)

## Project Overview

Driven by the need for reliable long-term energy forecasts in the deregulated Alberta Energy Market, our project delivers 12-hour ahead hourly energy price predictions. Our model aims to help businesses optimize their energy procurement strategy by providing insights into energy costs and enabling exploration of alternatives like sourcing from different suppliers, purchasing at different times, or even considering their own energy generation systems.

## Dashboard Features

Our dashboard provides an intuitive, real-time representation of energy price predictions, underpinned by a comprehensive breakdown of influencing factors.

- **Time Series Chart and Confidence Interval:** 24-hour time series chart with past and future energy prices, including a 95% confidence interval for predictions.
- **Interactive Features & Tool-Tips:** Enhanced user interaction with tool-tips providing specific data points on hover.
- **Interactive Bar Chart & Explanatory Text:** Deeper insights into the model's predictions with accompanying explanatory text highlighting key influencing factors.
- **Prediction Influences & Contextual Information:** Detailed breakdown of influencing factors for predictions, with contextual plots of key price influencers. To provide additional context, the dashboard breaks down the influences that affect our predictions. Furthermore, we have discovered four key global influencing factors strongly correlated with the price: the proportion of total energy generated by gas and wind, as well as the available reserves of each to meet surges in demand. Time series plots of these features are included in the bottom right section of the dashboard, offering users valuable insights into their impact on the predicted energy prices. 

## Installation

Ensure you have the following tools installed:

- [Tableau](https://www.tableau.com/)
- [Databricks Tableau driver](https://www.databricks.com/spark/odbc-drivers-download?_gl=1*wbycmt*_gcl_au*MTExNDA4MjAzOC4xNjg1Mzg0MjQw&_ga=2.190062569.311368728.1687321881-777036860.1685384240)
- [VS Code](https://code.visualstudio.com/)

## Usage

### README files

- [README file for repository structure](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/Repo_README.md)
- [README file for Tableau configuration](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/dashboard/Tableau_README.md)

### Local Setup

Follow the instructions below to run the prediction pipeline locally.

1. Clone the repo:

```bash
git clone https://github.com/slalom-ubc-mds/Power-Price-Prediction.git
```

2. Navigate to the project:

```bash
cd Power-Price-Prediction
```

3. Install and activate the required environment:

```bash
conda env create --file environment.yml
conda activate power_price_pred
```

4. Reset and clean the existing analysis results from directories by running the below command from the project root directory:

```bash
make clean
```

5. Raw data required to run the pipeline is already downloaded and saved to [folder](https://github.com/slalom-ubc-mds/Power-Price-Prediction/tree/main/data/raw). If you would like to get the latest data, navigate to [tableau](https://public.tableau.com/app/profile/market.analytics/viz/AnnualStatistics_16161854228350/Introduction) and click on the 7th tab which says Data Download Instructions and follow the guidelines.

   Data should be downloaded from the following sections:

    - Price & AIL
    - System & Regional Load
    - Generation
    - Interties

6. Run the prediction pipeline using Makefile. Note that the entire pipeline takes approximately one hour to run on an Intel i7 12700H, 14 Cores, 16 GB RAM for the test set from 26th May 2023 to 30th May 2023. The original model is trained using data from January 1st, 2021 to January 31st, 2023. Therefore, please keep the following in mind when setting up your training parameters:

    - `N_ESTIMATORS`: This parameter denotes the number of boosting stages the model will go through. You can tweak this number to balance model performance and training time according to your requirements.

    - `DEVICE`: Specify the hardware you want to use for training the model. If your system supports GPU processing, set this to 'gpu' for faster training. If not, or if you prefer to use the CPU, the default value is 'cpu'.

```bash
make all N_ESTIMATORS=1000 DEVICE=cpu
```

> You can limit the number of estimators for a quicker (~10 minutes) test run.

```bash
make all N_ESTIMATORS=1 DEVICE=cpu
```

### Deployment to Databricks

To deploy the prediction pipeline to Databricks, follow the instructions below:

1. Publish scripts in [src/databricks_prediction_pipeline](https://github.com/slalom-ubc-mds/Power-Price-Prediction/tree/main/src/databricks_prediction_pipeline) on Databricks cluster with [SKTIME](https://www.sktime.net/en/latest/installation.html) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) installed.
2. Remove the comment sign in the first line of [pipeline_helper.py](https://github.com/slalom-ubc-mds/Power-Price-Prediction/tree/main/src/databricks_prediction_pipeline/pipeline_helper.py)
3. Run [prediction_pipeline.ipynb](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/src/databricks_prediction_pipeline/prediction_pipeline.ipynb) to train the model and generate the initial set of predictions.
4. Schedule [update_predict_pipeline.ipynb](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/src/databricks_prediction_pipeline/update_predict_pipeline.ipynb) as a Databricks job to update predictions at your preferred frequency.

To configure Tableau for local or Databricks data sources, refer to these guides:

- [Local Tableau Configuration](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/dashboard/Tableau_README.md#connect-tableau-with-local-files)
- [Databricks Tableau Configuration](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/dashboard/Tableau_README.md#connect-tableau-with-databricks)

Currently, the pipeline is deployed at [Slalom's Databricks Account](https://univbritcol-slalom-capstone23.cloud.databricks.com/login.html?o=8254429304025469) under the folder `Workspace/Shared/Power_Price_Prediction_Pipeline`. The dashboard is published at [Tableau Public](https://public.tableau.com/app/profile/gaoxiang.wang8077/viz/AESOVersion6_0/PredictedDash?publish=yes). Due to licensing restrictions, the dashboard Tableau Public does not update automatically. Once the restrictions are lifted, please refer to the previous section to set up Tableau on Databricks.

### Dependencies

For the python dependencies and the conda environment creation file, please check [here](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/environment.yml)
