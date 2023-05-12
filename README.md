[![pages-build-deployment](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/slalom-ubc-mds/Power-Price-Prediction/actions/workflows/pages/pages-build-deployment)

# Power-Price-Prediction

Power Price Prediction: A Short-Term Forecast​  Forecasting the price of power in Alberta using Open Source data​

# Proposal report

The proposal report can be found [here](https://slalom-ubc-mds.github.io/Power-Price-Prediction/proposal.html)

To render the report locally, follow the below steps:

Clone the project repo:

```
git clone https://github.com/slalom-ubc-mds/Power-Price-Prediction.git
```

Navigate to the project repo:

```
cd Power-Price-Prediction
```

Install the conda environment for required dependencies:

```
conda env create --file power_environment.yaml
```

Activate the conda environment:

```
conda activate power_price_pred
```

Render the report:

```
jupyter-book build notebooks/proposal_jupyter_book/
```

The rendered report can be found locally at `notebooks/proposal_jupyter_book/_build/html/`
