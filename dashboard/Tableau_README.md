## Connect Tableau with local files:

1. Follow the local setup in the root directory [instruction](https://github.com/slalom-ubc-mds/Power-Price-Prediction/tree/main#local-setup)
2. Navigate to `src/databricks_prediction_pipeline/prediction_pipeline.ipynb`
3. Set `IS_LOCAL = True`
4. Run the notebook and the initial set of predictions will be stored in `databricks_assets/` folder
5. Load the [workbook](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/dashboard/prediction_dashboard_local.twbx) to Tableau
6. Click **Data Source** at the bottom left.
7. You will be asked to choose files.
8. Select the corresponding CSV file from `databricks_assets/` folder.
 
    (`shap_explain.csv` under `shap_explain`, `predicted_price.csv` under predicted_price, `shap.csv` under shap)
9. Repeat the above steps for the other two data sources.
10. Click **Predicted Dash** at the bottom.
11. Click **Data** on the top and choose **Refresh All Extracts**. Tableau will be up to date.

![Local Files Connection](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/img/local.gif)


## Connect Tableau with Databricks:

1. Generate a [Databricks Access Token](https://docs.databricks.com/dev-tools/auth.html#databricks-personal-access-tokens-for-users).
2. Load the [workbook](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/dashboard/prediction_dashboard_local.twbx) to Tableau. Click **Data Source** at the bottom left.
3. Choose one of the data sources (`predicted_price`, `shap`, `shap_explain`).
4. Click **Add** next to the connection.
5. Choose **Databricks**.
6. Type `univbritcol-slalom-capstone23.cloud.databricks.com` as the **Host Name** and `sql/protocolv1/o/8254429304025469/0525-003634-7o6qyuy3` as the **HTTP PATH**.
7. Choose **Personal Access Token** as the authentication method and copy the token from Databricks.
8. Choose `hive_metastore` from the **Catalog**.
9. Click the search sign and choose `default` from the **Database**.
10. Click the search sign in **Table**.
11. Select the corresponding SQL tables from the **Table**.

    (`shap_explain` under `shap_explain`, `predicted_price` under predicted_price, `shap` under shap)
13. Drag the Table to the center.
14. Repeat the above steps for the other two data sources.
15. Click **Predicted Dash** at the bottom.
16. Click **Data** on the top and choose **Refresh All Extracts**. Tableau will be up to date.

![Databricks Connection](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/img/databricks.gif)
