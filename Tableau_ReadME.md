## Connect Tableau with local files:

1. Click **Data Source** at the bottom left.
2. Choose one of the data sources (`predicted_price`, `shap`, `shap_explain`).
3. Click **Add** next to the connection.
4. Choose **Text File**.
5. Select the corresponding CSV file from your local computer.
6. Drag the file from **Files** to the center.
7. Repeat the above steps for the other two data sources.
8. Click **Predicted Dash** at the bottom.
9. Click **Data** on the top and choose **Refresh All Extracts**. Tableau will be up to date.

![Local Files Connection](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/img/local.gif)


## Connect Tableau with Databricks:

1. Generate a [Databricks Access Token](https://docs.databricks.com/dev-tools/auth.html#databricks-personal-access-tokens-for-users).
2. Click **Data Source** at the bottom left.
3. Choose one of the data sources (`predicted_price`, `shap`, `shap_explain`).
4. Click **Add** next to the connection.
5. Choose **Databricks**.
6. Type `univbritcol-slalom-capstone23.cloud.databricks.com` as the **Host Name** and `sql/protocolv1/o/8254429304025469/0525-003634-7o6qyuy3` as the **HTTP PATH**.
7. Choose **Personal Access Token** as the authentication method and copy the token from Databricks.
8. Choose `hive_metastore` from the **Catalog**.
9. Click the search sign and choose `default` from the **Database**.
10. Click the search sign in **Tables**.
11. Select the corresponding SQL tables from the **Table**.
12. Drag the Table to the center.
13. Repeat the above steps for the other two data sources.
14. Click **Predicted Dash** at the bottom.
15. Click **Data** on the top and choose **Refresh All Extracts**. Tableau will be up to date.

![Databricks Connection](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/main/img/databricks.gif)




 
