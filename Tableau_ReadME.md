# Connect Tableau with local files:

* Click Data Source at bottom left
* Choose one of the data sources (predicted_pric, shap, shap_explain)
* Click 'add' next to the connection
* Choose Text File
* Select the corresponding csv file from your local computer
* Drag the file from Files to center
* Repeated above steps for the other two data sources
* Click Predicted Dash at the bottom
* Click Data On the top and choose `Refresh All Extracts`. The Tableau will be up tp date.
![](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/gw_readme_update/img/local.gif)


# Connect Tableau with Databricks:
* Generate [Databrick Access Token](https://docs.databricks.com/dev-tools/auth.html#databricks-personal-access-tokens-for-users)
* Click Data Source at bottom left
* Choose one of the data sources (predicted_pric, shap, shap_explain)
* Click 'add' next to the connection
* Choose Databricks
* Type `univbritcol-slalom-capstone23.cloud.databricks.com` as Host Name and `sql/protocolv1/o/8254429304025469/0525-003634-7o6qyuy3` as HTTP PATH
* Choose `Personal Access Token` as authentication and copy the token from Databricks
* Choose hive_meeetastore from Catalog
* click search sign and choose default from Database
* click search sign in Tables 
* Select the corresponding sql tables from Table
* Drag the Table to center
* Repeated above steps for the other two data sources
* Click Predicted Dash at the bottom
* Click Data On the top and choose `Refresh All Extracts`. The Tableau will be up tp date.
![](https://github.com/slalom-ubc-mds/Power-Price-Prediction/blob/gw_readme_update/img/databricks.gif)



 
