## Repository Structure

We have the following folders in our git repository -

- `dashboard` - Contains the Tableau workbook and the Tableau_README.md which explains the instructions to connect Tableau to local files as well as with Databricks

- `data` - Contains two folders. The first one is the raw folder which contains all the raw data files. Then we have a processed folder that contains all the processed files. The complete features and target data that we have used in our pipeline can be found in the `complete_data` folder and the training and test set can be found in their respective folders `train` and `test`.

- `databricks_assets` -

- `docs` - Contains all the intermediate files generated to host the Jupyter Notebook for the report

-  `img` - Contains the GIFs that showcase the instructions to follow for Tableau configuration

-  `notebooks` - Contains the notebooks related to our modeling pipeline.

      - `benchmark_error` - Contains the notebook which can be used to generate the error made by AESO in their 6-step predictions
      - `cross-validation` - Contains the cross-validation notebooks for each experimented model
      - `jupyter_book_reports`- Contains the files related to the report
      - `local_testing_pipelines` - Contains the notebook which evaluates our model for the report
      - `utils`- All the scripts that contain the helper functions for our pipeline are stored here

- `results` - All the results files are stored here

- `src` - Contains all the scripts which are part of our pipeline
    - `cv_test_results` - Script to move the cross-validation results from the pickle file to the results folder
    - `data_preprocessing` - Script for data preprocessing and feature engineering
    - `databricks_prediction_pipeline` - Scripts that are hosted in data bricks
    - `local_prediction_pipeline` - Scripts that generate the predictions locally

- `tests` - Contains all the test scripts 
