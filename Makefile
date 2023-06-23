# Makefile
# Author: Arjun Radhakrishnan
# Date: 2023-06-23

# This file runs all of the script by order, to reproduce all the results in the repository folder.
# `Make all` will run all of the scripts and render the final report of the project.
# 'Make clean` will remove all generated files and folders.

# all
# run all of the scripts and render the final report of the project.
all : doc/power_price_prediction_report.html

# To test if the pipeline is working, use the following command:
test :
	make model_train_start_date='2022-12-01' predict_until='2023-02-02' n_estimators=1

# To run the whole pipeline on a limited test set, use the following command:
example :
	make model_train_start_date='2021-01-02' predict_until='2023-02-02' n_estimators=1000

# preprocessing the raw data, and save the clean data
data/processed/complete_data/features.csv data/processed/complete_data/target.csv data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv: src/data_preprocessing/data_preprocessing.py data/raw/intertie.csv data/raw/ail_price.csv data/raw/gen.csv data/raw/region_load.csv
	python -W ignore src/data_preprocessing/data_preprocessing.py

# EDA: save Exploratory Data Analysis results
results/eda/images/target_proportion.jpg results/eda/images/corr_plot.png : src/eda_script.py data/processed/credit_train_df.csv
	python -W ignore src/eda_script.py --processed_data_path 'data/processed/credit_train_df.csv' --eda_result_path 'results/'

# generate model predictions and save the results
src\local_testing_scripts\predictions_plot.html src\local_testing_scripts\rolling_predictions.csv: src\local_testing_scripts\local_testing_script.py data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv
	python -W ignore src/local_testing_scripts/local_testing_script.py --model_train_start_date=$(model_train_start_date) --predict_until=$(predict_until) --n_estimators=$(n_estimators)

# clean
# remove all generated files but preserve the directories
clean :
	rm -rf data/processed/
	rm -rf data/raw/
