# Makefile
# Author: Arjun Radhakrishnan
# Date: 2023-06-23

# This file runs all of the script by order, to reproduce all the results in the repository folder.
# `make all` will run all of the scripts and render the final report of the project.
# `make clean` will remove all generated files and folders.

# all
# run all of the scripts and render the final report of the project.
all : results/predictions_plot.html

# To test if the pipeline is working, use the following command:
test :
	make all model_train_start_date='2022-12-01' predict_until='2023-02-02' n_estimators=1

# To run the whole pipeline on a limited test set, use the following command:
example :
	make all model_train_start_date='2021-01-02' predict_until='2023-02-02' n_estimators=1000

# preprocessing the raw data, and save the clean data
data/processed/complete_data/features.csv data/processed/complete_data/target.csv data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv: src/data_preprocessing/data_preprocessing.py data/raw/intertie.csv data/raw/ail_price.csv data/raw/gen.csv data/raw/region_load.csv
	python -W ignore src/data_preprocessing/data_preprocessing.py

# get cross validation results
results/cv_results.csv: src/cross_validation_results/get_cross_validation_results.py
	python -W ignore src/cross_validation_results/get_cross_validation_results.py

# train the model and save the predictions
results/predictions_plot.html results/rolling_predictions.csv results/rolling_predictions_rmse.csv: data/processed/complete_data/features.csv data/processed/complete_data/target.csv data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv results/cv_results.csv
	python src/local_testing_scripts/generate_predictions.py --model_train_start_date=2022-12-01 --predict_until=2023-02-02 --n_estimators=1 --device=cpu 

# render the final report


# clean
# remove all generated files but preserve the directories
clean :
	rm -rf data/processed/
	rm -rf results/
