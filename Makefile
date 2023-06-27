# Makefile
# Author: Arjun Radhakrishnan
# Date: 2023-06-23

# This file runs all of the script by order, to reproduce all the results in the repository folder.
# `make all` will run all of the scripts and render the final report of the project.
# `make clean` will remove all generated files and folders.

# Default parameter values
MODEL_TRAIN_START_DATE ?= 2022-12-01
PREDICT_UNTIL ?= 2023-02-02
N_ESTIMATORS ?= 1
DEVICE ?= cpu

# all
# run all of the scripts and render the final report of the project.
all : notebooks\jupyter_book_reports\_build\html\index.html

# run tests
# run all of the tests in the test folder
pytest :
	pytest tests/data_preprocessing_tests.py
	pytest tests/generate_predictions_tests.py
	pytest tests/pipeline_helper_test.py
	pytest tests/preprocess_helper_tests.py

# preprocessing the raw data, and save the clean data
data/processed/complete_data/features.csv data/processed/complete_data/target.csv data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv: src/data_preprocessing/data_preprocessing.py data/raw/intertie.csv data/raw/ail_price.csv data/raw/gen.csv data/raw/region_load.csv
	python -W ignore src/data_preprocessing/data_preprocessing.py

# get cross validation results
results/cv_results.csv: src/cv_test_results/get_cv_test_results.py
	python -W ignore src/cv_test_results/get_cv_test_results.py

# train the model and save the predictions
results/predictions_plot.html results/rolling_predictions.csv results/rolling_predictions_rmse.csv: data/processed/complete_data/features.csv data/processed/complete_data/target.csv data/processed/train/y_train.csv data/processed/train/X_train.csv data/processed/test/y_test.csv data/processed/test/X_test.csv results/cv_results.csv
	python src/local_prediction_pipeline/generate_predictions.py --model_train_start_date=$(MODEL_TRAIN_START_DATE) --predict_until=$(PREDICT_UNTIL) --n_estimators=$(N_ESTIMATORS) --device=$(DEVICE)

# render the final report 
notebooks\jupyter_book_reports\_build\html\index.html: results/predictions_plot.html results/rolling_predictions.csv results/rolling_predictions_rmse.csv
	pytest tests/data_preprocessing_tests.py
	pytest tests/generate_predictions_tests.py
	pytest tests/pipeline_helper_test.py
	pytest tests/preprocess_helper_tests.py
	jupyter-book build notebooks/jupyter_book_reports/

# clean
# remove all generated files but preserve the directories
clean :
	rm -rf data/processed/
	rm -rf results/
