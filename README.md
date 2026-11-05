# Machine Learning Classification Comparison — Learning Project

**BITS Pilani M.Tech AIML Machine Learning Assignment 2 (guided practice version).**

## Problem statement
Compare five supervised classification algorithms on the same publicly available binary classification dataset and evaluate them on held-out observations. The Streamlit application loads saved models and evaluates uploaded test CSV data without retraining.

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, via `sklearn.datasets.load_breast_cancer`.
569 instances, 30 numerical features, benign and malignant classes. In this project `target=malignant` is the positive class (1). Not for medical diagnosis.
Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Reproduce the analysis
1. Run `pip install -r requirements.txt`.
2. Run notebook `ML_Assignment_2_Guided.ipynb` or `python train_models.py` in BITS Virtual Lab.
3. Check `model_comparison.csv`, `test_data.csv`, and `model/`.
4. Run `streamlit run app.py`, upload `test_data.csv`, and select a model.

Training uses a stratified 80/20 split (random_state 42); scaling is fitted only on the training set inside sklearn pipelines. The positive class for precision, recall and F1 is malignant.

## GitHub repository link
Add your repository URL here after reviewing, testing and uploading your own project.

## Comparison of models
See `model_comparison.csv` for unrounded scores. These numbers were obtained in a local reference execution; verify them in your BITS Virtual Lab before reporting.

## Observations to complete in your own words
Write one observation for Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM, referencing your six computed metrics. Explain your selection of the best model and possible limitations of a single hold-out split.

## Streamlit link
Add your tested deployment URL here after deployment.

## Integrity note
This is a teaching scaffold, not a finished submission. BITS requires evidence of execution in its Virtual Lab, authentic observations, and an original presentation. Do not copy-paste this project unchanged for submission.
