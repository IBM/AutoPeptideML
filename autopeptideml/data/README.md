# AutoPeptideML output Guide

## 1. Introduction

Thank you for using AutoPeptideML for developing you own peptide bioactivity models. This file should help you to understand the output of the tool and interpret the results you have just obtained. It is recommend to include this whole directory without any modifications as part of the supplementary information of any publication where models developed using AutoPeptideML are reported as it contains all necessary information for reproducing results.

## 2. Output contents

There are 6 subdirectories that comprise the output from AutoPeptideML. 

1. `best_configs`: Contains information regarding the best sets of hyper-parameters found for all models comprising the final ensemble. The format of each of the configuration files is `JSON`.
2. `ensemble`: Contains the weights for the models in the ensemble. The files are in binary format and are created using the `joblib` package.
3. `figures`: Contains representations of the main evaluation metrics that are publication ready. This figures have lines giving an orientation as to how good a model is. There are three thresholds: 1) useless model, where its performance is comparable to random classification, 2) decent model, where a model could be considered useful, and 3) good model, where the performance of the model is actually good. Evidently, categories (2) and (3) are subjective and depends on the difficulty of the task at hand and whether there are any other models available performing the same task. These values are provided for orientative purposes and to simplify the analysis of the results.
4. `folds`: Contains the files with the samples in the `n` cross-validation folds, column `Y` refers to whether the sample is positive (1) or negative (0) for the target bioactivity. Files are in `CSV` format.
5. `evaluation_data`: Raw evaluation contains the raw evaluation data for both cross-validation of all models in the ensemble and the evaluation of the ensemble itself against the hold-out independent test set. The files are in `CSV` format and are included to allow users to calculate their own custom metrics.
6. `splits`: Contains the files with train/evaluation sets in `CSV` format. Again, `Y` column refers to whether the sample is positive (1) or negative (0) for the target bioactivity.

## 3. Guide to interpret results

### 3.1. Figures
https://towardsdatascience.com/visualize-machine-learning-metrics-like-a-pro-b0d5d7815065