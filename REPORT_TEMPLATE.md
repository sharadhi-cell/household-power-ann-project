# Project Report Template

## Title
ANN-Based Classification of Individual Household Electric Power Consumption Using MLP and Autoencoder

## Abstract
This project develops and evaluates artificial neural network approaches for classifying household electric power consumption into low, medium, and high categories. The UCI time-series dataset is cleaned, chronologically partitioned, and transformed using training-only preprocessing. A baseline MLP, a regularized proposed MLP, and an autoencoder-based classifier are compared. Model selection uses validation data; the final test period remains held out until evaluation. Performance is assessed using accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.

**Replace this paragraph with the actual measured results after running the code. Do not insert guessed values.**

## 1. Introduction
Household energy monitoring can support energy awareness and demand-side management. Minute-level power measurements allow analysis of changing consumption patterns. ANN models can learn nonlinear relationships among electrical variables and calendar features.

## 2. Problem Statement
The original dataset does not provide predefined categorical consumption labels. This project defines three consumption categories from active-power quantiles estimated on the training period. The classification task is to assign each record to Low, Medium, or High.

## 3. Objectives
1. Clean and preprocess the raw UCI dataset.
2. Create leakage-resistant chronological train/validation/test partitions.
3. Implement a baseline and proposed MLP in TensorFlow/Keras.
4. Train an autoencoder and use its latent representation for classification.
5. Compare learning rate, batch size, architecture, and dropout.
6. Evaluate unseen data and analyze class-wise errors and limitations.

## 4. Dataset Description
Source: UCI Machine Learning Repository, dataset 235. The dataset contains 2,075,259 one-minute observations from one French household between December 2006 and November 2010. Features include global active/reactive power, voltage, current intensity, and three appliance sub-metering channels. The source has missing measurement values.

Citation: Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

## 5. Preprocessing
- Parse semicolon-separated data and convert `?` to missing values.
- Parse and sort timestamps; remove duplicate timestamps.
- Forward-fill numeric readings up to five consecutive rows, then drop unresolved missing rows.
- Derive cyclic hour, day-of-week, and month features.
- Split chronologically into 70% train, 15% validation, and 15% test.
- Derive class thresholds from the training period only.
- Exclude `Global_active_power` from inputs because it defines the class label; other electrical readings may remain correlated with the label.
- Fit standardization parameters on training features only.

Document the actual row counts and time ranges from `outputs/run_metadata.json`.

## 6. Model Architectures
### 6.1 Baseline MLP
Input layer, two ReLU hidden layers (32 and 16 units), and a three-unit softmax output.

### 6.2 Proposed MLP
Hidden layers with 128 and 64 units, batch normalization, dropout, L2 regularization, and three-class softmax output.

### 6.3 Autoencoder-based classifier
A 32-unit encoder layer compresses input into an 8-dimensional latent representation. A decoder reconstructs the standardized input. A separate classifier maps the learned latent features to three classes.

## 7. Training and Experimental Design
The script uses mini-batches of 256 (and 1024 in one controlled experiment) with Stochastic Gradient Descent (SGD) and momentum 0.9. Learning-rate experiments compare 0.01 and 0.003.

Early stopping monitors validation loss. Experiments vary learning rate, batch size, hidden width, and dropout. Validation results guide selection; test data must not be used for tuning.

## 8. Evaluation Metrics
- Accuracy: proportion of correct predictions.
- Precision: fraction of predicted members of a class that are correct.
- Recall: fraction of actual class members detected.
- F1-score: harmonic mean of precision and recall.
- Macro averages: arithmetic mean across the three classes.
- Confusion matrix: class-by-class counts of actual versus predicted labels.

## 9. Results
Fill the tables using the generated CSV files.

### 9.1 Final test comparison
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline MLP | Insert measured value | Insert measured value | Insert measured value | Insert measured value |
| Proposed MLP | Insert measured value | Insert measured value | Insert measured value | Insert measured value |
| Autoencoder classifier | Insert measured value | Insert measured value | Insert measured value | Insert measured value |

Source: `outputs/final_test_metrics.csv`.

### 9.2 Controlled experiments
Summarize which configuration achieved the best validation macro-F1. Source: `outputs/controlled_experiments_validation.csv`.

### 9.3 Confusion matrix and class-wise performance
Insert the generated PNGs and discuss which classes are confused most frequently. Source: `outputs/classwise_test_metrics.csv`.

## 10. Discussion
Discuss whether the proposed model improves macro-F1 over the baseline, whether validation and test performance differ, which class is hardest to identify, and whether the autoencoder representation helps. Support every claim with generated outputs.

## 11. Error Analysis
Inspect misclassified timestamps in each `*_test_predictions.csv`. Compare confidence, true class, and predicted class. Investigate whether errors cluster around transitions between consumption levels, missing-value intervals, or changes in daily/seasonal behavior. Do not infer causes without checking the records.

## 12. Generalization and Practical Limitations
- The dataset covers one household and a historical period.
- Derived quantile labels are relative to training consumption, not universal kW thresholds.
- Active power defines the class label and is therefore excluded from inputs. Other electrical measurements may still act as correlated proxies. For forecasting, define labels for a future time interval.
- Time-series autocorrelation and distribution shift may reduce performance in future periods.
- Imputation can smooth spikes.
- An autoencoder optimizes reconstruction and may not preserve class-discriminative information.
- Training energy and hardware should be recorded before making sustainability claims.

## 13. Conclusion
Complete this section only after obtaining actual results. State which model performed best on the held-out test set, whether it improved over the baseline, the main confusion pattern, and one concrete next step.

## 14. References
1. Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
2. TensorFlow/Keras documentation: https://www.tensorflow.org/guide/keras
