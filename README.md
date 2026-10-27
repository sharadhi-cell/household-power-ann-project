# ANN-Based Classification of Household Electric Power Consumption

## Project objective
Build and compare TensorFlow/Keras neural networks that classify household active power as Low, Medium, or High. Compare a baseline MLP, a proposed MLP, and an autoencoder representation followed by a classifier.

## Dataset
UCI Machine Learning Repository, dataset 235:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The source contains 2,075,259 one-minute records and missing numeric observations. Original data are primarily continuous/time-series; the three classification labels are derived for this project from the training split's one-third and two-thirds quantiles of `Global_active_power`.

## Requirements
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Download automatically from the official UCI archive:
```bash
python household_power_ann.py --epochs 25
```

Or pass a local source file:
```bash
python household_power_ann.py --data ./household_power_consumption.txt --epochs 25
```

Fast smoke test (debug only; not for final results):
```bash
python household_power_ann.py --max-rows 50000 --epochs 2 --skip-experiments
```

## Methodology
1. Parse semicolon-separated source data; convert `?` to missing values.
2. Parse timestamps, sort chronologically, remove duplicate timestamps.
3. Forward-fill numeric gaps up to five consecutive records; drop remaining rows with missing numeric values. This is a causal fill and does not use future readings.
4. Create cyclic hour/day-of-week/month features.
5. Split chronologically: 70% training, 15% validation, 15% test.
6. Define low/medium/high labels using training-period 33.3rd and 66.7th percentiles of active power only.
7. Fit `StandardScaler` on training features only and apply it to validation/test.
8. Train a baseline MLP, a regularized wider MLP, and an autoencoder plus latent-space classifier.
9. Conduct controlled MLP experiments changing one factor at a time where possible: learning rate, batch size, width/depth, and dropout.
10. Use validation metrics for model/parameter comparisons; evaluate final candidates on the held-out chronological test split.
11. Export test predictions, aggregate and class-wise metrics, confusion matrices, learning curves, metadata, and models.

## Architecture
### Baseline MLP
Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Softmax(3).
### Proposed MLP
Input -> Dense(128, ReLU + L2) -> BatchNorm -> Dropout(0.20) -> Dense(64, ReLU + L2) -> Dropout(0.20) -> Softmax(3).
### Autoencoder
Input -> Dense(32, ReLU) -> latent Dense(8, ReLU) -> Dense(32, ReLU) -> reconstruction. Train to reconstruct standardized input using training data. Freeze/use encoder representation to train a three-class supervised classifier.

## Optimization
Keras uses mini-batches (`batch_size` 256 or experimental 1024) with `tf.keras.optimizers.SGD` and momentum 0.9. Learning-rate experiments compare 0.01 with 0.003.

## Metrics
Accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, and confusion matrices. Macro averaging gives equal weight to each class. Use confusion matrices to identify which consumption levels are confused.

## Output files
The `outputs/` directory includes:
- `run_metadata.json`
- `controlled_experiments_validation.csv`
- `final_test_metrics.csv`
- `classwise_test_metrics.csv`
- per-model test predictions CSV
- confusion matrix PNGs
- learning curve PNGs
- saved Keras models and scaler parameters

## Report writing
Do not write metric numbers into the report until the script has completed. Insert the actual values from `outputs/final_test_metrics.csv` and classwise metrics. Treat any expected trends as hypotheses, not results.

## Important limitations
- One household and one historical period; results may not generalize to other homes or current appliances.
- Labels are derived from the training distribution and are not official UCI ground truth.
- The class labels are derived from contemporaneous active power, but active power itself is excluded from input features. Other electrical measurements may still be strongly correlated with it. For a real early-warning system, define the target over a future interval and use only information available before that interval.
- Adjacent minute records are highly correlated. Chronological split is more realistic than random split but cannot eliminate distribution drift.
- Forward-fill can smooth abrupt changes; the five-minute cap and dropped unresolved rows should be documented.
- The autoencoder is trained to reconstruct inputs, not directly to separate classes; compressed features may or may not improve classification.
- Neural networks are not automatically energy-efficient; report training hardware/time if sustainability impact is discussed.
