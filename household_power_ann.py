"""
ANN-Based Classification of Individual Household Electric Power Consumption
Energy & Sustainability | TensorFlow/Keras

Usage:
    pip install -r requirements.txt
    python household_power_ann.py --data household_power_consumption.txt

If --data is omitted, the script downloads the official UCI ZIP archive.
The script writes metrics, plots, model files, and experiment logs to ./outputs.

Important:
- Labels are derived from training-period quantiles of Global_active_power.
- Splits are chronological: train -> validation -> test.
- The test set is evaluated only after model/experiment selection.
- Reported metrics are computed by running this script; this file does not
  contain fabricated results.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
TARGET = "Global_active_power"
FEATURES = [
    # Exclude Global_active_power from inputs because it defines the class label.
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2",
    "Sub_metering_3", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]
CLASS_NAMES = ["Low", "Medium", "High"]
UCI_ZIP = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_data_path(data_arg: str | None, workdir: Path) -> Path:
    if data_arg:
        path = Path(data_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    workdir.mkdir(parents=True, exist_ok=True)
    expected = workdir / "household_power_consumption.txt"
    if expected.exists():
        return expected

    print("Downloading official UCI dataset archive...")
    req = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        candidates = [n for n in zf.namelist()
                      if n.lower().endswith("household_power_consumption.txt")]
        if not candidates:
            raise RuntimeError("Could not find household_power_consumption.txt in UCI ZIP.")
        member = candidates[0]
        with zf.open(member) as src, open(expected, "wb") as dst:
            dst.write(src.read())
    return expected


def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"Reading data from: {path}")
    # UCI source uses semicolons and '?' for missing numeric values.
    df = pd.read_csv(
        path, sep=";", na_values="?", low_memory=False,
        dtype={"Date": "string", "Time": "string"}
    )
    required = ["Date", "Time", "Global_active_power", "Global_reactive_power",
                "Voltage", "Global_intensity", "Sub_metering_1",
                "Sub_metering_2", "Sub_metering_3"]
    missing_cols = sorted(set(required) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Expected UCI columns missing: {missing_cols}")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    numeric_cols = required[2:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Forward fill only numeric readings for short gaps; do not backfill from future.
    # Long initial gaps remain missing and are removed. The time series remains ordered.
    df[numeric_cols] = df[numeric_cols].ffill(limit=5)
    before = len(df)
    df = df.dropna(subset=numeric_cols).copy()
    print(f"Rows after parsing/deduplication/imputation: {len(df):,} "
          f"(removed {before - len(df):,} rows with unresolved missing values)")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    out = df.copy()
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    month = ts.dt.month - 1
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def chronological_split(df: pd.DataFrame):
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def make_labels(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    # Label boundaries are learned from the training period only.
    q1, q2 = train[TARGET].quantile([1/3, 2/3]).to_numpy()
    if not q1 < q2:
        raise ValueError("Training quantiles do not define three distinct classes.")
    def label(part):
        return np.digitize(part[TARGET].to_numpy(), bins=[q1, q2]).astype(np.int32)
    return label(train), label(val), label(test), float(q1), float(q2)


def prepare_arrays(train, val, test):
    y_train, y_val, y_test, q1, q2 = make_labels(train, val, test)

    # All preprocessing is fitted only on the training split.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].astype("float32"))
    x_val = scaler.transform(val[FEATURES].astype("float32"))
    x_test = scaler.transform(test[FEATURES].astype("float32"))

    return (x_train, x_val, x_test, y_train, y_val, y_test,
            scaler, q1, q2)


def build_mlp(input_dim, width1=64, width2=32, dropout=0.15,
              learning_rate=1e-3, l2=1e-4):
    inputs = tf.keras.Input(shape=(input_dim,), name="standardized_features")
    x = tf.keras.layers.Dense(
        width1, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        width2, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2)
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="mlp_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_autoencoder_classifier(input_dim, latent_dim=8, learning_rate=1e-3):
    inputs = tf.keras.Input(shape=(input_dim,), name="standardized_features")
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(latent)
    reconstruction = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = tf.keras.Model(inputs, reconstruction, name="autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss="mse"
    )

    # Encoder is trained on training data only; classification labels are supervised.
    encoder = tf.keras.Model(inputs, latent, name="encoder")
    clf_in = tf.keras.Input(shape=(latent_dim,), name="latent_features")
    z = tf.keras.layers.Dense(16, activation="relu")(clf_in)
    z = tf.keras.layers.Dropout(0.15)(z)
    clf_out = tf.keras.layers.Dense(3, activation="softmax")(z)
    classifier = tf.keras.Model(clf_in, clf_out, name="latent_classifier")
    classifier.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return autoencoder, encoder, classifier


def callbacks(path: Path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(path), monitor="val_loss", save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]


def fit_classifier(model, x_train, y_train, x_val, y_val, epochs, batch_size,
                   output_path, class_weight=None):
    return model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        shuffle=True, verbose=2,
        callbacks=callbacks(output_path),
        class_weight=class_weight
    )


def metrics_dict(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }


def evaluate_and_save(name, y_true, probabilities, outdir: Path):
    pred = np.argmax(probabilities, axis=1)
    metrics = metrics_dict(y_true, pred)
    metrics["model"] = name
    print(f"\n{name} TEST METRICS (unseen chronological test set)")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_true, pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
    ))
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name}: test confusion matrix")
    plt.tight_layout()
    plt.savefig(outdir / f"{name.lower()}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics, pred, cm


def plot_history(history, title, path):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "loss" in hist:
        axes[0].plot(hist["loss"], label="train")
    if "val_loss" in hist:
        axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    if "accuracy" in hist:
        axes[1].plot(hist["accuracy"], label="train")
    if "val_accuracy" in hist:
        axes[1].plot(hist["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiments(x_train, y_train, x_val, y_val, outdir, max_epochs, seed):
    """Controlled experiments use validation metrics only; test set is not passed in."""
    configs = [
        {"name": "mlp_lr_1e-2_bs256", "lr": 1e-2, "batch": 256, "w1": 64, "w2": 32, "dropout": .15},
        {"name": "mlp_lr_3e-3_bs256", "lr": 3e-3, "batch": 256, "w1": 64, "w2": 32, "dropout": .15},
        {"name": "mlp_lr_1e-2_bs1024", "lr": 1e-2, "batch": 1024, "w1": 64, "w2": 32, "dropout": .15},
        {"name": "mlp_wider_128_64", "lr": 1e-2, "batch": 256, "w1": 128, "w2": 64, "dropout": .15},
        {"name": "mlp_regularized_dropout30", "lr": 1e-2, "batch": 256, "w1": 64, "w2": 32, "dropout": .30},
    ]
    results = []
    for cfg in configs:
        print(f"\n=== Experiment: {cfg['name']} ===")
        set_seed(seed)
        model = build_mlp(
            x_train.shape[1], cfg["w1"], cfg["w2"], cfg["dropout"], cfg["lr"]
        )
        save_path = outdir / f"{cfg['name']}_best.keras"
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            epochs=max_epochs, batch_size=cfg["batch"], shuffle=True, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ), tf.keras.callbacks.TerminateOnNaN()]
        )
        val_prob = model.predict(x_val, batch_size=4096, verbose=0)
        val_pred = val_prob.argmax(axis=1)
        row = {
            **cfg,
            "best_val_loss": float(np.min(history.history["val_loss"])),
            **{f"val_{k}": v for k, v in metrics_dict(y_val, val_pred).items()}
        }
        results.append(row)
        model.save(save_path)
    frame = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    frame.to_csv(outdir / "controlled_experiments_validation.csv", index=False)
    print("\nControlled experiments ranked by validation macro-F1:")
    print(frame.to_string(index=False))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to household_power_consumption.txt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Optional debugging cap after cleaning; 0 means all rows")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = get_data_path(args.data, Path("data"))
    df = load_and_clean(data_path)
    df = add_calendar_features(df)
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()
        print(f"DEBUG MODE: using first {len(df):,} rows only")
    train, val, test = chronological_split(df)
    (x_train, x_val, x_test, y_train, y_val, y_test,
     scaler, q1, q2) = prepare_arrays(train, val, test)

    # Persist metadata to make experiment decisions auditable.
    metadata = {
        "seed": args.seed,
        "rows_after_cleaning": int(len(df)),
        "split_counts": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "time_ranges": {
            "train": [str(train.timestamp.min()), str(train.timestamp.max())],
            "validation": [str(val.timestamp.min()), str(val.timestamp.max())],
            "test": [str(test.timestamp.min()), str(test.timestamp.max())],
        },
        "features": FEATURES,
        "target": TARGET,
        "label_method": "training-period one-third and two-thirds quantiles",
        "training_quantile_thresholds_kw": {"q1": q1, "q2": q2},
        "missing_data_policy": "numeric forward-fill up to 5 rows, then drop unresolved rows",
        "note": "No test-based tuning. Labels and scaler fitted on training data only. Global_active_power is excluded from input features because it defines the target labels."
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    # Baseline: compact MLP with fixed hyperparameters.
    print("\n=== Baseline MLP ===")
    set_seed(args.seed)
    baseline = build_mlp(x_train.shape[1], 32, 16, dropout=0.0, learning_rate=1e-2, l2=0.0)
    hist_base = fit_classifier(
        baseline, x_train, y_train, x_val, y_val, args.epochs, 256,
        outdir / "baseline_mlp_best.keras", class_weight
    )
    plot_history(hist_base, "Baseline MLP", outdir / "baseline_learning_curves.png")

    # Proposed MLP.
    print("\n=== Proposed MLP ===")
    set_seed(args.seed)
    proposed = build_mlp(x_train.shape[1], 128, 64, dropout=0.20, learning_rate=1e-2, l2=1e-4)
    hist_prop = fit_classifier(
        proposed, x_train, y_train, x_val, y_val, args.epochs, 256,
        outdir / "proposed_mlp_best.keras", class_weight
    )
    plot_history(hist_prop, "Proposed MLP", outdir / "proposed_learning_curves.png")

    # Autoencoder: unsupervised reconstruction on training features only.
    print("\n=== Autoencoder representation + classifier ===")
    set_seed(args.seed)
    autoencoder, encoder, latent_clf = build_autoencoder_classifier(x_train.shape[1], latent_dim=8, learning_rate=1e-2)
    ae_history = autoencoder.fit(
        x_train, x_train, validation_data=(x_val, x_val),
        epochs=args.epochs, batch_size=256, shuffle=True, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]
    )
    plot_history(ae_history, "Autoencoder reconstruction", outdir / "autoencoder_learning_curves.png")
    z_train = encoder.predict(x_train, batch_size=4096, verbose=0)
    z_val = encoder.predict(x_val, batch_size=4096, verbose=0)
    z_test = encoder.predict(x_test, batch_size=4096, verbose=0)
    ae_clf_history = latent_clf.fit(
        z_train, y_train, validation_data=(z_val, y_val),
        epochs=args.epochs, batch_size=256, shuffle=True, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN()
        ],
        class_weight=class_weight
    )
    plot_history(ae_clf_history, "Autoencoder latent classifier", outdir / "ae_classifier_learning_curves.png")

    # Experiments use validation data only and are kept separate from final test evaluation.
    experiment_frame = None
    if not args.skip_experiments:
        experiment_frame = run_experiments(
            x_train, y_train, x_val, y_val, outdir, args.epochs, args.seed
        )

    # Final test evaluation. Test metrics do not select the model.
    test_results = []
    model_probs = {
        "Baseline_MLP": baseline.predict(x_test, batch_size=4096, verbose=0),
        "Proposed_MLP": proposed.predict(x_test, batch_size=4096, verbose=0),
        "Autoencoder_Classifier": latent_clf.predict(z_test, batch_size=4096, verbose=0),
    }
    for name, probs in model_probs.items():
        result, pred, cm = evaluate_and_save(name, y_test, probs, outdir)
        test_results.append(result)
        pd.DataFrame({
            "timestamp": test["timestamp"].astype(str).to_numpy(),
            "true_class": [CLASS_NAMES[i] for i in y_test],
            "predicted_class": [CLASS_NAMES[i] for i in pred],
            "correct": y_test == pred,
            "confidence": probs.max(axis=1)
        }).to_csv(outdir / f"{name.lower()}_test_predictions.csv", index=False)

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(outdir / "final_test_metrics.csv", index=False)
    print("\nFinal test comparison:")
    print(results_df.to_string(index=False))

    # Class-wise metrics and confusion matrix counts for error analysis.
    error_rows = []
    for name, probs in model_probs.items():
        pred = probs.argmax(axis=1)
        report = classification_report(
            y_test, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        )
        for cls in CLASS_NAMES:
            error_rows.append({
                "model": name, "class": cls,
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"]
            })
    pd.DataFrame(error_rows).to_csv(outdir / "classwise_test_metrics.csv", index=False)

    # Save fitted scaler parameters for inference/reproducibility.
    np.savez(
        outdir / "scaler_parameters.npz",
        mean=scaler.mean_, scale=scaler.scale_,
        feature_names=np.array(FEATURES), label_q1=q1, label_q2=q2
    )
    baseline.save(outdir / "baseline_mlp_final.keras")
    proposed.save(outdir / "proposed_mlp_final.keras")
    autoencoder.save(outdir / "autoencoder_final.keras")
    encoder.save(outdir / "encoder_final.keras")
    latent_clf.save(outdir / "autoencoder_classifier_final.keras")

    print(f"\nDone. All generated outputs are in: {outdir.resolve()}")
    print("Use actual output CSVs/plots in your report; do not invent metric values.")


if __name__ == "__main__":
    main()
