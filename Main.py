
import zipfile
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


zip_path = "data/task1_classic_classification.zip"

with zipfile.ZipFile(zip_path, "r") as archive:
    with archive.open(
        "task1_classic_classification/calms21_task1_train.json"
    ) as file:
        train_data = json.load(file)

print("Dataset loaded successfully!")

recordings = train_data["annotator-id_0"]
print("Number of recordings:", len(recordings))

behavior_names = {
    0: "Attack",
    1: "Investigation",
    2: "Mount",
    3: "Other"
}

recording_ids = list(recordings.keys())

train_ids, val_ids = train_test_split(
    recording_ids,
    test_size=0.2,
    random_state=42
)

print("Training recordings:", len(train_ids))
print("Validation recordings:", len(val_ids))


def rolling_average(values, window=5):
    padded = np.pad(
        values,
        (window - 1, 0),
        mode="edge"
    )

    return np.convolve(
        padded,
        np.ones(window, dtype=np.float32) / window,
        mode="valid"
    )


def build_dataset(ids):
    X_list = []
    y_list = []

    for recording_id in ids:
        recording = recordings[recording_id]

        keypoints = np.asarray(
            recording["keypoints"],
            dtype=np.float32
        )

        labels = np.asarray(
            recording["annotations"],
            dtype=np.int32
        )

        position_features = keypoints.reshape(
            keypoints.shape[0], -1
        )

        mouse0_neck = keypoints[:, 0, :, 3]
        mouse1_neck = keypoints[:, 1, :, 3]

        mouse0_nose = keypoints[:, 0, :, 0]
        mouse1_nose = keypoints[:, 1, :, 0]

        mouse0_speed = np.linalg.norm(
            np.diff(
                mouse0_neck,
                axis=0,
                prepend=mouse0_neck[:1]
            ),
            axis=1
        ) * 30

        mouse1_speed = np.linalg.norm(
            np.diff(
                mouse1_neck,
                axis=0,
                prepend=mouse1_neck[:1]
            ),
            axis=1
        ) * 30

        mouse_distance = np.linalg.norm(
            mouse0_neck - mouse1_neck,
            axis=1
        )

        nose_to_nose_distance = np.linalg.norm(
            mouse0_nose - mouse1_nose,
            axis=1
        )

        mouse0_nose_to_mouse1_neck = np.linalg.norm(
            mouse0_nose - mouse1_neck,
            axis=1
        )

        mouse1_nose_to_mouse0_neck = np.linalg.norm(
            mouse1_nose - mouse0_neck,
            axis=1
        )

        mouse0_rolling_speed = rolling_average(
            mouse0_speed,
            window=5
        )

        mouse1_rolling_speed = rolling_average(
            mouse1_speed,
            window=5
        )

        rolling_mouse_distance = rolling_average(
            mouse_distance,
            window=5
        )

        distance_change = np.diff(
            mouse_distance,
            prepend=mouse_distance[:1]
        )

        mouse0_acceleration = np.diff(
            mouse0_speed,
            prepend=mouse0_speed[:1]
        ) * 30

        features = np.column_stack([
            position_features,
            mouse0_speed,
            mouse1_speed,
            mouse_distance,
            nose_to_nose_distance,
            mouse0_nose_to_mouse1_neck,
            mouse1_nose_to_mouse0_neck,
            mouse0_rolling_speed,
            mouse1_rolling_speed,
            rolling_mouse_distance,
            distance_change,
            mouse0_acceleration
        ]).astype(np.float32)

        X_list.append(features)
        y_list.append(labels)

    return (
        np.vstack(X_list),
        np.concatenate(y_list)
    )


X_train, y_train = build_dataset(train_ids)
X_val, y_val = build_dataset(val_ids)

print("\nTraining features:", X_train.shape)
print("Validation features:", X_val.shape)

assert X_train.shape[1] == 39
assert X_val.shape[1] == 39

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("\nClass distribution:")

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(unique, counts):
    percentage = count / len(y_train) * 100

    print(
        f"{behavior_names[label]}: "
        f"{count} frames ({percentage:.2f}%)"
    )


configurations = [
    {
        "name": "Baseline",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05
    },
    {
        "name": "Experiment A",
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.03
    },
    {
        "name": "Experiment B",
        "n_estimators": 350,
        "max_depth": 8,
        "learning_rate": 0.03
    }
]

results = []
best_model = None
best_config = None
best_f1 = -1

for config in configurations:
    print(f"\nTraining {config['name']}...")

    candidate = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    candidate.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        verbose=False
    )

    predictions = candidate.predict(X_val)

    acc = accuracy_score(y_val, predictions)
    f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    results.append({
        "Model": config["name"],
        "Accuracy": acc,
        "Macro F1": f1
    })

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = candidate
        best_config = config

model = best_model

print("\nHyperparameter Comparison:")

for result in results:
    print(
        f"{result['Model']}: "
        f"Accuracy={result['Accuracy']:.4f}, "
        f"Macro F1={result['Macro F1']:.4f}"
    )

print("\nBest configuration:", best_config["name"])
print("Best parameters:", best_config)


y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    labels=[0, 1, 2, 3],
    zero_division=0
)

print("\nBest XGBoost Model Results:")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=[
        "Attack",
        "Investigation",
        "Mount",
        "Other"
    ],
    zero_division=0
))


Path("results").mkdir(exist_ok=True)

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3],
    normalize="true"
)

print("\nNormalized Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Attack",
        "Investigation",
        "Mount",
        "Other"
    ]
)

disp.plot(
    cmap="Blues",
    values_format=".2f"
)

plt.title("XGBoost Tuned Model - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/xgboost_tuned_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


num_frames = min(300, len(y_val))

actual = y_val[:num_frames]
predicted = y_pred[:num_frames]

time = np.arange(num_frames) / 30

plt.figure(figsize=(12, 5))

plt.step(
    time,
    actual,
    where="post",
    label="Actual Behavior",
    linewidth=2
)

plt.step(
    time,
    predicted,
    where="post",
    label="Predicted Behavior",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7
)

plt.yticks(
    [0, 1, 2, 3],
    ["Attack", "Investigation", "Mount", "Other"]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Behavior")
plt.title("XGBoost Tuned Model: Actual vs Predicted Behavior")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/xgboost_tuned_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


Path("models").mkdir(exist_ok=True)

model.save_model(
    "models/mouse_behavior_xgboost_tuned.json"
)

print("\nBest tuned XGBoost model saved successfully!")
print("\nProject completed!")
