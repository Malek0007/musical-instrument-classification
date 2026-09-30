"""Train and evaluate a custom CNN for guitar, piano, and violin photos.

Place this file next to custom_cnn_dataset/{guitar,piano,violin}/.
The final test set is evaluated only after a model is selected by validation accuracy.
"""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "custom_cnn_dataset"
OUTPUT_DIR = ROOT / "custom_cnn_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ["guitar", "piano", "violin"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# IMAGE_SIZE = (128, 128)
IMAGE_SIZE = (96, 96)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Keep this collection independent of CIFAR and other packaged ML datasets.
# Review near-duplicates and ambiguous photos manually before running.
records = []
for class_id, class_name in enumerate(CLASS_NAMES):
    folder = DATA_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"{class_name}: {len(files)} supported images")
    if len(files) != 100:
        raise ValueError(
            f"Expected 100 supported images in {folder}; found {len(files)}. "
            "Convert HEIC files to JPEG if necessary."
        )
    records.extend((p, class_id) for p in files)

paths = np.array([str(path) for path, _ in records])
labels = np.array([class_id for _, class_id in records])
all_indices = np.arange(len(records))
train_val_idx, test_idx = train_test_split(
    all_indices, test_size=0.15, random_state=SEED, stratify=labels
)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=20 / 85, random_state=SEED,
    stratify=labels[train_val_idx]
)

with (OUTPUT_DIR / "split_manifest.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "class", "partition"])
    for partition, indices in [
        ("train", train_idx), ("validation", val_idx), ("test", test_idx)
    ]:
        for i in indices:
            writer.writerow([paths[i], CLASS_NAMES[labels[i]], partition])

for name, indices in [("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
    counts = np.bincount(labels[indices], minlength=3)
    print(f"{name}: {len(indices)} total; per class {dict(zip(CLASS_NAMES, counts.tolist()))}")


def read_images(indices):
    images = []
    for i in indices:
        try:
            with Image.open(paths[i]) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                images.append(np.asarray(rgb.resize(IMAGE_SIZE), dtype=np.uint8))
        except Exception as exc:
            raise ValueError(f"Cannot read {paths[i]}: {exc}") from exc
    return np.stack(images), labels[indices]


x_train, y_train = read_images(train_idx)
x_val, y_val = read_images(val_idx)
x_test, y_test = read_images(test_idx)

# Three planned iterations. Change one hyperparameter at each step.
# CONFIGS = [
#     {"name": "v1_baseline", "learning_rate": 0.01, "dropout": 0.20},
#     {"name": "v2_dropout", "learning_rate": 0.01, "dropout": 0.40},
#     {"name": "v3_learning_rate", "learning_rate": 0.003, "dropout": 0.40},
# ]
CONFIGS = [
    {"name": "v1_baseline", "learning_rate": 0.001, "dropout": 0.30},
    {"name": "v2_dropout", "learning_rate": 0.001, "dropout": 0.50},
    {"name": "v3_learning_rate", "learning_rate": 0.0003, "dropout": 0.50},
]

def build_model(learning_rate, dropout):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(128, 128, 3)),
    #     # Augmentation only runs on training batches, helping with 195 training photos.
    #     tf.keras.layers.RandomFlip("horizontal"),
    #     tf.keras.layers.RandomRotation(0.05),
    #     tf.keras.layers.Rescaling(1.0 / 255),
    #     tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    #     tf.keras.layers.MaxPooling2D(2),
    #     tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    #     tf.keras.layers.MaxPooling2D(2),
    #     tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    #     tf.keras.layers.MaxPooling2D(2),
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(64, activation="relu"),
    #     tf.keras.layers.Dropout(dropout),
    #     tf.keras.layers.Dense(3, activation="softmax"),
    # ])
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96, 96, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(3, activation="softmax"),
])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
        


results = []
histories = {}
for config in CONFIGS:
    tf.keras.utils.set_random_seed(SEED)
    name = config["name"]
    model = build_model(config["learning_rate"], config["dropout"])
    if name == "v1_baseline":
        model.summary()
    checkpoint_path = OUTPUT_DIR / f"{name}.keras"
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=40,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy",
                mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=12,
    restore_best_weights=True
),
        ],
        verbose=1,
    )
    histories[name] = history.history
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    saved = tf.keras.models.load_model(checkpoint_path)
    _, train_accuracy = saved.evaluate(x_train, y_train, verbose=0)
    _, val_accuracy = saved.evaluate(x_val, y_val, verbose=0)
    row = {
        **config,
        "best_epoch": best_epoch,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
    }
    results.append(row)
    print(f"{name}: best epoch {best_epoch}, training {train_accuracy:.4f}, "
          f"validation {val_accuracy:.4f}")

# Select by validation accuracy only. Test images have not been evaluated yet.
winner = max(results, key=lambda row: row["val_accuracy"])
best_model = tf.keras.models.load_model(OUTPUT_DIR / f"{winner['name']}.keras")
_, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
predictions = np.argmax(best_model.predict(x_test, verbose=0), axis=1)
confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=3).numpy()

print("\nSelected by validation:", winner["name"])
print(f"Training accuracy:   {winner['train_accuracy']:.4f}")
print(f"Validation accuracy: {winner['val_accuracy']:.4f}")
print(f"Test accuracy:       {test_accuracy:.4f}")
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion)

with (OUTPUT_DIR / "iteration_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
with (OUTPUT_DIR / "test_predictions.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "actual", "predicted"])
    for i, actual, prediction in zip(test_idx, y_test, predictions):
        writer.writerow([paths[i], CLASS_NAMES[actual], CLASS_NAMES[prediction]])
with (OUTPUT_DIR / "summary.json").open("w") as f:
    json.dump({
        "selected_model": winner,
        "test_accuracy": float(test_accuracy),
        "class_names": CLASS_NAMES,
        "confusion_matrix": confusion.tolist(),
    }, f, indent=2)

plt.figure(figsize=(8, 5))
for name, history in histories.items():
    plt.plot(
        np.arange(1, len(history["val_accuracy"]) + 1),
        history["val_accuracy"], marker="o", markersize=3, label=name
    )
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Three-class CNN: hyperparameter iterations")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_curves.png", dpi=200)
plt.close()
