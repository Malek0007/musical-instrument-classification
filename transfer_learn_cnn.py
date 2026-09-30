"""Transfer learning on the same guitar/piano/violin split as Part 4."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "transfer_learning_results"
RESULTS_DIR.mkdir(exist_ok=True)

MANIFEST = ROOT / "custom_cnn_results" / "split_manifest.csv"
CLASS_NAMES = ["guitar", "piano", "violin"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (96, 96)
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Reuse exactly the same partitions as the custom CNN.
partitions = {"train": [], "validation": [], "test": []}

with MANIFEST.open(newline="") as file:
    for row in csv.DictReader(file):
        partitions[row["partition"]].append(
            (Path(row["path"]), CLASS_TO_ID[row["class"]])
        )


def load_partition(records):
    images = []
    labels = []

    for path, class_id in records:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize(IMAGE_SIZE)
            images.append(np.asarray(image, dtype=np.float32))
        labels.append(class_id)

    return np.stack(images), np.asarray(labels, dtype=np.int32)


x_train, y_train = load_partition(partitions["train"])
x_val, y_val = load_partition(partitions["validation"])
x_test, y_test = load_partition(partitions["test"])

print(f"Training: {len(x_train)} images")
print(f"Validation: {len(x_val)} images")
print(f"Test: {len(x_test)} images")


def build_model(learning_rate):
    # The pretrained feature extractor stays frozen.
    feature_extractor = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3),
    )
    feature_extractor.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = feature_extractor(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Change one setting to see whether validation accuracy improves.
CONFIGS = [
    {"name": "resnet_head_v1", "learning_rate": 0.001},
    {"name": "resnet_head_v2", "learning_rate": 0.0001},
]

results = []
histories = {}

for config in CONFIGS:
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(config["learning_rate"])
    checkpoint_path = RESULTS_DIR / f"{config['name']}.keras"

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=8,
            ),
        ],
        verbose=1,
    )

    histories[config["name"]] = history.history
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    saved_model = tf.keras.models.load_model(checkpoint_path)
    _, train_accuracy = saved_model.evaluate(x_train, y_train, verbose=0)
    _, val_accuracy = saved_model.evaluate(x_val, y_val, verbose=0)

    results.append({
        "name": config["name"],
        "learning_rate": config["learning_rate"],
        "best_epoch": best_epoch,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
    })

    print(
        f"{config['name']}: best epoch {best_epoch}, "
        f"training {train_accuracy:.4f}, validation {val_accuracy:.4f}"
    )

# Choose using validation accuracy. Evaluate the test set once afterward.
winner = max(results, key=lambda result: result["val_accuracy"])
best_model = tf.keras.models.load_model(
    RESULTS_DIR / f"{winner['name']}.keras"
)
_, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)

predictions = np.argmax(
    best_model.predict(x_test, verbose=0), axis=1
)
confusion = tf.math.confusion_matrix(
    y_test, predictions, num_classes=3
).numpy()

custom_cnn_test_accuracy = 0.7778

print("\nSelected transfer-learning model:", winner["name"])
print(f"Training accuracy:   {winner['train_accuracy']:.4f}")
print(f"Validation accuracy: {winner['val_accuracy']:.4f}")
print(f"Test accuracy:       {test_accuracy:.4f}")
print(f"Custom CNN test accuracy: {custom_cnn_test_accuracy:.4f}")
print(
    "Difference from custom CNN: "
    f"{test_accuracy - custom_cnn_test_accuracy:+.4f}"
)
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion)

# Additional test result for ResNet50 V1
v1_model = tf.keras.models.load_model(
    RESULTS_DIR / "resnet_head_v1.keras"
)

_, v1_test_accuracy = v1_model.evaluate(
    x_test, y_test, verbose=0
)

print(f"ResNet50 V1 test accuracy: {v1_test_accuracy:.4f}")
print(f"ResNet50 V2 test accuracy: {test_accuracy:.4f}")

with (RESULTS_DIR / "iteration_results.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

plt.figure(figsize=(8, 5))
for name, history in histories.items():
    epochs = np.arange(1, len(history["val_accuracy"]) + 1)
    plt.plot(epochs, history["val_accuracy"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Frozen ResNet50: classification head experiments")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "validation_accuracy.png", dpi=200)
plt.show()