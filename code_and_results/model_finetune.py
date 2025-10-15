
import os
import math
import json
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# ============ Config ============
CONFIG = {
    "model_name": "efficientnetb7",  # "resnet50" | "efficientnetb7" | "mobilenetv3"
    "data_dir": "D:/DeepFake/pythonProject1/Frames/FF/FF 600",  # root folder containing train/ val/ test/
    "epochs": 10,
    "batch_size": 32,
    "seed": 42,
    "base_trainable_at": -40,  # e.g., -30 to fine-tune last 30 layers after warmup; None = freeze all base
    "warmup_epochs": 3,  # epochs with base frozen before optional fine-tune
    "learning_rate": 1e-3,
    "fine_tune_lr": 2e-5,  # used after unfreezing
    "use_class_weights": False,  # set True if your classes are imbalanced
    "mixed_precision": False,  # set True if you have a GPU that benefits (e.g., Ampere)
    "output_dir": "D:/DeepFake/pythonProject1/Main/FF/efficientnetb7_1a",
}

# ============ Repro / Perf ============
tf.keras.utils.set_random_seed(CONFIG["seed"])
if CONFIG["mixed_precision"]:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


# ============ Model registry ============
def get_model_spec(name: str):
    name = name.lower()
    if name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return {
            "constructor": lambda input_shape: ResNet50(include_top=False, weights="imagenet", input_shape=input_shape),
            "preprocess": preprocess_input,
            "input_size": (224, 224),
            "pool": "avg"
        }
    elif name == "efficientnetb7":
        from tensorflow.keras.applications import EfficientNetB7
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return {
            "constructor": lambda input_shape: EfficientNetB7(include_top=False, weights="imagenet",
                                                              input_shape=input_shape),
            "preprocess": preprocess_input,
            "input_size": (600, 600),  # heavy but high accuracy
            "pool": "avg"
        }
    elif name == "mobilenetv3":
        # MobileNetV3Large available in tf >=2.8 as tensorflow.keras.applications.MobileNetV3Large
        from tensorflow.keras.applications import MobileNetV3Large
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        return {
            "constructor": lambda input_shape: MobileNetV3Large(include_top=False, weights="imagenet",
                                                                input_shape=input_shape),
            "preprocess": preprocess_input,
            "input_size": (224, 224),
            "pool": "avg"
        }
    else:
        raise ValueError("Unknown model_name. Use 'resnet50', 'efficientnetb7', or 'mobilenetv3'.")


#============ Data ============
def build_generators(data_dir, input_size, preprocess, batch_size, seed):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_aug = ImageDataGenerator(preprocessing_function=preprocess)
    test_aug = ImageDataGenerator(preprocessing_function=preprocess)

    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=input_size, batch_size=batch_size, seed=seed, class_mode="categorical", color_mode="rgb"
    )
    val_gen = val_aug.flow_from_directory(
        val_dir, target_size=input_size, batch_size=batch_size, seed=seed, class_mode="categorical", shuffle=False, color_mode="rgb"
    )
    test_gen = test_aug.flow_from_directory(
        test_dir, target_size=input_size, batch_size=batch_size, seed=seed, class_mode="categorical", shuffle=False, color_mode="rgb"
    )

    return train_gen, val_gen, test_gen

# def build_generators(data_dir, input_size, preprocess, batch_size, seed):
#     train_dir = os.path.join(data_dir, "train")
#     val_dir = os.path.join(data_dir, "validation")
#     test_dir = os.path.join(data_dir, "test")
#
#     # No augmentation, only preprocessing
#     train_aug = ImageDataGenerator(preprocessing_function=preprocess)
#     val_aug = ImageDataGenerator(preprocessing_function=preprocess)
#     test_aug = ImageDataGenerator(preprocessing_function=preprocess)
#
#     train_gen = train_aug.flow_from_directory(
#         train_dir, target_size=input_size, batch_size=batch_size, seed=seed,
#         class_mode="categorical", color_mode="rgb"
#     )
#     val_gen = val_aug.flow_from_directory(
#         val_dir, target_size=input_size, batch_size=batch_size, seed=seed,
#         class_mode="categorical", shuffle=False, color_mode="rgb"
#     )
#     test_gen = test_aug.flow_from_directory(
#         test_dir, target_size=input_size, batch_size=batch_size, seed=seed,
#         class_mode="categorical", shuffle=False, color_mode="rgb"
#     )
#
#     return train_gen, val_gen, test_gen


# ============ Head ============
def build_head(x, num_classes, dtype="float32", dropout=0.3, pool="avg"):
    if pool == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    elif pool == "max":
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype=dtype)(x)
    return out


# ============ Class weights (optional) ============
def compute_class_weights(generator):
    # Count samples per class
    counts = np.zeros(len(generator.class_indices), dtype=np.int64)
    for _, y in itertools.islice(zip(generator, itertools.count()),
                                 math.ceil(generator.samples / generator.batch_size)):
        counts += y.sum(axis=0).astype(int)
    total = counts.sum()
    weights = {i: float(total / (len(counts) * c)) for i, c in enumerate(counts)}
    return weights


# ============ Compile helper ============
def compile_model(model, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ============ Training pipeline ============
def train():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["output_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    spec = get_model_spec(CONFIG["model_name"])
    input_h, input_w = spec["input_size"]
    preprocess = spec["preprocess"]

    train_gen, val_gen, test_gen = build_generators(
        CONFIG["data_dir"], (input_h, input_w), preprocess, CONFIG["batch_size"], CONFIG["seed"]
    )
    num_classes = train_gen.num_classes

    base_model = spec["constructor"]((input_h, input_w, 3))
    base_model.trainable = False  # freeze for warmup

    inputs = layers.Input(shape=(input_h, input_w, 3))
    x = base_model(inputs, training=False)
    outputs = build_head(x, num_classes, dtype=("float32" if not CONFIG["mixed_precision"] else "float32"),
                         pool=spec["pool"])
    model = models.Model(inputs, outputs, name=f"{CONFIG['model_name']}_transfer")

    # Callbacks
    ckpt_path = os.path.join(CONFIG["output_dir"], "best_warmup.keras")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", mode="max"),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(CONFIG["output_dir"], "training_log.csv"))
    ]

    # Class weights if needed
    class_weights = None
    if CONFIG["use_class_weights"]:
        print("Computing class weights...")
        class_weights = compute_class_weights(train_gen)
        print("Class weights:", class_weights)

    # Warmup: train top head only
    compile_model(model, CONFIG["learning_rate"])
    history_warm = model.fit(
        train_gen,
        epochs=CONFIG["warmup_epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Optional fine-tuning
    history_ft = None
    if CONFIG["base_trainable_at"] is not None:
        # Unfreeze last N layers
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[CONFIG["base_trainable_at"]:]:
            layer.trainable = True

        ckpt_path_ft = os.path.join(CONFIG["output_dir"], "best_finetune.keras")
        callbacks_ft = [
            EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy", mode="max"),
            ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
            ModelCheckpoint(ckpt_path_ft, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
            CSVLogger(os.path.join(CONFIG["output_dir"], "training_log_ft.csv"))
        ]

        compile_model(model, CONFIG["fine_tune_lr"])
        history_ft = model.fit(
            train_gen,
            epochs=CONFIG["epochs"],
            validation_data=val_gen,
            callbacks=callbacks_ft,
            class_weight=class_weights,
            verbose=1
        )

    # Save final model
    final_path = os.path.join(CONFIG["output_dir"], "efficientnetb7.keras")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    # Evaluate on test
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Classification report
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1).argmax(axis=1)
    target_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Quick plots (saved as PNGs)
    import matplotlib.pyplot as plt

    def plot_history(hist, out_path_prefix):
        if hist is None: return
        h = hist.history
        # Accuracy
        plt.figure()
        plt.plot(h["accuracy"], label="train_acc")
        plt.plot(h["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch");
        plt.ylabel("Accuracy");
        plt.legend();
        plt.title("Accuracy")
        plt.savefig(out_path_prefix + "_acc.png", bbox_inches="tight")
        plt.close()
        # Loss
        plt.figure()
        plt.plot(h["loss"], label="train_loss")
        plt.plot(h["val_loss"], label="val_loss")
        plt.xlabel("Epoch");
        plt.ylabel("Loss");
        plt.legend();
        plt.title("Loss")
        plt.savefig(out_path_prefix + "_loss.png", bbox_inches="tight")
        plt.close()

    plot_history(history_warm, os.path.join(CONFIG["output_dir"], "warmup"))
    plot_history(history_ft, os.path.join(CONFIG["output_dir"], "finetune"))


if __name__ == "__main__":
    print("Using TensorFlow", tf.__version__)
    print("Config:", json.dumps(CONFIG, indent=2))
    train()
