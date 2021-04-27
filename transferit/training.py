"""Functions for training model."""

import glob
from pathlib import Path
from typing import List, Tuple

from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .callbacks import PlotLosses

BATCH_SIZE = 32
CLASS_MODE = "categorical"
EPOCHS = 40
PATIENCE = 2
LEARNING_RATE = 0.001


def train(
    train_folder: Path,
    dev_folder: Path,
    model_folder: Path,
    image_size: Tuple[int, int],
):

    class_list = sorted([Path(f).name for f in glob.glob(str(train_folder / "**"))])
    print(f" ◦ Train folder: {train_folder}")
    print(f" ◦ Dev folder:   {dev_folder}")
    print(f" ◦  {len(class_list):3,} classes in train set")
    print("")
    print("Classes (first five only):")
    for class_ in class_list[:5]:
        print(f" ◦  {class_}")
    if len(class_list) > 5:
        print(" ◦  ...")

    model_folder.mkdir(parents=True, exist_ok=False)

    print("")
    model_name = model_folder.name
    print(f"Model name: {model_name}")
    print("")
    print("Training data:")
    train_iterator = _build_train_iterator(train_folder, class_list, image_size)
    print("")
    print("Dev data:")
    dev_iterator = _build_dev_iterator(dev_folder, class_list, image_size)
    print("")

    model = _build_model(class_count=len(class_list), image_size=image_size)

    callbacks = _build_callbacks(model_folder, model_name)

    _ = model.fit(
        train_iterator,
        steps_per_epoch=train_iterator.samples // BATCH_SIZE,
        validation_data=dev_iterator,
        validation_steps=dev_iterator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        use_multiprocessing=False,
        callbacks=callbacks,
    )

    model.save(str(model_folder / f"{model_name}_final.hdf5"))
    print("")


def _build_train_iterator(folder, classes, image_size: Tuple[int, int]):
    train_data_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        channel_shift_range=0.2,
        brightness_range=[0.7, 1.1],
        validation_split=0,
        preprocessing_function=preprocess_input,
    )

    train_iterator = train_data_generator.flow_from_directory(
        folder,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=True,
        seed=42,
        classes=classes,
        subset="training",
    )

    return train_iterator


def _build_dev_iterator(folder, classes, image_size):
    dev_data_generator = ImageDataGenerator(
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False,
        validation_split=0,
        preprocessing_function=preprocess_input,
    )

    dev_iterator = dev_data_generator.flow_from_directory(
        folder,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=True,
        seed=42,
        classes=classes,
    )

    return dev_iterator


def _build_model(class_count: int, image_size: Tuple[int, int]) -> Model:
    base_model = ResNet50(
        input_shape=(image_size[0], image_size[1], 3),
        weights="imagenet",
        include_top=False,
        pooling="avg",
    )

    classifier = _build_transfer_classifier(base_model, class_count)

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    classifier.compile(
        optimizer=optimizer,
        loss=["categorical_crossentropy"],
        metrics=["acc"],
    )
    return classifier


def _build_transfer_classifier(encoder: Model, class_count: int) -> Model:
    for layer in encoder.layers:
        layer.trainable = False

    x = Dense(units=class_count, activation="softmax", name="output")(encoder.output)

    classifier = Model(encoder.input, x)
    return classifier


def _build_callbacks(output_folder: Path, model_name: str) -> List[Callback]:
    checkpoint_loss = ModelCheckpoint(
        str(output_folder / (model_name + "_best_loss.hdf5")),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    checkpoint_acc = ModelCheckpoint(
        str(output_folder / (model_name + "_best_acc.hdf5")),
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    plot_losses = PlotLosses(output_folder, model_name)
    csv_logger = CSVLogger(str(output_folder / (model_name + "_train_log.csv")))

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=PATIENCE,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    callbacks = [
        checkpoint_loss,
        checkpoint_acc,
        csv_logger,
        plot_losses,
        early_stopping,
    ]
    return callbacks
