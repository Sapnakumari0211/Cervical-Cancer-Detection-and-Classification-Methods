import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import build_transfer_model, compile_model
from config import IMG_SIZE, BATCH_SIZE, SEED


def main(data_dir, epochs, model_name):
    data_dir = Path(data_dir) / "classification"
    train_gen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, brightness_range=[0.8, 1.2])
    plain_gen = ImageDataGenerator()

    train = train_gen.flow_from_directory(data_dir / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED)
    val = plain_gen.flow_from_directory(data_dir / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED)
    test = plain_gen.flow_from_directory(data_dir / "test", target_size=IMG_SIZE, batch_size=1, class_mode="categorical", shuffle=False)

    model = build_transfer_model(num_classes=5, model_name=model_name)
    compile_model(model)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("../models/classification_model.h5", monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)

    predictions = np.argmax(model.predict(test), axis=1)
    y_true = test.classes
    labels = list(test.class_indices.keys())
    print("Classification report")
    print(classification_report(y_true, predictions, target_names=labels))
    print("Confusion matrix")
    print(confusion_matrix(y_true, predictions))
    model.save("../models/classification_model_final.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Processed dataset folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model_name", default="resnet50", choices=["resnet50", "efficientnetb0"])
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.model_name)
