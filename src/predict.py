import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.config import IMG_SIZE


def load_img_array(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_model_safe(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": preprocess_input,
            "function": preprocess_input
        },
        compile=False
    )


def predict_image(image_path, detection_model, classification_model, class_labels_json=None):
    arr = load_img_array(image_path)

    det_model = load_model_safe(detection_model)
    cls_model = load_model_safe(classification_model)

    det_labels = ["Abnormal", "Normal"]
    det_pred = det_model.predict(arr, verbose=0)[0]
    detection_label = det_labels[int(np.argmax(det_pred))]

    if class_labels_json and Path(class_labels_json).exists():
        labels = json.loads(Path(class_labels_json).read_text())
    else:
        labels = [
            "Dyskeratotic",
            "Koilocytotic",
            "Metaplastic",
            "Parabasal",
            "Superficial-Intermediate"
        ]

    cls_pred = cls_model.predict(arr, verbose=0)[0]
    class_label = labels[int(np.argmax(cls_pred))]

    return {
        "detection_result": detection_label,
        "detection_confidence": float(np.max(det_pred)),
        "classification_result": class_label,
        "classification_confidence": float(np.max(cls_pred)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--detection_model", default="../models/detection_model_final.h5")
    parser.add_argument("--classification_model", default="../models/classification_model_final.h5")
    parser.add_argument("--class_labels_json", default=None)

    args = parser.parse_args()

    result = predict_image(
        args.image,
        args.detection_model,
        args.classification_model,
        args.class_labels_json
    )

    print(json.dumps(result, indent=2))