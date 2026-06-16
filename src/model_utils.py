import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE


def build_transfer_model(num_classes: int, model_name: str = "resnet50", dropout: float = 0.4):
    if model_name.lower() == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input

    base.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Lambda(preprocess)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


def compile_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model
