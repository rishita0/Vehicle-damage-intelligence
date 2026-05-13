"""
models/damage_classifier.py
────────────────────────────
CNN-based vehicle damage severity classifier using Transfer Learning.
Uses MobileNetV2 as base (lightweight, fast, deployable).

Classes: Minor | Moderate | Severe
Input:   224x224 RGB image
Output:  severity class + confidence scores per class
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Class Labels ──────────────────────────────────────────────────────────────
CLASS_LABELS = ["Minor", "Moderate", "Severe"]
IMAGE_SIZE = (224, 224)


# ── Model Builder (TensorFlow / Keras) ────────────────────────────────────────
def build_model(num_classes: int = 3):
    """
    Build a MobileNetV2-based transfer learning classifier.
    Fine-tuned for vehicle damage severity classification.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base for transfer learning

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("✅ MobileNetV2 model built successfully")
    return model


def save_model(model, path: str = "./models/damage_classifier.h5"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    logger.info(f"✅ Model saved → {path}")


def load_model(path: str = "./models/damage_classifier.h5"):
    import tensorflow as tf
    if not Path(path).exists():
        logger.warning(f"⚠️  No saved model at {path}. Building fresh model.")
        return build_model()
    model = tf.keras.models.load_model(path)
    logger.info(f"✅ Model loaded from {path}")
    return model


# ── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_from_url(image_url: str) -> np.ndarray:
    """
    Download image from URL, validate, convert to RGB,
    resize to 224x224, normalize to [0,1].
    Returns preprocessed numpy array ready for inference.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        # Validate it's an actual image
        img.verify()
        img = Image.open(BytesIO(response.content))  # Re-open after verify

        # Convert to RGB (handles RGBA, grayscale, etc.)
        img = img.convert("RGB")

        # Resize
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

        # Normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)

        return arr

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch image from URL: {e}")
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


def preprocess_from_file(image_path: str) -> np.ndarray:
    """Preprocess a local image file."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Inference Engine ──────────────────────────────────────────────────────────
class DamageClassifier:
    """
    Main inference engine.
    Loads model once, runs predictions, returns structured results.
    """

    def __init__(self, model_path: str = "./models/damage_classifier.h5"):
        self.model = load_model(model_path)
        self.class_labels = CLASS_LABELS
        logger.info("🔧 DamageClassifier initialized")

    def predict_from_url(self, image_url: str) -> dict:
        """Run full inference pipeline from image URL."""
        image_array = preprocess_from_url(image_url)
        return self._run_inference(image_array, source=image_url)

    def predict_from_file(self, image_path: str) -> dict:
        """Run full inference pipeline from local file."""
        image_array = preprocess_from_file(image_path)
        return self._run_inference(image_array, source=image_path)

    def _run_inference(self, image_array: np.ndarray, source: str) -> dict:
        """Core inference + result structuring."""
        scores = self.model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(scores))
        severity_class = self.class_labels[predicted_index]
        confidence = float(scores[predicted_index])

        result = {
            "severity_class": severity_class,
            "confidence_score": round(confidence, 4),
            "severity_scores": {
                label: round(float(score), 4)
                for label, score in zip(self.class_labels, scores)
            },
            "source": source,
            "model_version": "MobileNetV2-v1.0"
        }

        logger.info(f"Prediction: {severity_class} ({confidence:.2%}) — {source}")
        return result


# ── Demo / Standalone Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building and testing damage classifier...")
    model = build_model()
    print(model.summary())

    # Simulate a prediction with random input
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input)
    print(f"\nTest prediction scores: {dict(zip(CLASS_LABELS, output[0].round(4)))}")
    print(f"Predicted class: {CLASS_LABELS[np.argmax(output)]}")
