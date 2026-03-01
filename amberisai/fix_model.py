"""
run_image_predict.py
====================
Standalone script that runs image prediction.
Called as a subprocess by Flask.
Usage: python run_image_predict.py <image_path>
Output: JSON to stdout
"""
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict(image_path):
    # ── PATH RESOLUTION ───────────────────────────────────────────────────────
    # This script lives in amberisai/ (Render root)
    # image_model_updated is at amberisai/image_model_updated/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try same-level first (Render: amberisai/image_model_updated)
    image_dir = os.path.join(script_dir, 'image_model_updated')

    # Fallback: one level up (local dev: repo_root/image_model_updated)
    if not os.path.exists(image_dir):
        image_dir = os.path.abspath(os.path.join(script_dir, '..', 'image_model_updated'))

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"image_model_updated directory not found. Tried: {image_dir}")

    # ── LOAD MODEL ────────────────────────────────────────────────────────────
    # Use tf_keras for TF2-compatible Teachable Machine h5 models
    # Avoids conflicts with standalone Keras 3.x
    try:
        from tf_keras.models import load_model
    except ImportError:
        # Fallback to tensorflow.keras if tf_keras not installed
        from tensorflow.keras.models import load_model

    from PIL import Image, ImageOps
    import numpy as np

    model = None
    for fname in ['keras_model.h5', 'keras_Model.h5']:
        model_path = os.path.join(image_dir, fname)
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            break

    if model is None:
        raise FileNotFoundError(f"No keras model .h5 file found in {image_dir}")

    labels_path = os.path.join(image_dir, 'labels.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.txt not found in {image_dir}")

    with open(labels_path, 'r') as f:
        class_names = f.readlines()

    # ── PREPROCESS IMAGE ──────────────────────────────────────────────────────
    np.set_printoptions(suppress=True)

    image      = Image.open(image_path).convert("RGB")
    image      = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized  = (image_array.astype(np.float32) / 127.5) - 1
    data        = np.expand_dims(normalized, axis=0)  # shape (1, 224, 224, 3)

    # ── PREDICT ───────────────────────────────────────────────────────────────
    prediction       = model.predict(data, verbose=0)
    index            = int(np.argmax(prediction))
    confidence_score = float(prediction[0][index])

    def clean(raw):
        raw = raw.strip()
        if len(raw) > 2 and raw[1] == ' ' and raw[0].isdigit():
            return raw[2:]
        return raw

    detected  = clean(class_names[index])
    all_probs = {
        clean(cn): round(float(prediction[0][i]), 4)
        for i, cn in enumerate(class_names)
    }

    result = {
        "module": "image",
        "detected_condition": detected,
        "confidence": round(confidence_score, 4),
        "all_probabilities": all_probs,
        "low_confidence_warning": confidence_score < 0.50,
        "meta": {
            "model":       "keras_teachable_machine",
            "image_size":  "224x224",
            "num_classes": len(class_names)
        }
    }

    print(json.dumps(result))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    try:
        predict(sys.argv[1])
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
