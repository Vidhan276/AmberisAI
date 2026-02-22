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

def predict(image_path):
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'image_model_updated')
    image_dir = os.path.abspath(image_dir)
    os.chdir(image_dir)

    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from PIL import Image, ImageOps
    import numpy as np

    # Model is already patched by fix_model.py — load directly
    for fname in ['keras_model.h5', 'keras_Model.h5']:
        if os.path.exists(fname):
            model = load_model(fname, compile=False)
            break
    else:
        raise FileNotFoundError("No keras model h5 file found in image_model_updated/")

    class_names = open('labels.txt', 'r').readlines()

    np.set_printoptions(suppress=True)

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)  # shape (1, 224, 224, 3)

    prediction = model.predict(data, verbose=0)
    index = int(np.argmax(prediction))
    confidence_score = float(prediction[0][index])

    def clean(raw):
        raw = raw.strip()
        if len(raw) > 2 and raw[1] == ' ' and raw[0].isdigit():
            return raw[2:]
        return raw

    detected = clean(class_names[index])
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
            "model": "keras_teachable_machine",
            "image_size": "224x224",
            "num_classes": len(class_names)
        }
    }

    print(json.dumps(result))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    predict(sys.argv[1])
