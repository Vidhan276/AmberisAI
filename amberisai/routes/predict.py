import os
import sys
import uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

from utils.file_handler import save_upload
from utils.visual_generator import generate_visual
from models.database import save_session

predict_bp = Blueprint('predict', __name__)

# ── PATH RESOLUTION (Render root = amberisai/) ────────────────────────────────
_ROUTES_DIR    = os.path.dirname(os.path.abspath(__file__))  # amberisai/routes
_AMBERISAI_DIR = os.path.dirname(_ROUTES_DIR)                # amberisai  ← deployment root

AUDIO_MODULE_DIR = os.path.join(_AMBERISAI_DIR, 'audio_module')
IMAGE_MODULE_DIR = os.path.join(_AMBERISAI_DIR, 'image_model_updated')

if _AMBERISAI_DIR not in sys.path:
    sys.path.insert(0, _AMBERISAI_DIR)


# ── AUDIO ─────────────────────────────────────────────────────────────────────
_audio_predictor = None

def get_audio_predictor():
    global _audio_predictor
    if _audio_predictor is None:
        from audio_module.audio_predictor import AudioPredictor
        model_dir = os.path.join(AUDIO_MODULE_DIR, 'models')
        _audio_predictor = AudioPredictor.get_instance(model_dir=model_dir)
    return _audio_predictor


# ── IMAGE ─────────────────────────────────────────────────────────────────────
_image_model       = None
_image_class_names = None

def get_image_model():
    global _image_model, _image_class_names
    if _image_model is None:
        from tf_keras.models import load_model

        for fname in ['keras_model.h5', 'keras_Model.h5']:
            model_path = os.path.join(IMAGE_MODULE_DIR, fname)
            if os.path.exists(model_path):
                break
        else:
            raise FileNotFoundError(f"keras_model.h5 not found in {IMAGE_MODULE_DIR}")

        labels_path = os.path.join(IMAGE_MODULE_DIR, 'labels.txt')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.txt not found in {IMAGE_MODULE_DIR}")

        _image_model = load_model(model_path, compile=False)

        with open(labels_path, 'r') as f:
            _image_class_names = f.readlines()

    return _image_model, _image_class_names


def _clean_label(raw: str) -> str:
    raw = raw.strip()
    if len(raw) > 2 and raw[1] == ' ' and raw[0].isdigit():
        return raw[2:]
    return raw


def predict_image_file(image_path: str) -> dict:
    from PIL import Image, ImageOps
    import numpy as np

    model, class_names = get_image_model()

    data  = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized  = (image_array.astype(np.float32) / 127.5) - 1
    data[0]     = normalized

    prediction = model.predict(data, verbose=0)
    index      = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    detected   = _clean_label(class_names[index])

    all_probs = {
        _clean_label(cn): round(float(prediction[0][i]), 4)
        for i, cn in enumerate(class_names)
    }

    return {
        "module": "image",
        "detected_condition": detected,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "low_confidence_warning": confidence < 0.50,
        "meta": {
            "model": "keras_teachable_machine",
            "image_size": "224x224",
            "num_classes": len(class_names)
        }
    }


# ── POST /predict-audio ───────────────────────────────────────────────────────
@predict_bp.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No audio file. Use key 'file'."}), 400

    file    = request.files['file']
    baby_id = request.form.get('baby_id', None)
    if baby_id:
        try:
            baby_id = int(baby_id)
        except ValueError:
            baby_id = None

    file_path, err = save_upload(file, file_type='audio')
    if err:
        return jsonify({"success": False, "error": err}), 400

    try:
        predictor = get_audio_predictor()
        analysis  = predictor.predict(file_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Audio prediction failed: {str(e)}"}), 500

    session_id        = f"sess_{uuid.uuid4().hex[:10]}"
    primary_condition = analysis.get('detected_condition', 'unknown')
    viz_url           = generate_visual(file_path, session_id, primary_condition, analysis)
    timestamp         = datetime.now(timezone.utc).isoformat()

    save_session(session_id=session_id, baby_id=baby_id, timestamp=timestamp,
                 audio_json=analysis, visualization_url=viz_url)

    return jsonify({
        "success":           True,
        "session_id":        session_id,
        "baby_id":           baby_id,
        "audio_analysis":    analysis,
        "visualization_url": viz_url,
        "timestamp":         timestamp
    }), 200


# ── POST /predict-image ───────────────────────────────────────────────────────
@predict_bp.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No image file. Use key 'file'."}), 400

    file    = request.files['file']
    baby_id = request.form.get('baby_id', None)
    if baby_id:
        try:
            baby_id = int(baby_id)
        except ValueError:
            baby_id = None

    file_path, err = save_upload(file, file_type='image')
    if err:
        return jsonify({"success": False, "error": err}), 400

    try:
        analysis = predict_image_file(file_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Image prediction failed: {str(e)}"}), 500

    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    timestamp  = datetime.now(timezone.utc).isoformat()

    save_session(session_id=session_id, baby_id=baby_id, timestamp=timestamp,
                 image_json=analysis)

    return jsonify({
        "success":        True,
        "session_id":     session_id,
        "baby_id":        baby_id,
        "image_analysis": analysis,
        "timestamp":      timestamp
    }), 200
