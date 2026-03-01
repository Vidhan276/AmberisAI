from flask import Flask, jsonify
from flask_cors import CORS
from routes.predict import predict_bp
from routes.auth import auth_bp
from routes.agent import agent_bp
from models.database import init_db
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/visuals'
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/visuals', exist_ok=True)
os.makedirs('data', exist_ok=True)
init_db()
app.register_blueprint(predict_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(agent_bp)

# ── ADD THESE TWO ROUTES ──────────────────────────────────────
@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/routes-debug")
def routes_debug():
    """Lists every registered route — helps find the correct URL"""
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify(routes), 200

@app.route("/debug-models")
def debug_models():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    
    audio_model_dir = os.path.join(base, "audio_module", "models")
    db_model_dir    = os.path.join(base, "models")
    
    return {
        "audio_module_models": {
            "path":   audio_model_dir,
            "exists": os.path.exists(audio_model_dir),
            "files":  os.listdir(audio_model_dir) if os.path.exists(audio_model_dir) else "NOT FOUND"
        },
        "db_models": {
            "path":   db_model_dir,
            "exists": os.path.exists(db_model_dir),
            "files":  os.listdir(db_model_dir) if os.path.exists(db_model_dir) else "NOT FOUND"
        }
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
