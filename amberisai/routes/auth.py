import json
from flask import Blueprint, request, jsonify
from models.database import (
    register_user, login_user, get_user,
    create_baby, get_baby, get_babies_by_user, update_baby,
    get_sessions_by_baby, get_sessions_by_user
)

auth_bp = Blueprint('auth', __name__)


# ── REGISTER ──────────────────────────────────────────────────────────────────
@auth_bp.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get('email') or '').strip()
    password = (data.get('password') or '').strip()
    full_name = (data.get('full_name') or '').strip()
    phone = (data.get('phone') or '').strip()

    if not email or not password or not full_name:
        return jsonify({
            "success": False,
            "error": "email, password and full_name are required"
        }), 400
    if len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must be at least 6 characters"
        }), 400
    if '@' not in email:
        return jsonify({
            "success": False,
            "error": "Invalid email format"
        }), 400

    result = register_user(email, password, full_name, phone)
    if not result["success"]:
        return jsonify(result), 409

    return jsonify({
        "success": True,
        "user_id": result["user_id"],
        "message": "Account created!"
    }), 201


# ── LOGIN ─────────────────────────────────────────────────────────────────────
@auth_bp.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get('email') or '').strip()
    password = (data.get('password') or '').strip()

    if not email or not password:
        return jsonify({
            "success": False,
            "error": "Email and password required"
        }), 400

    result = login_user(email, password)
    if not result["success"]:
        return jsonify(result), 401

    # Also return all babies for this user
    babies = get_babies_by_user(result["user_id"])
    return jsonify({
        "success": True,
        "user_id": result["user_id"],
        "full_name": result["full_name"],
        "email": result["email"],
        "babies": babies
    }), 200


# ── GET USER ──────────────────────────────────────────────────────────────────
@auth_bp.route('/auth/me/<int:user_id>', methods=['GET'])
def me(user_id):
    user = get_user(user_id)
    if not user:
        return jsonify({
            "success": False,
            "error": "User not found"
        }), 404

    babies = get_babies_by_user(user_id)
    return jsonify({
        "success": True,
        "user": user,
        "babies": babies
    }), 200


# ── CREATE BABY ───────────────────────────────────────────────────────────────
@auth_bp.route('/profile', methods=['POST'])
def create_profile():
    data = request.get_json(force=True, silent=True) or {}

    nickname = (data.get('nickname') or '').strip()
    age_days = data.get('age_days', 0)
    allergies = data.get('allergies', [])
    user_id = data.get('user_id')
    date_of_birth = data.get('date_of_birth')
    gender = data.get('gender')
    weight_kg = data.get('weight_kg')
    blood_group = data.get('blood_group')
    medical_notes = data.get('medical_notes', '')

    if not nickname:
        return jsonify({
            "success": False,
            "error": "nickname is required"
        }), 400

    baby_id = create_baby(
        nickname=nickname,
        age_days=age_days,
        allergies=allergies,
        user_id=user_id,
        date_of_birth=date_of_birth,
        gender=gender,
        weight_kg=weight_kg,
        blood_group=blood_group,
        medical_notes=medical_notes
    )
    return jsonify({
        "success": True,
        "baby_id": baby_id
    }), 201


# ── GET BABY ──────────────────────────────────────────────────────────────────
@auth_bp.route('/profile/<int:baby_id>', methods=['GET'])
def get_profile(baby_id):
    baby = get_baby(baby_id)
    if not baby:
        return jsonify({
            "success": False,
            "error": "Baby not found"
        }), 404

    return jsonify({
        "success": True,
        "baby": baby
    }), 200


# ── GET ALL BABIES FOR USER ───────────────────────────────────────────────────
@auth_bp.route('/profile/user/<int:user_id>', methods=['GET'])
def get_user_babies(user_id):
    babies = get_babies_by_user(user_id)
    return jsonify({
        "success": True,
        "babies": babies,
        "count": len(babies)
    }), 200


# ── UPDATE BABY ───────────────────────────────────────────────────────────────
@auth_bp.route('/profile/<int:baby_id>', methods=['PUT'])
def update_profile(baby_id):
    data = request.get_json(force=True, silent=True) or {}
    allowed = [
        'nickname', 'age_days', 'allergies', 'date_of_birth',
        'gender', 'weight_kg', 'blood_group', 'medical_notes'
    ]
    updates = {k: v for k, v in data.items() if k in allowed}

    if not updates:
        return jsonify({
            "success": False,
            "error": "No valid fields to update"
        }), 400

    update_baby(baby_id, **updates)
    return jsonify({
        "success": True,
        "message": "Profile updated"
    }), 200


# ── GET BABY SESSION HISTORY ──────────────────────────────────────────────────
@auth_bp.route('/history/baby/<int:baby_id>', methods=['GET'])
def baby_history(baby_id):
    limit = int(request.args.get('limit', 20))
    sessions = get_sessions_by_baby(baby_id, limit=limit)
    return jsonify({
        "success": True,
        "sessions": sessions,
        "count": len(sessions)
    }), 200


# ── GET USER SESSION HISTORY ──────────────────────────────────────────────────
@auth_bp.route('/history/user/<int:user_id>', methods=['GET'])
def user_history(user_id):
    limit = int(request.args.get('limit', 50))
    sessions = get_sessions_by_user(user_id, limit=limit)
    return jsonify({
        "success": True,
        "sessions": sessions,
        "count": len(sessions)
    }), 200


# ── GOOGLE / FIREBASE AUTH ────────────────────────────────────────────────────
@auth_bp.route('/auth/google', methods=['POST'])
def google_auth():
    import os
    import firebase_admin
    from firebase_admin import credentials, auth as fb_auth

    data = request.get_json(force=True, silent=True) or {}
    id_token = (data.get('id_token') or '').strip()

    if not id_token:
        return jsonify({
            "success": False,
            "error": "Missing id_token"
        }), 400

    # Initialize Firebase Admin SDK once
    if not firebase_admin._apps:
        sa_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT')
        if not sa_json:
            return jsonify({
                "success": False,
                "error": "Firebase not configured on server"
            }), 500
        try:
            cred = credentials.Certificate(json.loads(sa_json))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Firebase init failed: {str(e)}"
            }), 500

    # Verify the Firebase ID token
    try:
        decoded = fb_auth.verify_id_token(id_token)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Token verification failed: {str(e)}"
        }), 401

    email = decoded.get('email', '').lower().strip()
    full_name = decoded.get('name', email.split('@')[0])
    picture = decoded.get('picture', '')
    google_id = decoded.get('uid', '')

    if not email:
        return jsonify({
            "success": False,
            "error": "Could not get email from token"
        }), 400

    # Check if user already exists
    from models.database import get_user_by_email, register_google_user
    existing = get_user_by_email(email)

    if existing:
        babies = get_babies_by_user(existing['id'])
        return jsonify({
            "success": True,
            "user_id": existing['id'],
            "full_name": existing['full_name'],
            "email": existing['email'],
            "picture": picture,
            "is_new_user": False,
            "babies": babies
        }), 200

    result = register_google_user(email, full_name, google_id, picture)
    if not result["success"]:
        return jsonify(result), 500

    return jsonify({
        "success": True,
        "user_id": result["user_id"],
        "full_name": full_name,
        "email": email,
        "picture": picture,
        "is_new_user": True,
        "babies": []
    }), 201
