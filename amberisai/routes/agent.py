
"""
routes/agent.py
===============
POST /agent — AmberisAI's Nurse Amber AI agent.
Works with both Groq (gsk_) and DeepSeek (sk-) keys.
"""

import json
import os
import logging
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context

from agent.deepseek_client import stream_deepseek, validate_api_key
from agent.memory import build_full_context, add_to_chat_history, check_rate_limit
from agent.tools import classify_query, safe_escalation, build_messages_for_deepseek, format_sse_event

agent_bp = Blueprint('agent', __name__)
logger = logging.getLogger(__name__)


def _error_sse(message: str):
    payload = json.dumps({"error": message})
    return Response(
        f"data: {payload}\n\ndata: [DONE]\n\n",
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


@agent_bp.route('/agent', methods=['POST', 'OPTIONS'])
def agent():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return Response(status=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS"
        })

    # ── 1. PARSE INPUT ────────────────────────────────────────────────────────
    try:
        data = request.get_json(force=True, silent=True)
    except Exception:
        return _error_sse("Invalid JSON body")

    if not data:
        return _error_sse("Empty request body")

    query      = (data.get("query") or "").strip()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    session_id = data.get("session_id")
    baby_id    = data.get("baby_id")

    # ── 2. VALIDATE ───────────────────────────────────────────────────────────
    if not query:
        return _error_sse("Missing 'query' field")

    if not api_key:
        return _error_sse("GROQ_API_KEY not configured on server")

    # ── 3. RATE LIMITING ──────────────────────────────────────────────────────
    rate_key = str(baby_id) if baby_id else session_id or "anonymous"
    if not check_rate_limit(rate_key):
        return _error_sse("Rate limit exceeded (10 requests/minute). Please wait.")

    # ── 4. BUILD CONTEXT ──────────────────────────────────────────────────────
    try:
        context = build_full_context(
            session_id=session_id,
            baby_id=int(baby_id) if baby_id else None,
            query=query
        )
    except Exception as e:
        logger.error(f"[Agent] Context build failed: {e}")
        context = {"baby_profile": None, "latest_session": None,
                   "trends": None, "chat_history": [], "query": query}

    # ── 5. SAFETY CHECK ───────────────────────────────────────────────────────
    escalation_msg = safe_escalation(query, context)
    if escalation_msg:
        def emergency_stream():
            yield format_sse_event(json.dumps({"text": escalation_msg}))
            yield format_sse_event("[DONE]")
        return Response(
            stream_with_context(emergency_stream()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Access-Control-Allow-Origin": "*"}
        )

    # ── 6. CLASSIFY + BUILD MESSAGES ─────────────────────────────────────────
    query_type = classify_query(query)
    logger.info(f"[Agent] Type: {query_type} | Query: {query[:60]}")
    messages = build_messages_for_deepseek(context, query)

    # ── 7. STREAM RESPONSE ────────────────────────────────────────────────────
    def generate():
        full_response = []

        try:
            # Context hint if we have audio analysis
            session = context.get("latest_session")
            if session and session.get("audio_json"):
                audio = session["audio_json"]
                cond = audio.get("detected_condition") or audio.get("primary_condition", "")
                conf = audio.get("confidence", 0)
                if cond and conf:
                    hint = f"[Based on {cond.upper()} detection at {conf*100:.0f}% confidence] "
                    yield format_sse_event(json.dumps({"text": hint, "type": "context_hint"}))
                    full_response.append(hint)

            # Stream from Groq or DeepSeek — auto-detected from key
            for chunk in stream_deepseek(
                messages=messages,
                api_key=api_key,
                model=None,   # auto-detect model from key
                max_tokens=2048,
                temperature=0.65
            ):
                full_response.append(chunk)
                yield format_sse_event(json.dumps({"text": chunk}))

            # Disclaimer
            disclaimer = "\n\n*Not medical advice — consult your pediatrician for serious concerns.*"
            yield format_sse_event(json.dumps({"text": disclaimer}))
            full_response.append(disclaimer)

            # Save chat history
            full_text = "".join(full_response)
            add_to_chat_history(session_id or "default", "user", query)
            add_to_chat_history(session_id or "default", "assistant", full_text)

            yield format_sse_event("[DONE]")

        except Exception as e:
            logger.error(f"[Agent] Stream error: {e}\n{traceback.format_exc()}")
            err_msg = str(e)
            if "authentication" in err_msg.lower() or "401" in err_msg:
                yield format_sse_event(json.dumps({"error": "Invalid API key. Check your Groq/DeepSeek key."}))
            elif "402" in err_msg or "balance" in err_msg.lower():
                yield format_sse_event(json.dumps({"error": "Insufficient balance. Use a Groq key (free) from console.groq.com"}))
            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                yield format_sse_event(json.dumps({"error": "Rate limit hit. Wait a moment and try again."}))
            elif "404" in err_msg or "does not exist" in err_msg.lower():
                yield format_sse_event(json.dumps({"error": "Model not found. Make sure you are using a Groq key (gsk_) or valid DeepSeek key."}))
            else:
                yield format_sse_event(json.dumps({"error": f"Agent error: {err_msg[:120]}"}))
            yield format_sse_event("[DONE]")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )


@agent_bp.route('/agent/history', methods=['GET'])
def get_history():
    from agent.memory import get_chat_history
    session_id = request.args.get("session_id", "default")
    history = get_chat_history(session_id)
    return jsonify({"success": True, "history": history, "count": len(history)})


@agent_bp.route('/agent/context', methods=['GET'])
def get_context():
    session_id = request.args.get("session_id")
    baby_id = request.args.get("baby_id")
    context = build_full_context(
        session_id=session_id,
        baby_id=int(baby_id) if baby_id else None,
        query=""
    )
    return jsonify({"success": True, "context": context})
