"""
api/app.py
───────────
Flask REST API for the AI Vehicle Damage Intelligence Pipeline.

Endpoints:
  POST /predict              → Run damage classification on an image URL
  POST /predict/batch        → Batch classify multiple image URLs
  GET  /cases                → List all damage cases
  GET  /cases/<record_id>    → Get single case + report
  GET  /similar/<record_id>  → Find similar historical cases
  GET  /report/<record_id>   → Get RAG-generated report
  POST /report/generate      → Generate report for a case
  GET  /analytics/severity   → Severity distribution stats
  GET  /analytics/fraud      → Fraud flag stats
  GET  /health               → API health check
"""

import os
import sys
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.damage_classifier import DamageClassifier
from embeddings.vector_store import DamageVectorStore
from rag.report_generator import RAGReportGenerator
from ingestion.mongo_store import MongoStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Initialize Pipeline Components ────────────────────────────────────────────
logger.info("🚀 Initializing pipeline components...")

try:
    classifier = DamageClassifier()
    logger.info("✅ Classifier ready")
except Exception as e:
    logger.warning(f"⚠️  Classifier init failed (TensorFlow may not be installed): {e}")
    classifier = None

try:
    vector_store = DamageVectorStore()
    logger.info("✅ Vector store ready")
except Exception as e:
    logger.warning(f"⚠️  Vector store init failed: {e}")
    vector_store = None

try:
    report_generator = RAGReportGenerator(vector_store=vector_store)
    logger.info("✅ RAG report generator ready")
except Exception as e:
    logger.warning(f"⚠️  Report generator init failed: {e}")
    report_generator = None

mongo = MongoStore()
logger.info("✅ MongoDB store ready")


# ── Helper: Build Full Record ─────────────────────────────────────────────────
def build_record(image_url: str, prediction: dict, extra_meta: dict = None) -> dict:
    """Assemble a full pipeline record from prediction output."""
    fraud_score = 1.0 - prediction["confidence_score"] + 0.1
    if prediction["severity_class"] == "Severe" and prediction["confidence_score"] < 0.75:
        fraud_score = min(fraud_score + 0.25, 0.99)
    fraud_score = round(fraud_score, 4)
    fraud_flagged = fraud_score > 0.70

    return {
        "record_id": str(uuid.uuid4()),
        "claim_id": f"CLM-{uuid.uuid4().hex[:6].upper()}",
        "image_url": image_url,
        "vehicle": extra_meta.get("vehicle", {}) if extra_meta else {},
        "location": extra_meta.get("location", "Unknown") if extra_meta else "Unknown",
        "prediction": prediction,
        "fraud_analysis": {
            "fraud_score": fraud_score,
            "fraud_flagged": fraud_flagged,
            "flag_reason": "Low confidence + Severe claim mismatch" if fraud_flagged else None
        },
        "claim_status": "Flagged" if fraud_flagged else "Pending",
        "processing_status": "completed",
        "created_at": datetime.utcnow().isoformat(),
        "rag_report_generated": False
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "classifier": classifier is not None,
            "vector_store": vector_store is not None,
            "rag_generator": report_generator is not None,
            "mongodb": mongo.client is not None,
            "vector_count": vector_store.count() if vector_store else 0,
            "prediction_count": mongo.count_predictions()
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify damage severity from an image URL.
    Saves prediction to MongoDB + embeds in vector store.
    """
    data = request.get_json()
    if not data or "image_url" not in data:
        return jsonify({"error": "image_url is required"}), 400

    image_url = data["image_url"]
    extra_meta = {
        "vehicle": data.get("vehicle", {}),
        "location": data.get("location", "Unknown")
    }

    try:
        if classifier:
            prediction = classifier.predict_from_url(image_url)
        else:
            # Demo mode: return mock prediction
            import random
            classes = ["Minor", "Moderate", "Severe"]
            scores = [round(random.uniform(0.1, 0.9), 4) for _ in range(3)]
            total = sum(scores)
            scores = [round(s / total, 4) for s in scores]
            idx = scores.index(max(scores))
            prediction = {
                "severity_class": classes[idx],
                "confidence_score": scores[idx],
                "severity_scores": dict(zip(classes, scores)),
                "source": image_url,
                "model_version": "demo-mode"
            }

        record = build_record(image_url, prediction, extra_meta)

        # Save to MongoDB
        mongo.save_prediction(record)

        # Embed in vector store
        if vector_store:
            vector_store.upsert_record(record)

        return jsonify({
            "success": True,
            "record_id": record["record_id"],
            "claim_id": record["claim_id"],
            "prediction": record["prediction"],
            "fraud_analysis": record["fraud_analysis"],
            "claim_status": record["claim_status"]
        }), 201

    except ValueError as e:
        return jsonify({"error": str(e), "image_url": image_url}), 422
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal prediction error", "detail": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch classify multiple image URLs."""
    data = request.get_json()
    if not data or "images" not in data:
        return jsonify({"error": "images array is required"}), 400

    results = []
    errors = []

    for item in data["images"]:
        image_url = item.get("image_url")
        if not image_url:
            continue
        try:
            if classifier:
                prediction = classifier.predict_from_url(image_url)
            else:
                import random
                classes = ["Minor", "Moderate", "Severe"]
                s = sorted([random.random() for _ in range(3)], reverse=True)
                total = sum(s)
                s = [round(x / total, 4) for x in s]
                prediction = {
                    "severity_class": classes[0],
                    "confidence_score": s[0],
                    "severity_scores": dict(zip(classes, s)),
                    "model_version": "demo-mode"
                }

            record = build_record(image_url, prediction, item)
            mongo.save_prediction(record)
            if vector_store:
                vector_store.upsert_record(record)
            results.append({"record_id": record["record_id"], "prediction": prediction})
        except Exception as e:
            errors.append({"image_url": image_url, "error": str(e)})

    return jsonify({
        "processed": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors
    }), 200


@app.route("/cases", methods=["GET"])
def get_cases():
    """List all damage cases with optional filters."""
    limit = int(request.args.get("limit", 50))
    severity = request.args.get("severity")
    flagged = request.args.get("flagged")

    if severity:
        cases = mongo.get_predictions_by_severity(severity)
    elif flagged == "true":
        cases = mongo.get_flagged_predictions()
    else:
        cases = mongo.get_all_predictions(limit=limit)

    return jsonify({"count": len(cases), "cases": cases})


@app.route("/cases/<record_id>", methods=["GET"])
def get_case(record_id):
    """Get a single case by record_id."""
    case = mongo.get_prediction(record_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    report = mongo.get_report(record_id)
    return jsonify({"case": case, "report": report})


@app.route("/similar/<record_id>", methods=["GET"])
def get_similar(record_id):
    """Find top-K similar historical cases using vector search."""
    if not vector_store:
        return jsonify({"error": "Vector store not available"}), 503

    case = mongo.get_prediction(record_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    top_k = int(request.args.get("top_k", 3))
    similar = vector_store.similarity_search(case, top_k=top_k)
    fraud_analysis = vector_store.detect_fraud_outlier(case, top_k=5)

    return jsonify({
        "record_id": record_id,
        "similar_cases": similar,
        "fraud_outlier_analysis": fraud_analysis
    })


@app.route("/report/generate", methods=["POST"])
def generate_report():
    """Generate a RAG-powered inspection report for a case."""
    if not report_generator:
        return jsonify({"error": "Report generator not available"}), 503

    data = request.get_json()
    record_id = data.get("record_id")
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400

    case = mongo.get_prediction(record_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    report = report_generator.generate_report(case)
    mongo.save_report(report)

    # Mark as report generated
    if mongo.db is not None:
        mongo.db.damage_predictions.update_one(
            {"record_id": record_id},
            {"$set": {"rag_report_generated": True}}
        )

    return jsonify({"success": True, "report": report}), 201


@app.route("/report/<record_id>", methods=["GET"])
def get_report(record_id):
    """Retrieve a previously generated report."""
    report = mongo.get_report(record_id)
    if not report:
        return jsonify({"error": "Report not found. Generate one first via POST /report/generate"}), 404
    return jsonify({"report": report})


@app.route("/analytics/severity", methods=["GET"])
def severity_analytics():
    """Get severity distribution stats."""
    distribution = mongo.get_severity_distribution()
    total = sum(distribution.values())
    return jsonify({
        "distribution": distribution,
        "total": total,
        "percentages": {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in distribution.items()
        }
    })


@app.route("/analytics/fraud", methods=["GET"])
def fraud_analytics():
    """Get fraud flag stats."""
    return jsonify(mongo.get_fraud_stats())


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    logger.info(f"🚀 Starting Vehicle Damage API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
