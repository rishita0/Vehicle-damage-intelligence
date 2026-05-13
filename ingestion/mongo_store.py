"""
ingestion/mongo_store.py
─────────────────────────
MongoDB storage layer for:
- Raw prediction records
- RAG-generated reports
- Fraud analysis results
- Pipeline audit logs

Collections:
  damage_predictions   → Full prediction output per image
  damage_reports       → RAG-generated inspection reports
  audit_logs           → Pipeline run metadata
"""

import os
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "vehicle_damage_db")


class MongoStore:
    """
    MongoDB client wrapper for the vehicle damage pipeline.
    Handles predictions, reports, and audit logging.
    """

    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")  # Test connection
            self.db = self.client[MONGO_DB]
            self._setup_indexes()
            logger.info(f"✅ MongoDB connected → {MONGO_DB}")
        except ConnectionFailure:
            logger.warning("⚠️  MongoDB not available — using in-memory fallback")
            self.client = None
            self.db = None
            self._fallback_store = {"predictions": [], "reports": [], "audit_logs": []}

    def _setup_indexes(self):
        """Create indexes for fast querying."""
        if self.db is None:
            return
        self.db.damage_predictions.create_index("record_id", unique=True)
        self.db.damage_predictions.create_index("claim_id")
        self.db.damage_predictions.create_index([("created_at", DESCENDING)])
        self.db.damage_predictions.create_index("prediction.severity_class")
        self.db.damage_predictions.create_index("fraud_analysis.fraud_flagged")
        self.db.damage_reports.create_index("record_id")
        self.db.damage_reports.create_index([("generated_at", DESCENDING)])

    # ── Predictions ────────────────────────────────────────────────────────────
    def save_prediction(self, record: dict) -> str:
        """Save a damage prediction record. Returns record_id."""
        record["saved_at"] = datetime.utcnow().isoformat()

        if self.db is not None:
            try:
                self.db.damage_predictions.insert_one(record)
            except DuplicateKeyError:
                self.db.damage_predictions.replace_one(
                    {"record_id": record["record_id"]}, record
                )
        else:
            self._fallback_store["predictions"].append(record)

        logger.info(f"💾 Saved prediction: {record.get('record_id')}")
        return record.get("record_id")

    def get_prediction(self, record_id: str) -> Optional[dict]:
        """Retrieve a prediction by record_id."""
        if self.db is not None:
            result = self.db.damage_predictions.find_one(
                {"record_id": record_id}, {"_id": 0}
            )
            return result
        return next(
            (r for r in self._fallback_store["predictions"] if r["record_id"] == record_id),
            None
        )

    def get_all_predictions(self, limit: int = 100, skip: int = 0) -> list:
        """Retrieve recent predictions for dashboard."""
        if self.db is not None:
            return list(
                self.db.damage_predictions
                .find({}, {"_id": 0})
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
        return self._fallback_store["predictions"][-limit:]

    def get_predictions_by_severity(self, severity: str) -> list:
        """Filter predictions by severity class."""
        if self.db is not None:
            return list(
                self.db.damage_predictions.find(
                    {"prediction.severity_class": severity}, {"_id": 0}
                )
            )
        return [
            r for r in self._fallback_store["predictions"]
            if r.get("prediction", {}).get("severity_class") == severity
        ]

    def get_flagged_predictions(self) -> list:
        """Get all fraud-flagged predictions."""
        if self.db is not None:
            return list(
                self.db.damage_predictions.find(
                    {"fraud_analysis.fraud_flagged": True}, {"_id": 0}
                )
            )
        return [
            r for r in self._fallback_store["predictions"]
            if r.get("fraud_analysis", {}).get("fraud_flagged")
        ]

    # ── Reports ────────────────────────────────────────────────────────────────
    def save_report(self, report: dict) -> str:
        """Save a RAG-generated damage report."""
        report["saved_at"] = datetime.utcnow().isoformat()

        if self.db is not None:
            self.db.damage_reports.insert_one(report)
        else:
            self._fallback_store["reports"].append(report)

        logger.info(f"📄 Saved report for: {report.get('record_id')}")
        return report.get("record_id")

    def get_report(self, record_id: str) -> Optional[dict]:
        """Retrieve a report by record_id."""
        if self.db is not None:
            return self.db.damage_reports.find_one({"record_id": record_id}, {"_id": 0})
        return next(
            (r for r in self._fallback_store["reports"] if r["record_id"] == record_id),
            None
        )

    # ── Analytics ──────────────────────────────────────────────────────────────
    def get_severity_distribution(self) -> dict:
        """Get count of each severity class."""
        if self.db is not None:
            pipeline = [
                {"$group": {"_id": "$prediction.severity_class", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            results = list(self.db.damage_predictions.aggregate(pipeline))
            return {r["_id"]: r["count"] for r in results if r["_id"]}

        records = self._fallback_store["predictions"]
        dist = {"Minor": 0, "Moderate": 0, "Severe": 0}
        for r in records:
            s = r.get("prediction", {}).get("severity_class")
            if s in dist:
                dist[s] += 1
        return dist

    def get_fraud_stats(self) -> dict:
        """Get fraud flag statistics."""
        if self.db is not None:
            total = self.db.damage_predictions.count_documents({})
            flagged = self.db.damage_predictions.count_documents(
                {"fraud_analysis.fraud_flagged": True}
            )
            return {
                "total_cases": total,
                "flagged_cases": flagged,
                "flag_rate": round(flagged / total * 100, 1) if total > 0 else 0
            }

        records = self._fallback_store["predictions"]
        flagged = sum(1 for r in records if r.get("fraud_analysis", {}).get("fraud_flagged"))
        return {
            "total_cases": len(records),
            "flagged_cases": flagged,
            "flag_rate": round(flagged / len(records) * 100, 1) if records else 0
        }

    def count_predictions(self) -> int:
        if self.db is not None:
            return self.db.damage_predictions.count_documents({})
        return len(self._fallback_store["predictions"])

    # ── Audit Logging ──────────────────────────────────────────────────────────
    def log_pipeline_run(self, run_meta: dict):
        run_meta["logged_at"] = datetime.utcnow().isoformat()
        if self.db is not None:
            self.db.audit_logs.insert_one(run_meta)
        else:
            self._fallback_store["audit_logs"].append(run_meta)


# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = MongoStore()
    print(f"Total predictions stored: {store.count_predictions()}")
    print(f"Severity distribution: {store.get_severity_distribution()}")
    print(f"Fraud stats: {store.get_fraud_stats()}")
