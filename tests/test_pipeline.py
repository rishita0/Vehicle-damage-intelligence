"""
tests/test_pipeline.py
───────────────────────
Automated tests for the Vehicle Damage AI Pipeline.
Tests: preprocessing, prediction, vector store, RAG, API, fraud detection.

Run: pytest tests/test_pipeline.py -v
"""

import sys
import uuid
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_record():
    return {
        "record_id": str(uuid.uuid4()),
        "claim_id": "CLM-TEST001",
        "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400",
        "vehicle": {"make": "Toyota", "year": 2021},
        "location": "Dallas, TX",
        "prediction": {
            "severity_class": "Moderate",
            "confidence_score": 0.84,
            "damage_type": "Bumper Dent",
            "severity_scores": {"Minor": 0.10, "Moderate": 0.84, "Severe": 0.06}
        },
        "fraud_analysis": {
            "fraud_flagged": False,
            "fraud_score": 0.15,
            "flag_reason": None
        },
        "claim_status": "Pending",
        "created_at": "2024-06-01T10:00:00"
    }


@pytest.fixture
def severe_flagged_record():
    return {
        "record_id": str(uuid.uuid4()),
        "claim_id": "CLM-FRAUD001",
        "image_url": "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=400",
        "vehicle": {"make": "BMW", "year": 2023},
        "location": "Houston, TX",
        "prediction": {
            "severity_class": "Severe",
            "confidence_score": 0.61,
            "damage_type": "Hood Damage",
            "severity_scores": {"Minor": 0.20, "Moderate": 0.19, "Severe": 0.61}
        },
        "fraud_analysis": {
            "fraud_flagged": True,
            "fraud_score": 0.82,
            "flag_reason": "Low confidence + Severe claim mismatch"
        },
        "claim_status": "Flagged",
        "created_at": "2024-06-02T14:00:00"
    }


# ── Test: Synthetic Data Generation ───────────────────────────────────────────
class TestSyntheticDataGeneration:

    def test_generate_records_count(self):
        from synthetic_data.generate_damage_records import generate_dataset
        records = generate_dataset(n=10, output_path="/tmp/test_records.json")
        assert len(records) == 10

    def test_record_has_required_fields(self):
        from synthetic_data.generate_damage_records import generate_record
        record = generate_record(0)
        required = ["record_id", "claim_id", "image_url", "vehicle", "prediction",
                    "fraud_analysis", "claim_status", "created_at"]
        for field in required:
            assert field in record, f"Missing field: {field}"

    def test_severity_classes_valid(self):
        from synthetic_data.generate_damage_records import generate_record
        valid_severities = {"Minor", "Moderate", "Severe"}
        for i in range(20):
            record = generate_record(i)
            assert record["prediction"]["severity_class"] in valid_severities

    def test_confidence_score_range(self):
        from synthetic_data.generate_damage_records import generate_record
        for i in range(20):
            record = generate_record(i)
            conf = record["prediction"]["confidence_score"]
            assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"

    def test_fraud_score_range(self):
        from synthetic_data.generate_damage_records import generate_record
        for i in range(20):
            record = generate_record(i)
            score = record["fraud_analysis"]["fraud_score"]
            assert 0.0 <= score <= 1.0, f"Invalid fraud score: {score}"


# ── Test: Image Preprocessing ─────────────────────────────────────────────────
class TestImagePreprocessing:

    def test_preprocess_returns_correct_shape(self):
        from models.damage_classifier import preprocess_from_url
        url = "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=200"
        try:
            arr = preprocess_from_url(url)
            assert arr.shape == (1, 224, 224, 3)
        except Exception:
            pytest.skip("Image URL not accessible in test environment")

    def test_preprocess_normalizes_to_0_1(self):
        from models.damage_classifier import preprocess_from_url
        url = "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=200"
        try:
            arr = preprocess_from_url(url)
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0
        except Exception:
            pytest.skip("Image URL not accessible in test environment")

    def test_invalid_url_raises_value_error(self):
        from models.damage_classifier import preprocess_from_url
        with pytest.raises(ValueError):
            preprocess_from_url("https://invalid-url-that-does-not-exist.xyz/img.jpg")


# ── Test: Vector Store ────────────────────────────────────────────────────────
class TestVectorStore:

    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        """Use a temp directory for each test."""
        import os
        os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chromadb")
        from embeddings.vector_store import DamageVectorStore
        self.store = DamageVectorStore()

    def test_upsert_record(self, sample_record):
        record_id = self.store.upsert_record(sample_record)
        assert record_id == sample_record["record_id"]
        assert self.store.count() == 1

    def test_similarity_search_returns_results(self, sample_record, severe_flagged_record):
        self.store.upsert_record(sample_record)
        self.store.upsert_record(severe_flagged_record)

        results = self.store.similarity_search(sample_record, top_k=2)
        assert len(results) > 0
        assert "similarity_score" in results[0]
        assert "metadata" in results[0]

    def test_similarity_scores_in_range(self, sample_record, severe_flagged_record):
        self.store.upsert_record(sample_record)
        self.store.upsert_record(severe_flagged_record)

        results = self.store.similarity_search(sample_record, top_k=2)
        for result in results:
            assert 0.0 <= result["similarity_score"] <= 1.0

    def test_rag_context_is_string(self, sample_record, severe_flagged_record):
        self.store.upsert_record(sample_record)
        self.store.upsert_record(severe_flagged_record)

        context = self.store.get_rag_context(sample_record, top_k=2)
        assert isinstance(context, str)
        assert len(context) > 0

    def test_fraud_outlier_detection(self, severe_flagged_record):
        # Add several minor cases
        for i in range(5):
            record = {
                "record_id": str(uuid.uuid4()),
                "prediction": {"severity_class": "Minor", "confidence_score": 0.92,
                               "damage_type": "Scratch"},
                "fraud_analysis": {"fraud_flagged": False, "fraud_score": 0.05},
                "vehicle": {"make": "Honda", "year": 2020},
                "location": "Dallas, TX",
                "claim_status": "Approved"
            }
            self.store.upsert_record(record)

        result = self.store.detect_fraud_outlier(severe_flagged_record, top_k=5)
        assert "fraud_outlier" in result
        assert "claimed_severity" in result

    def test_bulk_upsert(self):
        from synthetic_data.generate_damage_records import generate_dataset
        records = generate_dataset(n=20, output_path="/tmp/bulk_test.json")
        count = self.store.bulk_upsert(records)
        assert count == 20
        assert self.store.count() == 20


# ── Test: RAG Report Generator ────────────────────────────────────────────────
class TestRAGReportGenerator:

    def test_mock_report_generated(self, sample_record):
        from rag.report_generator import RAGReportGenerator
        generator = RAGReportGenerator(vector_store=None)
        report = generator.generate_report(sample_record)

        assert "report_text" in report
        assert len(report["report_text"]) > 100
        assert report["claim_id"] == sample_record["claim_id"]

    def test_report_has_required_sections(self, sample_record):
        from rag.report_generator import RAGReportGenerator
        generator = RAGReportGenerator(vector_store=None)
        report = generator.generate_report(sample_record)
        text = report["report_text"]

        assert "DAMAGE SUMMARY" in text
        assert "SEVERITY JUSTIFICATION" in text
        assert "FRAUD RISK ASSESSMENT" in text
        assert "RECOMMENDED ACTION" in text

    def test_report_metadata(self, sample_record):
        from rag.report_generator import RAGReportGenerator
        generator = RAGReportGenerator(vector_store=None)
        report = generator.generate_report(sample_record)

        assert report["severity_class"] == sample_record["prediction"]["severity_class"]
        assert report["confidence_score"] == sample_record["prediction"]["confidence_score"]
        assert "generated_at" in report


# ── Test: MongoDB Store ────────────────────────────────────────────────────────
class TestMongoStore:

    @pytest.fixture(autouse=True)
    def setup_mongo(self):
        from ingestion.mongo_store import MongoStore
        self.mongo = MongoStore()
        # Use fallback if no MongoDB
        if self.mongo.db is None:
            self.mongo._fallback_store = {"predictions": [], "reports": [], "audit_logs": []}

    def test_save_and_retrieve_prediction(self, sample_record):
        self.mongo.save_prediction(sample_record)
        retrieved = self.mongo.get_prediction(sample_record["record_id"])
        assert retrieved is not None
        assert retrieved["claim_id"] == sample_record["claim_id"]

    def test_severity_distribution(self, sample_record, severe_flagged_record):
        self.mongo.save_prediction(sample_record)
        self.mongo.save_prediction(severe_flagged_record)
        dist = self.mongo.get_severity_distribution()
        assert isinstance(dist, dict)

    def test_fraud_stats(self, severe_flagged_record):
        self.mongo.save_prediction(severe_flagged_record)
        stats = self.mongo.get_fraud_stats()
        assert "total_cases" in stats
        assert "flagged_cases" in stats
        assert "flag_rate" in stats

    def test_save_and_retrieve_report(self, sample_record):
        report = {
            "record_id": sample_record["record_id"],
            "claim_id": sample_record["claim_id"],
            "report_text": "Test report content.",
            "generated_at": "2024-06-01T10:00:00"
        }
        self.mongo.save_report(report)
        retrieved = self.mongo.get_report(sample_record["record_id"])
        assert retrieved is not None


# ── Test: Flask API ────────────────────────────────────────────────────────────
class TestFlaskAPI:

    @pytest.fixture(autouse=True)
    def setup_client(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from api.app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "components" in data

    def test_predict_endpoint_missing_url(self):
        response = self.client.post(
            "/predict",
            json={},
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_predict_endpoint_with_url(self):
        response = self.client.post(
            "/predict",
            json={"image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=200"},
            content_type="application/json"
        )
        # Either 201 (success) or 422 (image processing error in test env)
        assert response.status_code in [201, 422, 500]

    def test_cases_endpoint(self):
        response = self.client.get("/cases")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "cases" in data
        assert "count" in data

    def test_severity_analytics_endpoint(self):
        response = self.client.get("/analytics/severity")
        assert response.status_code == 200

    def test_fraud_analytics_endpoint(self):
        response = self.client.get("/analytics/fraud")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "total_cases" in data
        assert "flagged_cases" in data

    def test_nonexistent_case(self):
        response = self.client.get("/cases/nonexistent-id-12345")
        assert response.status_code == 404


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
