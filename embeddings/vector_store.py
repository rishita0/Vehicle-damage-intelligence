"""
embeddings/vector_store.py
───────────────────────────
Converts damage prediction outputs into semantic vector embeddings
and stores them in ChromaDB for similarity search.

Flow:
  Prediction Result → Text Representation → Embedding → ChromaDB

Use Cases:
  1. Semantic similarity search — find past cases similar to new damage
  2. RAG context retrieval — fetch relevant cases to ground LLM reports
  3. Fraud detection — flag outliers by distance from nearest neighbors
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "vehicle_damage_cases"


# ── Text Representation Builder ───────────────────────────────────────────────
def build_damage_text(record: dict) -> str:
    """
    Convert a prediction record into a rich text string for embedding.
    Captures severity, confidence, damage type, location, fraud signal.
    """
    pred = record.get("prediction", record)  # Handle both raw + nested
    fraud = record.get("fraud_analysis", {})
    vehicle = record.get("vehicle", {})

    text = (
        f"Vehicle damage case. "
        f"Severity: {pred.get('severity_class', 'Unknown')}. "
        f"Confidence: {pred.get('confidence_score', 0):.0%}. "
        f"Damage type: {pred.get('damage_type', 'General damage')}. "
        f"Vehicle: {vehicle.get('make', 'Unknown')} {vehicle.get('year', '')}. "
        f"Location: {record.get('location', 'Unknown')}. "
        f"Fraud flagged: {fraud.get('fraud_flagged', False)}. "
        f"Fraud score: {fraud.get('fraud_score', 0):.2f}. "
        f"Claim status: {record.get('claim_status', 'Pending')}."
    )
    return text


# ── Vector Store Client ───────────────────────────────────────────────────────
class DamageVectorStore:
    """
    ChromaDB-backed vector store for vehicle damage cases.
    Handles upsert, similarity search, and RAG context retrieval.
    """

    def __init__(self):
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✅ VectorStore initialized — {self.collection.count()} records in index")

    def upsert_record(self, record: dict) -> str:
        """
        Embed a damage record and upsert into ChromaDB.
        Returns the vector ID.
        """
        record_id = record.get("record_id", str(uuid.uuid4()))
        text = build_damage_text(record)
        embedding = self.encoder.encode(text).tolist()

        pred = record.get("prediction", record)
        fraud = record.get("fraud_analysis", {})

        metadata = {
            "record_id": record_id,
            "severity_class": pred.get("severity_class", "Unknown"),
            "confidence_score": float(pred.get("confidence_score", 0)),
            "damage_type": pred.get("damage_type", "Unknown"),
            "fraud_flagged": str(fraud.get("fraud_flagged", False)),
            "fraud_score": float(fraud.get("fraud_score", 0)),
            "location": record.get("location", "Unknown"),
            "claim_status": record.get("claim_status", "Pending"),
            "created_at": record.get("created_at", datetime.utcnow().isoformat()),
            "text_representation": text
        }

        self.collection.upsert(
            ids=[record_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )

        logger.info(f"📌 Upserted vector: {record_id} [{pred.get('severity_class')}]")
        return record_id

    def bulk_upsert(self, records: list) -> int:
        """Upsert a list of records. Returns count inserted."""
        count = 0
        for record in records:
            try:
                self.upsert_record(record)
                count += 1
            except Exception as e:
                logger.error(f"Failed to upsert {record.get('record_id')}: {e}")
        logger.info(f"✅ Bulk upsert complete: {count}/{len(records)} records")
        return count

    def similarity_search(self, query_record: dict, top_k: int = 3) -> list[dict]:
        """
        Find top-K most similar historical damage cases.
        Used for: similar case retrieval, RAG context, fraud comparison.
        """
        query_text = build_damage_text(query_record)
        query_embedding = self.encoder.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "documents", "distances"]
        )

        similar_cases = []
        if results["ids"] and results["ids"][0]:
            for i, record_id in enumerate(results["ids"][0]):
                similar_cases.append({
                    "record_id": record_id,
                    "similarity_score": round(1 - results["distances"][0][i], 4),
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i]
                })

        logger.info(f"🔍 Found {len(similar_cases)} similar cases")
        return similar_cases

    def get_rag_context(self, query_record: dict, top_k: int = 3) -> str:
        """
        Build a formatted context string from similar cases for LLM RAG.
        Returns a structured text block ready to inject into a prompt.
        """
        similar_cases = self.similarity_search(query_record, top_k=top_k)

        if not similar_cases:
            return "No similar historical cases found."

        context_lines = ["=== SIMILAR HISTORICAL CASES FOR REFERENCE ===\n"]
        for i, case in enumerate(similar_cases, 1):
            meta = case["metadata"]
            context_lines.append(
                f"Case {i} (Similarity: {case['similarity_score']:.0%}):\n"
                f"  Severity: {meta['severity_class']} | "
                f"Confidence: {float(meta['confidence_score']):.0%}\n"
                f"  Damage: {meta['damage_type']} | Location: {meta['location']}\n"
                f"  Fraud Flagged: {meta['fraud_flagged']} | "
                f"Fraud Score: {float(meta['fraud_score']):.2f}\n"
                f"  Claim Status: {meta['claim_status']}\n"
            )

        return "\n".join(context_lines)

    def detect_fraud_outlier(self, query_record: dict, top_k: int = 5) -> dict:
        """
        Compare a new case against similar historical cases.
        Flag as potential fraud if severity is inconsistent with neighbors.
        """
        similar_cases = self.similarity_search(query_record, top_k=top_k)

        if not similar_cases:
            return {"fraud_outlier": False, "reason": "Insufficient historical data"}

        neighbor_severities = [c["metadata"]["severity_class"] for c in similar_cases]
        claimed_severity = query_record.get("prediction", {}).get("severity_class", "")

        severe_count = neighbor_severities.count("Severe")
        mismatch = claimed_severity == "Severe" and severe_count < 2

        avg_neighbor_fraud = sum(
            float(c["metadata"]["fraud_score"]) for c in similar_cases
        ) / len(similar_cases)

        return {
            "fraud_outlier": mismatch,
            "claimed_severity": claimed_severity,
            "neighbor_severities": neighbor_severities,
            "avg_neighbor_fraud_score": round(avg_neighbor_fraud, 4),
            "reason": "Severe claim inconsistent with similar cases" if mismatch else "Within normal range"
        }

    def count(self) -> int:
        return self.collection.count()


# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = DamageVectorStore()
    print(f"Records in store: {store.count()}")

    test_record = {
        "record_id": "test-001",
        "prediction": {
            "severity_class": "Moderate",
            "confidence_score": 0.82,
            "damage_type": "Bumper Dent"
        },
        "fraud_analysis": {"fraud_flagged": False, "fraud_score": 0.15},
        "vehicle": {"make": "Toyota", "year": 2020},
        "location": "Dallas, TX",
        "claim_status": "Pending"
    }

    store.upsert_record(test_record)
    context = store.get_rag_context(test_record)
    print("\nRAG Context Preview:")
    print(context)
