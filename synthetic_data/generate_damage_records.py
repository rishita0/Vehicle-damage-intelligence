"""
synthetic_data/generate_damage_records.py
─────────────────────────────────────────
Generates synthetic vehicle damage records for testing the pipeline.
Simulates image URLs, damage classifications, confidence scores,
and metadata — no real images needed to test the full pipeline.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

SEVERITY_CLASSES = ["Minor", "Moderate", "Severe"]
DAMAGE_TYPES = ["Bumper Dent", "Side Panel Scratch", "Hood Damage",
                "Windshield Crack", "Door Dent", "Roof Damage",
                "Fender Damage", "Quarter Panel Damage"]
VEHICLE_MAKES = ["Toyota", "Honda", "Ford", "BMW", "Tesla",
                 "Chevrolet", "Nissan", "Hyundai", "Kia", "Audi"]
CLAIM_STATUSES = ["Pending", "Approved", "Flagged", "Rejected"]
LOCATIONS = ["Dallas, TX", "Austin, TX", "Houston, TX", "Phoenix, AZ",
             "Los Angeles, CA", "Chicago, IL", "New York, NY", "Miami, FL"]

SAMPLE_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400",
    "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=400",
    "https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=400",
    "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=400",
    "https://images.unsplash.com/photo-1544636331-e26879cd4d9b?w=400",
]


def generate_record(index: int) -> dict:
    severity = random.choices(
        SEVERITY_CLASSES,
        weights=[0.45, 0.35, 0.20]  # Minor most common
    )[0]

    confidence = round(random.uniform(0.62, 0.99), 4)
    fraud_score = round(random.uniform(0.01, 0.95), 4)

    # Higher fraud score for Severe + low confidence
    if severity == "Severe" and confidence < 0.75:
        fraud_score = round(random.uniform(0.55, 0.95), 4)

    fraud_flagged = fraud_score > 0.70

    days_ago = random.randint(0, 180)
    created_at = datetime.utcnow() - timedelta(days=days_ago)

    return {
        "record_id": str(uuid.uuid4()),
        "claim_id": f"CLM-{random.randint(100000, 999999)}",
        "image_url": random.choice(SAMPLE_IMAGE_URLS),
        "vehicle": {
            "make": random.choice(VEHICLE_MAKES),
            "year": random.randint(2010, 2024),
            "vin": f"VIN{uuid.uuid4().hex[:12].upper()}"
        },
        "prediction": {
            "severity_class": severity,
            "confidence_score": confidence,
            "damage_type": random.choice(DAMAGE_TYPES),
            "severity_scores": {
                "Minor": round(random.uniform(0.01, 0.40), 4),
                "Moderate": round(random.uniform(0.01, 0.40), 4),
                "Severe": round(random.uniform(0.01, 0.40), 4)
            }
        },
        "fraud_analysis": {
            "fraud_score": fraud_score,
            "fraud_flagged": fraud_flagged,
            "flag_reason": "Low confidence + Severe claim mismatch" if fraud_flagged else None
        },
        "location": random.choice(LOCATIONS),
        "claim_status": "Flagged" if fraud_flagged else random.choice(["Pending", "Approved"]),
        "processing_status": "completed",
        "created_at": created_at.isoformat(),
        "rag_report_generated": False
    }


def generate_dataset(n: int = 100, output_path: str = "./data/raw/synthetic_damage_records.json"):
    records = [generate_record(i) for i in range(n)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"✅ Generated {n} synthetic damage records → {output_path}")

    # Print summary
    severities = [r["prediction"]["severity_class"] for r in records]
    flagged = sum(1 for r in records if r["fraud_analysis"]["fraud_flagged"])
    print(f"   Minor: {severities.count('Minor')} | "
          f"Moderate: {severities.count('Moderate')} | "
          f"Severe: {severities.count('Severe')}")
    print(f"   Fraud Flagged: {flagged} ({round(flagged/n*100, 1)}%)")
    return records


if __name__ == "__main__":
    generate_dataset(n=150)
