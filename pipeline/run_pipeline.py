"""
pipeline/run_pipeline.py
─────────────────────────
End-to-end pipeline orchestrator.

Runs the full pipeline:
  1. Generate or load synthetic damage records
  2. Run CNN inference (or load pre-computed predictions)
  3. Embed all records into ChromaDB vector store
  4. Generate RAG reports for flagged / high-severity cases
  5. Save everything to MongoDB
  6. Print pipeline summary

Usage:
  python pipeline/run_pipeline.py                  # Full pipeline
  python pipeline/run_pipeline.py --mode embed     # Just embed existing records
  python pipeline/run_pipeline.py --mode report    # Just generate reports
  python pipeline/run_pipeline.py --n 50           # Use 50 synthetic records
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data.generate_damage_records import generate_dataset
from embeddings.vector_store import DamageVectorStore
from rag.report_generator import RAGReportGenerator
from ingestion.mongo_store import MongoStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("pipeline")

SYNTHETIC_DATA_PATH = "./data/raw/synthetic_damage_records.json"


def run_ingestion(n: int = 100) -> list:
    """Step 1: Generate or load synthetic records."""
    logger.info(f"📥 Step 1: Ingestion — generating {n} synthetic records")
    records = generate_dataset(n=n, output_path=SYNTHETIC_DATA_PATH)
    return records


def run_embedding(records: list, vector_store: DamageVectorStore) -> int:
    """Step 2: Embed all records into ChromaDB."""
    logger.info(f"🧠 Step 2: Embedding {len(records)} records into vector store")
    count = vector_store.bulk_upsert(records)
    logger.info(f"✅ Embedded {count} records. Total in store: {vector_store.count()}")
    return count


def run_mongo_save(records: list, mongo: MongoStore) -> int:
    """Step 3: Save all records to MongoDB."""
    logger.info(f"💾 Step 3: Saving {len(records)} records to MongoDB")
    count = 0
    for record in records:
        try:
            mongo.save_prediction(record)
            count += 1
        except Exception as e:
            logger.warning(f"Skipped {record.get('record_id')}: {e}")
    logger.info(f"✅ Saved {count} records to MongoDB")
    return count


def run_report_generation(
    records: list,
    report_generator: RAGReportGenerator,
    mongo: MongoStore,
    target: str = "flagged"  # "flagged" | "severe" | "all"
) -> int:
    """Step 4: Generate RAG reports for target cases."""
    if target == "flagged":
        targets = [r for r in records if r.get("fraud_analysis", {}).get("fraud_flagged")]
    elif target == "severe":
        targets = [r for r in records if r.get("prediction", {}).get("severity_class") == "Severe"]
    else:
        targets = records

    logger.info(f"📄 Step 4: Generating RAG reports for {len(targets)} {target} cases")

    count = 0
    for record in targets[:20]:  # Cap at 20 to avoid API overuse in demo
        try:
            report = report_generator.generate_report(record)
            mongo.save_report(report)
            count += 1
            if count % 5 == 0:
                logger.info(f"   Generated {count}/{len(targets)} reports...")
        except Exception as e:
            logger.error(f"Report generation failed for {record.get('record_id')}: {e}")

    logger.info(f"✅ Generated {count} RAG reports")
    return count


def print_summary(mongo: MongoStore, vector_store: DamageVectorStore):
    """Print pipeline run summary."""
    print("\n" + "=" * 60)
    print("  PIPELINE RUN SUMMARY")
    print("=" * 60)
    print(f"  Timestamp:          {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Total predictions:  {mongo.count_predictions()}")
    print(f"  Vector embeddings:  {vector_store.count()}")

    dist = mongo.get_severity_distribution()
    print(f"\n  Severity Distribution:")
    for severity, count in dist.items():
        bar = "█" * (count // 2)
        print(f"    {severity:<10} {count:>4}  {bar}")

    fraud = mongo.get_fraud_stats()
    print(f"\n  Fraud Analysis:")
    print(f"    Total cases:    {fraud['total_cases']}")
    print(f"    Flagged cases:  {fraud['flagged_cases']} ({fraud['flag_rate']}%)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Vehicle Damage AI Pipeline")
    parser.add_argument("--mode", choices=["full", "embed", "report"], default="full")
    parser.add_argument("--n", type=int, default=100, help="Number of synthetic records")
    parser.add_argument("--report-target", choices=["flagged", "severe", "all"], default="flagged")
    args = parser.parse_args()

    logger.info("🚗 Vehicle Damage AI Pipeline starting...")
    start = datetime.utcnow()

    # Initialize components
    vector_store = DamageVectorStore()
    mongo = MongoStore()
    report_generator = RAGReportGenerator(vector_store=vector_store)

    if args.mode == "full":
        records = run_ingestion(n=args.n)
        run_mongo_save(records, mongo)
        run_embedding(records, vector_store)
        run_report_generation(records, report_generator, mongo, target=args.report_target)

    elif args.mode == "embed":
        # Load existing records
        with open(SYNTHETIC_DATA_PATH) as f:
            records = json.load(f)
        run_embedding(records, vector_store)

    elif args.mode == "report":
        records = mongo.get_all_predictions(limit=200)
        run_report_generation(records, report_generator, mongo, target=args.report_target)

    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info(f"⏱  Pipeline completed in {elapsed:.1f}s")
    print_summary(mongo, vector_store)


if __name__ == "__main__":
    main()
