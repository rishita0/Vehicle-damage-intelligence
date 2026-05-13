"""
rag/report_generator.py
────────────────────────
RAG-powered damage assessment report generator using Ollama (100% local).

Flow:
  1. New damage prediction arrives
  2. Retrieve top-K similar historical cases from ChromaDB (vector search)
  3. Build grounded prompt with retrieved context
  4. Ollama (llama3.2 local) generates a structured inspection report
  5. Report stored in MongoDB alongside prediction data

No API key required — runs entirely on your local machine via Ollama.

Setup (one-time):
  1. Download Ollama: https://ollama.com/download
  2. Run in terminal: ollama pull llama3.2
  3. Ollama runs automatically as a background service on Windows
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
MAX_REPORT_TOKENS = int(os.getenv("MAX_REPORT_TOKENS", 800))


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_rag_prompt(prediction: dict, rag_context: str) -> str:
    """Build a structured RAG prompt grounding the LLM in retrieved cases."""
    pred = prediction.get("prediction", prediction)
    fraud = prediction.get("fraud_analysis", {})
    vehicle = prediction.get("vehicle", {})

    return f"""You are an expert vehicle damage assessment analyst for an insurance company.
Generate accurate, professional damage inspection reports grounded in historical case data.
Be specific, concise, and actionable. Use the provided historical cases as reference only.

CURRENT CASE:
Claim ID:        {prediction.get('claim_id', 'N/A')}
Vehicle:         {vehicle.get('make', 'Unknown')} ({vehicle.get('year', 'N/A')})
Location:        {prediction.get('location', 'Unknown')}
Severity Class:  {pred.get('severity_class', 'Unknown')}
Confidence:      {pred.get('confidence_score', 0):.0%}
Damage Type:     {pred.get('damage_type', 'General damage')}
Fraud Score:     {fraud.get('fraud_score', 0):.2f}
Fraud Flagged:   {fraud.get('fraud_flagged', False)}
Flag Reason:     {fraud.get('flag_reason', 'None')}

{rag_context}

Generate the report in this EXACT format:

**DAMAGE SUMMARY**
[2-3 sentences describing the observed damage]

**SEVERITY JUSTIFICATION**
[2 sentences explaining why this severity class was assigned]

**SIMILAR CASE REFERENCES**
[1-2 sentences noting patterns from retrieved historical cases]

**FRAUD RISK ASSESSMENT**
[1-2 sentences on fraud indicators and risk level]

**RECOMMENDED ACTION**
[One clear action: Approve / Investigate Further / Escalate to Senior Adjuster / Reject]
"""


# ── Ollama Client ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call the local Ollama REST API.
    Ollama exposes a simple HTTP endpoint — no SDK needed.
    """
    import requests

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": MAX_REPORT_TOKENS,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama.\n"
            "  1. Download: https://ollama.com/download (Windows installer)\n"
            "  2. Run: ollama pull llama3.2\n"
            "  3. Ollama starts automatically on Windows after install."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out. Try: ollama pull llama3.2:1b (smaller)")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def check_ollama_available() -> dict:
    """Check if Ollama is running and the configured model is pulled."""
    import requests
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        model_ready = any(OLLAMA_MODEL in m for m in models)
        return {
            "available": True,
            "models": models,
            "target_model_ready": model_ready,
            "target_model": OLLAMA_MODEL
        }
    except Exception:
        return {"available": False, "models": [], "target_model_ready": False}


# ── RAG Report Generator ──────────────────────────────────────────────────────
class RAGReportGenerator:
    """
    Generates grounded LLM damage assessment reports using:
    - ChromaDB for context retrieval (RAG)
    - Ollama llama3.2 for local LLM generation — 100% free, no API key
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store

        status = check_ollama_available()
        if status["available"] and status["target_model_ready"]:
            logger.info(f"✅ Ollama ready — model: {OLLAMA_MODEL}")
            self.ollama_ready = True
        elif status["available"]:
            logger.warning(
                f"⚠️  Ollama running but '{OLLAMA_MODEL}' not pulled.\n"
                f"   Run: ollama pull {OLLAMA_MODEL}\n"
                f"   Available models: {status['models']}"
            )
            self.ollama_ready = False
        else:
            logger.warning(
                "⚠️  Ollama not running — using mock reports.\n"
                "   Install from: https://ollama.com/download\n"
                "   Then: ollama pull llama3.2"
            )
            self.ollama_ready = False

    def generate_report(self, prediction: dict) -> dict:
        """Retrieve RAG context from ChromaDB, generate grounded Ollama report."""

        # Step 1: Retrieve similar cases from vector store
        rag_context = "No historical context available."
        similar_cases = []

        if self.vector_store and self.vector_store.count() > 0:
            rag_context = self.vector_store.get_rag_context(prediction, top_k=RAG_TOP_K)
            similar_cases = self.vector_store.similarity_search(prediction, top_k=RAG_TOP_K)

        # Step 2: Build grounded prompt
        prompt = build_rag_prompt(prediction, rag_context)

        # Step 3: Generate with Ollama or fallback
        if self.ollama_ready:
            try:
                report_text = call_ollama(prompt)
                model_used = f"ollama/{OLLAMA_MODEL}"
            except Exception as e:
                logger.warning(f"Ollama call failed, using mock: {e}")
                report_text = self._mock_report(prediction)
                model_used = "mock-fallback"
        else:
            report_text = self._mock_report(prediction)
            model_used = "mock-generator"

        return {
            "claim_id": prediction.get("claim_id", "N/A"),
            "record_id": prediction.get("record_id", "N/A"),
            "generated_at": datetime.utcnow().isoformat(),
            "severity_class": prediction.get("prediction", {}).get("severity_class"),
            "confidence_score": prediction.get("prediction", {}).get("confidence_score"),
            "fraud_flagged": prediction.get("fraud_analysis", {}).get("fraud_flagged", False),
            "rag_context_used": len(similar_cases) > 0,
            "similar_cases_retrieved": len(similar_cases),
            "report_text": report_text,
            "model_used": model_used
        }

    def _mock_report(self, prediction: dict) -> str:
        """Fallback report when Ollama is not running."""
        pred = prediction.get("prediction", {})
        severity = pred.get("severity_class", "Unknown")
        damage = pred.get("damage_type", "vehicle damage")
        confidence = pred.get("confidence_score", 0)
        fraud = prediction.get("fraud_analysis", {})

        return f"""**DAMAGE SUMMARY**
The vehicle shows evidence of {severity.lower()} {damage.lower()}. Visual analysis of the submitted image indicates damage consistent with the assigned severity classification. The affected area requires professional assessment before repair authorization.

**SEVERITY JUSTIFICATION**
The model assigned {severity} severity with {confidence:.0%} confidence based on visual damage patterns. This aligns with the observed damage type and extent of affected vehicle components.

**SIMILAR CASE REFERENCES**
Historical cases with similar damage profiles were retrieved from the vector database. The pattern is consistent with typical {severity.lower()} severity claims for this vehicle type and damage category.

**FRAUD RISK ASSESSMENT**
{'This claim is flagged for fraud review. The Severe severity claim is inconsistent with the low model confidence and nearest-neighbor historical cases. Manual adjuster review required before approval.' if fraud.get('fraud_flagged') else 'No significant fraud indicators detected. Claimed severity and model confidence are consistent with similar historical cases.'}

**RECOMMENDED ACTION**
{'Escalate to Senior Adjuster' if fraud.get('fraud_flagged') else 'Approve' if severity == 'Minor' else 'Investigate Further'}

---
[Mock report — start Ollama and run "ollama pull llama3.2" to enable local LLM generation.]"""


# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Checking Ollama status...")
    status = check_ollama_available()
    print(f"  Running:       {status['available']}")
    print(f"  Models pulled: {status['models']}")
    print(f"  {OLLAMA_MODEL} ready: {status['target_model_ready']}")

    generator = RAGReportGenerator(vector_store=None)

    test = {
        "claim_id": "CLM-TEST001",
        "record_id": "test-001",
        "vehicle": {"make": "Toyota", "year": 2021},
        "location": "Dallas, TX",
        "prediction": {"severity_class": "Moderate", "confidence_score": 0.84,
                       "damage_type": "Bumper Dent"},
        "fraud_analysis": {"fraud_flagged": False, "fraud_score": 0.12, "flag_reason": None}
    }

    report = generator.generate_report(test)
    print("\n" + "=" * 60)
    print(report["report_text"])
    print(f"\nModel: {report['model_used']}")
