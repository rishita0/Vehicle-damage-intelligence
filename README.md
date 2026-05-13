# 🚗 AI-Powered Vehicle Damage Intelligence Pipeline

> **Classify damage. Embed predictions. Retrieve similar cases. Generate grounded reports.**

A production-style AI data engineering pipeline that classifies vehicle damage severity from images using CNNs, converts prediction outputs into vector embeddings for semantic search, and applies **Retrieval-Augmented Generation (RAG)** to produce grounded, auditable inspection reports.

---

## 🏗️ Architecture

```
Image URL Input
      │
      ▼
┌─────────────────────────────┐
│   Image Preprocessing       │  ← PIL, NumPy
│   URL fetch → RGB → resize  │
│   → normalize → validate    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   CNN Damage Classifier     │  ← TensorFlow / MobileNetV2
│   Minor / Moderate / Severe │
│   + Confidence Score        │
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐    ┌───────────────────────┐
│ MongoDB  │    │  Vector Embedding     │  ← sentence-transformers
│ Storage  │    │  Prediction → Text    │
│ + Audit  │    │  → Embedding → ChromaDB│
└──────────┘    └───────────┬───────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
            ┌──────────────┐  ┌────────────┐
            │ Similarity   │  │ RAG Report │  ← Ollama llama3.2
            │ Search       │  │ Generator  │
            │ Top-K Cases  │  │ + Context  │
            └──────────────┘  └────────────┘
                                    │
                            ┌───────┴────────┐
                            │  Fraud Flag    │
                            │  Analytics     │
                            │  Dashboard API │
                            └────────────────┘
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| CNN Classification | TensorFlow, MobileNetV2, Keras |
| Vector Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (local, persistent) |
| RAG / LLM | Ollama (llama3.2 — local, free, no API key) |
| Storage | MongoDB |
| API | Flask, Flask-CORS |
| Testing | pytest |
| Language | Python 3.9+ |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/vehicle-damage-ai.git
cd vehicle-damage-ai

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install Ollama (one-time, free)

```bash
# Download Ollama for Windows: https://ollama.com/download
# Then pull the model:
ollama pull llama3.2
# Runs as a background service on Windows — no API key needed!
```

### 3. Configure Environment

```bash
cp .env.example .env
# Defaults work out of the box — no API keys required!
```

### 3. Run the Full Pipeline

```bash
# Generate 100 synthetic records → embed → store → generate reports
python pipeline/run_pipeline.py --n 100

# Or with specific options:
python pipeline/run_pipeline.py --n 50 --report-target severe
```

### 4. Start the API Server

```bash
python api/app.py
# API running at http://localhost:5000
```

### 5. Run Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Pipeline health + component status |
| `POST` | `/predict` | Classify damage from image URL |
| `POST` | `/predict/batch` | Batch classify multiple URLs |
| `GET` | `/cases` | List all damage cases |
| `GET` | `/cases/<id>` | Get single case + report |
| `GET` | `/similar/<id>` | Find similar cases (vector search) |
| `POST` | `/report/generate` | Generate RAG inspection report |
| `GET` | `/report/<id>` | Retrieve generated report |
| `GET` | `/analytics/severity` | Severity distribution stats |
| `GET` | `/analytics/fraud` | Fraud flag analytics |

### Example: Classify a Damage Image

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-image-url.jpg",
    "vehicle": {"make": "Toyota", "year": 2021},
    "location": "Dallas, TX"
  }'
```

**Response:**
```json
{
  "success": true,
  "record_id": "uuid-here",
  "claim_id": "CLM-ABC123",
  "prediction": {
    "severity_class": "Moderate",
    "confidence_score": 0.84,
    "severity_scores": {"Minor": 0.10, "Moderate": 0.84, "Severe": 0.06}
  },
  "fraud_analysis": {
    "fraud_score": 0.15,
    "fraud_flagged": false
  }
}
```

### Example: Generate RAG Report

```bash
curl -X POST http://localhost:5000/report/generate \
  -H "Content-Type: application/json" \
  -d '{"record_id": "uuid-from-predict"}'
```

---

## 📁 Project Structure

```
vehicle-damage-ai/
│
├── models/
│   └── damage_classifier.py        # MobileNetV2 CNN + inference engine
│
├── embeddings/
│   └── vector_store.py             # ChromaDB vector store + similarity search
│
├── rag/
│   └── report_generator.py         # RAG report generation (GPT-4 + ChromaDB)
│
├── ingestion/
│   └── mongo_store.py              # MongoDB storage + analytics queries
│
├── api/
│   └── app.py                      # Flask REST API
│
├── pipeline/
│   └── run_pipeline.py             # End-to-end pipeline orchestrator
│
├── synthetic_data/
│   └── generate_damage_records.py  # Synthetic data generator
│
├── tests/
│   └── test_pipeline.py            # Full pytest test suite
│
├── data/
│   ├── raw/                        # Raw ingested records
│   ├── processed/                  # Cleaned prediction outputs
│   └── chromadb/                   # Persistent vector store
│
├── .env.example                    # Environment config template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🧠 How RAG Works in This Pipeline

1. **New damage image arrives** → CNN classifies severity + generates prediction record
2. **Prediction → Text** → converts structured output into semantic text description
3. **Text → Embedding** → sentence-transformers encodes it into a vector
4. **Vector stored** in ChromaDB alongside metadata
5. **On report generation** → retrieve top-3 similar historical cases from ChromaDB
6. **Build grounded prompt** → inject retrieved cases as context into GPT-4 prompt
7. **GPT-4 generates report** grounded in real historical data → no hallucination risk
8. **Report saved** to MongoDB for auditability

---

## 🔍 Outputs & Business Value

| Output | Value |
|---|---|
| **Damage severity classification** | Automates adjuster triage |
| **Confidence scores** | Flags uncertain predictions for human review |
| **Similar case retrieval** | Benchmark new claims against historical data |
| **RAG inspection reports** | Replaces manual adjuster notes with consistent AI reports |
| **Fraud score + flag** | Detects Severe claims with low model confidence |
| **Severity trend analytics** | Operational insight for claims managers |
| **Vector outlier detection** | Identifies anomalous claims beyond normal patterns |

---

## 🔮 Future Roadmap

- [ ] Fine-tune MobileNetV2 on real vehicle damage dataset (CARVANA, Stanford Cars)
- [ ] Add Pinecone cloud vector store option
- [ ] Streamlit dashboard for claims adjuster UI
- [ ] PDF report export with `fpdf2`
- [ ] Airflow DAG for scheduled batch processing
- [ ] Docker Compose setup for full-stack deployment
- [ ] Role-based API authentication

---

## 👩‍💻 Author

**Rishita Reddy**  
Master's in Business Analytics — University of Texas at Dallas  
Data Engineer  
📧 rrmaryada@gmail.com

---

*Classify. Embed. Retrieve. Report.*
