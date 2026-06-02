# 🚗 RentalGuard AI — Car Rental Damage Intelligence & Profitability Platform

> **Detect damage. Estimate costs. Flag fraud. Protect profit.**

A production-style AI data engineering platform built for car rental companies. Classifies vehicle damage severity from images using CNNs, detects repeat fraud claimants using vector similarity search, estimates repair costs before claims are submitted, and generates RAG-powered inspection reports — all running 100% locally with no paid APIs.

---

## 💡 Business Problem

Car rental companies face three major financial risks on every damage claim:

- **Fraud** — same renter filing repeated claims, pre-existing damage filed as new, exaggerated repair costs
- **Unknown costs** — repair bills arrive after claims are approved with no pre-estimate baseline
- **Premium mispricing** — insurance daily rates are not adjusted after a renter's claim history builds up

RentalGuard AI solves all three by combining CNN damage classification, vector-based fraud detection, financial analytics, and AI-generated reports into one unified platform.

---

## 🏗️ Architecture

```
Damage Image URL + Renter/Vehicle Data
              │
              ▼
┌─────────────────────────────────┐
│      Image Preprocessing        │  ← PIL, NumPy
│  URL fetch → RGB → resize       │
│  → normalize → validate         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     CNN Damage Classifier       │  ← TensorFlow / MobileNetV2
│  Minor / Moderate / Severe      │
│  + Confidence Score             │
└──────────────┬──────────────────┘
               │
      ┌────────┴──────────┐
      ▼                   ▼
┌───────────┐    ┌────────────────────────┐
│  MongoDB  │    │   Vector Embedding     │  ← sentence-transformers
│  Storage  │    │   Prediction → Text    │
│  + Audit  │    │   → Embedding → ChromaDB│
└───────────┘    └────────────┬───────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
    ┌────────────────┐ ┌────────────┐ ┌──────────────────┐
    │ Renter Fraud   │ │ RAG Report │ │  Model Fraud     │
    │ Flag           │ │ Generator  │ │  Flag            │
    │ (per person)   │ │ Ollama     │ │  (per car model) │
    └────────┬───────┘ └─────┬──────┘ └────────┬─────────┘
             │               │                 │
             └───────────────┼─────────────────┘
                             ▼
          ┌──────────────────────────────────────┐
          │         Analytics Engine             │
          │  • Pre-claim repair cost estimate    │
          │  • Repair cost by damage × model     │
          │  • Insurance P&L per renter          │
          │  • Fleet revenue vs claim losses     │
          │  • Premium increase recommendation   │
          └──────────────────┬───────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │       Streamlit Dashboard            │  ← Free, local
          │  • Fleet profitability view          │
          │  • Fraud alert center                │
          │  • Renter risk profiles              │
          │  • Car model cost breakdown          │
          │  • Premium adjustment reports        │
          └──────────────────────────────────────┘
```

---

## 📦 Tech Stack

> ✅ Every tool in this stack is **100% free and open source** — no paid APIs, no subscriptions, no cloud costs.

| Layer | Technology | Cost |
|---|---|---|
| CNN Classification | TensorFlow, MobileNetV2, Keras | Free (Apache 2.0) |
| Image Processing | Pillow, NumPy | Free (open source) |
| Vector Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free (Apache 2.0) |
| Vector Database | ChromaDB (local, persistent) | Free (Apache 2.0) |
| RAG / LLM | Ollama llama3.2 — runs locally | Free (no API key) |
| Storage | MongoDB Community Edition | Free (local) |
| Analytics | pandas, scikit-learn | Free (open source) |
| API | Flask, Flask-CORS | Free (BSD) |
| Dashboard | Streamlit | Free (Apache 2.0) |
| PDF Reports | fpdf2 | Free (LGPL) |
| Testing | pytest | Free (MIT) |
| Language | Python 3.9+ | Free |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourname/rentalguard-ai.git
cd rentalguard-ai

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
# After install, pull the model:
ollama pull llama3.2
# Ollama runs as a background service on Windows automatically — no API key needed
```

### 3. Install MongoDB (one-time, free)

```bash
# Download MongoDB Community Edition: https://www.mongodb.com/try/download/community
# Or run via Docker:
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 4. Configure Environment

```bash
cp .env.example .env
# Defaults work out of the box — no API keys required
```

### 5. Generate Synthetic Data & Run Pipeline

```bash
# Generate 150 synthetic rental claims → embed → store → generate reports
python pipeline/run_pipeline.py --n 150

# Run only report generation for flagged claims
python pipeline/run_pipeline.py --mode report --report-target flagged
```

### 6. Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
# Dashboard opens at http://localhost:8501
```

### 7. Start the API Server

```bash
python api/app.py
# API running at http://localhost:5000
```

### 8. Run Tests

```bash
pytest tests/ -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Pipeline health + component status |
| `POST` | `/predict` | Classify damage from image URL |
| `POST` | `/predict/batch` | Batch classify multiple URLs |
| `GET` | `/cases` | List all damage cases |
| `GET` | `/cases/<id>` | Get single case + RAG report |
| `GET` | `/similar/<id>` | Find similar historical cases (vector search) |
| `POST` | `/report/generate` | Generate RAG inspection report via Ollama |
| `GET` | `/report/<id>` | Retrieve a generated report |
| `GET` | `/renter/<renter_id>/profile` | Renter risk profile + claim history |
| `GET` | `/renter/<renter_id>/premium` | Recommended premium increase after claim |
| `GET` | `/vehicle/<model>/costs` | Repair cost breakdown by damage type for a model |
| `GET` | `/analytics/severity` | Severity distribution across all claims |
| `GET` | `/analytics/fraud` | Fraud flag stats (renter + model flags separately) |
| `GET` | `/analytics/pnl` | Total insurance revenue vs total claim payouts |
| `GET` | `/analytics/model-roi` | Profitability ranking of car models in fleet |

### Example: Classify a Damage Image

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-damage-image.jpg",
    "renter_id": "RNT-001",
    "vehicle": {"make": "Toyota", "model": "Camry", "year": 2022},
    "insurance_plan": "CDW",
    "daily_premium": 18.99,
    "rental_days": 7,
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
    "damage_type": "Bumper Dent",
    "severity_scores": {"Minor": 0.10, "Moderate": 0.84, "Severe": 0.06}
  },
  "cost_estimate": {
    "estimated_repair_cost": 820.00,
    "cost_range": {"low": 650.00, "high": 1100.00},
    "based_on_similar_cases": 12
  },
  "financial_summary": {
    "premium_collected": 132.93,
    "estimated_payout": 820.00,
    "net_position": -687.07,
    "profitable": false
  },
  "fraud_analysis": {
    "renter_flagged": false,
    "renter_prior_claims": 0,
    "model_flagged": false,
    "model_claim_count": 2,
    "combined_fraud_score": 0.15
  },
  "premium_recommendation": {
    "current_daily_rate": 18.99,
    "recommended_daily_rate": 24.50,
    "increase_pct": 29.0,
    "reason": "Claim payout exceeds premiums collected. Moderate severity + first claim."
  }
}
```

### Example: Get Fleet P&L Analytics

```bash
curl http://localhost:5000/analytics/pnl
```

**Response:**
```json
{
  "total_premium_revenue": 48320.00,
  "total_claim_payouts": 31450.00,
  "net_profit": 16870.00,
  "profit_margin_pct": 34.9,
  "claims_by_severity": {"Minor": 62, "Moderate": 38, "Severe": 15},
  "fraud_flagged_claims": 11,
  "fraud_flagged_pct": 9.4
}
```

---

## 📁 Project Structure

```
rentalguard-ai/
│
├── models/
│   └── damage_classifier.py          # MobileNetV2 CNN + inference engine
│
├── embeddings/
│   └── vector_store.py               # ChromaDB vector store + similarity search
│
├── rag/
│   └── report_generator.py           # RAG report generation (Ollama llama3.2 + ChromaDB)
│
├── ingestion/
│   └── mongo_store.py                # MongoDB storage + analytics queries
│
├── analytics/
│   ├── cost_estimator.py             # Pre-claim repair cost estimation
│   ├── pnl_engine.py                 # Revenue vs payout P&L calculations
│   ├── premium_recommender.py        # Post-claim premium increase logic
│   ├── fraud_profiler.py             # Renter + model fraud flagging (separate)
│   └── model_roi.py                  # Car model profitability rankings
│
├── api/
│   └── app.py                        # Flask REST API
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard (free, local)
│
├── pipeline/
│   └── run_pipeline.py               # End-to-end pipeline orchestrator
│
├── synthetic_data/
│   └── generate_rental_claims.py     # Synthetic rental + claim data generator
│
├── tests/
│   ├── test_pipeline.py              # Pipeline + model tests
│   ├── test_analytics.py             # Cost, P&L, premium tests
│   └── test_fraud.py                 # Fraud flagging tests
│
├── data/
│   ├── raw/                          # Raw ingested claim records
│   ├── processed/                    # Cleaned prediction outputs
│   ├── curated/                      # Analytics-ready datasets
│   └── chromadb/                     # Persistent vector store
│
├── .env.example                      # Environment config template
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # MongoDB + optional services
└── README.md
```

---

## 🧠 How RAG Works in This Pipeline

1. **Damage image arrives** → CNN classifies severity + generates structured prediction
2. **Prediction → Text** → converts to semantic description including vehicle, damage type, severity, fraud signals
3. **Text → Embedding** → sentence-transformers encodes it as a vector (runs locally, no API)
4. **Vector stored** in ChromaDB with full metadata
5. **On report generation** → retrieve top-3 similar historical cases from ChromaDB
6. **Grounded prompt built** → similar cases injected as context into Ollama llama3.2 prompt
7. **Ollama generates report** locally — no internet, no cost, no hallucination risk from lack of context
8. **Report + financials saved** to MongoDB for full auditability

---

## 🔍 Outputs & Business Value

| Output | Business Question Answered |
|---|---|
| **Damage severity classification** | How bad is this car — Minor scratch or needs full panel replacement? |
| **Pre-claim repair cost estimate** | Before we open a claim — how much will this cost us? |
| **Repair cost by damage type × car model** | Does a bumper dent cost $400 on a Corolla but $1,200 on a BMW? |
| **Renter fraud flag** (separate) | Is this same renter filing claims repeatedly across any vehicle? |
| **Model fraud flag** (separate) | Is this specific car model showing suspiciously frequent damage? |
| **Combined fraud flag** | Same renter + same model = highest risk signal |
| **Insurance P&L per renter** | This renter paid $133 in premiums — we're paying out $820. Are we losing money? |
| **Fleet profitability dashboard** | Total revenue vs total losses — which models make vs cost us money? |
| **Premium increase recommendation** | After this claim, how much should we raise their daily insurance rate? |
| **RAG inspection reports** | Auto-written damage report grounded in similar historical cases via Ollama |

---

## 💰 Premium Increase Logic

```
Base increase   = (claim_payout - premiums_collected) / expected_remaining_days
Risk buffer     = 10% baseline
                + 10% if renter has 2+ prior claims
                + 15% if severity is Severe
                + 25% if fraud flagged

New daily rate  = current_rate + base_increase + risk_buffer
```

**Example:**
- Renter pays $18.99/day CDW × 7 days = $132.93 collected
- Claim payout = $820.00
- Shortfall = $687.07 over expected 90 remaining rental days = +$7.63/day
- Risk buffer (first claim, Moderate) = +10% = +$1.90/day
- **Recommended new rate = $28.52/day**

---

## 🚨 Fraud Flagging Rules

```
RENTER FLAG    → same renter_id with 2+ claims on ANY vehicle model
MODEL FLAG     → same vehicle make+model with 3+ claims from ANY renter
COMBINED FLAG  → same renter_id + same make+model = highest risk (escalate immediately)

Fraud score increases when:
  → Claimed severity is Severe but model confidence < 0.75
  → Repair cost estimate far exceeds historical average for that damage type
  → Vector similarity to known fraud cases is high
```

---

## 📊 Streamlit Dashboard Modules

| Module | What it shows |
|---|---|
| **Fleet P&L Overview** | Total premiums collected vs total claim payouts, net profit/loss |
| **Damage Severity Trends** | Minor / Moderate / Severe breakdown over time across all claims |
| **Fraud Alert Center** | Renter flags and model flags listed separately, sortable by risk score |
| **Renter Risk Profiles** | Per-renter claim history, fraud score, current premium, recommended rate |
| **Car Model Cost Breakdown** | Repair cost per damage type per vehicle make/model |
| **Model ROI Ranking** | Which car models are most profitable vs most costly in the fleet |
| **Premium Adjustment Report** | All renters due for a rate increase with recommended new daily rate |

---

---

## 👩‍💻 Author

**Rishita Reddy**
Master's in Business Analytics — University of Texas at Dallas
Data Engineer
📧 rrmaryada@gmail.com

---

*Detect damage. Estimate costs. Flag fraud. Protect profit.*
