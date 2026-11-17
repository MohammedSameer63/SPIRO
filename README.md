# SPIRO — Smart Waste Management System

> AI-powered waste reporting, classification, and municipal waste collection management.

SPIRO combines **React**, **FastAPI**, **PostgreSQL**, and **Machine Learning** (YOLOv11 ONNX) to help citizens report waste, automatically classify it using AI, and enable workers to collect and verify waste efficiently.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + Vite |
| Backend | FastAPI + SQLAlchemy + Python 3.12 |
| Database | PostgreSQL 16 |
| ML Inference | YOLOv11n ONNX Runtime (CPU) |
| QR Verification | Browser camera API (html5-qrcode) |

---

## Project Structure

```
SPIRO/
├── backend/                  # FastAPI backend
│   ├── app/                  # Application code
│   │   ├── api/              # API routes
│   │   ├── core/             # Config, database, exceptions
│   │   ├── models/           # SQLAlchemy models
│   │   ├── repositories/     # Data access layer
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic (ML, coins, QR)
│   │   └── main.py           # FastAPI entrypoint
│   ├── models/exports/       # ONNX ML model files
│   ├── database/             # schema.sql, seed.sql
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── api/              # API client functions
│   │   ├── components/       # Reusable UI components
│   │   ├── layouts/          # Page layouts
│   │   ├── pages/            # Page components
│   │   └── theme.ts          # Design tokens
│   ├── package.json
│   └── vite.config.ts
├── ml/                       # ML subsystem (training, configs, dataset engineering)
├── docker-compose.yml        # PostgreSQL + Backend (Docker Compose)
├── docs/DEPLOYMENT.md        # Production deployment guide
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Node.js (LTS) + npm
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/MohammedSameer63/SPIRO.git
cd SPIRO
```

### Step 2: Set up the database

Create a PostgreSQL database and run the schema:

```bash
# Create the database (adjust user/password as needed)
createdb -U postgres spiro

# Run the schema
psql -U postgres -d spiro -f backend/database/schema.sql

# Seed initial data (wards, waste categories)
psql -U postgres -d spiro -f backend/database/seed.sql
```

### Step 3: Configure environment

Copy the example environment files and edit them:

```bash
cd backend
cp .env.example .env
# Edit backend/.env with your DATABASE_URL, JWT_SECRET_KEY, etc.

cd ../frontend
cp .env.example .env
# Edit frontend/.env if needed (VITE_API_BASE_URL)
```

**backend/.env** key variables:
```
DATABASE_URL=postgresql://user:password@localhost:5432/spiro
JWT_SECRET_KEY=change-this-to-a-long-random-secret
ML_SERVICE_API_KEY=my-long-random-secret
ML_ENABLED=true
ML_MOCK_MODE=false
ML_CONFIDENCE_THRESHOLD=0.25
```

### Step 4: Start the backend

```bash
cd backend
source .venv/bin/activate  # or: .venv\Scripts\activate  on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API: `http://127.0.0.1:8000`  
API docs: `http://127.0.0.1:8000/docs`

### Step 5: Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

---

## ML Inference Pipeline

SPIRO uses a YOLOv11n ONNX model to automatically classify waste from uploaded images:

- **Detection:** YOLOv11n detects waste objects in images
- **Classification:** Maps detected objects to waste categories (Recoverable, Organic, Hazardous, Sanitary, E-Waste, Non-Recoverable, Reject)
- **Confidence:** Each prediction includes a confidence score

The ML model runs inside the FastAPI backend via ONNX Runtime (CPU). No separate ML service is required.

---

## QR Validation Feature

Workers can verify waste pickups by scanning the citizen's household QR code:

- Each household has a unique QR code (shown on the citizen dashboard)
- Workers scan the QR at pickup time
- The backend verifies the QR signature and that it matches the report's household
- Verified pickups earn the citizen bonus green coins

---

## Green Coins Feature

Citizens earn green coins when their reports are completed:

- **10 coins** per completed report
- **+5 bonus** if the pickup was QR-validated
- Citizens can view their balance and transaction history in the wallet

---

## Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Citizen | `citizen@example.com` | `citizen123` |
| Worker | `worker@example.com` | `worker123` |
| Admin | `admin@example.com` | `admin123` |

---

## License

This project is developed as part of a Bachelor of Engineering Final Year Project.