# SPIRO — Smart Waste Management System

> AI-powered waste reporting, classification, and municipal waste collection management.

SPIRO is a full-stack platform combining **React**, **FastAPI**, **PostgreSQL**, and **Machine Learning** (YOLOv11 ONNX) to improve waste reporting, classification, and municipal waste collection management.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Quick Start (Backend + Frontend)](#quick-start-backend--frontend)
5. [Demonstration Guide (Demo to your Guide)](#demonstration-guide-demo-to-your-guide)
6. [ML Inference Pipeline](#ml-inference-pipeline)
7. [QR Validation Feature](#qr-validation-feature)
8. [Green Coins Feature](#green-coins-feature)
9. [Deployment (Backend Docker + PostgreSQL)](#deployment-backend-docker--postgresql)
10. [Git & Deployment Strategy](#git--deployment-strategy)
11. [License](#license)

---

## 1. Project Overview

SPIRO lets **citizens** submit waste reports with photos. The backend runs **ML inference** (YOLOv11 ONNX) on the uploaded image to classify the waste type and assign a disposal stream. **Workers** view pending reports, scan the citizen's household QR code at pickup to validate collection, and mark reports complete. Citizens earn **green coins** for verified collections. **Admins** manage workers, wards, schedules, and view audit trails.

### How the system works

```
Citizen submits report (photo)
        │
        ▼
Backend runs ML inference (YOLOv11 ONNX)
        │
        ▼
Prediction saved (category + confidence)
        │
        ▼
Worker accepts report from queue
        │
        ▼
Worker scans citizen's household QR at pickup
        │
        ▼
Worker marks report Complete
        │
        ▼
Citizen earns Green Coins
```

---

## 2. Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + Vite + Lucide Icons |
| Backend | FastAPI + SQLAlchemy + Python 3.12 |
| Database | PostgreSQL 16 |
| ML Inference | YOLOv11n ONNX Runtime (CPU) |
| QR Scanning | html5-qrcode (browser camera API) |

---

## 3. Project Structure

```
SPIRO/
├── backend/                  # FastAPI backend
│   ├── app/                  # Application code
│   │   ├── api/              # API routes
│   │   ├── core/             & Config, database, exceptions
│   │   ├── models/           & SQLAlchemy models
│   │   ├── repositories/     & Data access layer
│   │   ├── schemas/          & Pydantic schemas
│   │   ├── services/         & Business logic (ML, coins, QR)
│   │   └── main.py           & FastAPI app entrypoint
│   ├── models/exports/       # ONNX model files
│   ├── database/             # schema.sql, seed.sql
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── api/              # API client functions
│   │   ├── components/       & Reusable UI components
│   │   ├── layouts/          & Page layouts
│   │   ├── pages/            & Page components
│   │   └── theme.ts          & Design tokens
│   ├── package.json
│   └── vite.config.ts
├── ml/                       # ML subsystem (training, configs, dataset engineering)
├── docker-compose.yml        # PostgreSQL + Backend (Docker Compose)
├── docs/                     # Documentation
├── .gitignore
└── README.md
```

---

## 4. Quick Start (Backend + Frontend)

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Node.js (LTS) + npm
- Git

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd SPIRO
```

### Step 2: Backend setup

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from example)
cp .env.example .env
# Edit .env with your DATABASE_URL, JWT_SECRET_KEY, etc.

# Initialize the database
psql -U postgres -d spiro -f database/schema.sql
psql -U postgres -d spiro -f database/seed.sql

# Start the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://127.0.0.1:8000`. API docs at `http://127.0.0.1:8000/docs`.

### Step 3: Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## 5. Demonstration Guide (Demo to your Guide)

This walkthrough shows the complete end-to-end flow: **citizen submits report → AI classifies waste → worker validates pickup via QR → citizen earns green coins**.

### Prerequisites before demo
- Backend running on `http://127.0.0.1:8000`
- Frontend running on `http://localhost:5173`
- Database seeded (`schema.sql` + `seed.sql`)
- Test accounts created:
  - **Citizen**: `citizen@example.com` / `citizen123`
  - **Worker**: `worker@example.com` / `worker123`
  - **Admin**: `admin@example.com` / `admin123`

### Step 1: Citizen submits a waste report

1. Open `http://localhost:5173` in a browser.
2. Register/login as a citizen.
3. Go to **Submit Report**.
4. Upload a waste photo (or use the camera).
5. Enter a description.
6. Click **Submit Report**.

**What happens:**
- The image is uploaded to the backend.
- The backend runs **YOLOv11n ONNX inference** on the image.
- The waste category is classified (e.g., Recoverable, Organic, Hazardous).
- A prediction is saved with confidence score.
- The report appears in the worker queue.

### Step 2: Citizen views AI result

1. Go to **My Reports**.
2. Click on a report to view details.
3. The **AI Classification** section shows:
   - Detected category (e.g., "Recoverable")
   - Confidence percentage
   - Disposal guidance
   - Estimated green coins

### Step 3: Worker accepts and validates pickup

1. Log in as a worker (`worker@example.com` / `worker123`).
2. Go to **Report Queue**.
3. Click **Accept** on a report.
4. Go to **Active Reports**.
5. Click **Verify Pickup**.
6. **Scan the citizen's Household QR code** (citizen opens their dashboard → My QR).
7. Click **Mark Completed**.

### Step 4: Citizen earns Green Coins

1. Citizen logs back in.
2. Goes to **Dashboard** or **Wallet**.
3. Sees the green coin balance updated.
4. Can redeem coins for rewards (displayed in wallet).

### Step 5: Admin oversight

1. Log in as admin (`admin@example.com` / `admin123`).
2. Go to **Dashboard** for overview stats.
3. Go to **Workers** to manage worker-ward assignments.
4. Go to **AI/ML** to view model information.
5. Go to **Audit** to see the full transaction history.

---

## 6. ML Inference Pipeline

### Architecture

```
Citizen Uploads Image
        │
        ▼
FastAPI Backend
        │
        ▼
YOLOv11n ONNX Runtime (CPU)
        │
        ▼
Detection + Classification
        │
        ▼
Waste Category + Confidence
        │
        ▼
Stored in PostgreSQL
```

### Model details

- **Architecture:** YOLOv11n (nano)
- **Format:** ONNX (exported from Ultralytics)
- **Classes:** 34 waste classes
- **Input:** 640×640 RGB, normalized to 0–1
- **Execution:** CPU via ONNX Runtime

### How inference works

1. Citizen uploads an image.
2. Backend loads the image and preprocesses it (letterbox resize, BGR→RGB, normalize to 0–1).
3. The ONNX model runs inference via ONNX Runtime CPU.
4. Detections are decoded, filtered by confidence threshold, and NMS is applied.
5. The top detection's class is mapped to a waste stream (Recoverable, Organic, Hazardous, Sanitary, E-Waste, Non-Recoverable, Reject).
6. The prediction is saved to the database with confidence score.

### Confidence threshold

Configured via `ML_CONFIDENCE_THRESHOLD` in `.env` (default: 0.25).

If no objects are detected above the threshold, the report gets an **"Unclassified"** result, and a worker will manually classify it during collection.

---

## 7. QR Validation Feature

### How it works

1. Each household has a unique QR code (displayed on the citizen's dashboard).
2. The worker scans the QR at pickup time.
3. The QR payload is HMAC-signed to prevent tampering.
4. The backend verifies:
   - QR signature is valid
   - QR household matches the report's household
   - Report is assigned to the scanning worker
   - Report status allows pickup validation
5. If valid, the pickup is marked as validated.
6. When the report is completed, the citizen earns bonus green coins.

### API endpoints

- `GET /households/{id}/qr` — Get signed QR code (data URL PNG)
- `POST /worker/reports/{report_id}/validate-pickup` — Validate pickup with QR payload

---

## 8. Green Coins Feature

### How it works

1. Citizen submits a report.
2. Worker marks the report as **Completed**.
3. Citizen earns green coins:
   - **Base:** 10 coins per completed report
   - **Bonus:** +5 coins if pickup was QR-validated
4. Citizen can view balance and transaction history in the wallet.

### API endpoint

- `GET /coins/me` — Get citizen's wallet balance and transaction history

---

## 9. Deployment (Backend Docker + PostgreSQL)

### Using Docker Compose

```bash
# Copy environment template
cp backend/.env.example backend/.env
# Edit backend/.env with your secrets

# Build and run
docker compose up --build -d
```

This starts:
- **PostgreSQL 16** (persistent volume `pgdata`)
- **FastAPI Backend** (persistent volume `uploads` for images)

### Initialize database (first time only)

```bash
docker compose exec db psql -U spiro -d spiro -f - < backend/database/schema.sql
docker compose exec db psql -U spiro -d spiro -f - < backend/database/seed.sql
```

### Access

- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Production notes

- Replace all secrets in `.env` with strong random values.
- Set `CORS_ORIGINS` to your production frontend domain.
- For production, consider moving uploaded images to object storage (S3/GCS).
- Use a managed PostgreSQL database for production.

---

## 10. Git & Deployment Strategy

### Repository setup

```bash
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: SPIRO smart waste management system"

# Create repository on GitHub/GitLab, then:
git remote add origin <repository-url>
git push -u origin main
```

### Branch strategy

```
main          → Production-ready code
develop       → Integration branch
feature/*     → Feature development (e.g., feature/ml-pipeline)
```

### Deployment workflow

1. Develop on feature branches.
2. Merge to `develop` after testing.
3. Merge `develop` to `main` for releases.
4. Tag releases: `git tag v1.0.0`
5. Deploy `main` to production server.

### Environment variables for production

Create `.env` files on the production server:

```env
# backend/.env
DATABASE_URL=postgresql://user:password@host:5432/spiro
JWT_SECRET_KEY=<long-random-secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
ML_SERVICE_API_KEY=<long-random-secret>
ML_ENABLED=true
ML_MOCK_MODE=false
ML_CONFIDENCE_THRESHOLD=0.25
CORS_ORIGINS=https://your-frontend-domain.com
QR_SECRET_KEY=<long-random-secret>
COINS_PER_COMPLETION=10
COINS_QR_BONUS=5
```

```env
# frontend/.env
VITE_API_BASE_URL=https://your-backend-domain.com/api/v1
```

---

## 11. License

This project is developed as part of a Bachelor of Engineering Final Year Project.

---

## Support

For issues or questions, please open a GitHub issue in the repository.