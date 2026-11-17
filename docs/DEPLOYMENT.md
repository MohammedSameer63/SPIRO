# SPIRO Deployment Guide (simple setup)

No Kubernetes, no separate ML service. The ONNX model runs **inside**
the FastAPI backend, exactly as in development.

```
Internet
   │  (static hosting, e.g. Netlify / Vercel / S3)
   ▼
┌────────────┐   HTTPS   ┌──────────────┐
│  Frontend  │ ────────▶ │   Backend    │──▶ PostgreSQL (managed or compose)
│ React/Vite │           │ FastAPI+ONNX │──▶ uploads/ (persistent volume)
└────────────┘           └──────────────┘
```

## 1. Backend (Docker)

```bash
cp backend/.env.example backend/.env
# Edit backend/.env: DATABASE_URL, JWT_SECRET_KEY, QR_SECRET_KEY,
# ML_SERVICE_API_KEY, CORS_ORIGINS (your frontend domain), DB_PASSWORD
```

Run with compose (Postgres + API):

```bash
docker compose up --build -d
```

Initialize the database once (schema + waste categories + wards):

```bash
# If Postgres is the compose service:
docker compose exec db psql -U spiro -d spiro -f - < backend/database/schema.sql
docker compose exec db psql -U spiro -d spiro -f - < backend/database/seed.sql

# Or with a managed database:
psql "$DATABASE_URL" -f backend/database/schema.sql
psql "$DATABASE_URL" -f backend/database/seed.sql
```

Upgrading an **existing** database (adds QR + coin tables only) —
run the standalone migration as a privileged user (table owner or
superuser; a restricted app user will get `permission denied`):

```bash
sudo -u postgres psql -d spiro -f backend/database/migrate_qr_coins.sql
```

The file is idempotent (`IF NOT EXISTS`), so re-running is safe.
Until it is applied, queue/reports keep working but pickup validation,
coin awards and the wallet return `503` with this message instead of a
`500` traceback.

The image (`backend/Dockerfile`) already contains:

- `app/`, `configs/`, `models/exports/*.onnx`
- `uvicorn app.main:app --host 0.0.0.0 --port 8000` (no `--reload`)
- `/health` endpoint (used by the compose healthcheck)

Uploaded report images live in `uploads/` — the compose file mounts a
named volume there so images survive container replacements. For a
long-term production setup, move this to object storage (S3/GCS);
the DB only stores the `image_url` path, so the migration is contained
in `FileStorageService`.

## 2. Frontend (static hosting)

```bash
cd frontend
cp .env.example .env   # local dev only
npm install
VITE_API_BASE_URL=https://your-backend-domain/api/v1 npm run build
```

Deploy `frontend/dist/` to any static host. The app reads
`VITE_API_BASE_URL` at **build time** — rebuild if the backend URL changes.
Image URLs resolve against the same origin automatically
(`src/api/config.ts`), so no extra config is needed for uploads.

Local dev is unchanged: `npm run dev` + the Vite `/api` proxy, or the
default `http://127.0.0.1:8000/api/v1`.

## 3. New features in this release

**Household QR validation**
- `GET /api/v1/households/{id}/qr` → `{ householdId, qrPayload, qrCode }`
  (citizens: own household; workers/admins: any). QR payloads are
  HMAC-signed with `QR_SECRET_KEY`.
- `POST /api/v1/worker/reports/{id}/validate-pickup` with
  `{ "qr_payload": "<scanned string>" }` — worker scans the citizen's
  QR at collection. The backend checks the signature, that the QR
  household owns the report, and that the report is assigned to the
  scanning worker.
- Citizen dashboard shows the household QR; worker Active Reports page
  has the validate-pickup input.

**Green coins**
- Awarded automatically when a worker marks a report COMPLETED:
  `COINS_PER_COMPLETION` (default 10) + `COINS_QR_BONUS` (default 5)
  when the pickup was QR-validated. Awards are idempotent (one per
  report, enforced by lookup + unique index).
- `GET /api/v1/coins/me` → `{ balance, total_earned, transactions }`.
- Citizen dashboard shows the balance; report details show coins earned.

## 4. Checklist before going live

- [ ] All secrets in `backend/.env` replaced (JWT, QR, ML API key, DB password)
- [ ] `CORS_ORIGINS` includes the hosted frontend domain (exact origin, https)
- [ ] `schema.sql` + `seed.sql` applied to the production database
- [ ] `DEBUG` off / `ML_MOCK_MODE=false` with the real `.onnx` in the image
- [ ] Uploads on a persistent volume (compose does this by default)
- [ ] Frontend built with the production `VITE_API_BASE_URL`
