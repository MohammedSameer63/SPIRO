# SPIRO

> **AI-Powered Smart Waste Management System**

SPIRO is a smart waste management platform that combines **React**, **FastAPI**, **PostgreSQL**, and **Machine Learning** to improve waste reporting, classification, and municipal waste collection management.

---

# Team

| Name            | Role                                    |
| --------------- | --------------------------------------- |
| Mohammed Sameer | Backend Developer & System Architecture |
| Sindhu          | Machine Learning                        |
| Rithvik         | Frontend Development                    |

---

# Tech Stack

## Frontend

* React
* TypeScript
* Vite

## Backend

* FastAPI
* SQLAlchemy
* PostgreSQL
* Python

## Machine Learning

* TensorFlow / PyTorch (TBD)
* OpenCV

## Database

* PostgreSQL

## Version Control

* Git
* GitHub

---

# Project Structure

```text
SPIRO/
│
├── frontend/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   │
│   ├── tests/
│   ├── requirements.txt
│   ├── .env
│   └── README.md
│
├── ml-service/
│
├── database/
│   └── schema.sql
│
├── docs/
│
├── README.md
└── .gitignore
```

---

# Prerequisites

Install the following before starting development:

* Git
* Python 3.12+
* PostgreSQL 16+
* Node.js (LTS)
* npm

Verify installation:

```bash
git --version
python3 --version
pip3 --version
node --version
npm --version
psql --version
```

---

# Clone the Repository

```bash
git clone https://github.com/<organization-or-username>/SPIRO.git
```

```bash
cd SPIRO
```

---

# Backend Setup

Move to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file inside the `backend` directory.

Example:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/spiro

SECRET_KEY=change_this_before_production

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

# PostgreSQL Setup

Install PostgreSQL.

Create the database:

```sql
CREATE DATABASE spiro;
```

Enable UUID support:

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

Run the schema:

```bash
psql -U postgres -d spiro -f ../database/schema.sql
```

Verify:

```sql
\dt
```

---

# Run Backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload
```

Backend:

```
http://127.0.0.1:8000
```

Swagger API Documentation:

```
http://127.0.0.1:8000/docs
```

---

# Frontend Setup

> **Coming Soon**

After the frontend is added:

```bash
cd frontend
npm install
npm run dev
```

---

# ML Service

> **Coming Soon**

Development will begin after backend APIs are finalized.

---

# Git Workflow

Do **not** commit directly to `main`.

Create a feature branch:

```bash
git checkout develop
```

```bash
git pull origin develop
```

```bash
git checkout -b feature/<feature-name>
```

Examples:

```text
feature/auth

feature/reports

feature/ml

feature/frontend
```

Commit your changes:

```bash
git add .
git commit -m "Describe your changes"
```

Push:

```bash
git push origin feature/<feature-name>
```

Create a Pull Request to `develop`.

---

# Current Status

## Completed

* Project structure
* Backend setup
* PostgreSQL setup
* Initial documentation

## In Progress

* Database schema
* REST API development

## Planned

* Authentication
* Waste reporting
* ML prediction
* Worker assignment
* Admin dashboard
* Deployment

---

# Development Notes

* Use Python virtual environments.
* Never commit `.env` files.
* Never commit `.venv`.
* Keep feature branches focused on a single task.
* Update documentation when adding new modules.

---

# License

This project is developed as part of the Bachelor of Engineering Final Year Project at Bangalore Institute of Technology.
