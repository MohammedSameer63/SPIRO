from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers.auth import router as auth_router
from app.routers.reports import router as reports_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SPIRO API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded waste images
app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads",
)


app.include_router(
    auth_router,
    prefix="/api/v1",
)


app.include_router(
    reports_router,
    prefix="/api/v1",
)


@app.get("/")
def home():
    return {
        "message": "SPIRO Backend API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }