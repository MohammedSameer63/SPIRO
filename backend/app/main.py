from fastapi import FastAPI

app = FastAPI(
    title="SPIRO API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "SPIRO Backend Running Successfully"
    }
