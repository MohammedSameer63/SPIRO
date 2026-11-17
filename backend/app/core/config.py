from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    project_name: str = "SPIRO REST API"
    api_version: str = "1.0"
    api_prefix: str = "/api/v1"
    debug: bool = True

    # Database
    database_url: str

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # File Uploads
    upload_dir: str = "uploads"
    max_image_size_mb: int = 10

    # CORS — comma-separated frontend origins.
    # Local dev covers `npm run dev`; add the hosted frontend domain.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Extra safety net for phone/LAN demos: any http(s) origin on the
    # Vite dev port is accepted (covers every hotspot without edits).
    # Tighten this (or leave empty) for real production deployments.
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?"

    # ML Models
    ml_models_dir: str = "models"
    yolo_onnx_path: str = "models/exports/spiro_yolov11n_best.onnx"
    effnet_onnx_path: str = "models/exports/spiro_effnetv2_s_best_effnet.onnx"
    pipeline_config_path: str = "configs/pipeline/inference_pipeline.yaml"
    ml_confidence_threshold: float = 0.25
    ml_enabled: bool = True
    ml_mock_mode: bool = False  # Real YOLO model available

    # Internal Services
    ml_service_api_key: str

    # Household QR validation (HMAC signing of QR payloads)
    qr_secret_key: str = "spiro-dev-qr-secret-change-before-production"

    # Green coins (citizen rewards)
    coins_per_completion: int = 10
    coins_qr_bonus: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
