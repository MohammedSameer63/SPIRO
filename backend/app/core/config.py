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


    # ML Models
    yolo_model_path: str = "models/spiro_yolo11n.pt"
    efficientnet_model_path: str = "models/efficientnetv2.pth"

    # Internal Services
    ml_service_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
