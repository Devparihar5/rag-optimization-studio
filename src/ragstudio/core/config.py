from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RAG Optimization Studio"
    environment: str = "development"
    max_experiments: int = 200


settings = Settings()
