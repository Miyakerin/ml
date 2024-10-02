import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
print("loading dotenv")
load_dotenv(".env")

class FastAPISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FASTAPI_", extra="ignore")
    port: int = 5050
    host: str = "0.0.0.0"


class Settings(BaseSettings):
    fastapi_settings: FastAPISettings = FastAPISettings()


settings = Settings()