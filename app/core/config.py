import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1"
    )

    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    CHROMA_PATH = os.getenv(
        "CHROMA_PATH", "./storage/chroma"
    )

    DATABASE_URL = os.getenv("DATABASE_URL")

settings = Settings()