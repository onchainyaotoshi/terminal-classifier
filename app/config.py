import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.port: int = int(os.getenv("PORT", "8980"))
        self.api_key: str = os.getenv("API_KEY", "")
        self.cpu_cores: int = int(os.getenv("CPU_CORES", "1"))


settings = Settings()

# Set thread limits for PyTorch
os.environ["OMP_NUM_THREADS"] = str(settings.cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(settings.cpu_cores)
