from fastapi import FastAPI

from app.api.routes import health
from app.config import get_settings

settings = get_settings()

app = FastAPI(title="memevid", version="0.1.0")

app.include_router(health.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "memevid API is running", "environment": settings.environment}