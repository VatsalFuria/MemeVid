from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness check (PROJECT_SPEC.md §8 API sketch)."""
    return {"status": "ok"}