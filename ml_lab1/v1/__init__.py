from v1.simple_router import simple_router
from fastapi import APIRouter

router = APIRouter(prefix="/api")
router.include_router(simple_router)