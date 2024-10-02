import uvicorn
from fastapi import FastAPI
from src.core.settings import settings
from src.core.models.classificator import *
from v1 import router
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.fastapi_settings.host, port=settings.fastapi_settings.port)