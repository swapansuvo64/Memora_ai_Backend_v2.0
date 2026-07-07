import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.gallery_routes import router as gallery_router
from config.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database connection...")
    await init_db()
    yield

app = FastAPI(
    title="MemoraAI User Service",
    description="Gallery management and Upload endpoint service for MemoraAI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gallery_router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "user-service"}
