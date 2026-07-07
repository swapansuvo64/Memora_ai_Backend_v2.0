import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.agent_routes import router as agent_router
from config.db import init_db
from worker import RedisEventWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize background Redis subscriber worker
worker = RedisEventWorker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database connection...")
    await init_db()
    logger.info("Starting background event subscription loop...")
    worker.start()
    yield
    logger.info("Stopping background event subscription loop...")
    worker.stop()

app = FastAPI(
    title="MemoraAI Agent Service",
    description="Face recognition and Bedrock RAG vector search agent service for MemoraAI",
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

app.include_router(agent_router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "agent-service"}
