import json
import logging
import redis
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize connection
try:
    r_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True
    )
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    r_client = None

def publish_event(channel: str, event_data: dict) -> bool:
    if r_client is None:
        logger.error("Redis client is not initialized, cannot publish event")
        return False
    try:
        payload = json.dumps(event_data)
        r_client.publish(channel, payload)
        logger.info(f"Published event to channel {channel}: {event_data}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish event to Redis: {e}")
        return False
