import logging
from supabase._async.client import create_client, AsyncClient
from config.settings import settings

logger = logging.getLogger(__name__)

supabase_client: AsyncClient = None

async def init_db() -> AsyncClient:
    global supabase_client
    try:
        logger.info(f"Initializing async Supabase client for {settings.SUPABASE_URL}...")
        supabase_client = await create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        # Test connection by making a small select query
        await supabase_client.table("users").select("id").limit(1).execute()
        logger.info("Async Supabase client initialized successfully")
        return supabase_client
    except Exception as e:
        logger.error(f"Database connection error: failed to initialize Supabase client. Details: {e}")
        raise RuntimeError(f"Database connection error: failed to initialize Supabase client. Details: {str(e)}") from e

async def get_db() -> AsyncClient:
    global supabase_client
    if supabase_client is None:
        await init_db()
    return supabase_client
