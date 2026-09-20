from dotenv import load_dotenv
import os
import ssl
import certifi

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")

print("MongoDB URL loaded:", bool(MONGODB_URL))

from motor.motor_asyncio import AsyncIOMotorClient
from backend.config import settings
import logging

logger = logging.getLogger("backend.database")

class Database:
    client: AsyncIOMotorClient = None
    db = None
    is_connected: bool = False

db_instance = Database()

async def connect_to_mongo():
    try:
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}...")
        # First try with certifi CA bundle (recommended for Atlas)
        try:
            db_instance.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                tls=True,
                tlsCAFile=certifi.where(),
            )
            db_instance.db = db_instance.client[settings.DATABASE_NAME]
            await db_instance.client.admin.command('ping')
        except Exception as ssl_err:
            logger.warning(f"[WARNING] Primary TLS connection failed: {ssl_err}. Retrying with relaxed TLS settings...")
            # Fallback: relax TLS for older Python SSL stacks on Windows
            try:
                db_instance.client = AsyncIOMotorClient(
                    settings.MONGODB_URL,
                    serverSelectionTimeoutMS=8000,
                    connectTimeoutMS=8000,
                    tls=True,
                    tlsAllowInvalidCertificates=True,
                    tlsInsecure=True,
                )
                db_instance.db = db_instance.client[settings.DATABASE_NAME]
                await db_instance.client.admin.command('ping')
            except Exception as tls_err2:
                logger.warning(f"[WARNING] Relaxed TLS also failed: {tls_err2}. Trying without TLS...")
                db_instance.client = AsyncIOMotorClient(
                    settings.MONGODB_URL,
                    serverSelectionTimeoutMS=8000,
                    connectTimeoutMS=8000,
                )
                db_instance.db = db_instance.client[settings.DATABASE_NAME]
                await db_instance.client.admin.command('ping')

        db_instance.is_connected = True
        logger.info(f"Successfully connected to MongoDB. Database: {settings.DATABASE_NAME}")
    except Exception as e:
        db_instance.is_connected = False
        logger.warning(
            f"[WARNING] Could not connect to MongoDB: {e}\n"
            "   The API server will still start, but database-dependent endpoints will run in offline mode.\n"
            "   To fix: check your MONGODB_URL and network/firewall settings."
        )
        # Do NOT re-raise — allow the app to start in degraded mode

async def close_mongo_connection():
    if db_instance.client:
        logger.info("Closing MongoDB connection...")
        db_instance.client.close()
        logger.info("MongoDB connection closed.")

async def get_db():
    """Returns the DB instance, or None if unavailable (callers handle gracefully)."""
    # Lazy reconnect: if startup connection failed or reload race occurred, try once more
    if not db_instance.is_connected or db_instance.db is None:
        await connect_to_mongo()
    if not db_instance.is_connected or db_instance.db is None:
        logger.warning("get_db() called but database is unavailable — returning None.")
        return None
    return db_instance.db
