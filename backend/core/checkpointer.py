import os
import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("SUPABASE_DB_URI")

pool = AsyncConnectionPool(conninfo=DB_URI, max_size=10, open=False)

checkpointer: AsyncPostgresSaver | None = None   # abhi None, lifespan mein set hoga

def run_checkpointer_migrations():
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        PostgresSaver(conn).setup()

async def init_checkpointer():
    """Ye sirf lifespan ke andar call hoga, jab event loop already chal raha hoga."""
    global checkpointer
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()   # async version ka setup — tables migrate karega
    return checkpointer