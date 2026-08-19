import contextlib
from typing import AsyncGenerator
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.core.config import settings

@contextlib.asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """
    Yields an active PostgreSQL checkpointer.
    Runs checkpointer.setup() on startup to ensure database tables are created.
    """
    async with AsyncPostgresSaver.from_conn_string(settings.POSTGRES_URI) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
