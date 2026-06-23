from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.shared.logging import get_logger

logger = get_logger(__name__)


def get_checkpointer(database_url: str | None = None) -> BaseCheckpointSaver:
    """
    Return a checkpointer appropriate for the current environment.

    When *database_url* is provided the function attempts to connect to
    Postgres and returns a PostgresSaver.  If the connection fails, or the
    required packages are not installed, it logs a warning and falls back
    to an in-process MemorySaver so the graph can still run without a DB.
    """
    if database_url:
        try:
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = psycopg.connect(database_url, autocommit=True)
            saver: BaseCheckpointSaver = PostgresSaver(conn)
            saver.setup()  # type: ignore[attr-defined]
            logger.info("checkpointing: connected to Postgres at %s", database_url)
            return saver

        except ImportError:
            logger.warning(
                "checkpointing: postgres packages not installed "
                "(install with: uv pip install '.[postgres]'); "
                "falling back to MemorySaver"
            )
        except Exception as exc:
            logger.warning(
                "checkpointing: failed to connect to Postgres (%s); "
                "falling back to MemorySaver",
                exc,
            )

    logger.info("checkpointing: using in-memory MemorySaver (no persistence)")
    return MemorySaver()
