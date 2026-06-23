import uuid


def generate_research_id() -> str:
    """
    Return a unique research session identifier.

    Attempts to use UUID version 7 (time-ordered, sortable) from the
    ``uuid6`` package.  Falls back to UUID version 4 when the package is
    not installed — the result is still unique, just not time-sortable.

    Install optional support:  uv pip install uuid6
    """
    try:
        from uuid6 import uuid7  # type: ignore[import-untyped]

        return str(uuid7())
    except ImportError:
        return str(uuid.uuid4())
