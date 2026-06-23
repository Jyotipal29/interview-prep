import re
import uuid

from src.shared.ids import generate_research_id

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def test_generate_research_id_returns_string() -> None:
    assert isinstance(generate_research_id(), str)


def test_generate_research_id_is_uuid_format() -> None:
    rid = generate_research_id()
    assert _UUID_RE.match(rid), f"Not a UUID: {rid!r}"


def test_generate_research_id_is_unique() -> None:
    ids = {generate_research_id() for _ in range(100)}
    assert len(ids) == 100


def test_generate_research_id_is_parseable() -> None:
    rid = generate_research_id()
    parsed = uuid.UUID(rid)
    assert str(parsed) == rid
