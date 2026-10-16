from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, TypeVar


class JsonRecord(Protocol):
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRecord: ...


RecordT = TypeVar("RecordT", bound=JsonRecord)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def read_jsonl(path: str | Path, model: type[RecordT]) -> list[RecordT]:
    source = Path(path)
    records: list[RecordT] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(payload, dict):
                raise ValueError("record must be a JSON object")
            records.append(model.from_dict(payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSONL record at {source}:{line_number}: {error}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[JsonRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), allow_nan=False, sort_keys=True) + "\n")
    return destination
