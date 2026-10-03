from __future__ import annotations

import json
from dataclasses import dataclass

KINDS = {"text", "code", "log", "json"}


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _fields(data: dict, allowed: set[str], where: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Unknown {where} fields: {', '.join(sorted(extra))}")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    kind: str = "text"
    pinned: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Each chunk needs a nonempty string id")
        if not isinstance(self.text, str):
            raise ValueError(f"Chunk {self.id!r} text must be a string")
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError(f"Chunk {self.id!r} kind must be text, code, log, or json")
        if type(self.pinned) is not bool:
            raise ValueError(f"Chunk {self.id!r} pinned must be a boolean")
        if self.source is not None and not isinstance(self.source, str):
            raise ValueError(f"Chunk {self.id!r} source must be a string")

    def to_dict(self) -> dict:
        result = {"id": self.id, "text": self.text, "kind": self.kind, "pinned": self.pinned}
        if self.source is not None:
            result["source"] = self.source
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        if not isinstance(data, dict):
            raise ValueError("Each chunk must be an object")
        _fields(data, {"id", "text", "kind", "pinned", "source"}, "chunk")
        if "id" not in data or "text" not in data:
            raise ValueError("Each chunk requires id and text")
        return cls(**data)


@dataclass(frozen=True)
class ContextPacket:
    query: str
    chunks: tuple[Chunk, ...] = ()
    instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ValueError("query must be a string")
        if not isinstance(self.chunks, (tuple, list)) or not all(
            isinstance(chunk, Chunk) for chunk in self.chunks
        ):
            raise ValueError("chunks must contain Chunk objects")
        if not isinstance(self.instructions, (tuple, list)) or not all(
            isinstance(item, str) for item in self.instructions
        ):
            raise ValueError("instructions must be a list of strings")
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        ids = [chunk.id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk ids must be unique")

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "instructions": list(self.instructions),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def render(self) -> str:
        return serialize(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> ContextPacket:
        if not isinstance(data, dict):
            raise ValueError("Context packet must be an object")
        _fields(data, {"query", "instructions", "chunks"}, "packet")
        if "query" not in data or not isinstance(data.get("chunks", []), list):
            raise ValueError("Packet requires query and a chunks array")
        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            raise ValueError("instructions must be a list of strings")
        return cls(
            query=data["query"],
            instructions=tuple(instructions),
            chunks=tuple(Chunk.from_dict(item) for item in data.get("chunks", [])),
        )
