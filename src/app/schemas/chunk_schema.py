from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "text": self.text, "metadata": self.metadata}