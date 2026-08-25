from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar


@dataclass
class Genome:
    genome_type: ClassVar[str] = "base"

    def as_dict(self) -> dict[str, float]:
        raise NotImplementedError()

    def from_dict(self, data: dict[str, float]) -> "Genome":
        raise NotImplementedError()

    def as_string(self) -> str:
        raise NotImplementedError()

    def mutate(self, **kwargs) -> "Genome":
        raise NotImplementedError()

    @classmethod
    def random(cls) -> "Genome":
        raise NotImplementedError()
