"""Artifact types of the Ebola scenario: protective equipment, a health center, remains."""

from typing import Tuple

from terralingua.environment.artifact import Artifact, register_artifact_type


@register_artifact_type("ppe")
class PPEArtifact(Artifact):
    """Protective equipment. Carrying it lowers the chance of catching the sickness."""

    description = "Protective equipment that lowers the chance of catching the sickness."

    def __init__(self, *args, protection: float = 0.1, **kwargs):
        self.protection = float(protection)
        super().__init__(*args, **kwargs)

    @property
    def actions(self) -> dict:
        return {}

    def interact(self, agent_name: str, action: str, params: dict, timestamp: int) -> str:
        return ""

    def passive_effect(self, timestamp: int, agent_name: str) -> str:
        return "Protective equipment. While you carry it, the sickness passes to you less easily."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""

    def serialize(self) -> dict:
        data = super().serialize()
        data["protection"] = self.protection
        return data

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.protection = float(data.get("protection", 0.1))
        return artifact


@register_artifact_type("health_center")
class HealthCenterArtifact(Artifact):
    """A fixed place. Sick beings near it may heal and die less often."""

    description = "A health center that cares for the sick beings near it."

    def __init__(
        self, *args, radius: int = 1, heal_probability: float = 0.0,
        hazard_multiplier: float = 0.5, **kwargs,
    ):
        self.radius = int(radius)
        self.heal_probability = float(heal_probability)
        self.hazard_multiplier = float(hazard_multiplier)
        kwargs["movable"] = False
        super().__init__(*args, **kwargs)

    @property
    def actions(self) -> dict:
        return {}

    def interact(self, agent_name: str, action: str, params: dict, timestamp: int) -> str:
        return ""

    def passive_effect(self, timestamp: int, agent_name: str) -> str:
        return str(self.payload) if self.payload else "A health center."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(
            radius=self.radius,
            heal_probability=self.heal_probability,
            hazard_multiplier=self.hazard_multiplier,
        )
        return data

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.radius = int(data.get("radius", 1))
        artifact.heal_probability = float(data.get("heal_probability", 0.0))
        artifact.hazard_multiplier = float(data.get("hazard_multiplier", 0.5))
        return artifact


@register_artifact_type("remains")
class RemainsArtifact(Artifact):
    """The remains of a being that died sick. They stay infectious until buried or gone."""

    description = "The remains of a dead being."

    def __init__(
        self, *args, strain: str = "virus", host_name: str = "", buriable_after: int = 0,
        infection_id: str = "", **kwargs,
    ):
        self.strain = str(strain)
        self.host_name = str(host_name)
        self.buriable_after = int(buriable_after)
        self.infection_id = str(infection_id)
        kwargs["movable"] = False
        super().__init__(*args, **kwargs)

    @property
    def actions(self) -> dict:
        return {}

    def interact(self, agent_name: str, action: str, params: dict, timestamp: int) -> str:
        return ""

    def passive_effect(self, timestamp: int, agent_name: str) -> str:
        return f"The remains of {self.host_name} lie here."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(
            strain=self.strain, host_name=self.host_name, buriable_after=self.buriable_after,
            infection_id=self.infection_id,
        )
        return data

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.strain = str(data.get("strain", "virus"))
        artifact.host_name = str(data.get("host_name", ""))
        artifact.buriable_after = int(data.get("buriable_after", 0))
        artifact.infection_id = str(data.get("infection_id", ""))
        return artifact
