from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Set, Tuple

import numpy as np
import tiktoken

MAX_TEXT_ARTIFACT_SIZE = 500

# {max_text_artifact_size} is substituted with the configured limit when the
# create_artifact action is offered (env._get_avail_actions).
ARTIFACT_TYPE = {
    "text": "Any alfanumeric data stored in a physical marker. Maximum size is {max_text_artifact_size} tokens.",
}

ArtifactCreationError = ValueError


class Artifact:
    # Whether agents can act on the artifact: pick it up, drop it, give it
    # away or use the actions it offers. Non-interactable artifacts can only
    # be affected by the environment itself.
    interactable = True
    # Multiplier on the carrier's probability of contracting an infection
    infection_protection = 1.0

    def __init__(
        self,
        name: str,
        payload: Any,
        lifespan: int | float,
        pose: Tuple[int, int],
        creator: str,
        creation_time: int,
    ):
        self.name = name
        self.art_type = None
        valid, error_message = self.verify_payload(payload=payload)
        if valid:
            self.payload = payload
        else:
            raise ArtifactCreationError(
                f"Invalid payload for artifact type {self.art_type}: {error_message}"
            )
        self.pose = pose
        self.creator = creator
        self.lifespan = lifespan
        self.remaining_time = np.inf if lifespan == -1 else lifespan
        # Agents that interfaced with it. Just for tracking
        self.users: Dict[str, Set[int]] = defaultdict(set)
        self.creation_time = creation_time
        self.version_creation_time = creation_time
        self.deletion_time: int | None = None
        self.past_versions = []
        self.version = 0

    @property
    @abstractmethod
    def actions(self) -> dict:
        raise NotImplementedError("Must specify artifact actions")

    def serialize(self) -> dict:
        serialized = {
            "name": self.name,
            "art_type": self.art_type,
            "payload": self.payload,
            "lifespan": "inf" if self.lifespan == np.inf else self.lifespan,
            "pose": (int(self.pose[0]), int(self.pose[1])),
            "creator_tag": self.creator,
            "users_tag": {user: list(ts) for user, ts in self.users.items()},
            "creation_time": self.creation_time,
            "past_versions": self.past_versions,
            "version": self.version,
            "version_creation_time": self.version_creation_time,
        }
        if self.deletion_time is not None:
            serialized["deletion_time"] = self.deletion_time
        else:
            serialized["remaining_time"] = (
                "inf" if self.remaining_time == np.inf else self.remaining_time
            )
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        name = data["name"]
        payload = data["payload"]
        lifespan = np.inf if data["lifespan"] == "inf" else int(data["lifespan"])
        pose = (data["pose"][0], data["pose"][1])
        creator = data["creator_tag"]
        users = defaultdict(set)
        for user, ts in data["users_tag"].items():
            users[user] = set(ts)
        creation_time = data["creation_time"]
        if "deletion_time" in data:
            deletion_time = data["deletion_time"]
        else:
            remaining_time = (
                np.inf if data["remaining_time"] == "inf" else data["remaining_time"]
            )
        past_versions = data.get("past_versions", [])
        version = data.get("version", 0)

        artifact = cls(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        artifact.users = users
        artifact.deletion_time = deletion_time if "deletion_time" in data else None
        artifact.remaining_time = remaining_time if "remaining_time" in data else None
        artifact.past_versions = past_versions
        artifact.version = version
        return artifact

    @abstractmethod
    def interact(self, agent_name: str, action: str, params: dict, timestamp: int):
        raise NotImplementedError(
            "interact method not implemented for base Artifact class"
        )

    @abstractmethod
    def passive_effect(self, timestamp: int, agent_name: str):
        """This is the effect that the artifact has on the agents that just step on it"""
        raise NotImplementedError(
            "passive_effect method not implemented for base Artifact class"
        )

    @abstractmethod
    def verify_payload(self, payload) -> Tuple[bool, str]:
        """Verify that the payload is valid for the artifact type"""
        raise NotImplementedError(
            "verify_payload method not implemented for base Artifact class"
        )


class TextArtifact(Artifact):
    """An artifact that contains text.
    Agents can act on it by modifying its content or destroing the artifact.
    Passive effect: read the content
    """

    def __init__(
        self,
        name: str,
        payload: str,
        lifespan: int | float,
        pose: Tuple[int, int],
        creator: str,
        creation_time: int,
        max_size: int = MAX_TEXT_ARTIFACT_SIZE,
    ):
        self.payload_encoder = tiktoken.get_encoding("cl100k_base")
        self.max_size = max_size  # before super(): payload is validated there
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "text"
        self.creation_cost = 0

    @property
    def actions(self):
        return {
            f"destroy_artifact_{self.name}": {
                "description": f"Destroys {self.name} artifact",
                "params": {},
            },
            f"modify_artifact_{self.name}": {
                "description": f"Modifies the content of {self.name} artifact",
                "params": {
                    "payload": "New content of the artifact",
                    "lifespan": "New lifespan of the artifact",
                },
            },
        }

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        return f"Artifact {self.name} content: {self.payload}"

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        self.users[agent_name].add(timestamp)
        if action not in self.actions:
            return f"Unknown action: {action} - Available actions: {self.actions}"
        if action == f"modify_artifact_{self.name}":
            valid, error_message = self.verify_payload(params.get("payload", ""))
            if not valid:
                return f"Failed to modify artifact {self.name}: {error_message}"

            # Do not change the name. This ensure uniqueness of the artifacts
            past_version = {
                "payload": self.payload,
                "lifespan": "inf" if self.lifespan == np.inf else self.lifespan,
                "name": self.name,
                "version": self.version,
                "version_creation_time": self.version_creation_time,
            }
            self.past_versions.append(past_version)

            self.version += 1
            self.version_creation_time = timestamp
            self.payload = params.get("payload", "")
            self.lifespan = params.get("lifespan", self.lifespan)
            self.remaining_time = np.inf if self.lifespan == -1 else self.lifespan
            return f"Artifact {self.name} updated"
        if action == f"destroy_artifact_{self.name}":
            self.remaining_time = 0
            return f"Artifact {self.name} destroyed"
        return ""

    def verify_payload(self, payload) -> Tuple[bool, str]:
        payload = str(payload)
        if not isinstance(payload, str):
            return False, "Payload must be a string for TextArtifact"
        token_count = len(self.payload_encoder.encode(payload))
        if token_count > self.max_size:
            return (
                False,
                f"Payload exceeds maximum token limit of {self.max_size} tokens (got {token_count} tokens)",
            )
        return True, ""

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["max_size"] = self.max_size
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        # Construct with an empty payload so a payload larger than the default
        # limit survives a checkpoint written under a larger configured limit.
        data = dict(data)
        payload = data["payload"]
        data["payload"] = ""
        artifact = super().deserialize(data)
        artifact.max_size = data.get("max_size", MAX_TEXT_ARTIFACT_SIZE)
        artifact.payload = payload
        return artifact


class PPEArtifact(Artifact):
    """Personal protective equipment.

    While an agent carries one, its probability of contracting a viral
    artifact is multiplied by ``infection_protection`` (< 1). Seeded by the
    environment only — agents cannot create it, but they can pick it up,
    drop it or give it away, and it drops on the map when its host dies.
    """

    def __init__(
        self,
        name: str = "PPE",
        payload: str = "Personal protective equipment: Whoever carries it is far less likely to catch infections from others.",
        lifespan: int | float = -1,
        pose: Tuple[int, int] = (0, 0),
        creator: str = "environment",
        creation_time: int = 0,
        protection: float = 0.1,
    ):
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "ppe"
        self.infection_protection = protection

    @property
    def actions(self):
        return {}

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        return f"Artifact {self.name}: {self.payload}"

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        return f"Artifact {self.name} offers no actions."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["protection"] = self.infection_protection
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.infection_protection = data.get("protection", 0.1)
        return artifact


class HealthCenterArtifact(Artifact):
    """A fixed treatment site on the grid.

    Immovable: agents cannot pick it up, drop it, give it away or act on it.
    Always active within ``radius`` cells (Chebyshev; the default 1 covers its
    cell plus the 8 around it): every infected agent there has a per-step
    ``heal_probability`` of losing its viral artifacts, and every sick one
    dies at ``hazard_multiplier`` x the usual hazard — supportive care, which
    improves the odds of surviving the illness rather than curing it.
    """

    interactable = False

    def __init__(
        self,
        name: str,
        payload: str = "A health center. Sick beings near it receive care: they are more likely to survive the sickness.",
        lifespan: int | float = -1,
        pose: Tuple[int, int] = (0, 0),
        creator: str = "environment",
        creation_time: int = 0,
        heal_probability: float = 0.2,
        hazard_multiplier: float = 1.0,
        radius: int = 1,
    ):
        super().__init__(
            name=name,
            payload=payload,
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "health_center"
        self.heal_probability = float(heal_probability)
        self.hazard_multiplier = float(hazard_multiplier)
        self.radius = int(radius)

    @property
    def actions(self):
        return {}

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        return f"Artifact {self.name}: {self.payload}"

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        return f"Artifact {self.name} cannot be acted upon."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        return True, ""

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["heal_probability"] = self.heal_probability
        serialized["hazard_multiplier"] = self.hazard_multiplier
        serialized["radius"] = self.radius
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.heal_probability = float(data.get("heal_probability", 0.2))
        artifact.hazard_multiplier = float(data.get("hazard_multiplier", 1.0))
        artifact.radius = int(data.get("radius", 1))
        return artifact


class ViralArtifact(Artifact):
    """A virus-like artifact that lives only inside agent inventories.

    It has no content and cannot be created by agents: the environment adds it
    directly to an agent's inventory. Agents cannot act on it in any way
    (no pickup/drop/give/modify/destroy). Each step the environment can spread
    copies of it to nearby agents, and hosts consume energy faster. The
    infection ends when the artifact's lifespan runs out. When its host dies,
    it is dropped on the map for a set amount of time, during which it keeps
    spreading to nearby agents.

    An infection has two phases. While ``incubation`` steps remain the host is
    a silent carrier: it behaves normally, pays no extra energy and infects
    nobody. Once the countdown reaches zero the artifact is ``symptomatic`` and
    everything above applies. ``lifespan`` measures the symptomatic period
    only — ``remaining_time`` does not tick while the artifact is incubating —
    so the infectious window is the same length whatever the latency was.
    """

    interactable = False

    def __init__(
        self,
        name: str,
        lifespan: int | float,
        pose: Tuple[int, int],
        creator: str,
        creation_time: int,
        strain: str | None = None,
        incubation: int = 0,
        payload: Any = "",  # ignored, viral artifacts have no content
    ):
        super().__init__(
            name=name,
            payload="",
            lifespan=lifespan,
            pose=pose,
            creator=creator,
            creation_time=creation_time,
        )
        self.art_type = "viral"
        # Name of the strain this artifact belongs to. Copies keep the strain
        # of their source, so an agent cannot host the same strain twice.
        self.strain = strain if strain is not None else name
        # Steps left before the host develops symptoms. 0 means symptomatic now.
        self.incubation = incubation
        # Symptomatic steps already lived through. The environment compares it
        # to viral_mobile_days to split the illness into an early ambulatory
        # ("dry") phase and the bedridden ("wet") one.
        self.days_symptomatic = 0
        # Set at the host's death: last step of the mourning period, before
        # which the remains refuse burial. -1 means always buriable.
        self.buriable_after = -1
        # Set when the host dies: how the artifact presents to agents and the
        # dashboard ("remains_of_<host>"). The internal name never changes, so
        # transmission chains keyed on it stay intact.
        self.display_name: str | None = None

    @property
    def symptomatic(self) -> bool:
        """Whether the infection has run its incubation and shows symptoms."""
        return self.incubation <= 0

    @property
    def actions(self):
        # Hosts cannot act on a viral artifact
        return {}

    def passive_effect(self, timestamp: int, agent_name: str):
        self.users[agent_name].add(timestamp)
        return f"Artifact {self.name} cannot be interacted with."

    def interact(
        self, agent_name: str, action: str, params: dict, timestamp: int
    ) -> str:
        return f"Artifact {self.name} cannot be acted upon."

    def verify_payload(self, payload) -> Tuple[bool, str]:
        # Viral artifacts carry no content
        return True, ""

    def spawn_copy(
        self, name: str, pose: Tuple[int, int], timestamp: int, incubation: int = 0
    ):
        """Creates the copy of this artifact that infects a new host.

        The copy draws its own incubation rather than inheriting the source's:
        how long the last host took to fall ill says nothing about this one.
        """
        return ViralArtifact(
            name=name,
            lifespan=self.lifespan,
            pose=pose,
            creator=self.creator,
            creation_time=timestamp,
            strain=self.strain,
            incubation=incubation,
        )

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["strain"] = self.strain
        serialized["incubation"] = self.incubation
        serialized["days_symptomatic"] = self.days_symptomatic
        serialized["buriable_after"] = self.buriable_after
        if self.display_name:
            serialized["display_name"] = self.display_name
        return serialized

    @classmethod
    def deserialize(cls, data: dict):
        artifact = super().deserialize(data)
        artifact.strain = data.get("strain", artifact.name)
        # Runs predating the incubation phase have none: they were symptomatic
        # from the moment they were created.
        artifact.incubation = data.get("incubation", 0)
        artifact.days_symptomatic = data.get("days_symptomatic", 0)
        artifact.buriable_after = data.get("buriable_after", -1)
        artifact.display_name = data.get("display_name")
        return artifact
