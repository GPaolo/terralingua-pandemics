"""The Ebola mechanic: infection, sickness, care, death, remains, and burial.

Every rule runs inside the three Mechanic hooks. The world never learns what an
infection is. Infection state lives in `self.state`, which the world saves and
restores with its checkpoint.

An infection has three phases. It incubates silently, then the host is
feverish and still able to act, then bedridden. Transmission happens by
standing near a sick host or unburied remains, by giving or taking energy or
handing over an artifact with a sick host, and by burying remains or standing
beside the grave. Protective equipment lowers the chance. A health center may
heal and lowers the death hazard. Recovery gives immunity.
"""

import json
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field, model_validator
from terralingua.environment.mechanic import Mechanic

from pandemics.artifacts import HealthCenterArtifact, PPEArtifact, RemainsArtifact
from pandemics.state_log import WorldStateLogger

FEVERISH_NOTICE = (
    "You have fallen ill: you are feverish and weak, and something is "
    "draining your energy. You can still move, eat and act."
)
BEDRIDDEN_NOTICE = (
    "You are sick. Something is living inside you and draining your energy. "
    "You are too weak to move or to take energy from others, and you have no "
    "appetite: food no longer restores you. You can still speak, and other "
    "beings can still give you energy."
)
NO_APPETITE_NOTICE = (
    "You are standing on food but you have no appetite. It gave you nothing."
)
RECOVERED_NOTICE = (
    "You have survived the sickness and recovered. Your body now resists "
    "it: you cannot catch the sickness again, even from the sick or from "
    "remains."
)

# Every being-to-being exchange is contact: an adjacent being, not merely one in view.
CONTACT_TEXT = {
    "give": (
        "Transfer some of your energy to a being on a cell adjacent to yours. This is physical contact.",
        "target",
        "Name of a being on a cell next to yours to give energy to.",
    ),
    "take": (
        "Steal energy from a being on a cell adjacent to yours. This is physical contact.",
        "target",
        "Name of a being on a cell next to yours to steal energy from.",
    ),
    "give_artifact": (
        "Gives an artifact from the inventory to a being on a cell adjacent to yours. This is physical contact.",
        "target_agent",
        "Name of a being on a cell next to yours to give the artifact to.",
    ),
}


class HealthCenterOptions(BaseModel):
    """One health center, seeded at the start."""

    model_config = ConfigDict(extra="forbid")

    pose: list[int] | str = Field(
        [25, 25], description="Grid cell [row, col] or graph node id."
    )
    radius: int = Field(
        1, ge=0, description="Distance within which beings receive care."
    )
    heal_probability: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Chance per step that a sick being within reach heals.",
    )
    hazard_multiplier: float = Field(
        0.5,
        ge=0,
        description="Multiplier on the daily death chance of sick beings within reach.",
    )
    name: str = "Health Center"
    payload: str = (
        "A health center. Sick beings near it receive supportive care: they are "
        "far more likely to survive the sickness, but it does not cure. Recovery "
        "still takes its time."
    )

    @model_validator(mode="after")
    def _cell_has_two_coordinates(self):
        if isinstance(self.pose, list) and len(self.pose) != 2:
            raise ValueError("health_center.pose must be [row, col] or a node id")
        return self


class EpidemicOptions(BaseModel):
    """Settings of the Ebola scenario, given as run.scenario_options."""

    model_config = ConfigDict(extra="forbid")

    init_infected: int = Field(1, ge=0, description="Beings infected at the outbreak.")
    outbreak_step: int = Field(0, ge=0, description="Step of the outbreak.")
    incubation_min: int = Field(2, ge=0, description="Shortest silent phase, in steps.")
    incubation_max: int = Field(15, ge=0, description="Longest silent phase, in steps.")
    infection_duration: int = Field(
        13, ge=-1, description="Sick steps until recovery. -1 means never."
    )
    mobile_days: int = Field(
        4, ge=0, description="Sick steps during which the host can still move."
    )
    feverish_multiplier: float = Field(
        0.3,
        ge=0,
        description="Multiplier on infection_probability while the host is feverish and still mobile. Bedridden hosts and remains spread at the full chance.",
    )
    infection_radius: int = Field(
        1, ge=0, description="Distance at which the sickness passes between beings."
    )
    infection_probability: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Base chance, per step and per source, that a being within infection_radius of a bedridden host or of unburied remains catches it. Every other exposure multiplies this chance.",
    )
    contact_multiplier: float = Field(
        1.8,
        ge=0,
        description="Multiplier on infection_probability for the extra exposure when energy or an artifact changes hands with a sick host.",
    )
    energy_multiplier: float = Field(
        6.0,
        ge=1,
        description="Energy a sick host loses per step, as a multiple of the normal loss.",
    )
    case_fatality: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Share of the sick who die of it. The daily chance is derived from this and rises over the sick period, so deaths come late. With infection_duration -1 it is the chance per bedridden step instead.",
    )
    ppe_protection: float = Field(
        0.1,
        ge=0,
        le=1,
        description="Multiplier on the infection chance of a being carrying protective equipment.",
    )
    ppe_role: str = Field(
        "health_worker",
        description="Persona role whose beings start with protective equipment.",
    )
    ppe_per_worker: int = Field(
        20,
        ge=0,
        description="Protective equipment items each being of that role starts with.",
    )
    burials: bool = Field(True, description="Beings next to remains may bury them.")
    burial_infection_multiplier: float = Field(
        1.9,
        ge=0,
        description="Multiplier on infection_probability for the one exposure the burier takes handling the remains.",
    )
    burial_bystander_multiplier: float = Field(
        1.5,
        ge=0,
        description="Multiplier on infection_probability for the one exposure each being next to the remains takes when they are buried, whether or not it meant to attend.",
    )
    funeral_announcements: bool = Field(
        True,
        description="Announce a death that leaves remains to beings within earshot.",
    )
    funeral_announcement_radius: int = Field(
        10, ge=-1, description="How far the announcement reaches. -1 means everyone."
    )
    funeral_mourning_days: int = Field(
        0, ge=0, description="Steps during which remains refuse burial."
    )
    remains_lifespan: int = Field(
        10,
        ge=-1,
        description="Steps that remains stay on the ground. -1 means forever.",
    )
    health_center: HealthCenterOptions | None = Field(
        None, description="A health center to seed at the start, or null."
    )
    personas_path: str = Field(
        "personas.json",
        description="JSON list of personas with name, persona, count, and role. Relative to the scenario folder.",
    )

    @model_validator(mode="after")
    def _ordered(self):
        if self.incubation_min > self.incubation_max:
            raise ValueError("incubation_min cannot exceed incubation_max")
        return self


class Epidemic(Mechanic):
    name = "epidemic"

    def __init__(self, options: EpidemicOptions):
        super().__init__()
        self.options = options
        self.state.update(
            {
                "infections": {},
                "recovered": {},
                "count": 0,
                "reminders": [],
                "seeded": False,
                "identities": {},
                "assigned": 0,
            }
        )
        self._personas: List[dict] | None = None
        self.world_log: WorldStateLogger | None = None

    # ---------- identity ----------
    def personas(self) -> List[dict]:
        """The persona entries of the file, expanded by count, in file order."""
        if self._personas is None:
            path = Path(self.options.personas_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if not path.exists():
                raise FileNotFoundError(f"personas file not found: {path}")
            entries = json.loads(path.read_text())
            expanded = []
            for entry in entries:
                if isinstance(entry, str):
                    entry = {"persona": entry}
                for i in range(int(entry.get("count", 1))):
                    expanded.append(
                        {
                            "persona": str(entry["persona"]),
                            "name": entry.get("name") if i == 0 else None,
                            "role": entry.get("role"),
                        }
                    )
            self._personas = expanded
        return self._personas

    def human_name(self, env, tag: str) -> str:
        """A human first name, fixed by the tag, unique among the beings so far."""
        used = {i["name"] for i in self.state["identities"].values()} | set(
            env.agent_names.values()
        )
        fake = Faker()
        fake.seed_instance(zlib.crc32(tag.encode()))
        name = fake.first_name()
        while name in used:
            name = fake.first_name()
        return name

    def identity(self, env, tag: str) -> dict:
        """Personas go to the first beings created, in file order; everyone gets a human name."""
        ident = self.state["identities"].get(tag)
        if ident is None:
            index = self.state["assigned"]
            self.state["assigned"] = index + 1
            personas = self.personas()
            entry = personas[index] if index < len(personas) else {}
            ident = {
                "name": entry.get("name") or self.human_name(env, tag),
                "persona": entry.get("persona", ""),
                "role": entry.get("role"),
            }
            self.state["identities"][tag] = ident
            if env.logger is not None:
                env.logger.log(
                    time=env.step_count,
                    event_type="IDENTITY",
                    agent_tag=tag,
                    agent_name=ident["name"],
                    role=ident["role"],
                    persona=ident["persona"],
                )
        return {"name": ident["name"], "persona": ident["persona"]}

    def appetite(self, env) -> None:
        """Bedridden hosts leave food untouched; tell them when they stand on some."""
        for tag in env.agent_registry:
            if self.health(tag) == "bedridden":
                env.no_appetite.add(tag)
                if env.agent_pos[tag] in env.food:
                    env.note(tag, "Action outcome", NO_APPETITE_NOTICE)
            else:
                env.no_appetite.discard(tag)

    # ---------- queries ----------
    def health(self, tag: str) -> str:
        """healthy, incubating, feverish, or bedridden."""
        infection = self.state["infections"].get(tag)
        if infection is None:
            return "healthy"
        if infection["incubation"] > 0:
            return "incubating"
        if infection["days_symptomatic"] < self.options.mobile_days:
            return "feverish"
        return "bedridden"

    def symptomatic(self, tag: str) -> bool:
        return self.health(tag) in ("feverish", "bedridden")

    def susceptible(self, env, tag: str) -> bool:
        return (
            tag in env.agent_registry
            and tag not in self.state["infections"]
            and tag not in self.state["recovered"]
        )

    def protection(self, env, tag: str) -> float:
        """Lowest protection factor among the carried protective equipment. 1 means none."""
        factors = [
            env.artifacts[name].protection
            for name in env.agent_inventories.get(tag, ())
            if isinstance(env.artifacts.get(name), PPEArtifact)
        ]
        return min(factors) if factors else 1.0

    def remains_on_map(self, env) -> Dict[str, tuple]:
        """Name -> position of every remains artifact lying on the map."""
        found = {}
        for name, artifact in env.artifacts.items():
            location = env.artifact_location.get(name)
            if (
                isinstance(artifact, RemainsArtifact)
                and artifact.remaining_time > 0
                and location
                and location[0] == "map"
            ):
                found[name] = location[1]
        return found

    def remains_in_reach(self, env, tag: str) -> List[str]:
        origin = env.agent_pos[tag]
        return sorted(
            name
            for name, pos in self.remains_on_map(env).items()
            if env.distance(origin, pos) <= 1
        )

    @staticmethod
    def rng(env) -> np.random.Generator:
        if env.rng is None:
            env.rng = np.random.default_rng()
        return env.rng

    def compass(self, env, origin, target) -> str:
        """Where the target lies, seen from the origin, in plain words."""
        if not (isinstance(origin, tuple) and isinstance(target, tuple)):
            return f"at {target}"
        size = getattr(env, "grid_size", None)
        dr, dc = target[0] - origin[0], target[1] - origin[1]
        if size:
            if dr > size // 2:
                dr -= size
            elif dr < -(size // 2):
                dr += size
            if dc > size // 2:
                dc -= size
            elif dc < -(size // 2):
                dc += size
        parts = []
        if dr:
            parts.append(
                f"{abs(dr)} cell{'s' if abs(dr) != 1 else ''} {'up' if dr < 0 else 'down'}"
            )
        if dc:
            parts.append(
                f"{abs(dc)} cell{'s' if abs(dc) != 1 else ''} {'right' if dc > 0 else 'left'}"
            )
        return " and ".join(parts) if parts else "right here"

    # ---------- state changes ----------
    def infect(
        self,
        env,
        tag: str,
        source_tag: str | None,
        source_kind: str,
        strain: str = "virus",
    ):
        if not self.susceptible(env, tag):
            return None
        parent = self.source_infection(env, source_tag, source_kind)
        self.state["count"] += 1
        infection_id = f"{strain}_i{self.state['count']}"
        o = self.options
        incubation = int(self.rng(env).integers(o.incubation_min, o.incubation_max + 1))
        self.state["infections"][tag] = {
            "id": infection_id,
            "strain": strain,
            "incubation": incubation,
            "days_symptomatic": 0,
            "acquired_at": env.step_count,
            "source": source_tag,
        }
        env.logger.log(
            time=env.step_count,
            event_type="VIRAL_INFECTION",
            agent_tag=tag,
            agent_name=env.agent_names[tag],
            source_tag=source_tag,
            source_name=env.agent_names.get(source_tag) if source_tag else None,
            source_kind=source_kind,
            source_infection_id=parent,
            strain=strain,
            infection_id=infection_id,
            incubation=incubation,
        )
        return infection_id

    def source_infection(
        self, env, source_tag: str | None, source_kind: str
    ) -> str | None:
        """The id of the infection a new one comes from: the host's, or the one the remains carry."""
        if source_tag is not None:
            source = self.state["infections"].get(source_tag)
            return source["id"] if source else None
        if ":" in source_kind:
            remains = env.artifacts.get(source_kind.split(":", 1)[1])
            return getattr(remains, "infection_id", None) or None
        return None

    def recover(self, env, tag: str, cause: str) -> None:
        known = self.symptomatic(tag)
        infection = self.state["infections"].pop(tag)
        self.state["recovered"][tag] = known
        env.logger.log(
            time=env.step_count,
            event_type="VIRAL_HEALED",
            agent_tag=tag,
            agent_name=env.agent_names[tag],
            infection_id=infection["id"],
            cause=cause,
            known=known,
        )

    def expose(
        self, env, tag: str, probability: float, source_tag, source_kind: str
    ) -> bool:
        """One chance to catch the sickness. Logged, so the analysis counts exposures exactly."""
        if not self.susceptible(env, tag):
            return False
        protection = self.protection(env, tag)
        infected = self.rng(env).random() < probability * protection
        env.logger.log(
            time=env.step_count,
            event_type="VIRAL_EXPOSURE",
            agent_tag=tag,
            agent_name=env.agent_names[tag],
            source_tag=source_tag,
            source_kind=source_kind,
            probability=float(probability),
            protection=float(protection),
            infected=bool(infected),
        )
        if infected:
            self.infect(env, tag, source_tag, source_kind)
        return bool(infected)

    # ---------- hooks ----------
    def on_menu(self, env, tag: str, menu: dict) -> dict:
        health = self.health(tag)
        adjacent = sorted(env.agent_names[t] for t in env.agents_within(tag, 1))
        for action, (description, param, param_text) in CONTACT_TEXT.items():
            if action not in menu:
                continue
            if not adjacent:
                menu.pop(action)
                continue
            menu[action]["description"] = description
            menu[action]["params"][param] = {
                "description": param_text,
                "choices": adjacent,
            }
        if health == "bedridden":
            menu.pop("take", None)
            if "move" in menu:
                menu["move"] = {
                    "description": "You are too sick to move. You can only stay where you are.",
                    "params": {
                        "direction": "Must be 'stay': you are too sick to move."
                    },
                }
            return menu
        if health == "feverish" and "move" in menu:
            menu["move"]["description"] += (
                " You are feverish and weak, but still able to walk."
            )
        if self.options.burials:
            names = self.remains_in_reach(env, tag)
            if names:
                menu["bury"] = {
                    "description": (
                        "Bury remains lying on the ground next to you, removing them from the world. "
                        "Handling remains is risky: you may catch the sickness."
                    ),
                    "params": {
                        "name": {
                            "description": "Name of the remains artifact to bury (must be on your cell or an adjacent one).",
                            "choices": names,
                        }
                    },
                }
        return menu

    def on_action(self, env, tag: str, action: str, params: dict) -> str | None:
        bedridden = self.health(tag) == "bedridden"
        if action == "bury":
            if bedridden:
                return "You are too sick to bury anything."
            return self.bury(env, tag, params)
        if bedridden and action == "move" and params.get("direction", "stay") != "stay":
            return "You are too sick to move. You stayed where you are."
        if bedridden and action == "take":
            return "You are too sick to take energy from another being."
        if action in CONTACT_TEXT:
            return self.refuse_distant_contact(env, tag, action, params)
        return None

    def refuse_distant_contact(
        self, env, tag: str, action: str, params: dict
    ) -> str | None:
        """Give, take, and give_artifact reach only a being on an adjacent cell."""
        param = CONTACT_TEXT[action][1]
        target_name = str(params.get(param, ""))
        target = env.name_to_tag.get(target_name)
        if target is None or target == tag or target not in env.agent_pos:
            return None  # the world's own handler explains these cases
        if env.distance(env.agent_pos[tag], env.agent_pos[target]) <= 1:
            return None
        if action == "give_artifact":
            return f"Failed. Target being {target_name} is not on a cell adjacent to yours."
        return (
            f"Cannot {action} energy: {target_name} is not on a cell adjacent to yours."
        )

    def on_step(self, env, infos: dict) -> None:
        o = self.options
        step = env.step_count
        if self.world_log is None and hasattr(env, "grid_size"):
            self.world_log = WorldStateLogger(
                env.log_path / "world_state.jsonl",
                env.grid_size,
                env._max_food_value,
                append=self.state["seeded"],
            )
        if not self.state["seeded"]:
            self.seed_fixtures(env)
        self.handle_deaths(env)
        self.touch(env)
        self.advance_incubation(env)
        care = self.care(env)
        self.spread(env)
        self.advance_symptoms(env)
        if step == o.outbreak_step and o.init_infected > 0:
            candidates = sorted(
                tag for tag in env.agent_registry if self.susceptible(env, tag)
            )
            chosen = self.rng(env).permutation(len(candidates))[: o.init_infected]
            for index in chosen:
                self.infect(env, candidates[int(index)], None, "outbreak")
        for tag in list(self.state["infections"]):
            if tag not in env.agent_registry or not self.symptomatic(tag):
                continue
            env.agent_energy[tag] -= o.energy_multiplier - 1
            hazard = self.death_hazard(tag) * care.get(tag, 1.0)
            if hazard > 0 and self.rng(env).random() < hazard:
                env.kill(tag, "sickness")
        self.appetite(env)
        self.notices(env)
        self.record_state(env)

    # ---------- rules ----------
    def seed_fixtures(self, env) -> None:
        """Place the health center and hand protective equipment to its role, once."""
        o = self.options
        self.state["seeded"] = True
        center = o.health_center
        if center is not None:
            pose = tuple(center.pose) if isinstance(center.pose, list) else center.pose
            grid_size = getattr(env, "grid_size", None)
            if grid_size and not all(0 <= int(c) < grid_size for c in pose):
                raise ValueError(
                    f"health_center.pose {list(pose)} lies outside the {grid_size}x{grid_size} grid"
                )
            env.seed_artifact(
                pose,
                "health_center",
                center.name,
                center.payload,
                -1,
                movable=False,
                radius=center.radius,
                heal_probability=center.heal_probability,
                hazard_multiplier=center.hazard_multiplier,
            )
        if o.ppe_per_worker <= 0:
            return
        for tag, ident in sorted(self.state["identities"].items()):
            if ident.get("role") != o.ppe_role or tag not in env.agent_registry:
                continue
            for i in range(o.ppe_per_worker):
                env.seed_artifact(
                    env.agent_pos[tag],
                    "ppe",
                    f"PPE_{i + 1:02d}",
                    "Protective equipment.",
                    -1,
                    to_inventory=tag,
                    protection=o.ppe_protection,
                )

    def handle_deaths(self, env) -> None:
        """Turn the sick dead of the previous step into remains and announce them."""
        o = self.options
        step = env.step_count
        for record in env.deaths:
            infection = self.state["infections"].pop(record["tag"], None)
            self.state["recovered"].pop(record["tag"], None)
            if infection is None or record["position"] is None:
                continue
            pos = record["position"]
            # Remains spread for remains_lifespan - 1 steps.
            lifespan = (
                o.remains_lifespan - 1 if o.remains_lifespan > 0 else o.remains_lifespan
            )
            _, name = env.seed_artifact(
                pos,
                "remains",
                f"remains_of_{record['name']}",
                "",
                lifespan,
                movable=False,
                strain=infection["strain"],
                host_name=record["name"],
                buriable_after=step + o.funeral_mourning_days,
                infection_id=infection["id"],
            )
            if not o.funeral_announcements or name is None:
                continue
            if o.funeral_mourning_days > 0:
                d = o.funeral_mourning_days
                mourning = f" The mourning lasts {d} day{'s' if d != 1 else ''}; only then may the remains be buried."
                # Written one step early, so the note arrives with the first step that allows the burial.
                if d > 1:
                    self.state["reminders"].append(
                        [record["name"], pos, step + d - 1, name]
                    )
            else:
                mourning = " The remains may be buried."
            for tag in self.within_earshot(env, pos):
                env.note(
                    tag,
                    "Deaths",
                    f"{record['name']} has died. Their remains lie unburied "
                    f"{self.compass(env, env.agent_pos[tag], pos)}.{mourning}",
                )

    def within_earshot(self, env, pos) -> List[str]:
        radius = self.options.funeral_announcement_radius
        return sorted(
            tag
            for tag, agent_pos in env.agent_pos.items()
            if radius < 0 or env.distance(agent_pos, pos) <= radius
        )

    def touch(self, env) -> None:
        """Energy that changes hands is a contact: each sick side exposes the other."""
        o = self.options
        for giver, receiver, _kind in env.transfers:
            for source, target in ((giver, receiver), (receiver, giver)):
                if source in self.state["infections"] and self.symptomatic(source):
                    factor = (
                        o.feverish_multiplier
                        if self.health(source) == "feverish"
                        else 1.0
                    )
                    self.expose(
                        env,
                        target,
                        o.infection_probability * o.contact_multiplier * factor,
                        source,
                        "touch",
                    )

    def care(self, env) -> Dict[str, float]:
        """Health centers heal and lower the death hazard of the sick within reach."""
        multipliers: Dict[str, float] = {}
        for name, artifact in env.artifacts.items():
            location = env.artifact_location.get(name)
            if (
                not isinstance(artifact, HealthCenterArtifact)
                or not location
                or location[0] != "map"
            ):
                continue
            for tag, pos in list(env.agent_pos.items()):
                if env.distance(pos, location[1]) > artifact.radius:
                    continue
                env.note(
                    tag,
                    "Nearby facilities",
                    artifact.passive_effect(env.step_count, tag),
                )
                if tag in self.state["infections"]:
                    multipliers[tag] = min(
                        multipliers.get(tag, 1.0), artifact.hazard_multiplier
                    )
                    if self.rng(env).random() < artifact.heal_probability:
                        self.recover(env, tag, "care")
        return multipliers

    def spread(self, env) -> None:
        """Sick hosts and unburied remains expose the beings around them."""
        o = self.options
        step = env.step_count
        for source, infection in list(self.state["infections"].items()):
            if source not in env.agent_registry or infection["acquired_at"] == step:
                continue
            if not self.symptomatic(source):
                continue
            factor = o.feverish_multiplier if self.health(source) == "feverish" else 1.0
            for target in env.agents_within(source, o.infection_radius):
                self.expose(
                    env, target, o.infection_probability * factor, source, "proximity"
                )
        for name, pos in self.remains_on_map(env).items():
            for target, agent_pos in list(env.agent_pos.items()):
                if env.distance(agent_pos, pos) <= o.infection_radius:
                    self.expose(
                        env, target, o.infection_probability, None, f"remains:{name}"
                    )

    def advance_incubation(self, env) -> None:
        """Count down the silent phase of infections that did not start this step."""
        step = env.step_count
        for tag, infection in self.state["infections"].items():
            if (
                tag in env.agent_registry
                and infection["acquired_at"] != step
                and infection["incubation"] > 0
            ):
                infection["incubation"] -= 1

    def advance_symptoms(self, env) -> None:
        """Count the sick steps after the spread pass, and recover in time."""
        o = self.options
        step = env.step_count
        for tag, infection in list(self.state["infections"].items()):
            if (
                tag not in env.agent_registry
                or infection["acquired_at"] == step
                or infection["incubation"] > 0
            ):
                continue
            infection["days_symptomatic"] += 1
            if (
                o.infection_duration >= 0
                and infection["days_symptomatic"] >= o.infection_duration
            ):
                self.recover(env, tag, "recovery")

    def record_state(self, env) -> None:
        """Append this step's world to world_state.jsonl, before the death phase.

        Grid worlds only: the file holds cells, and the viewer draws a grid.
        """
        if self.world_log is None:
            return
        agents = {}
        n_infected = n_sick = n_bedridden = 0
        for tag in env.agent_registry:
            pos = env.agent_pos.get(tag)
            if pos is None:
                continue
            health = self.health(tag)
            infected = health != "healthy"
            sick = health in ("feverish", "bedridden")
            bedridden = health == "bedridden"
            n_infected += infected
            n_sick += sick
            n_bedridden += bedridden
            inventory = env.agent_inventories.get(tag, ())
            energy = env.agent_energy[tag]
            time_left = env.agent_time[tag]
            agents[tag] = [
                pos[0],
                pos[1],
                float(energy) if np.isfinite(energy) else None,
                float(time_left) if np.isfinite(time_left) else None,
                len(inventory),
                int(infected),
                int(sick),
                sum(
                    isinstance(env.artifacts.get(name), PPEArtifact)
                    for name in inventory
                ),
                int(tag in self.state["recovered"]),
                int(bedridden),
            ]
        artifacts = (
            (pos[0], pos[1], name, env.artifacts[name].art_type)
            for pos, names in env.pos_artifacts.items()
            for name in names
            if name in env.artifacts
        )
        self.world_log.log_step(
            t=env.step_count,
            agents=agents,
            food=env.food,
            artifacts=artifacts,
            food_total=float(sum(env.food.values())),
            n_infected=n_infected,
            n_sick=n_sick,
            n_bedridden=n_bedridden,
        )

    def death_hazard(self, tag: str) -> float:
        """Chance that the host dies this step, before care. Deaths are spread over
        the sick period with a weight rising by day, so a share case_fatality dies in all."""
        o = self.options
        day = self.state["infections"][tag]["days_symptomatic"]
        if o.infection_duration < 0:
            return o.case_fatality if self.health(tag) == "bedridden" else 0.0
        days = o.infection_duration - 1  # the last sick step recovers before the roll
        if days < 1 or day > days:
            return 0.0
        total = days * (days + 1)
        return min(
            1.0, 2 * o.case_fatality * day / (total - o.case_fatality * (day - 1) * day)
        )

    def bury(self, env, tag: str, params: dict) -> str:
        o = self.options
        ref = str(params.get("name", ""))
        if ref not in self.remains_in_reach(env, tag):
            return f"There are no remains named '{ref}' within reach to bury."
        artifact = env.artifacts[ref]
        if env.step_count < artifact.buriable_after:
            days_left = artifact.buriable_after - env.step_count
            return (
                f"The mourning for {artifact.host_name} has not ended: they may be buried "
                f"in {days_left} day{'s' if days_left != 1 else ''}."
            )
        pos = env.artifact_location[ref][1]
        infected = self.expose(
            env,
            tag,
            o.infection_probability * o.burial_infection_multiplier,
            None,
            f"burial:{ref}",
        )
        attendees = [
            other
            for other, other_pos in sorted(env.agent_pos.items())
            if other != tag and env.distance(other_pos, pos) <= 1
        ]
        for other in attendees:
            self.expose(
                env,
                other,
                o.infection_probability * o.burial_bystander_multiplier,
                None,
                f"funeral:{ref}",
            )
        artifact.remaining_time = 0
        env.logger.log(
            time=env.step_count,
            event_type="BURIAL",
            agent_tag=tag,
            agent_name=env.agent_names[tag],
            remains=ref,
            host_name=artifact.host_name,
            position=pos,
            attendees=attendees,
            infected=infected,
        )
        return f"You buried {ref}."

    def notices(self, env) -> None:
        step = env.step_count
        due = [r for r in self.state["reminders"] if r[2] == step]
        self.state["reminders"] = [r for r in self.state["reminders"] if r[2] > step]
        for name, pos, _due, art_name in due:
            if art_name not in env.artifacts:
                continue
            pos = tuple(pos) if isinstance(pos, list) else pos
            for tag in self.within_earshot(env, pos):
                env.note(
                    tag,
                    "Deaths",
                    f"The mourning for {name} has ended: their remains "
                    f"({self.compass(env, env.agent_pos[tag], pos)}) may now be buried.",
                )
        for tag in env.agent_registry:
            state = self.health(tag)
            if state == "feverish":
                env.note(tag, "Health", FEVERISH_NOTICE)
            elif state == "bedridden":
                env.note(tag, "Health", BEDRIDDEN_NOTICE)
            elif state == "healthy" and self.state["recovered"].get(tag):
                env.note(tag, "Health", RECOVERED_NOTICE)
