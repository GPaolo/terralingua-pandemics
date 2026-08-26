# survival_parallel_env.py

import json
import pickle
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import tiktoken

from core.environment.actions import ACTION_TEXT
from core.environment.artifact import (
    ARTIFACT_TYPE,
    MAX_TEXT_ARTIFACT_SIZE,
    Artifact,
    ArtifactCreationError,
    HealthCenterArtifact,
    PPEArtifact,
    TextArtifact,
    ViralArtifact,
)
from core.environment.env_logger import Event, JSONLogger
from core.environment.world_logger import WorldStateLogger

MOVE_DICT = {
    "stay": (0, 0),
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


AGENT_INPUT_TYPE = ["text"]
AVAILABLE_DEAD_AGENT_FOOD = ["single", "area", "none"]

#: What an agent hosting a *symptomatic* viral artifact is told about itself,
#: every step, in the "Additional info from the environment" block of its
#: prompt. Deliberately describes the symptoms rather than naming the virus:
#: nothing in the system prompt announces that sickness exists, so a healthy
#: agent still has to work out what is happening to the being that stopped
#: moving. An agent still incubating is told nothing at all — see
#: OpenGridWorld._count_sick.
SICK_AGENT_NOTICE = (
    "You are sick. Something is living inside you and draining your energy. "
    "You are too weak to move or to take energy from others, and you have no "
    "appetite: food no longer restores you. You can still speak, and other "
    "beings can still give you energy."
)


class OpenGridWorld:
    """Same grid rules as before, plus agent registry & coded messages."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 100,
        vision_radius: int = 2,
        init_agent_energy: int = 100,
        lifespan: int = 100,
        init_food: int = 1250,
        max_food_value: float = 10.0,
        food_decay_rate: float = 0.05,
        food_decay_amount: float = 1.0,
        food_spawn_rate: int = 3,
        log_path: Path | str | None = None,
        dead_agent_food: str = "single",  # "single" | "area" | "none"
        use_inventory: bool = False,
        use_colors: bool = False,
        reproduction_cost: int = 50,
        artifact_creation_cost: int = 0,
        artifact_creation: bool = True,
        reproduction_allowed: bool = True,
        food_zones: int | List[Tuple[int, int]] | None = None,
        static_food: bool = False,
        food_mechanism: bool = True,
        verbose: int = 1,  # 0: warnings only, 1: key events, 2: per-step debug
        inert_artifacts: bool = False,
        viral_init_infected: int = 0,
        viral_outbreak_step: int = 0,
        viral_lifespan: int = -1,
        viral_incubation_min: int = 2,
        viral_incubation_max: int = 21,
        viral_dropped_lifespan: int = 20,
        viral_infection_radius: int = 1,
        viral_infection_probability: float = 0.3,
        viral_energy_multiplier: float = 2.0,
        viral_death_probability: float = 0.0,
        ppe_protection: float = 0.1,
        burials: bool = False,
        burial_infection_multiplier: float = 2.0,
        max_message_size: int = -1,  # tokens per message; -1: unlimited
        max_text_artifact_size: int = MAX_TEXT_ARTIFACT_SIZE,  # tokens per text artifact
        init_artifacts: str | Path | List[dict] | None = None,
        parent_authored_genome: bool = False,
        log_world_state: bool = True,
    ):
        # grid/world params
        # ---------------------------
        self.verbose = verbose
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        if init_agent_energy < 0:
            self.init_agent_energy = np.inf
        else:
            self.init_agent_energy = init_agent_energy
        self.lifespan = lifespan
        self.dead_agent_food = dead_agent_food
        self.use_inventory = use_inventory
        self.use_colors = use_colors
        self.reproduction_cost = reproduction_cost
        self.artifact_creation_cost = artifact_creation_cost
        self.artifact_creation = artifact_creation
        self.inert_artifacts = inert_artifacts
        # Viral artifacts: viral_init_infected agents get infected at
        # step viral_outbreak_step (0 initial infected disables the mechanic)
        self.viral_init_infected = viral_init_infected
        self.viral_outbreak_step = viral_outbreak_step
        self.viral_lifespan = viral_lifespan
        # Each infection draws its own incubation from [min, max]: the host is a
        # silent, harmless carrier until it runs out. viral_lifespan then counts
        # the symptomatic period, so latency does not eat the infectious window.
        self.viral_incubation_min = viral_incubation_min
        self.viral_incubation_max = viral_incubation_max
        self.viral_dropped_lifespan = viral_dropped_lifespan
        self.viral_infection_radius = viral_infection_radius
        self.viral_infection_probability = viral_infection_probability
        self.viral_energy_multiplier = viral_energy_multiplier
        # Per-step chance that a symptomatic agent dies of the sickness
        self.viral_death_probability = viral_death_probability
        # Multiplier on the contraction probability of an agent carrying PPE
        self.ppe_protection = ppe_protection
        # Beings next to remains (a ground viral artifact) may bury them,
        # removing the artifact at burial_infection_multiplier x the usual
        # contact risk of catching the infection themselves
        self.burials = burials
        self.burial_infection_multiplier = burial_infection_multiplier
        self.max_message_size = max_message_size
        self.max_text_artifact_size = max_text_artifact_size
        self._token_encoder = None  # lazy: only needed when messages are capped
        # Artifacts seeded by the environment itself, as a list of entries
        # {name, payload, pose, lifespan, step} or a path to a JSON file
        # containing such a list
        self.init_artifacts = self._load_init_artifacts(init_artifacts)
        self.reproduction_allowed = reproduction_allowed
        self.parent_authored_genome = parent_authored_genome
        self.food_mechanism = food_mechanism
        if not self.food_mechanism:
            self.dead_agent_food = "none"

        if self.dead_agent_food not in AVAILABLE_DEAD_AGENT_FOOD:
            raise ValueError(
                f"dead_agent_food must be one of {AVAILABLE_DEAD_AGENT_FOOD}, got {self.dead_agent_food}"
            )

        self._init_food_count = init_food
        # max value per new/respawned tile
        self._max_food_value = max_food_value
        # how much value to subtract from each tile per step
        self._food_decay_amount = food_decay_amount
        self._food_decay_rate = food_decay_rate
        self._food_spawn_rate = food_spawn_rate
        self.food_zones = food_zones
        self.static_food = static_food
        self.log_path = Path(log_path) if log_path is not None else Path(".")
        self.logger = JSONLogger(self.log_path / "open_gridworld.log")
        # Per-step world snapshot, so a run can be followed live and replayed
        # afterwards without reading any pickle.
        self.world_logger = (
            WorldStateLogger(
                self.log_path / "world_state.jsonl",
                grid_size=grid_size,
                max_food_value=max_food_value,
            )
            if log_world_state
            else None
        )
        # ---------------------------

        # Runtime state
        # ---------------------------
        self.rng = None

        self.food: Dict[Tuple[int, int], float] = {}  # {(x, y): value} for food tiles
        self.food_count: List[float] = []
        self.food_distribution = None
        self._food_cdf = None  # rebuilt lazily by _sample_density_cell
        self.empty_food: List[Tuple[int, int]] = []

        self.artifacts: Dict[str, Artifact] = {}  # {artifact_name: Artifact}
        # {(x, y): [artifact_names]} for tiles
        self.artifacts_map: Dict[Tuple, Set[str]] = defaultdict(set)
        # {agent_tag: [artifact_names]}
        self.agent_inventories: Dict[str, Set[str]] = defaultdict(set)
        # {agent_tag: infections cleared while alive (cure or natural expiry)}
        self.agent_recoveries: Dict[str, int] = defaultdict(int)
        self.expired_artifacts: List[Artifact] = []
        # Total number of viral infections so far. Used to name infection copies uniquely.
        self.viral_infection_count = 0
        # Configured init artifacts not yet placed in the environment
        self._pending_init_artifacts: List[dict] = list(self.init_artifacts)

        self.agent_pos: Dict[str, Tuple[int, int]] = {}
        self.agent_trajectories: Dict[str, List[Tuple[int, int]]] = {}
        self.agent_avail_actions: Dict[str, Dict[str, dict]] = {}
        self.pos_to_agent: Dict[Tuple[int, int], str] = {}
        self.agent_energy: Dict[str, float] = {}
        self.agent_time: Dict[str, float] = {}
        self.agent_spawn: Dict[str, List[str]] = {}
        self.agent_names: Dict[str, str] = {}
        # {agent_tag: persona role}, only for beings whose persona declares one
        self.agent_roles: Dict[str, str] = {}
        self.agent_colors: Dict[str, str] = {}

        self.msg_raw: Dict[str, str | np.ndarray] = {}
        self.chat = {}
        self.agent_registry = {}  # {name: "numerical" | "text"}
        self.step_count = 0
        # ---------------------------

        # Rendering
        self._pygame_inited = False
        self._cell_size = 10  # size of each cell in pixels
        default = self.grid_size * self._cell_size
        self._window_size = (default, default)
        self._screen = None

        # --- Gym spaces ---
        self.obs_shape = [2 * vision_radius + 1, 2 * vision_radius + 1]

    # ---------- PettingZoo API helpers ----------
    def action_space(self, agent):  # type: ignore
        return ACTION_TEXT

    # ---------- public helpers ----------
    def add_agent(
        self,
        agent_tag: str,
        agent_name: str,
        agent_type: str,
        position: tuple | None = None,
        agent_role: str | None = None,
    ):
        """Register an external agent object before reset()."""
        if agent_type not in AGENT_INPUT_TYPE:
            raise ValueError(
                f"Unknown agent type {agent_type}. Must be one of {AGENT_INPUT_TYPE}."
            )
        if agent_tag in self.agent_registry:
            raise ValueError(
                f"Agent {self.agent_names[agent_tag]}({agent_tag}) already exists in the environment."
            )
        self.agent_registry[agent_tag] = agent_type
        self.agent_names[agent_tag] = agent_name
        if agent_role:
            self.agent_roles[agent_tag] = agent_role
        self._place_agent(agent_tag, p=position)
        if self.verbose >= 2:
            print(
                f"Adding agent {self.agent_names[agent_tag]}({agent_tag}) of type {agent_type} at position {self.agent_pos[agent_tag]}."
            )
        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.AGENT_ADDED,
                agent_tag=agent_tag,
                position=self.agent_pos[agent_tag],
                agent_name=self.agent_names[agent_tag],
                agent_type=agent_type,
                agent_role=agent_role,
            )
        avail_actions = self._get_avail_actions(agent_tag=agent_tag)
        observation = self._build_obs(agent_tag)
        return observation, {"available_actions": avail_actions}

    def add_artifact(
        self,
        pose,
        art_type,
        art_name,
        payload,
        creator,
        lifespan,
        to_inventory=None,
        artifact_params=None,
    ) -> str:
        """Adds an artifact to the environment.

        creator is normally an agent tag; creators that are not registered
        agents (e.g. "environment" for seeded artifacts) pay no energy.
        If to_inventory is an agent tag, the artifact is placed directly in
        that agent's inventory instead of on the map at pose.
        """
        is_agent_creator = creator in self.agent_energy
        if (
            is_agent_creator
            and self.agent_energy[creator] < self.artifact_creation_cost
        ):
            return f"Failed. Agent does not have enough energy. Required: {self.artifact_creation_cost}"

        if to_inventory is not None and to_inventory not in self.agent_registry:
            return f"Failed. No agent {to_inventory} to receive the artifact"

        # Agents can only create the types advertised in ARTIFACT_TYPE
        if is_agent_creator and art_type not in ARTIFACT_TYPE:
            return f"Artifact type: {art_type} is not a valid type. Only artifact valid types are: {list(ARTIFACT_TYPE.keys())}"

        if art_type == "text":
            while art_name in self.artifacts:
                art_name = f"{art_name}_1"
            try:
                artifact = TextArtifact(
                    name=art_name,
                    payload=payload,
                    pose=pose,
                    creator=creator,
                    lifespan=lifespan,
                    creation_time=self.step_count,
                    max_size=self.max_text_artifact_size,
                )
            except ArtifactCreationError as e:
                return str(e)
        elif art_type == "ppe":
            while art_name in self.artifacts:
                art_name = f"{art_name}_1"
            # No payload given -> keep the class's default description
            artifact = PPEArtifact(
                name=art_name,
                pose=pose,
                creator=creator,
                lifespan=lifespan,
                creation_time=self.step_count,
                protection=self.ppe_protection,
                **({"payload": payload} if payload else {}),
            )
        elif art_type == "health_center":
            if to_inventory is not None:
                return (
                    "Failed. A health center is fixed to the map and cannot be carried."
                )
            while art_name in self.artifacts:
                art_name = f"{art_name}_1"
            params = artifact_params or {}
            artifact = HealthCenterArtifact(
                name=art_name,
                pose=pose,
                creator=creator,
                lifespan=lifespan,
                creation_time=self.step_count,
                heal_probability=params.get("heal_probability", 0.2),
                radius=params.get("radius", 1),
                **({"payload": payload} if payload else {}),
            )
        else:
            return f"Artifact type: {art_type} is not a valid type. Only artifact valid types are: {list(ARTIFACT_TYPE.keys())}"

        if is_agent_creator:
            self.agent_energy[creator] -= self.artifact_creation_cost
        self.artifacts[art_name] = artifact

        if to_inventory is not None:
            self.agent_inventories[to_inventory].add(art_name)
            artifact.users[to_inventory].add(self.step_count)
            status = (
                f"Created artifact {art_name} of type {art_type} in "
                f"{self.agent_names[to_inventory]}'s inventory."
            )
        else:
            self.artifacts_map[pose].add(art_name)
            status = (
                f"Created artifact {art_name} of type {art_type} at position {pose}."
            )

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.ARTIFACT_ADDED,
                artifact=artifact.serialize(),
                position=pose,
                agent_tag=creator,
                agent_name=self.agent_names.get(creator, creator),
                possessor_tag=to_inventory,
            )
        return status

    def _draw_incubation(self) -> int:
        """Steps this new infection stays latent before showing symptoms."""
        if self.viral_incubation_max <= 0:
            return 0
        if self.rng is None:
            self.rng = np.random.default_rng()
        return int(
            self.rng.integers(self.viral_incubation_min, self.viral_incubation_max + 1)
        )

    def infect_agent(
        self,
        agent_tag: str,
        strain: str = "virus",
        source_artifact: ViralArtifact | None = None,
        source_tag: str | None = None,
    ) -> str | None:
        """Infects an agent by adding a viral artifact to its inventory.

        If source_artifact is given, the new artifact is a copy of it (the
        infection spread from another agent); otherwise a fresh artifact of
        the given strain is seeded by the environment.
        Returns the name of the created artifact, or None if the agent
        already hosts the strain.
        """
        if agent_tag not in self.agent_registry:
            raise ValueError(f"Agent {agent_tag} not found in the environment.")

        if source_artifact is not None:
            strain = source_artifact.strain
        if strain in self._hosted_viral_strains(agent_tag):
            return None
        # Recovery grants immunity: a healed agent never catches it again
        if self.agent_recoveries.get(agent_tag, 0) > 0:
            return None

        # Unique name for the new infection
        self.viral_infection_count += 1
        art_name = f"{strain}_i{self.viral_infection_count}"
        while art_name in self.artifacts:
            self.viral_infection_count += 1
            art_name = f"{strain}_i{self.viral_infection_count}"

        pose = self.agent_pos[agent_tag]
        incubation = self._draw_incubation()
        if source_artifact is not None:
            artifact = source_artifact.spawn_copy(
                name=art_name,
                pose=pose,
                timestamp=self.step_count,
                incubation=incubation,
            )
        else:
            lifespan = np.inf if self.viral_lifespan == -1 else self.viral_lifespan
            artifact = ViralArtifact(
                name=art_name,
                lifespan=lifespan,
                pose=pose,
                creator="environment",
                creation_time=self.step_count,
                strain=strain,
                incubation=incubation,
            )

        artifact.users[agent_tag].add(self.step_count)
        self.artifacts[art_name] = artifact
        self.agent_inventories[agent_tag].add(art_name)

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.VIRAL_INFECTION,
                agent_tag=agent_tag,
                agent_name=self.agent_names[agent_tag],
                source_tag=source_tag,
                source_name=self.agent_names.get(source_tag),
                source_artifact=(
                    source_artifact.name if source_artifact is not None else None
                ),
                strain=strain,
                incubation=incubation,
                artifact=artifact.serialize(),
            )
        return art_name

    # ---------- env lifecycle ----------
    def reset(self, agent_tag, position=None):
        """Resets a single agent in the environment.

        Args:
            agent (_type_): _description_
            seed (_type_, optional): _description_. Defaults to None.
        """
        if agent_tag not in self.agent_registry:
            raise ValueError(
                f"Agent {self.agent_names[agent_tag]} not found in the environment."
            )

        self._place_agent(agent_tag, p=position)
        if self.verbose >= 1:
            print(
                f"Agent {self.agent_names[agent_tag]} reset. Respawning at position {self.agent_pos[agent_tag]}."
            )

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.RESET_AGENT,
                agent_tag=agent_tag,
                agent_name=self.agent_names[agent_tag],
                position=self.agent_pos[agent_tag],
            )

    def restart_env(self, seed=None, **options):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # fresh world
        self.agent_pos = {}
        self.agent_trajectories = {}
        self.agent_energy = {}
        self.msg_raw = {}
        self.pos_to_agent = {}
        self.agent_spawn = {}
        self.food.clear()
        self.step_count = 0
        self.chat = {}
        self.food_count = []
        self.agent_inventories = defaultdict(set)
        self.agent_recoveries = defaultdict(int)
        self.artifacts_map = defaultdict(set)
        self.artifacts = {}
        self.expired_artifacts = []
        self.viral_infection_count = 0
        self._pending_init_artifacts = list(self.init_artifacts)

        poses = options.get("agent_poses")
        agent_poses = {tag: None for tag in self.agent_registry}
        if poses is not None:
            agent_poses.update(poses)

        # place agents
        for tag in self.agent_registry:
            self._place_agent(tag, p=agent_poses[tag])

        # Step-0 init artifacts are placed before the initial observations
        self._seed_init_artifacts()

        if self.food_mechanism:
            self._seed_initial_food()
            self.food_count.append(sum(self.food.values()))

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.ENV_RESET,
                num_agents=len(self.agent_registry),
                agent_poses=self.agent_pos,
                agent_types=self.agent_registry,
                food_count=len(self.food),
                agent_names=self.agent_names,
                agent_roles=self.agent_roles,
            )

        infos = {}
        for agent_tag in self.agent_registry:
            avail_actions = self._get_avail_actions(agent_tag=agent_tag)
            infos[agent_tag] = {"available_actions": avail_actions}
            # Matters on --resume: agents sick before the checkpoint would
            # otherwise get their first prompt back without the notice.
            if self._count_sick(agent_tag) > 0:
                infos[agent_tag]["Health"] = SICK_AGENT_NOTICE

        self._log_world_state()

        return self._observe_all(), infos

    @staticmethod
    def _note_outcome(infos: dict, agent_tag: str, message: str):
        """Appends to an agent's "Action outcome" instead of overwriting it.

        A sick agent can trip several restrictions in the same step (it tried to
        move *and* it is standing on food it cannot eat), and each of those is
        the only feedback it gets about why nothing happened.
        """
        previous = infos[agent_tag].get("Action outcome", "")
        infos[agent_tag]["Action outcome"] = f"{previous} {message}".strip()

    def _count_viral(self, agent_tag: str) -> int:
        """How many viral artifacts this agent is currently hosting.

        Counts both phases. An agent still incubating is infected but not yet
        sick — for anything the agent can feel or do, use _count_sick.
        """
        return sum(
            1
            for name in self.agent_inventories[agent_tag]
            if isinstance(self.artifacts.get(name), ViralArtifact)
        )

    def _count_sick(self, agent_tag: str) -> int:
        """How many of this agent's infections have finished incubating.

        This, not _count_viral, is what gates behaviour: an incubating host
        moves, eats, steals and spreads nothing, and is told nothing.
        """
        return sum(
            1
            for name in self.agent_inventories[agent_tag]
            if isinstance(self.artifacts.get(name), ViralArtifact)
            and self.artifacts[name].symptomatic
        )

    def _count_ppe(self, agent_tag: str) -> int:
        """How many PPE artifacts this agent carries (protection does not stack)."""
        return sum(
            1
            for name in self.agent_inventories[agent_tag]
            if isinstance(self.artifacts.get(name), PPEArtifact)
        )

    def _log_world_state(self):
        """Append the current world to ``world_state.jsonl``.

        Called at the end of ``reset`` (for t=0) and at the end of every ``step``.
        """
        if self.world_logger is None:
            return

        agents = {}
        n_infected = 0
        n_sick = 0
        for tag in self.agent_registry:
            pos = self.agent_pos.get(tag)
            if pos is None:
                continue
            n_viral = self._count_viral(tag)
            n_sick_tag = self._count_sick(tag)
            n_infected += n_viral > 0
            n_sick += n_sick_tag > 0
            # A negative init_agent_energy means infinite energy, which is not
            # representable in JSON; null reads as "unbounded" to the consumer.
            energy = self.agent_energy[tag]
            agents[tag] = [
                pos[0],
                pos[1],
                float(energy) if np.isfinite(energy) else None,
                self.agent_time[tag],
                len(self.agent_inventories[tag]),
                n_viral,
                n_sick_tag,
                self._count_ppe(tag),
                self.agent_recoveries.get(tag, 0),
            ]

        # artifacts_map is a defaultdict, so a stale read can leave empty sets.
        # Dropped viral artifacts go out under their remains display name and a
        # "remains" kind, so the dashboard matches what the beings see.
        def _kind(artifact):
            if isinstance(artifact, ViralArtifact):
                return "remains"
            return getattr(artifact, "art_type", None) or "text"

        artifacts = (
            (
                pos[0],
                pos[1],
                getattr(self.artifacts.get(name), "display_name", None) or name,
                _kind(self.artifacts.get(name)),
            )
            for pos, names in self.artifacts_map.items()
            for name in names
        )

        self.world_logger.log_step(
            t=self.step_count,
            agents=agents,
            food=self.food,
            artifacts=artifacts,
            food_total=sum(self.food.values()),
            n_infected=n_infected,
            n_sick=n_sick,
        )

    # ---------- core mechanics ----------
    def step(self, actions):
        """
        `actions` may omit some agents (those on cooldown, etc.).
        Missing agents simply keep previous message and perform no move.
        """

        # ---- default bookkeeping ----
        rewards = {a: 0.0 for a in self.agent_registry}
        infos = {a: {} for a in self.agent_registry}
        done_dict = {a: False for a in self.agent_registry}

        # ---- Process actions from acting agents ----
        # ================================
        for agent, act in actions.items():
            if agent not in self.agent_registry:
                raise ValueError(
                    f"Agent {self.agent_names[agent]}({agent}) not found in the environment."
                )

            move = (0, 0)
            # A viral artifact past its incubation makes an agent sick: it
            # cannot move, steal energy or eat until the infection clears.
            # _get_avail_actions already withholds those affordances;
            # re-checking here keeps the rule true even for a hand-driven
            # step() and, unlike the silent coercion below, tells the agent why
            # nothing happened. An agent still incubating is unaffected.
            sick = self._count_sick(agent) > 0

            # Get action name, message and params
            # ---------------------------
            # NOTE Cooldown is internal to the agent, not the environment.
            action_name = act.get("action", "move")
            message = act.get("message", "")
            action_params = act.get("params", {})

            if self.max_message_size >= 0 and message:
                if self._token_encoder is None:
                    self._token_encoder = tiktoken.get_encoding("cl100k_base")
                toks = self._token_encoder.encode(str(message))
                if len(toks) > self.max_message_size:
                    message = self._token_encoder.decode(toks[: self.max_message_size])
                    self._note_outcome(
                        infos,
                        agent,
                        f"Your message was cut off at {self.max_message_size} tokens.",
                    )

            # Intercept before the availability gate below: it withholds these
            # from a sick agent, so the gate would already have coerced this to
            # a silent stay and the agent would never learn why.
            if sick and action_name in ("take", "bury"):
                self._note_outcome(
                    infos,
                    agent,
                    "You are too sick to take energy from another being."
                    if action_name == "take"
                    else "You are too sick to bury anything.",
                )
                rewards[agent] -= 1
                action_name = "move"
                action_params = {"direction": "stay"}

            if action_name not in self.agent_avail_actions[agent]:
                if self.verbose >= 1:
                    print(
                        f"{self.agent_names[agent]}({agent}) - Unknown action: {action_name} - Available actions: {list(self.agent_avail_actions[agent].keys())}"
                    )
                action_name = "move"
                action_params["direction"] = "stay"

            avail_params = self.agent_avail_actions[agent][action_name].get(
                "params", {}
            )

            if not set(action_params.keys()) == set(avail_params.keys()):
                if self.verbose >= 1:
                    print(
                        f"{self.agent_names[agent]}({agent}) - Provided action params: {list(action_params.keys())} do not match required {avail_params}"
                    )
                action_name = "move"
                action_params["direction"] = "stay"

            # Handle Move
            # ---------------------------
            if action_name == "move":
                try:
                    move = MOVE_DICT.get(action_params.get("direction", "stay"), (0, 0))
                except:
                    move = (0, 0)
                if sick and move != (0, 0):
                    move = (0, 0)
                    self._note_outcome(
                        infos,
                        agent,
                        "You are too sick to move. You stayed where you are.",
                    )
            # ---------------------------

            # Handle Energy exchange
            # ---------------------------
            elif action_name == "bury":
                ref = str(action_params.get("name", ""))
                # Remains present under their display name; accept either it
                # or the internal artifact name.
                art_name = next(
                    (
                        n
                        for n in self._ground_viral_nearby(agent)
                        if n == ref
                        or getattr(self.artifacts.get(n), "display_name", None) == ref
                    ),
                    None,
                )
                artifact = self.artifacts.get(art_name) if art_name else None
                if not isinstance(artifact, ViralArtifact):
                    self._note_outcome(
                        infos,
                        agent,
                        f"There are no remains named '{ref}' within reach to bury.",
                    )
                    rewards[agent] -= 1
                else:
                    # Handling the remains is one extra exposure at scaled risk,
                    # rolled before removal so the copy has a source; PPE still
                    # protects the digger.
                    if self.rng is None:
                        self.rng = np.random.default_rng()
                    risk = (
                        self.viral_infection_probability
                        * self.burial_infection_multiplier
                        * self._infection_protection(agent)
                    )
                    caught = None
                    if self.rng.random() < risk:
                        caught = self.infect_agent(
                            agent_tag=agent, source_artifact=artifact
                        )
                    pos = None
                    for cell in list(self.artifacts_map):
                        if art_name in self.artifacts_map[cell]:
                            pos = cell
                            self.artifacts_map[cell].discard(art_name)
                            break
                    self.artifacts.pop(art_name, None)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self._note_outcome(
                        infos,
                        agent,
                        f"You buried {artifact.display_name or art_name}.",
                    )
                    # Say nothing while a caught infection incubates
                    if caught is not None and self.artifacts[caught].symptomatic:
                        infos.setdefault(agent, {})["Infection"] = (
                            f"Artifact {caught} appeared in your inventory."
                        )
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.BURIAL,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        artifact=artifact.serialize(),
                        pose=pos,
                        infected=caught is not None,
                    )

            elif action_name in ("give", "take"):
                nearby_agents = self._get_nearby_agents(agent)
                # must be in vision radius
                target_name = action_params.get("target")
                target_tag = None
                for tag, name in self.agent_names.items():
                    if name == target_name:
                        target_tag = tag
                        break

                if target_tag in nearby_agents:
                    a_pos = self.agent_pos[agent]
                    tgtpos = self.agent_pos[target_tag]
                    if (
                        abs(a_pos[0] - tgtpos[0]) <= self.vision_radius
                        and abs(a_pos[1] - tgtpos[1]) <= self.vision_radius
                    ):
                        # perform transfer
                        amount = max(float(action_params.get("amount", 0.0)), 0.0)
                        if action_name == "give":
                            energy_source = float(self.agent_energy[agent])
                            xfer = min(amount, energy_source)
                            self.agent_energy[agent] -= xfer
                            self.agent_energy[target_tag] += xfer
                            rewards[agent] += xfer * 0.5
                            rewards[target_tag] += xfer
                            infos[target_tag]["Energy received"] = (
                                f"{self.agent_names[agent]} gave you {xfer:g} energy."
                            )
                            self.logger.log(
                                time=self.step_count,
                                event_type=Event.GIFT_ENERGY,
                                agent_tag=agent,
                                agent_name=self.agent_names[agent],
                                target_tag=target_tag,
                                target_name=self.agent_names[target_tag],
                                amount=xfer,
                                step=self.step_count,
                                target_final_energy=self.agent_energy[target_tag],
                                final_energy=self.agent_energy[agent],
                            )
                        else:  # take
                            energy_target = float(self.agent_energy[target_tag])
                            stolen = min(amount, energy_target)
                            self.agent_energy[target_tag] -= stolen
                            self.agent_energy[agent] += stolen
                            rewards[agent] += stolen
                            rewards[target_tag] -= stolen * 2
                            infos[target_tag]["Energy stolen"] = (
                                f"Someone took {stolen:g} energy from you."
                            )
                            self.logger.log(
                                time=self.step_count,
                                event_type=Event.TAKE_ENERGY,
                                agent_tag=agent,
                                agent_name=self.agent_names[agent],
                                target_tag=target_tag,
                                target_name=self.agent_names[target_tag],
                                amount=stolen,
                                step=self.step_count,
                                target_final_energy=self.agent_energy[target_tag],
                                final_energy=self.agent_energy[agent],
                            )
                    else:
                        self._note_outcome(
                            infos,
                            agent,
                            f"Cannot {action_name} energy to {target_name} as not nearby",
                        )
                        rewards[agent] -= 1
            # ---------------------------

            # Change color
            # ---------------------------
            elif action_name == "set_color":
                selected_color = action_params.get("color")
                self.agent_colors[agent] = selected_color
                infos[agent]["Color selection"] = (
                    f"Success. Current color {selected_color}"
                )
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.SET_COLOR,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    color=selected_color,
                )
            # ---------------------------

            # Handle reproduction
            # ---------------------------
            elif action_name == "reproduce":
                # subtract 50 energy for attempting to reproduce
                self.agent_energy[agent] -= self.reproduction_cost
                spawn_pose = self._random_free_neigh(center=self.agent_pos[agent])

                if self.agent_energy[agent] >= 0 and spawn_pose is not None:
                    offspring_name = action_params.get("name")
                    already_present = False
                    while offspring_name in list(self.agent_names.values()):
                        offspring_name = f"{offspring_name}_1"
                        already_present = True

                    offspring_idx = 0
                    new_agent_idx = f"{agent}_{offspring_idx}"
                    while new_agent_idx in self.agent_spawn[agent]:
                        offspring_idx += 1
                        new_agent_idx = f"{agent}_{offspring_idx}"

                    self.agent_spawn[agent].append(new_agent_idx)

                    self.add_agent(
                        agent_tag=new_agent_idx,
                        agent_name=offspring_name,
                        agent_type=self.agent_registry[agent],
                        position=spawn_pose,
                    )
                    # Give additional energy
                    additional_energy = min(
                        int(action_params.get("energy", 0)),
                        self.agent_energy[agent],
                    )

                    self.agent_energy[new_agent_idx] += additional_energy
                    self.agent_energy[agent] -= additional_energy

                    if self.verbose >= 1:
                        print(f"""Agent {offspring_name}({new_agent_idx}) is born from {self.agent_names[agent]}({agent}).
{offspring_name}({new_agent_idx}) energy: {self.agent_energy[new_agent_idx]}
{self.agent_names[agent]}({agent}) energy: {self.agent_energy[agent]}""")

                    reprod_info = {
                        "status": "successful",
                        "child_name": offspring_name,
                        "child_tag": new_agent_idx,
                        "child_type": "text",
                    }
                    if self.parent_authored_genome:
                        reprod_info["offspring_genome"] = str(
                            action_params.get("offspring_genome", "")
                        )
                    if already_present:
                        reprod_info["note"] = (
                            f"An agent with name {action_params.get('name')} was already present. So Offspring has been named: {offspring_name}"
                        )

                    infos[agent] = {"reproduction": reprod_info}
                    infos[new_agent_idx] = {}
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.AGENT_REPRODUCED,
                        step=self.step_count,
                        child_name=offspring_name,
                        child_tag=new_agent_idx,
                        successful=True,
                        energy_gifted=additional_energy,
                        final_energy=self.agent_energy[agent],
                        child_energy=self.agent_energy[new_agent_idx],
                    )

                elif self.agent_energy[agent] < 0:
                    infos[agent] = {
                        "reproduction": {
                            "status": "failed",
                            "reason": "Not enough energy",
                        }
                    }
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.AGENT_REPRODUCED,
                        step=self.step_count,
                        successful=False,
                        fail_reason="No energy",
                    )
                elif spawn_pose is None:
                    infos[agent] = {
                        "reproduction": {
                            "status": "failed",
                            "reason": "No free space around",
                        }
                    }
                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.AGENT_REPRODUCED,
                        step=self.step_count,
                        successful=False,
                        fail_reason="No space",
                    )
            # ---------------------------

            # Artifact creation
            # ---------------------------
            elif action_name == "create_artifact":
                pose = self.agent_pos[agent]
                art_type = action_params.get("type", "text")
                art_name = action_params.get("name", f"{agent}_artifact")
                payload = action_params.get("payload", "")
                lifespan = int(action_params.get("lifespan", -1))
                if lifespan == -1:
                    lifespan = np.inf
                else:
                    lifespan += 1  # to offset the fact that we reduce it later
                status = self.add_artifact(
                    pose=pose,
                    art_type=art_type,
                    art_name=art_name,
                    payload=payload,
                    creator=agent,
                    lifespan=lifespan,
                )
                infos[agent]["Artifact creation status"] = status
            # ---------------------------

            # Artifact pickup
            # ---------------------------
            elif action_name == "pickup_artifact":
                art_to_pickup = action_params.get("name")
                pose = self.agent_pos[agent]

                if art_to_pickup in self.artifacts:
                    if art_to_pickup in self.artifacts_map[pose]:
                        if not self.artifacts[art_to_pickup].interactable:
                            status = (
                                f"Failed. Artifact {art_to_pickup} cannot be picked up"
                            )
                        else:
                            # Record interaction
                            self.artifacts[art_to_pickup].users[agent].add(
                                self.step_count
                            )
                            # Remove from the map
                            self.artifacts_map[pose].remove(art_to_pickup)
                            # Put it in inventory
                            self.agent_inventories[agent].add(art_to_pickup)
                            status = "Success"
                    else:
                        status = f"Failed. No artifact with name {art_to_pickup} at current position"
                else:
                    status = f"Failed. Artifact {art_to_pickup} does not exist"

                infos[agent]["Artifact pickup status"] = status
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ARTIFACT_PICKUP,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    artifact_name=art_to_pickup,
                    status=status,
                    pose=pose,
                )
            # ---------------------------

            # Artifact drop
            # ---------------------------
            elif action_name == "drop_artifact":
                art_to_drop = action_params.get("name")
                pose = self.agent_pos[agent]

                if art_to_drop in self.artifacts:
                    if art_to_drop in self.agent_inventories[agent]:
                        if not self.artifacts[art_to_drop].interactable:
                            status = f"Failed. Artifact {art_to_drop} cannot be dropped"
                        else:
                            # Record interaction
                            self.artifacts[art_to_drop].users[agent].add(
                                self.step_count
                            )
                            # Remove from inventory
                            self.agent_inventories[agent].remove(art_to_drop)
                            # Put it in map
                            self.artifacts_map[pose].add(art_to_drop)
                            status = "Success"
                    else:
                        status = (
                            f"Failed. No artifact with name {art_to_drop} in inventory"
                        )
                else:
                    status = f"Failed. Artifact {art_to_drop} does not exist"

                infos[agent]["Artifact drop status"] = status
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.ARTIFACT_DROP,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    artifact_name=art_to_drop,
                    status=status,
                    pose=pose,
                )
            # ---------------------------

            # Artifact gift
            # ---------------------------
            elif action_name == "give_artifact":
                nearby_agents = self._get_nearby_agents(agent)
                art_to_gift = action_params.get("artifact_name")

                # Find target agent
                target_name = action_params.get("target_agent")
                target_tag = None
                for tag, name in self.agent_names.items():
                    if name == target_name:
                        target_tag = tag
                        break

                if target_tag is None:
                    status = f"Failed. No being with name {target_name}"
                elif target_tag in nearby_agents:
                    if art_to_gift in self.agent_inventories[agent]:
                        if not self.artifacts[art_to_gift].interactable:
                            status = (
                                f"Failed. Artifact {art_to_gift} cannot be given away"
                            )
                        else:
                            # Remove from inventory
                            self.agent_inventories[agent].remove(art_to_gift)
                            # Put in target inventory
                            self.agent_inventories[target_tag].add(art_to_gift)
                            # Record interactions
                            self.artifacts[art_to_gift].users[agent].add(
                                self.step_count
                            )
                            self.artifacts[art_to_gift].users[target_tag].add(
                                self.step_count
                            )
                            status = "Success"
                            infos[target_tag]["Artifact received"] = (
                                f"{self.agent_names[agent]} gave you "
                                f"the artifact {art_to_gift}."
                            )
                    else:
                        status = (
                            f"Failed. No artifact with name {art_to_gift} in inventory"
                        )
                else:
                    status = f"Failed. Target being {target_name} not nearby."

                infos[agent]["Artifact give status"] = status
                self.logger.log(
                    time=self.step_count,
                    event_type=Event.GIVE_ARTIFACT,
                    agent_tag=agent,
                    agent_name=self.agent_names[agent],
                    target_tag=target_tag,
                    target_name=target_name,
                    artifact_name=art_to_gift,
                    status=status,
                )
            # ---------------------------

            # Artifact active interaction
            # These actions are all artifact specific so we just check which artifact offers
            # ---------------------------
            else:
                pose = self.agent_pos[agent]
                interactable_artifacts = []
                interactable_artifacts.extend(self.artifacts_map[pose])
                interactable_artifacts.extend(self.agent_inventories[agent])

                artifact_found = False
                for art_name in interactable_artifacts:
                    if (
                        self.artifacts[art_name].interactable
                        and action_name in self.artifacts[art_name].actions
                    ):
                        artifact_found = True
                        break

                if artifact_found:
                    # No check cause we already checked before that the action was available
                    interaction_result = self.artifacts[art_name].interact(
                        agent_name=agent,
                        action=action_name,
                        params=action_params,
                        timestamp=self.step_count,
                    )
                    infos[agent]["Artifact interaction result"] = interaction_result

                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.ARTIFACT_INTERACTION,
                        action=action_name,
                        action_params=action_params,
                        result=interaction_result,
                        artifact=self.artifacts[art_name].serialize(),
                    )
                else:
                    infos[agent]["Artifact interaction result"] = (
                        f"No artifact with corresponding action {action_name} found"
                    )
            # ---------------------------

            self.msg_raw[agent] = message
            if len(message):
                self.chat.setdefault(self.step_count, list()).append(
                    f"{self.agent_names[agent]}: {message}"
                )

            # Move and eat
            # ---------------------------
            new_pose = self.wrap_xy(
                x=self.agent_pos[agent][0] + move[0],
                y=self.agent_pos[agent][1] + move[1],
            )

            if not move == (0, 0):
                # Agent cannot move to a position occupied by another agent
                if new_pose not in self.agent_pos.values():
                    self._update_agent_pos(agent=agent, new_pos=new_pose)
                else:
                    blocker = next(
                        t for t, p in self.agent_pos.items() if p == new_pose
                    )
                    self._note_outcome(
                        infos,
                        agent,
                        f"You bumped into {self.agent_names[blocker]} "
                        "and stayed where you are.",
                    )
                    # New pose is current agent position
                    new_pose = self.agent_pos[agent]
                    # small penalty for trying to move into another agent
                    rewards[agent] -= 0.5

            # food / energy
            # A sick agent has no appetite: the food is left untouched on the
            # cell, still there for others and for the agent once it recovers.
            if new_pose in self.food:
                if sick:
                    self._note_outcome(
                        infos,
                        agent,
                        "You are standing on food but you have no appetite. "
                        "It gave you nothing.",
                    )
                else:
                    val = self.food.pop(new_pose)
                    self.agent_energy[agent] += val
                    rewards[agent] += val
            # ---------------------------

            # Artifact passive effects
            # ---------------------------
            # Map
            if not self.inert_artifacts:
                passive_effects = []
                for art_name in self.artifacts_map[new_pose]:
                    effect = self.artifacts[art_name].passive_effect(
                        timestamp=self.step_count, agent_name=agent
                    )
                    passive_effects.append(effect)

                    self.logger.log(
                        time=self.step_count,
                        agent_tag=agent,
                        agent_name=self.agent_names[agent],
                        event_type=Event.ARTIFACT_PASSIVE_INTERACTION,
                        result=effect,
                        artifact=self.artifacts[art_name].serialize(),
                    )
                if passive_effects:
                    infos[agent][
                        "Passive interaction result - Artifacts at position"
                    ] = passive_effects

                # Inventory
                passive_effects = []
                for art_name in self.agent_inventories[agent]:
                    # A virus the agent is still incubating stays out of its
                    # own prompt: this effect names the artifact outright.
                    if self._is_hidden_infection(art_name):
                        continue
                    effect = self.artifacts[art_name].passive_effect(
                        timestamp=self.step_count, agent_name=agent
                    )
                    passive_effects.append(effect)
                    # We do not log these ones...
                if passive_effects:
                    infos[agent][
                        "Passive interaction result - Artifacts in inventory"
                    ] = passive_effects
            # ---------------------------
        # ================================

        # World events
        # ================================
        # Handle artifacts expiration
        # ---------------------------
        # Map
        updated_artifacts = defaultdict(set)
        for cell, artifacts in self.artifacts_map.items():
            for art_name in artifacts:
                self.artifacts[art_name].remaining_time -= 1
                if self.artifacts[art_name].remaining_time <= 0:
                    artifact = self.artifacts.pop(art_name)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ARTIFACT_REMOVED,
                        artifact=artifact.serialize(),
                        pose=cell,
                    )
                else:
                    updated_artifacts[cell].add(art_name)
        self.artifacts_map = updated_artifacts

        # Inventory
        updated_inventory = defaultdict(set)
        for agent_tag, inventory in self.agent_inventories.items():
            for art_name in inventory:
                artifact = self.artifacts[art_name]
                # An infection that is still incubating burns its latency, not
                # its lifespan: viral_lifespan counts symptomatic steps only, so
                # a long incubation does not shorten the infectious window.
                if isinstance(artifact, ViralArtifact) and artifact.incubation > 0:
                    artifact.incubation -= 1
                elif artifact.remaining_time is not None:
                    artifact.remaining_time -= 1

                if self.artifacts[art_name].remaining_time <= 0:
                    artifact = self.artifacts.pop(art_name)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    # An infection running out its lifespan is a recovery. Say
                    # so, or the agent keeps believing it cannot move: it was
                    # told every sick step that it was frozen.
                    if isinstance(artifact, ViralArtifact):
                        self.agent_recoveries[agent_tag] += 1
                        if artifact.symptomatic:
                            infos.setdefault(agent_tag, {})["Health"] = (
                                "You have recovered from the sickness. Your "
                                "strength is back: you can move and eat again."
                            )
                    self.logger.log(
                        time=self.step_count,
                        event_type=Event.ARTIFACT_REMOVED,
                        artifact=artifact.serialize(),
                        possessor_tag=agent_tag,
                        possessor_name=self.agent_names[agent_tag],
                    )
                else:
                    updated_inventory[agent_tag].add(art_name)
        self.agent_inventories = updated_inventory

        # Verify that artifacts are unique. And if they are not, we just leave those in the map.
        self._cleanup_artifact_duplicates()
        # ---------------------------

        # Viral artifacts heal, spread and outbreak
        # ---------------------------
        # Healing first: an infection caught this step gets no instant cure.
        if not self.inert_artifacts:
            self._heal_at_health_centers(infos)
            self._spread_viral_artifacts(infos)
        self._seed_viral_outbreak(infos)
        # ---------------------------

        # Environment-seeded artifacts
        # ---------------------------
        self._seed_init_artifacts()
        # ---------------------------

        # ---- Handle energy and time loss for all agents ----
        # Symptomatic viral artifacts multiply the energy consumed per step.
        # Incubating ones cost nothing extra: a silent carrier feels fine.
        for a in self.agent_registry:
            energy_loss = 1
            if not self.inert_artifacts:
                for art_name in self.agent_inventories[a]:
                    artifact = self.artifacts.get(art_name)
                    if isinstance(artifact, ViralArtifact) and artifact.symptomatic:
                        energy_loss *= self.viral_energy_multiplier
            self.agent_energy[a] -= energy_loss
            self.agent_time[a] -= 1

        # ---- handle deaths ----
        # Agents with energy <= 0 are considered dead.
        # They leave behind food in their position.
        # They are removed from the environment and their position.
        # We store their last observation so we can return it.
        # Each symptomatic step is also one lethality roll, at a hazard that
        # ramps with how long the agent has been sick (_death_hazard).
        succumbed = set()
        if self.viral_death_probability > 0 and not self.inert_artifacts:
            if self.rng is None:
                self.rng = np.random.default_rng()
            succumbed = {
                a
                for a in self.agent_registry
                if self.rng.random() < self._death_hazard(a)
            }
        dead = {
            a: self._build_obs(a)
            for a in self.agent_registry
            if self.agent_energy[a] <= 0 or self.agent_time[a] <= 0 or a in succumbed
        }
        for a in dead:
            self._kill(a, reason="sickness" if a in succumbed else None)
            done_dict[a] = True
            rewards[a] -= 100

        # ---- handle food ----
        if self.food_mechanism:
            self._decay_and_respawn_food()
        # ================================

        observations = self._observe_all()
        observations.update(dead)  # include dead agents' last obs

        if self.verbose >= 2:
            print("Getting Available actions")
        for agent_tag in self.agent_registry:
            avail_actions = self._get_avail_actions(agent_tag=agent_tag)
            if agent_tag not in infos:
                infos[agent_tag] = {}
            infos[agent_tag]["available_actions"] = avail_actions
            if self._count_sick(agent_tag) > 0:
                infos[agent_tag]["Health"] = SICK_AGENT_NOTICE
            if self.use_colors:
                agent_color = self.agent_colors.get(agent_tag, "no color")
                infos[agent_tag]["Your color"] = agent_color
            if self.verbose >= 2:
                print("Avail actions for agent: ", agent_tag)
                print(infos[agent_tag]["available_actions"].keys())
                print()

        if self.verbose >= 2:
            print("Gotten avail actions for all agents")

        self.step_count += 1
        self.food_count.append(sum(self.food.values()))
        self._log_world_state()

        return observations, rewards, done_dict, done_dict, infos

    # ---------- internal helpers ----------
    def _assert_artifact_uniqueness(self, art_name):
        appearances = defaultdict(list)
        in_map = 0
        for pos, names in self.artifacts_map.items():
            if art_name in names:
                appearances["map"].append(pos)
                in_map += 1

        in_inv = 0
        for tag, names in self.agent_inventories.items():
            if art_name in names:
                appearances["inv"].append(tag)
                in_inv += 1

        if (in_map + in_inv) != 1:
            print(
                f"❌❌❌ WARNING\n Artifact {art_name} in {in_map} map cells and {in_inv} inventories"
            )
            return appearances
        else:
            return None

    def _cleanup_artifact_duplicates(self):
        """
        Comprehensive cleanup to ensure each artifact appears in exactly one location.

        Strategy:
        - If artifact is on map: keep on map, remove from all inventories
        - If artifact is on map multiple times: keep first occurrence, remove others
        - If artifact is only in inventories: keep first occurrence, remove others
        """
        for art_name in list(self.artifacts.keys()):
            appearances = self._assert_artifact_uniqueness(art_name)

            if appearances is None:
                continue  # Artifact is unique, no action needed

            # Count occurrences
            map_positions = appearances.get("map", [])
            inventory_agents = appearances.get("inv", [])

            # Case 1: Artifact appears on map (possibly multiple times) AND in inventories
            # → Keep on map, remove from inventories
            if map_positions and inventory_agents:
                for agent_tag in inventory_agents:
                    self.agent_inventories[agent_tag].discard(art_name)
                # If on map multiple times, keep only first
                if len(map_positions) > 1:
                    for pos in map_positions[1:]:
                        self.artifacts_map[pos].discard(art_name)

            # Case 2: Artifact appears on map multiple times only
            # → Keep first occurrence, remove others
            elif map_positions and len(map_positions) > 1:
                for pos in map_positions[1:]:
                    self.artifacts_map[pos].discard(art_name)

            # Case 3: Artifact appears in multiple inventories only
            # → Keep first occurrence, remove others
            elif inventory_agents and len(inventory_agents) > 1:
                for agent_tag in inventory_agents[1:]:
                    self.agent_inventories[agent_tag].discard(art_name)

            # Verify it's now unique
            final_check = self._assert_artifact_uniqueness(art_name)
            if final_check is not None:
                # Should never happen, but log if it does
                print(f"❌ CRITICAL: Failed to fix duplicate artifact {art_name}")
                print(f"   Still appears in: {final_check}")

    def _update_agent_pos(self, agent: str, new_pos: Tuple[int, int]):
        """Update the position of an agent in the grid. Useful to keep the pos_to_agent properly updated"""
        if agent not in self.agent_registry:
            raise ValueError(
                f"Agent {self.agent_names[agent]}({agent}) not found in the environment."
            )
        old_pos = self.agent_pos.get(agent, None)
        self.agent_trajectories[agent].append((int(new_pos[0]), int(new_pos[1])))
        self.agent_pos[agent] = new_pos
        if old_pos is not None:
            self.pos_to_agent.pop(old_pos, None)
        self.pos_to_agent[(int(new_pos[0]), int(new_pos[1]))] = agent

    def _place_agent(self, name, p=None):
        """Place an agent at a position in the grid."""
        if p is None:
            p = self._random_free_pos()
        else:
            if any(np.array_equal(p, q) for q in self.agent_pos.values()):
                raise ValueError(f"Position {p} is already occupied by another agent.")
        self.agent_trajectories[name] = []
        self._update_agent_pos(agent=name, new_pos=p)
        self.agent_energy[name] = self.init_agent_energy
        self.agent_time[name] = self.lifespan
        self.agent_spawn[name] = list()

    def _random_free_neigh(self, center):
        neighbours = [
            (center[0] + c[0], center[1] + c[1])
            for c in [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ]
        ]
        free_cells = [
            p
            for p in neighbours
            if (0 <= p[0] < self.grid_size)  # inside grid
            and (0 <= p[1] < self.grid_size)
            and (p not in self.pos_to_agent.keys())
        ]
        if free_cells:
            if self.rng is None:
                self.rng = np.random.default_rng()
            p = free_cells[int(self.rng.integers(len(free_cells)))]
            return (p[0], p[1])
        else:
            return None

    def _random_free_pos(self):
        """Find a random position in the grid that is not occupied by any agent."""
        if self.rng is None:
            self.rng = np.random.default_rng()
        while True:
            p = tuple(int(v) for v in self.rng.integers(1, self.grid_size - 1, size=2))
            if p not in self.agent_pos.values():
                return p

    def _get_food_distribution(self):
        if self._init_food_count <= 0:
            raise ValueError("_init_food_count must be > 0.")
        if self._init_food_count > self.grid_size**2:
            raise ValueError("_init_food_count exceeds number of grid cells.")

        if self.rng is None:
            self.rng = np.random.default_rng()

        if self.food_zones is None:
            density = np.full((self.grid_size, self.grid_size), 1 / self.grid_size**2)
            centers = []
        else:
            if isinstance(self.food_zones, int):
                ys = self.rng.integers(0, self.grid_size, size=self.food_zones)
                xs = self.rng.integers(0, self.grid_size, size=self.food_zones)
                centers = [(int(x), int(y)) for x, y in zip(xs, ys)]
            else:
                centers = self.food_zones

            sigma = float(getattr(self, "food_sigma", 2.0))
            if sigma <= 0:
                raise ValueError("food_sigma must be > 0.")

            w = np.full(len(centers), 1.0 / len(centers), dtype=np.float64)
            yy, xx = np.mgrid[0 : self.grid_size, 0 : self.grid_size]
            density = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

            for (cx, cy), wt in zip(centers, w):
                dx = np.abs(xx - cx)
                dy = np.abs(yy - cy)
                dx = np.minimum(dx, self.grid_size - dx)
                dy = np.minimum(dy, self.grid_size - dy)

                r2 = dx * dx + dy * dy  # squared toroidal distance
                g = np.exp(-0.5 * r2 / (sigma * sigma))
                density += wt * g

            total = density.sum()
            if total <= 0 or not np.isfinite(total):
                raise RuntimeError(
                    "Density normalization failed (sum <= 0 or non-finite)."
                )
            density /= total

        self.food_distribution = density
        self._food_cdf = None
        return density

    def _sample_density_cell(self) -> Tuple[int, int]:
        """One i.i.d. draw from the (fixed) food density; returns an (x, y) cell."""
        if self.food_distribution is None:
            self._get_food_distribution()
        if self._food_cdf is None:
            cdf = np.cumsum(self.food_distribution.ravel().astype(np.float64))  # type: ignore
            cdf[-1] = 1.0  # guard rounding so searchsorted stays in range
            self._food_cdf = cdf
        idx = int(np.searchsorted(self._food_cdf, self.rng.random(), side="right"))
        y, x = divmod(idx, self.grid_size)
        return (int(x), int(y))

    def _seed_initial_food(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

        density = self._get_food_distribution()

        # Hard count: zones fill first, surplus spills onto low-probability cells.
        flat_p = density.ravel()
        nnz = int(np.count_nonzero(flat_p))
        if nnz >= self._init_food_count:
            idx = self.rng.choice(
                self.grid_size**2,
                size=self._init_food_count,
                replace=False,
                p=flat_p,
            )
        else:
            # tail underflowed to zero: take every cell with mass, rest uniform
            idx_nz = np.flatnonzero(flat_p)
            idx_zero = self.rng.choice(
                np.flatnonzero(flat_p == 0),
                size=self._init_food_count - nnz,
                replace=False,
            )
            idx = np.concatenate([idx_nz, idx_zero])
        ys, xs = np.divmod(idx, self.grid_size)
        self.food = {(int(x), int(y)): self._max_food_value for x, y in zip(xs, ys)}
        self.empty_food = []

    def _respawn_food_one(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

        if not self.static_food:
            # One draw from the fixed density; occupied cell -> the spawn fails
            # this step, so spawns never drift out of the zones as they fill.
            p = self._sample_density_cell()
            if (
                p in self.food
                or p in self.artifacts_map
                or p in self.agent_pos.values()
            ):
                return
            self.food[p] = self._max_food_value
        else:
            if len(self.empty_food):
                idx = self.rng.integers(0, len(self.empty_food))
                p = self.empty_food.pop(idx)
                self.food[p] = self._max_food_value

    def _decay_and_respawn_food(self):
        """Decay food values and respawn expired ones"""
        if self.rng is None:
            self.rng = np.random.default_rng()

        # Spawning below must still run when the map is bare
        if len(self.food) > 0:
            food_positions = np.array(list(self.food.keys()))
            food_values = np.array(list(self.food.values()))

            # randomly choose which ones decay this step
            to_decay = self.rng.random(food_values.shape[0]) < self._food_decay_rate
            food_values[to_decay] -= self._food_decay_amount

            # find expired food
            expired = food_values <= 0
            expired_inds = np.where(expired)[0]
            decayed_alive = to_decay & ~expired

            # Update food values
            for i in np.where(decayed_alive)[0]:
                pos = tuple(food_positions[i])
                self.food[pos] = food_values[i]  # type: ignore
            # Remove expired food
            for idx in expired_inds:
                self.food.pop(tuple(food_positions[idx]), None)
                if self.static_food:
                    self.empty_food.append(tuple(food_positions[idx]))

        # Spawn food
        n_new_food = self.rng.poisson(self._food_spawn_rate)
        for _ in range(n_new_food):
            self._respawn_food_one()

    def _kill(self, agent, reason=None):
        """
        Remove an agent from the environment.
        Dead agents leave behind food in their position.
        """
        announcement = f"☠️ Agent {self.agent_names[agent]}({agent}) died "
        if reason == "sickness":
            announcement += "of the sickness."
        elif self.agent_time[agent] <= 0:
            announcement += "of old age."
            reason = "old age"
        elif self.agent_energy[agent] <= 0:
            announcement += "of hunger."
            reason = "hunger"
        else:
            reason = reason or "unknown"
            announcement += f"({reason})."
        if self.verbose >= 1:
            print(announcement)
        self.agent_registry.pop(agent, None)
        energy = self.agent_energy.pop(agent, 0)
        self.agent_time.pop(agent, None)
        self.msg_raw.pop(agent, None)
        self.agent_recoveries.pop(agent, None)
        pos = self.agent_pos.pop(agent, None)
        if pos is not None:
            # Artifacts are dropped at agent's death position
            artifacts = self.agent_inventories.pop(agent, None)
            if artifacts:
                for art_name in artifacts:
                    artifact = self.artifacts.get(art_name)
                    # Viral artifacts dropped at their host's death only
                    # survive on the map for viral_dropped_lifespan steps.
                    # A corpse is contagious whatever stage its host reached,
                    # so the incubation ends here: that keeps
                    # viral_dropped_lifespan meaning exactly what it says and
                    # spares the map pass a second clock to tick.
                    if isinstance(artifact, ViralArtifact):
                        artifact.pose = pos
                        artifact.incubation = 0
                        artifact.remaining_time = (
                            np.inf
                            if self.viral_dropped_lifespan == -1
                            else self.viral_dropped_lifespan
                        )
                        # Present as the corpse it is; the internal name stays
                        # so transmission chains keyed on it survive.
                        artifact.display_name = f"remains_of_{self.agent_names[agent]}"
                    if art_name not in self.artifacts_map[pos]:
                        self.artifacts_map[pos].add(art_name)
                    else:
                        if self.verbose >= 1:
                            print(
                                f"⚠️  Artifact {art_name} already at death position {pos}, not dropping duplicate"
                            )

            self.pos_to_agent.pop(pos, None)
            if self.dead_agent_food == "single":
                if pos in self.food:
                    self.food[pos] += max(self._max_food_value, energy)
                else:
                    self.food[pos] = max(self._max_food_value, energy)
            elif self.dead_agent_food == "area":
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        # Add food around the agent's position
                        food_pos = (pos[0] + i, pos[1] + j)
                        if (
                            0 <= food_pos[0] < self.grid_size
                            and 0 <= food_pos[1] < self.grid_size
                        ):
                            if food_pos in self.food:
                                self.food[food_pos] += self._max_food_value
                            else:
                                self.food[food_pos] = self._max_food_value
            elif self.dead_agent_food == "none":
                pass

        if self.logger:
            self.logger.log(
                time=self.step_count,
                event_type=Event.AGENT_DIED,
                agent_tag=agent,
                agent_name=self.agent_names[agent],
                position=pos,
                step=self.step_count,
                energy=energy,
                reason=reason,
            )

    def _count_food_agents_nearby(self, pos):
        """Count how many food tiles are within the vision radius of a position."""
        food_count = 0
        agent_count = 0
        r = self.vision_radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = pos[0] + dx, pos[1] + dy
                if (
                    0 <= gx < self.grid_size
                    and 0 <= gy < self.grid_size
                    and (gx, gy) in self.food
                ):
                    food_count += 1

                if (
                    0 <= gx < self.grid_size
                    and 0 <= gy < self.grid_size
                    and (gx, gy) in self.pos_to_agent
                ):
                    agent_count += 1
        return food_count, agent_count

    # ---------- observation ----------
    def _observe_all(self) -> dict[str, np.ndarray | dict]:
        """Builds observations for all agents in the environment."""
        return {a: self._build_obs(a) for a in self.agent_registry}

    def _build_obs(self, agent: str) -> np.ndarray | dict:
        """Builds the observation for a given agent.
        Returns either a numerical vector or a dict for text-based agents.
        """
        x, y = self.agent_pos[agent]
        r = self.vision_radius
        # Observation is just provided as a dict of lists.
        # It's the agent that shows how to represent it (e.g. list or grid)
        messages = {}
        observation = defaultdict(list)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = self.wrap_xy(x + dx, y + dy)
                obs_x, obs_y = dx + r, dy + r
                rel_pos = (dx, dy)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    # FOOD
                    if (gx, gy) in self.food:
                        val = self.food[(gx, gy)]
                        observation[rel_pos].append(str(val))

                    # AGENT
                    if (gx, gy) in self.pos_to_agent:
                        a2 = self.pos_to_agent[(int(gx), int(gy))]
                        if a2 != agent:
                            if self.use_colors:
                                agent_descr = f"{self.agent_names[a2]}({self.agent_colors.get(a2, 'no color')})"
                            else:
                                agent_descr = self.agent_names[a2]
                            observation[rel_pos].append(agent_descr)  # type: ignore

                            # Add message if it exists
                            msg = self.msg_raw.get(a2, "")
                            if len(msg):
                                messages[self.agent_names[a2]] = msg

                    # ARTIFACT
                    if not self.inert_artifacts:
                        for art_name in self.artifacts_map[(gx, gy)]:
                            observation[rel_pos].append(
                                self._artifact_label(art_name)
                            )
                # CELLS OUTSIDE OF MAP
                else:
                    observation[rel_pos].append("X")

        inventory_list = [
            f"A({self.artifacts[art].art_type}): {self.artifacts[art].name}"
            for art in self.agent_inventories[agent]
            if not self._is_hidden_infection(art)
        ]

        complete_obs = {
            "observation": observation,
            "message": messages,
            "energy": self.agent_energy[agent],
            "time": self.agent_time[agent],
            "inventory": inventory_list,
            "vision_radius": self.vision_radius,  # Passing it here as this can change
        }
        return complete_obs

    def _is_hidden_infection(self, art_name: str) -> bool:
        """Whether an artifact must be kept out of its own host's prompt.

        An incubating infection is silent: the host has no symptoms and must
        have no way to tell it is carrying anything. Everything in ``infos``
        and ``obs["inventory"]`` is rendered verbatim into the prompt, so the
        artifact has to be filtered out at the source rather than hoping the
        agent does not read it. Once symptomatic it is shown as normal — by
        then the host is being told it is sick anyway.
        """
        artifact = self.artifacts.get(art_name)
        return isinstance(artifact, ViralArtifact) and not artifact.symptomatic

    def _artifact_label(self, art_name: str) -> str:
        """How an artifact reads in an observation.

        A viral artifact dropped at its host's death presents as that
        being's remains, not as a labelled virus.
        """
        artifact = self.artifacts[art_name]
        display = getattr(artifact, "display_name", None)
        if display:
            return f"A(remains): {display}"
        return f"A({artifact.art_type}): {artifact.name}"

    def _ground_viral_nearby(self, agent_tag: str) -> List[str]:
        """Names of viral artifacts on the ground within reach (own cell + 8 neighbours)."""
        x, y = self.agent_pos[agent_tag]
        names = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = self.wrap_xy(x + dx, y + dy)
                for n in self.artifacts_map.get(cell, ()):
                    if isinstance(self.artifacts.get(n), ViralArtifact):
                        names.append(n)
        return names

    def _heal_at_health_centers(self, infos: dict):
        """Cures infected agents inside health centers.

        A center is always active within its radius (default 1: its cell plus
        the 8 around it). Each infected agent in the area has heal_probability
        per step of losing every hosted viral artifact. Curing a
        still-incubating infection says nothing: the host never knew it was
        carrying anything.
        """
        centers = [
            a for a in self.artifacts.values() if isinstance(a, HealthCenterArtifact)
        ]
        if not centers or not self.agent_registry:
            return
        if self.rng is None:
            self.rng = np.random.default_rng()
        for center in centers:
            for tag in list(self.agent_registry):
                if (
                    self._toroidal_distance(center.pose, self.agent_pos[tag])
                    > center.radius
                ):
                    continue
                # Everyone in the area is told what the place is, not only
                # those standing on its cell.
                infos.setdefault(tag, {}).setdefault("Nearby facilities", []).append(
                    center.passive_effect(timestamp=self.step_count, agent_name=tag)
                )
                hosted = [
                    n
                    for n in self.agent_inventories[tag]
                    if isinstance(self.artifacts.get(n), ViralArtifact)
                ]
                if not hosted or self.rng.random() >= center.heal_probability:
                    continue
                was_sick = any(self.artifacts[n].symptomatic for n in hosted)
                center.users[tag].add(self.step_count)
                self.agent_recoveries[tag] += len(hosted)
                for n in hosted:
                    artifact = self.artifacts.pop(n)
                    artifact.deletion_time = self.step_count
                    self.expired_artifacts.append(artifact)
                    self.agent_inventories[tag].discard(n)
                    if self.logger:
                        self.logger.log(
                            time=self.step_count,
                            event_type=Event.VIRAL_HEALED,
                            agent_tag=tag,
                            agent_name=self.agent_names[tag],
                            spot=center.name,
                            artifact=artifact.serialize(),
                        )
                if was_sick:
                    infos.setdefault(tag, {})["Health"] = (
                        f"You have been treated at {center.name} and recovered "
                        "from the sickness."
                    )
                    if self.verbose >= 1:
                        print(
                            f"⚕️ {self.agent_names[tag]}({tag}) healed at {center.name} ⚕️"
                        )

    def _death_hazard(self, agent_tag: str) -> float:
        """Per-step chance this agent dies of the sickness.

        Zero at symptom onset, ramping linearly to viral_death_probability at
        the end of the infectious window (constant at the maximum when
        viral_lifespan is -1, since there is no window to ramp over).
        """
        frac = 0.0
        for name in self.agent_inventories[agent_tag]:
            artifact = self.artifacts.get(name)
            if not isinstance(artifact, ViralArtifact) or not artifact.symptomatic:
                continue
            if artifact.lifespan in (np.inf, -1) or artifact.remaining_time == np.inf:
                frac = 1.0
            else:
                frac = max(
                    frac, 1.0 - max(artifact.remaining_time, 0) / artifact.lifespan
                )
        return self.viral_death_probability * frac

    def _infection_protection(self, agent_tag: str) -> float:
        """Multiplier on the agent's probability of contracting an infection.

        The best (lowest) protection carried wins; carrying several protective
        artifacts does not stack.
        """
        return min(
            (
                self.artifacts[a].infection_protection
                for a in self.agent_inventories[agent_tag]
                if a in self.artifacts
            ),
            default=1.0,
        )

    def _hosted_viral_strains(self, agent_tag: str) -> Set[str]:
        """Strains of the viral artifacts currently in an agent's inventory.

        Both phases count: an incubating host is already immune to re-catching
        the strain it is carrying.
        """
        strains = set()
        for art_name in self.agent_inventories[agent_tag]:
            artifact = self.artifacts.get(art_name)
            if isinstance(artifact, ViralArtifact):
                strains.add(artifact.strain)
        return strains

    def _toroidal_distance(self, pos_a, pos_b) -> int:
        """Chebyshev distance between two cells on the wrapping grid."""
        dx = abs(pos_a[0] - pos_b[0])
        dy = abs(pos_a[1] - pos_b[1])
        return max(min(dx, self.grid_size - dx), min(dy, self.grid_size - dy))

    def _load_init_artifacts(
        self, init_artifacts: str | Path | List[dict] | None
    ) -> List[dict]:
        """Loads and validates the artifacts the environment seeds by itself.

        Accepts a list of entries or a path to a JSON file containing one.
        Each entry is {name, payload, pose, agent, role, lifespan, step}:
        only name is required. The artifact appears on the map at pose
        (default: a random free cell) or in inventories: agent (tag or name)
        targets one being, role targets every living being with that persona
        role, both together the named being required to hold the role.
        lifespan defaults to -1 (never expires) and step to 0 (seeded before
        the first timestep).
        """
        if init_artifacts is None:
            return []
        if isinstance(init_artifacts, (str, Path)):
            with open(init_artifacts) as f:
                init_artifacts = json.load(f)
        if not isinstance(init_artifacts, list):
            raise ValueError(
                f"init_artifacts must be a list of entries, got {type(init_artifacts)}"
            )

        validated = []
        for i, entry in enumerate(init_artifacts):
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(
                    f"init_artifacts[{i}] must be a dict with at least a 'name' key, got {entry}"
                )
            art_type = entry.get("type", "text")
            if art_type not in ("text", "ppe", "health_center"):
                raise ValueError(
                    f"init_artifacts[{i}] ({entry['name']}): only 'text', 'ppe' and "
                    f"'health_center' artifacts can be seeded, got type '{art_type}'. "
                    "Viral artifacts are seeded through the "
                    "viral_init_infected/viral_outbreak_step parameters."
                )
            pose = entry.get("pose")
            agent = entry.get("agent")
            if pose is not None and agent is not None:
                raise ValueError(
                    f"init_artifacts[{i}] ({entry['name']}): pose and agent are mutually "
                    "exclusive; use pose for the map, agent for an inventory"
                )
            if pose is not None:
                if len(pose) != 2:
                    raise ValueError(
                        f"init_artifacts[{i}] ({entry['name']}): pose must be [x, y], got {pose}"
                    )
                x, y = int(pose[0]), int(pose[1])
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    raise ValueError(
                        f"init_artifacts[{i}] ({entry['name']}): pose {pose} outside the "
                        f"{self.grid_size}x{self.grid_size} grid"
                    )
                pose = [x, y]
            lifespan = int(entry.get("lifespan", -1))
            if lifespan != -1 and lifespan <= 0:
                raise ValueError(
                    f"init_artifacts[{i}] ({entry['name']}): lifespan must be > 0 or -1, got {lifespan}"
                )
            step = int(entry.get("step", 0))
            if step < 0:
                raise ValueError(
                    f"init_artifacts[{i}] ({entry['name']}): step cannot be negative, got {step}"
                )
            role = entry.get("role")
            if role is not None and pose is not None:
                raise ValueError(
                    f"init_artifacts[{i}] ({entry['name']}): role targets "
                    "inventories; it cannot be combined with pose"
                )
            extra = {}
            if art_type == "health_center":
                if agent is not None or role is not None:
                    raise ValueError(
                        f"init_artifacts[{i}] ({entry['name']}): a health center is "
                        "fixed to the map; use pose, not agent or role"
                    )
                heal_probability = float(entry.get("heal_probability", 0.2))
                radius = int(entry.get("radius", 1))
                if not 0.0 <= heal_probability <= 1.0:
                    raise ValueError(
                        f"init_artifacts[{i}] ({entry['name']}): heal_probability must be in [0, 1]"
                    )
                if radius < 0:
                    raise ValueError(
                        f"init_artifacts[{i}] ({entry['name']}): radius cannot be negative"
                    )
                extra = {"heal_probability": heal_probability, "radius": radius}
            validated.append(
                {
                    "name": str(entry["name"]),
                    "type": art_type,
                    "payload": str(entry.get("payload", "")),
                    "pose": pose,
                    "agent": str(agent) if agent is not None else None,
                    "role": str(role) if role is not None else None,
                    "lifespan": lifespan,
                    "step": step,
                    **extra,
                }
            )
        return validated

    def _resolve_agent_ref(self, agent_ref: str) -> str | None:
        """Resolves an agent tag or name to the tag of a living agent."""
        if agent_ref in self.agent_registry:
            return agent_ref
        for tag, name in self.agent_names.items():
            if name == agent_ref and tag in self.agent_registry:
                return tag
        return None

    def _seed_init_artifacts(self):
        """Places the configured init artifacts whose seeding step has been reached."""
        remaining = []
        for entry in self._pending_init_artifacts:
            if entry["step"] > self.step_count:
                remaining.append(entry)
                continue

            # Inventory targets: agent (tag or name), role (every living agent
            # with it), or both (the named agent, required to hold the role).
            agent_ref, role = entry.get("agent"), entry.get("role")
            targets: List[str | None] = [None]
            if agent_ref is not None or role is not None:
                if agent_ref is not None:
                    resolved = self._resolve_agent_ref(agent_ref)
                    targets = [resolved] if resolved is not None else []
                    if role is not None:
                        targets = [
                            t for t in targets if self.agent_roles.get(t) == role
                        ]
                else:
                    targets = [
                        t
                        for t in self.agent_registry
                        if self.agent_roles.get(t) == role
                    ]
                if not targets:
                    print(
                        f"⚠️  Failed to seed artifact {entry['name']}: no living "
                        f"agent matching {agent_ref or ''}"
                        f"{' role ' + role if role else ''} at step {self.step_count}"
                    )
                    continue

            lifespan = np.inf if entry["lifespan"] == -1 else entry["lifespan"]
            params = {
                k: entry[k] for k in ("heal_probability", "radius") if k in entry
            } or None
            for to_inventory in targets:
                if to_inventory is not None:
                    pose = self.agent_pos[to_inventory]
                else:
                    pose = entry["pose"]
                    pose = tuple(pose) if pose is not None else self._random_free_pos()
                status = self.add_artifact(
                    pose=pose,
                    art_type=entry.get("type", "text"),
                    art_name=entry["name"],
                    payload=entry["payload"],
                    creator="environment",
                    lifespan=lifespan,
                    to_inventory=to_inventory,
                    artifact_params=params,
                )
                if status.startswith("Created"):
                    where = (
                        f"in {self.agent_names[to_inventory]}'s inventory"
                        if to_inventory is not None
                        else f"at {pose}"
                    )
                    if self.verbose >= 1:
                        print(f"🌱 Seeded artifact {entry['name']} {where} 🌱")
                else:
                    print(f"⚠️  Failed to seed artifact {entry['name']}: {status}")
        self._pending_init_artifacts = remaining

    def _seed_viral_outbreak(self, infos: dict):
        """Infects viral_init_infected random agents at step viral_outbreak_step."""
        if self.viral_init_infected <= 0 or self.step_count != self.viral_outbreak_step:
            return
        if not self.agent_registry:
            return
        if self.rng is None:
            self.rng = np.random.default_rng()

        candidates = list(self.agent_registry)
        n_infected = min(self.viral_init_infected, len(candidates))
        chosen = self.rng.choice(len(candidates), size=n_infected, replace=False)
        for idx in chosen:
            agent_tag = candidates[int(idx)]
            art_name = self.infect_agent(agent_tag=agent_tag)
            if art_name is not None:
                # An index case that is still incubating is told nothing — see
                # _spread_viral_artifacts. The console line below is for the
                # operator, not the agent.
                if self.artifacts[art_name].symptomatic:
                    infos.setdefault(agent_tag, {})["Infection"] = (
                        f"Artifact {art_name} appeared in your inventory."
                    )
                if self.verbose >= 1:
                    print(
                        f"🦠 Viral outbreak: {self.agent_names[agent_tag]}({agent_tag}) infected with {art_name} 🦠"
                    )

    def _spread_viral_artifacts(self, infos: dict):
        """Spreads viral artifacts to agents nearby their current location.

        Every *symptomatic* viral artifact — hosted in an agent's inventory or
        lying on the map after its host died — has a
        viral_infection_probability chance of copying itself into the inventory
        of each agent within viral_infection_radius cells. An agent cannot host
        the same strain twice. Artifacts still incubating transmit nothing, so
        a silent carrier is genuinely harmless until its symptoms start.

        Distance is the Chebyshev distance on the torus (_toroidal_distance),
        so the default radius of 1 means transmission by contact only: the 8
        cells directly adjacent to the source, diagonals included and wrapping
        across the grid seam.

        Note hosts are immobilised and stop foraging while sick, so they are
        stationary sources and healthy agents have to walk into them. Realised
        R0 therefore falls below the free-mixing estimate annotated in
        run_viral_experiment.sh; measure it with analysis_scripts/compute_r0.py
        rather than trusting the formula.
        """
        # Snapshot current infections so that copies created now do not
        # spread in the same step. Sources are (host_tag, position, artifact),
        # with host_tag None for artifacts lying on the map.
        infections = []
        for host in self.agent_registry:
            for art_name in self.agent_inventories[host]:
                artifact = self.artifacts.get(art_name)
                if isinstance(artifact, ViralArtifact) and artifact.symptomatic:
                    infections.append((host, self.agent_pos[host], artifact))
        for pos, art_names in self.artifacts_map.items():
            for art_name in art_names:
                artifact = self.artifacts.get(art_name)
                if isinstance(artifact, ViralArtifact) and artifact.symptomatic:
                    infections.append((None, pos, artifact))
        if not infections:
            return

        if self.rng is None:
            self.rng = np.random.default_rng()

        hosted_strains = {
            tag: self._hosted_viral_strains(tag) for tag in self.agent_registry
        }
        protection = {
            tag: self._infection_protection(tag) for tag in self.agent_registry
        }
        for host, source_pos, artifact in infections:
            for target in self.agent_registry:
                if target == host:
                    continue
                if artifact.strain in hosted_strains[target]:
                    continue
                distance = self._toroidal_distance(source_pos, self.agent_pos[target])
                if distance > self.viral_infection_radius:
                    continue
                if (
                    self.rng.random()
                    >= self.viral_infection_probability * protection[target]
                ):
                    continue

                art_name = self.infect_agent(
                    agent_tag=target, source_artifact=artifact, source_tag=host
                )
                if art_name is None:
                    continue
                hosted_strains[target].add(artifact.strain)
                # Say nothing while the new infection incubates: everything in
                # infos is rendered verbatim into the agent's prompt, so this
                # message alone would give the whole latent phase away.
                if self.artifacts[art_name].symptomatic:
                    infos.setdefault(target, {})["Infection"] = (
                        f"Artifact {art_name} appeared in your inventory."
                    )

    def _get_nearby_agents(self, agent_tag: str) -> List[str]:
        agent_position = self.agent_pos[agent_tag]
        x, y = agent_position
        r = self.vision_radius
        nearby_agents = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = x + dx, y + dy
                if (gx, gy) in self.pos_to_agent and (gx, gy) != agent_position:
                    nearby_agents.append(self.pos_to_agent[(gx, gy)])
        return nearby_agents

    def _get_avail_actions(self, agent_tag: str):
        agent_position = self.agent_pos[agent_tag]

        # A viral artifact past its incubation makes an agent sick: it cannot
        # move or steal energy (and, in step(), it stops eating). "move" itself
        # stays on the list, restricted to "stay", so that a sick agent alone in
        # an empty patch is never left with an empty action list — the prompt
        # would then offer it nothing to choose and every reply would fail to
        # parse. An agent still incubating keeps the full action set: nothing
        # here may hint that it is carrying anything.
        sick = self._count_sick(agent_tag) > 0

        available_actions = {"move": deepcopy(ACTION_TEXT["move"])}
        if sick:
            available_actions["move"]["description"] = (
                "You are too sick to move. You can only stay where you are."
            )
            available_actions["move"]["params"]["direction"] = (
                "Must be 'stay': you are too sick to move."
            )

        # Colors
        # ---------------------------
        if self.use_colors:
            available_actions["set_color"] = deepcopy(ACTION_TEXT["set_color"])
        # ---------------------------

        # Agent interactions
        # ---------------------------
        x, y = agent_position
        r = self.vision_radius
        nearby_agents = False
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                gx, gy = x + dx, y + dy
                if (gx, gy) in self.pos_to_agent and (gx, gy) != agent_position:
                    nearby_agents = True
                    break
            if nearby_agents:
                break

        if nearby_agents and self.food_mechanism:
            available_actions["give"] = deepcopy(ACTION_TEXT["give"])
            # A sick agent is too weak to steal, but can still give energy away
            if not sick:
                available_actions["take"] = deepcopy(ACTION_TEXT["take"])
        # ---------------------------

        # Burials: remains within reach, and the digger must be well
        # ---------------------------
        if self.burials and not sick and self._ground_viral_nearby(agent_tag):
            available_actions["bury"] = deepcopy(ACTION_TEXT["bury"])
        # ---------------------------

        # Artifacts
        # ---------------------------
        if (
            self.artifact_creation
            and self.agent_energy[agent_tag] >= self.artifact_creation_cost
        ):
            available_actions["create_artifact"] = deepcopy(
                ACTION_TEXT["create_artifact"]
            )
            # .replace, not .format: the string embeds a dict repr full of braces
            available_actions["create_artifact"]["params"]["payload"] = (
                available_actions["create_artifact"]["params"]["payload"].replace(
                    "{max_text_artifact_size}", str(self.max_text_artifact_size)
                )
            )
            if self.artifact_creation_cost > 0:
                available_actions["create_artifact"]["description"] += (
                    f" It costs {self.artifact_creation_cost} energy."
                )

        # Only give actions if the agents use the inventory
        if not self.inert_artifacts:
            for art_name in self.artifacts_map[agent_position]:
                if self.artifacts[art_name].interactable:
                    available_actions.update(self.artifacts[art_name].actions)

            if self.use_inventory:
                # Non-interactable artifacts (e.g. viral) cannot be picked up
                pickupable = [
                    art_name
                    for art_name in self.artifacts_map[agent_position]
                    if self.artifacts[art_name].interactable
                ]
                if pickupable:
                    available_actions["pickup_artifact"] = deepcopy(
                        ACTION_TEXT["pickup_artifact"]
                    )

                if self.agent_inventories[agent_tag]:
                    # Non-interactable artifacts (e.g. viral) cannot be
                    # dropped or given away
                    transferable = [
                        art_name
                        for art_name in self.agent_inventories[agent_tag]
                        if self.artifacts[art_name].interactable
                    ]
                    if transferable:
                        available_actions["drop_artifact"] = deepcopy(
                            ACTION_TEXT["drop_artifact"]
                        )
                    for art_name in self.agent_inventories[agent_tag]:
                        if self.artifacts[art_name].interactable:
                            available_actions.update(self.artifacts[art_name].actions)

                    if transferable and nearby_agents:
                        available_actions["give_artifact"] = deepcopy(
                            ACTION_TEXT["give_artifact"]
                        )
        # ---------------------------

        # Reproduction
        # ---------------------------
        if (
            self.reproduction_allowed
            and self.agent_energy[agent_tag] >= self.reproduction_cost
        ):
            action = deepcopy(ACTION_TEXT["reproduce"])
            if self.food_mechanism:
                params = {
                    "energy": action["params"]["energy"].format(
                        reproduction_cost=str(self.reproduction_cost)
                    ),
                    "name": action["params"]["name"],
                }
            else:
                params = {
                    "name": action["params"]["name"],
                }
            if self.parent_authored_genome:
                params["offspring_genome"] = action["params"]["offspring_genome"]
            available_actions["reproduce"] = {
                "description": action["description"].format(
                    reproduction_cost=str(self.reproduction_cost)
                ),
                "params": params,
            }
        # ---------------------------

        self.agent_avail_actions[agent_tag] = available_actions
        return available_actions

    def wrap_xy(self, x: int, y: int) -> tuple[int, int]:
        return x % self.grid_size, y % self.grid_size

    # ---------- rendering (optional) ----------

    def render(self, mode="human"):
        assert mode in ("ascii", "rgb_array", "human"), mode

        if mode == "ascii":
            grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
            for fx, fy in self.food:
                grid[fx, fy] = "F"
            for a, (x, y) in self.agent_pos.items():
                grid[x, y] = a[0].upper()
            print("\n".join(" ".join(row) for row in grid))
            return

        # --- Init pygame (no window: rgb_array draws on offscreen surfaces) ---
        if not self._pygame_inited:
            pygame.init()
            self._sidebar_width = 300
            default_size = self.grid_size * self._cell_size
            self._window_size = (default_size + self._sidebar_width, default_size)
            self._font = pygame.font.SysFont(None, 15)  # body / wrapping
            self._font_hdr = pygame.font.SysFont(None, 17, bold=True)  # section headers
            self._font_tag = pygame.font.SysFont(
                None, 13, bold=True
            )  # small caps labels
            self._scroll_offset = 0
            self._msg_log = []
            self._seen_msgs = set()
            self._pygame_inited = True

        # --- Append messages for the step ---
        if self.step_count not in self._seen_msgs:
            messages = self.chat.get(self.step_count - 1)
            if messages:
                self._msg_log.append(f"Step {self.step_count - 1}")
                self._msg_log.extend(messages)
                self._msg_log.append("--")
            self._seen_msgs.add(self.step_count)

        # --- Common draw config ---
        line_height = 18
        x_margin = 10
        y_start = 30
        max_width = self._sidebar_width - 2 * x_margin

        wrapped_lines = []
        for raw_line in self._msg_log:
            wrapped_lines.extend(self._wrap_text(raw_line or "", max_width))

        # --- Draw sidebar to a surface ---
        SB_BG = (252, 252, 252)
        SB_STRIP = (236, 236, 236)
        SB_ACCENT = (70, 130, 180)
        SB_TEXT = (35, 35, 35)
        SB_DIVIDER = (218, 218, 218)

        def draw_sidebar_surface(height):
            surface = pygame.Surface((self._sidebar_width, height))
            surface.fill(SB_BG)

            # Left accent bar
            pygame.draw.rect(surface, SB_ACCENT, pygame.Rect(0, 0, 4, height))

            # ── Header ──────────────────────────────────────
            hdr_h = 38
            pygame.draw.rect(
                surface, SB_STRIP, pygame.Rect(4, 0, self._sidebar_width - 4, hdr_h)
            )
            step_surf = self._font_hdr.render(f"Step  {self.step_count}", True, SB_TEXT)
            surface.blit(step_surf, (12, (hdr_h - step_surf.get_height()) // 2))
            pygame.draw.line(
                surface, SB_DIVIDER, (4, hdr_h), (self._sidebar_width, hdr_h)
            )
            y = hdr_h + 1

            # ── Chat log ────────────────────────────────────
            msg_lh = 17
            visible_height = height - y
            max_lines = visible_height // msg_lh
            start_idx = max(0, len(wrapped_lines) - max_lines - self._scroll_offset)
            end_idx = start_idx + max_lines
            visible_lines = wrapped_lines[start_idx:end_idx]

            for line in visible_lines:
                if y + msg_lh > height:
                    break
                if line == "--":
                    pygame.draw.line(
                        surface,
                        SB_DIVIDER,
                        (12, y + msg_lh // 2),
                        (self._sidebar_width - 12, y + msg_lh // 2),
                    )
                elif line.startswith("Step ") and line[5:].strip().isdigit():
                    text_surf = self._font_tag.render(line, True, SB_ACCENT)
                    surface.blit(
                        text_surf, (12, y + (msg_lh - text_surf.get_height()) // 2)
                    )
                else:
                    text_surf = self._font.render(line, True, SB_TEXT)
                    surface.blit(
                        text_surf, (12, y + (msg_lh - text_surf.get_height()) // 2)
                    )
                y += msg_lh

            return surface

        # --- Determine grid pixel dimensions ---
        if mode == "rgb_array":
            grid_pixel_w = self.grid_size * self._cell_size
            grid_pixel_h = self.grid_size * self._cell_size
        else:
            grid_pixel_w = self._window_size[0] - self._sidebar_width
            grid_pixel_h = self._window_size[1]

        cell_w = grid_pixel_w / self.grid_size
        cell_h = grid_pixel_h / self.grid_size

        # --- Draw grid surface ---
        # Colors from publication-ready palette
        BG_COLOR = (245, 245, 245)
        GRID_LINE_COLOR = (225, 225, 225)
        VISION_COLOR = (225, 225, 225)
        FOOD_LIGHT = (120, 200, 120)
        FOOD_DARK = (34, 139, 34)
        ARTIFACT_COLOR = (180, 60, 70)
        VIRAL_COLOR = (150, 40, 160)
        INCUBATING_COLOR = (250, 178, 25)
        AGENT_PALETTE = {"numerical": (70, 130, 180), "text": (40, 90, 140)}

        cw = max(1, int(cell_w))
        ch = max(1, int(cell_h))

        grid_surf = pygame.Surface((grid_pixel_w, grid_pixel_h))
        grid_surf.fill(BG_COLOR)

        # Vision radius — solid gray cells
        r = self.vision_radius
        vision_cells = set()
        for agent in self.agent_registry:
            cx, cy = self.agent_pos[agent]
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    vision_cells.add(self.wrap_xy(cx + dx, cy + dy))
        for gx, gy in vision_cells:
            pygame.draw.rect(
                grid_surf,
                VISION_COLOR,
                pygame.Rect(int(gy * cell_w), int(gx * cell_h), cw, ch),
            )

        # Food — full cell, varying green
        for (x, y), val in self.food.items():
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue
            ratio = max(0.0, min(1.0, float(val) / float(self._max_food_value)))
            color = tuple(
                int(FOOD_LIGHT[i] + (FOOD_DARK[i] - FOOD_LIGHT[i]) * ratio)
                for i in range(3)
            )
            pygame.draw.rect(
                grid_surf,
                color,
                pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
            )

        # Artifacts — full cell, amber (viral ones in purple)
        for (x, y), arts in self.artifacts_map.items():
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size and len(arts):
                if any(
                    isinstance(self.artifacts.get(art_name), ViralArtifact)
                    for art_name in arts
                ):
                    color = VIRAL_COLOR
                else:
                    color = ARTIFACT_COLOR
                pygame.draw.rect(
                    grid_surf,
                    color,
                    pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
                )

        # Agents — full cell, colored (sick ones purple, incubating ones amber)
        for agent, agent_type in self.agent_registry.items():
            if self._count_sick(agent) > 0:
                color = VIRAL_COLOR
            elif self._count_viral(agent) > 0:
                color = INCUBATING_COLOR
            else:
                color = AGENT_PALETTE.get(agent_type, (100, 100, 100))
            x, y = self.agent_pos[agent]
            pygame.draw.rect(
                grid_surf,
                color,
                pygame.Rect(int(y * cell_w), int(x * cell_h), cw, ch),
            )

        # Grid lines (drawn last, on top of everything)
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                grid_surf,
                GRID_LINE_COLOR,
                (int(i * cell_w), 0),
                (int(i * cell_w), grid_pixel_h),
            )
            pygame.draw.line(
                grid_surf,
                GRID_LINE_COLOR,
                (0, int(i * cell_h)),
                (grid_pixel_w, int(i * cell_h)),
            )

        # --- RGB array output ---
        if mode == "rgb_array":
            sidebar_surf = draw_sidebar_surface(grid_pixel_h)
            combined = pygame.Surface(
                (grid_pixel_w + self._sidebar_width, grid_pixel_h)
            )
            combined.blit(grid_surf, (0, 0))
            combined.blit(sidebar_surf, (grid_pixel_w, 0))
            return pygame.surfarray.array3d(combined).transpose((1, 0, 2))

        # --- Human display ---
        if self._screen is None:
            self._screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE)
            pygame.display.set_caption("OpenGridWorld")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_inited = False
                self._screen = None
                return
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._screen = pygame.display.set_mode(
                    self._window_size, pygame.RESIZABLE
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._scroll_offset = max(0, self._scroll_offset - 3)
                elif event.key == pygame.K_DOWN:
                    self._scroll_offset += 3
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self._scroll_offset = max(0, self._scroll_offset - 3)
                elif event.y < 0:
                    self._scroll_offset += 3

        self._screen.blit(grid_surf, (0, 0))  # type: ignore
        sidebar_surf = draw_sidebar_surface(self._window_size[1])
        self._screen.blit(sidebar_surf, (grid_pixel_w, 0))  # type: ignore
        pygame.display.flip()
        return pygame.surfarray.array3d(grid_surf).transpose((1, 0, 2))

    def _wrap_text(self, text, max_width):
        """Wrap a single string into multiple lines that fit within max_width."""
        words = text.split(" ")
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if self._font.size(test_line)[0] <= max_width:
                line = test_line
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    @staticmethod
    def _serialize(obj):
        """Helper function to serialize objects for saving state."""
        if obj is None:
            return None

        if isinstance(obj, np.random.Generator):
            return {
                "__type__": "rng",
                "state": obj.bit_generator.state,
            }

        if isinstance(obj, str):
            return {
                "__type__": "str",
                "data": obj,
            }

        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.tolist(),
            }

        if isinstance(obj, Artifact):
            return {
                "__type__": "artifact",
                "data": obj.serialize(),
            }

        raise TypeError(f"Type {type(obj)} not serializable")

    @staticmethod
    def _deserialize(data):
        """Helper function to deserialize objects from saved state."""
        if data is None:
            return None

        if data["__type__"] == "str":
            return data["data"]

        if data["__type__"] == "rng":
            rng = np.random.default_rng()
            rng.bit_generator.state = data["state"]
            return rng

        if data["__type__"] == "ndarray":
            return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])

        if data["__type__"] == "artifact":
            art_type = data["data"].pop("art_type", None)
            if art_type == "text":
                return TextArtifact.deserialize(data["data"])
            if art_type == "viral":
                return ViralArtifact.deserialize(data["data"])
            if art_type == "ppe":
                return PPEArtifact.deserialize(data["data"])
            if art_type == "health_center":
                return HealthCenterArtifact.deserialize(data["data"])
            return Artifact.deserialize(data["data"])

        raise TypeError(f"Type {data['__type__']} not deserializable")

    def get_state_ckpt(self):
        """Save the state checkpoint"""

        def enc_pos(pos):
            return f"{pos[0]}:{pos[1]}"

        rng = (
            self._serialize(self.rng)
            if self.rng is not None
            else {"__type__": "rng", "state": None}
        )
        food = {enc_pos(pos): v for pos, v in self.food.items()}
        artifacts_map = {
            enc_pos(pos): list(arts) for pos, arts in self.artifacts_map.items()
        }
        agent_trajectories = {
            agent: [enc_pos(p) for p in traj]
            for agent, traj in self.agent_trajectories.items()
        }

        state_ckpt = {
            "rng": rng,
            "food": food,
            "food_count": list(self.food_count),
            "food_distribution": self._serialize(self.food_distribution),
            "empty_food": self.empty_food,
            "artifacts": {
                name: self._serialize(art) for name, art in self.artifacts.items()
            },
            "artifacts_map": artifacts_map,
            "agent_inventories": {
                agent: list(inv) for agent, inv in self.agent_inventories.items()
            },
            "agent_recoveries": dict(self.agent_recoveries),
            "expired_artifacts": [
                self._serialize(art) for art in self.expired_artifacts
            ],
            "viral_infection_count": self.viral_infection_count,
            "pending_init_artifacts": self._pending_init_artifacts,
            "agent_pos": {agent: enc_pos(pos) for agent, pos in self.agent_pos.items()},
            "agent_trajectories": agent_trajectories,
            "agent_avail_actions": self.agent_avail_actions,
            "pos_to_agent": {
                enc_pos(pos): agent for pos, agent in self.pos_to_agent.items()
            },
            "agent_energy": self.agent_energy,
            "agent_time": self.agent_time,
            "agent_spawn": self.agent_spawn,
            "agent_names": self.agent_names,
            "agent_roles": self.agent_roles,
            "agent_colors": self.agent_colors,
            "msg_raw": {
                agent: self._serialize(msg) for agent, msg in self.msg_raw.items()
            },
            "chat": self.chat,
            "agent_registry": self.agent_registry,
            "step_count": self.step_count,
            "logger_save_path": self.logger.save_path if self.logger else None,
            "logger_data": self.logger._sanitize(self.logger.data)
            if self.logger
            else None,
        }
        return state_ckpt

    def set_state_ckpt(self, state_ckpt):
        """Load the state checkpoint"""
        self.step_count = state_ckpt["step_count"]
        self.logger.log(
            time=self.step_count,
            event_type=Event.SET_STATE_CKPT,
        )

        def parse_pos(s):
            x, y = s.split(":")
            return (int(x), int(y))

        tmp = self._deserialize(state_ckpt["rng"])
        if not isinstance(tmp, np.random.Generator):
            raise TypeError("Expected numpy Generator for rng")
        self.rng = tmp

        self.food = {parse_pos(pos): v for pos, v in state_ckpt["food"].items()}
        self.food_count = list(state_ckpt["food_count"])
        self.food_distribution = self._deserialize(state_ckpt["food_distribution"])

        self.empty_food = state_ckpt["empty_food"]
        self.artifacts = {
            name: art
            for name, art in (
                (name, self._deserialize(art))
                for name, art in state_ckpt["artifacts"].items()
            )
            if isinstance(art, Artifact)
        }
        self.artifacts_map = defaultdict(set)
        for pos, arts in state_ckpt["artifacts_map"].items():
            self.artifacts_map[parse_pos(pos)] = set(arts)
        self.agent_inventories = defaultdict(set)
        for agent, inv in state_ckpt["agent_inventories"].items():
            self.agent_inventories[agent] = set(inv)
        self.agent_recoveries = defaultdict(int, state_ckpt.get("agent_recoveries", {}))
        self.expired_artifacts = [
            art
            for art in (
                self._deserialize(data) for data in state_ckpt["expired_artifacts"]
            )
            if isinstance(art, Artifact)
        ]
        self.viral_infection_count = state_ckpt.get("viral_infection_count", 0)
        self._pending_init_artifacts = state_ckpt.get("pending_init_artifacts", [])
        self.agent_pos = {
            agent: parse_pos(pos) for agent, pos in state_ckpt["agent_pos"].items()
        }
        self.agent_trajectories = {
            agent: [parse_pos(p) for p in traj]
            for agent, traj in state_ckpt["agent_trajectories"].items()
        }
        self.agent_avail_actions = state_ckpt["agent_avail_actions"]
        self.pos_to_agent = {
            parse_pos(pos): agent for pos, agent in state_ckpt["pos_to_agent"].items()
        }

        self.agent_energy = state_ckpt["agent_energy"]
        self.agent_time = state_ckpt["agent_time"]
        self.agent_spawn = state_ckpt["agent_spawn"]
        self.agent_names = state_ckpt["agent_names"]
        self.agent_roles = state_ckpt.get("agent_roles", {})
        self.agent_colors = state_ckpt["agent_colors"]
        self.msg_raw = {}
        for agent, msg in state_ckpt["msg_raw"].items():
            deserialized = self._deserialize(msg)
            if isinstance(deserialized, (str, np.ndarray)):
                self.msg_raw[agent] = deserialized
        self.chat = state_ckpt["chat"]
        self.agent_registry = state_ckpt["agent_registry"]

        logger_save_path = state_ckpt.get("logger_save_path", None)
        if logger_save_path is not None:
            self.logger = JSONLogger(logger_save_path)
            self.logger.data = state_ckpt.get("logger_data", {})

        if self.world_logger is not None:
            # Resuming continues an existing run, so append rather than truncate.
            # The logger writes a fresh keyframe on its first line, which gives a
            # reader a clean seek point at the resume step.
            self.world_logger.close()
            self.world_logger = WorldStateLogger(
                Path(logger_save_path).parent / "world_state.jsonl"
                if logger_save_path is not None
                else self.log_path / "world_state.jsonl",
                grid_size=self.grid_size,
                max_food_value=self._max_food_value,
                append=True,
            )

    def close(self):
        if self.verbose >= 1:
            print("Saving environment...")
        self.logger.log(
            time=self.step_count,
            event_type=Event.END_RUN,
        )
        self.save_state(self.log_path / "env_state.pkl")

        with open(self.logger.save_path.parent / "messages.json", "w") as f:
            json.dump(self.chat, f, indent=4)

        active_artifacts = []
        for art_name, artifact in self.artifacts.items():
            art_dict = artifact.serialize()
            for agent, inv in self.agent_inventories.items():
                if art_name in inv:
                    art_dict["owner"] = agent
            active_artifacts.append(art_dict)

        expired_artifacts = [
            artifact.serialize() for artifact in self.expired_artifacts
        ]
        with open(self.logger.save_path.parent / "artifacts.json", "w") as f:
            json.dump(
                {
                    "active": active_artifacts,
                    "expired": expired_artifacts,
                },
                f,
                indent=4,
            )

        with open(self.logger.save_path.parent / "food_counts.json", "w") as f:
            json.dump(self.food_count, f, indent=4)

        with open(self.logger.save_path.parent / "agent_names.json", "w") as f:
            json.dump(self.agent_names, f, indent=4)

        with open(self.logger.save_path.parent / "agent_trajectories.pkl", "wb") as f:
            pickle.dump(self.agent_trajectories, f)

        if self.logger:
            self.logger.close()
        if self.world_logger:
            self.world_logger.close()
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False

    def save_state(self, filepath: str | Path):
        """Saves the environment state to a file."""
        if self.verbose >= 1:
            print("Saving environment state at:", filepath)
        state_ckpt = self.get_state_ckpt()
        with open(filepath, "wb") as f:
            pickle.dump(state_ckpt, f)
        if self.verbose >= 1:
            print("Environment state saved at: ", filepath)

    def load_state(self, filepath: str | Path | None = None):
        """Loads the environment state from a file."""
        if filepath is None:
            filepath = self.log_path / "env_state.pkl"

        assert Path(filepath).exists(), f"State file {filepath} does not exist."
        if self.verbose >= 1:
            print("Loading environment state from:", filepath)
        with open(filepath, "rb") as f:
            state_ckpt = pickle.load(f)
        self.set_state_ckpt(state_ckpt)
        if self.verbose >= 1:
            print("Environment state loaded from:", filepath)


if __name__ == "__main__":
    from PIL import Image

    log_path = Path("logs/test_env")

    env = OpenGridWorld(food_zones=[(10, 10)], grid_size=50, log_path=log_path)
    image_path = log_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    env.restart_env()
    for i in range(100):
        env.step({})
        rgb = env.render(mode="rgb_array")
        img = Image.fromarray(rgb)  # type: ignore
        img.save(image_path / f"step_{i:04d}.png")

    env.close()
    env.load_state()
    for i in range(100, 150):
        env.step({})
        rgb = env.render(mode="rgb_array")
        img = Image.fromarray(rgb)  # type: ignore
        img.save(image_path / f"step_{i:04d}.png")
    env.close()
