from argparse import Namespace
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import List, Tuple

from core.agents.prompt_templates import AVAILABLE_EX_MOTIVATIONS, OBS_STYLE
from core.environment.env import AVAILABLE_DEAD_AGENT_FOOD
from core.genome import AVAILABLE_GENOMES
from core.utils import ROOT


@dataclass
class AgentConfig:
    agents_name_prefix: str = field(
        default="being",
        metadata={"help": "Prefix for agent names (e.g. being0, being1)"},
    )
    exogenous_motivation: str = field(
        default="base",
        metadata={
            "help": "Type of exogenous motivations",
            "choices": AVAILABLE_EX_MOTIVATIONS,
        },
    )
    genome: str = field(
        default="ocean_5",
        metadata={
            "help": "Agent genome type",
            "choices": AVAILABLE_GENOMES,
        },
    )
    hereditary_persona: bool = field(
        default=False,
        metadata={
            "help": "Children inherit their parent's persona "
            "(parent A on two-parent reproduction)"
        },
    )
    internal_memory_size: int = field(
        default=150,
        metadata={"help": "Size in tokens of internal memory"},
    )
    max_history: int = field(
        default=1,
        metadata={"help": "Max number of interactions stored per agent"},
    )
    model: str = field(
        default="claude-sonnet-4-6",
        metadata={"help": "Model name used for agent decision making"},
    )
    obs_style: str = field(
        default="list",
        metadata={"help": "Observation style", "choices": list(OBS_STYLE.keys())},
    )
    personas: str | None = field(
        default=None,
        metadata={
            "help": "Path to a JSON file with personas: a list of {persona, name, "
            "role, count} entries (or plain strings) assigned to the initial agents "
            "in order; name renames the agent everywhere (tag is unchanged), role "
            "gives it a distinct marker shape on the dashboard map"
        },
    )
    use_colors: bool = field(
        default=False,
        metadata={"help": "Allow agents to choose their own color"},
    )
    use_internal_memory: bool = field(
        default=True,
        metadata={"help": "Use agent-internal memory"},
    )
    use_inventory: bool = field(
        default=True,
        metadata={"help": "Enable inventory system"},
    )

    def __post_init__(self):
        avail_obs_styles = list(OBS_STYLE.keys())
        if self.obs_style not in avail_obs_styles:
            raise ValueError(
                f"Obs_style is {self.obs_style} - Available: {avail_obs_styles}"
            )

        assert self.genome in AVAILABLE_GENOMES, (
            f"Genome must be one of {AVAILABLE_GENOMES}, got {self.genome}"
        )

        assert self.exogenous_motivation in AVAILABLE_EX_MOTIVATIONS, (
            f"Exogenous motivation must be one of {AVAILABLE_EX_MOTIVATIONS}, got {self.exogenous_motivation}"
        )


@dataclass
class EnvConfig:
    agent_lifespan: int = field(default=100, metadata={"help": "Max lifespan"})
    artifact_creation: bool = field(
        default=True, metadata={"help": "Enable artifact creation"}
    )
    artifact_creation_cost: int = field(
        default=0, metadata={"help": "Cost to create artifact"}
    )
    burials: bool = field(
        default=False,
        metadata={
            "help": "Beings next to remains (a ground viral artifact) get a "
            "'bury' action that removes the artifact from the world"
        },
    )
    burial_infection_multiplier: float = field(
        default=2.0,
        metadata={
            "help": "Burying scales the burier's viral_infection_probability "
            "by this factor for that one exposure (PPE still applies)"
        },
    )
    dead_agent_food: str = field(
        default="single",
        metadata={
            "help": "Food from dead agents",
            "choices": AVAILABLE_DEAD_AGENT_FOOD,
        },
    )
    food_decay_rate: float = field(default=0.05, metadata={"help": "Food decay rate"})
    food_mechanism: bool = field(
        default=True, metadata={"help": "Enable energy mechanic"}
    )
    food_spawn_rate: int = field(default=1, metadata={"help": "Food spawn per step"})
    genome_mutation_rate: float = field(
        default=0.5,
        metadata={"help": "Per-gene mutation rate applied to the child genome on reproduction"},
    )
    food_zones: int | List[Tuple[int, int]] | None = field(
        default=None,
        metadata={
            "help": "Food zones. Accepts integer OR list of 'x,y' pairs",
            "autocoerce": "food_zones",
        },
    )
    grid_size: int = field(default=50, metadata={"help": "Grid dimension"})
    inert_artifacts: bool = field(
        default=False, metadata={"help": "Artifacts cannot be interacted with"}
    )
    init_agents: int = field(default=20, metadata={"help": "Initial agent count"})
    init_artifacts: str | None = field(
        default=None,
        metadata={
            "help": "Path to a JSON file with artifacts seeded by the environment: "
            "a list of {name, type, payload, pose, lifespan, step} entries "
            "(type: 'text', 'ppe' or 'health_center'; health centers also "
            "take heal_probability and cover their cell plus the 8 around it)"
        },
    )
    init_human_agents: int = field(
        default=0, metadata={"help": "Initial human agent count"}
    )
    init_agent_energy: int = field(
        default=50, metadata={"help": "Initial energy per agent"}
    )
    init_food: int = field(default=100, metadata={"help": "Initial food count"})
    max_message_size: int = field(
        default=-1,
        metadata={
            "help": "Max tokens per agent message; longer ones are cut off "
            "and the agent is told (-1 = unlimited)"
        },
    )
    max_text_artifact_size: int = field(
        default=500,
        metadata={"help": "Max tokens a text artifact can hold"},
    )
    min_agents: int = field(default=0, metadata={"help": "Minimum agent population"})
    reproduction_allowed: bool = field(
        default=True, metadata={"help": "Enable reproduction"}
    )
    reproduction_cost: int = field(
        default=50, metadata={"help": "Energy cost to reproduce"}
    )
    ppe_protection: float = field(
        default=0.1,
        metadata={
            "help": "Multiplier on the infection probability of an agent carrying "
            "a PPE artifact (0 = full immunity, 1 = no protection)"
        },
    )
    static_food: bool = field(
        default=False, metadata={"help": "Food always spawns in same positions"}
    )
    funeral_announcements: bool = field(
        default=False,
        metadata={
            "help": "Announce deaths that leave remains, with directions, so "
            "gathering at the corpse is a choice"
        },
    )
    funeral_announcement_radius: int = field(
        default=-1,
        metadata={
            "help": "How far funeral news travels on its own (-1: the whole "
            "map); beings further away only learn by word of mouth"
        },
    )
    funeral_attendance_multiplier: float = field(
        default=0.0,
        metadata={
            "help": "Beings beside the grave when remains are buried roll an "
            "exposure at this multiple of viral_infection_probability "
            "(the digger uses burial_infection_multiplier; 0 = digger only)"
        },
    )
    funeral_mourning_days: int = field(
        default=0,
        metadata={
            "help": "Steps remains refuse burial after the death, so mourners "
            "can gather first (0: buriable at once)"
        },
    )
    viral_contact_multiplier: float = field(
        default=1.0,
        metadata={
            "help": "Touching (give/take energy) scales viral_infection_probability "
            "by this factor for that exposure (PPE still applies)"
        },
    )
    viral_dropped_lifespan: int = field(
        default=20,
        metadata={
            "help": "Steps a viral artifact dropped at its host's death survives on the map (-1 = forever)"
        },
    )
    viral_death_probability: float = field(
        default=0.0,
        metadata={
            "help": "Death chance per symptomatic step at the END of the "
            "infectious window; the hazard ramps linearly from 0 at symptom "
            "onset (0 = only starvation kills)"
        },
    )
    viral_energy_multiplier: float = field(
        default=2.0,
        metadata={
            "help": "Energy consumption multiplier for each viral artifact hosted (K)"
        },
    )
    viral_incubation_max: int = field(
        default=21,
        metadata={
            "help": "Max steps between infection and symptoms (silent, non-infectious phase)"
        },
    )
    viral_incubation_min: int = field(
        default=2,
        metadata={
            "help": "Min steps between infection and symptoms (silent, non-infectious phase)"
        },
    )
    viral_infection_probability: float = field(
        default=0.3,
        metadata={
            "help": "Per-step probability that a viral artifact spreads to a nearby agent"
        },
    )
    viral_infection_radius: int = field(
        default=1,
        metadata={
            "help": "Max cell distance at which viral artifacts spread (1 = contact)"
        },
    )
    viral_init_infected: int = field(
        default=0,
        metadata={
            "help": "Agents infected at the viral outbreak (0 disables viral artifacts)"
        },
    )
    viral_lifespan: int = field(
        default=-1,
        metadata={"help": "Steps a viral infection lasts in an agent's inventory (-1 = forever)"},
    )
    viral_mobile_days: int = field(
        default=0,
        metadata={
            "help": "First symptomatic days the host stays ambulatory ('dry' phase): "
            "it can still move, eat and act. 0 = bedridden at symptom onset"
        },
    )
    viral_mobile_infectiousness: float = field(
        default=1.0,
        metadata={
            "help": "Multiplier on viral_infection_probability while the host is "
            "in its ambulatory days"
        },
    )
    viral_outbreak_step: int = field(
        default=0,
        metadata={"help": "Timestep at which the viral outbreak happens"},
    )
    verbose: int = field(
        default=1,
        metadata={
            "help": "Terminal chatter from the environment: 0 warnings only, "
            "1 key events (deaths, births, infections, seeding), 2 per-step debug"
        },
    )
    vision_radius: int = field(default=6, metadata={"help": "Vision radius"})

    def __post_init__(self):
        assert self.dead_agent_food in AVAILABLE_DEAD_AGENT_FOOD, (
            f"Dead agent food must be one of {AVAILABLE_DEAD_AGENT_FOOD}, got {self.dead_agent_food}"
        )

        assert self.min_agents <= self.init_agents, (
            "min_agents cannot be greater than init_agents"
        )

        assert 0.0 <= self.viral_infection_probability <= 1.0, (
            "viral_infection_probability must be in [0, 1]"
        )
        assert self.viral_infection_radius >= 0, (
            "viral_infection_radius cannot be negative"
        )
        assert self.viral_energy_multiplier > 0, (
            "viral_energy_multiplier must be positive"
        )
        assert 0.0 <= self.ppe_protection <= 1.0, (
            "ppe_protection must be in [0, 1]"
        )
        assert self.burial_infection_multiplier >= 0, (
            "burial_infection_multiplier cannot be negative"
        )
        assert self.viral_contact_multiplier >= 0, (
            "viral_contact_multiplier cannot be negative"
        )
        assert self.funeral_attendance_multiplier >= 0, (
            "funeral_attendance_multiplier cannot be negative"
        )
        assert self.funeral_mourning_days >= 0, (
            "funeral_mourning_days cannot be negative"
        )
        assert self.viral_mobile_days >= 0, (
            "viral_mobile_days cannot be negative"
        )
        assert 0.0 <= self.viral_mobile_infectiousness <= 1.0, (
            "viral_mobile_infectiousness must be in [0, 1]"
        )
        assert 0.0 <= self.viral_death_probability <= 1.0, (
            "viral_death_probability must be in [0, 1]"
        )
        assert self.verbose in (0, 1, 2), "verbose must be 0, 1 or 2"
        assert self.max_text_artifact_size > 0, (
            "max_text_artifact_size must be positive"
        )
        assert self.viral_init_infected >= 0, (
            "viral_init_infected cannot be negative"
        )
        assert self.viral_incubation_min >= 0, (
            "viral_incubation_min cannot be negative"
        )
        assert self.viral_incubation_min <= self.viral_incubation_max, (
            "viral_incubation_min cannot be greater than viral_incubation_max"
        )


@dataclass
class RunConfig:
    ckpt_interval: int = field(default=100, metadata={"help": "Checkpoint interval"})
    empty_countdown: int = field(default=20, metadata={"excluded": True})
    exp_description: str = field(
        default="", metadata={"help": "Experiment description"}
    )
    exp_name: str | None = field(
        default='TEST', metadata={"help": "Experiment name", "arg_type": str}
    )
    live_render: bool = field(
        default=False, metadata={"help": "Render simulation live"}
    )
    log_world_state: bool = field(
        default=True,
        metadata={
            "help": "Log per-step world state to world_state.jsonl (used by the dashboard)"
        },
    )
    max_parallel_workers: int = field(
        default=20, metadata={"help": "Max worker threads"}
    )
    max_ts: int = field(default=3000, metadata={"help": "Max simulation timesteps"})
    ports: tuple = field(
        default=(9000, 9001, 9002, 9003, 9010, 9011, 9012),
        metadata={
            "help": "Ports hosting LLM models",
            "arg_type": int,
            "nargs": "+",
        },
    )
    save_root: str | None = field(
        default=None, metadata={"help": "Output directory root", "arg_type": str}
    )
    save_video: bool = field(default=True, metadata={"help": "Save video"})
    video_fps: int = field(default=10, metadata={"help": "FPS for video output"})

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.save_root is None:
            self.save_root = str(ROOT)

        self.ports = tuple(self.ports)


@dataclass
class ExperimentConfig:
    agent: AgentConfig
    env: EnvConfig
    run: RunConfig

    def to_json(self):
        return {
            "agent": asdict(self.agent),
            "env": asdict(self.env),
            "run": asdict(self.run),
        }


def build_config(args: dict | Namespace) -> ExperimentConfig:
    agent = AgentConfig()
    env = EnvConfig()
    run = RunConfig()

    if isinstance(args, Namespace):
        args = vars(args)

    for k, v in args.items():
        if v is None or k == "resume":
            continue
        if hasattr(agent, k):
            agent = replace(agent, **{k: v})
        elif hasattr(env, k):
            env = replace(env, **{k: v})
        elif hasattr(run, k):
            run = replace(run, **{k: v})

    return ExperimentConfig(agent, env, run)
