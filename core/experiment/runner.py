import json
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import numpy as np
from faker import Faker
from PIL import Image

from core.agents.human_agent import HumanAgent
from core.agents.llm_agent import LLMAgent
from core.environment.env import OpenGridWorld
from core.experiment.checkpoint import CheckpointManager
from core.experiment.config import ExperimentConfig
from core.experiment.llm_router import LLMRouter
from core.genome.no_traits import Genome as NoTraitsGenome
from core.genome.ocean_5 import Genome as Ocean5Genome
from core.genome.sentence_directed import Genome as SentenceDirectedGenome
from core.utils.generic import create_video
from core.utils.llm_utils import select_with_retry


def _draw_names(n: int, exclude: set) -> List[str]:
    """Returns n unique human first names, avoiding those in exclude."""
    fake = Faker()
    names: List[str] = []
    seen = set(exclude)
    while len(names) < n:
        name = fake.first_name()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def load_personas(path: str | None) -> List[dict]:
    """Loads a personas JSON file and expands it into a per-agent list.

    The file is a list whose entries are either plain persona strings or
    {"persona": str, "name": str, "role": str, "count": int} dicts (name and
    role optional, count defaults to 1). Returns one {"persona", "name",
    "role"} dict per agent, in file order; name/role are None when not given.
    With count > 1 the explicit name goes to the first agent and the rest get
    unique Faker-generated names; the role is shared by the whole group.
    """
    if not path:
        return []
    with open(path) as f:
        entries = json.load(f)
    dict_entries = [e for e in entries if isinstance(e, dict)]
    explicit = {e["name"] for e in dict_entries if e.get("name")}
    needed = sum(
        int(e.get("count", 1)) - 1 for e in dict_entries if e.get("name")
    )
    pool = _draw_names(needed, explicit)
    personas: List[dict] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"persona": entry}
        count = int(entry.get("count", 1))
        name = entry.get("name")
        for i in range(count):
            expanded = name if i == 0 or not name else pool.pop()
            personas.append(
                {
                    "persona": entry["persona"],
                    "name": expanded,
                    "role": entry.get("role"),
                }
            )
    return personas


class SimulationRunner:
    def __init__(self, params: ExperimentConfig, resume: bool = False):
        print("Initializing SimulationRunner...")
        self.params = params
        self.resume = resume

        run_cfg = self.params.run
        self.exp_logdir = Path(run_cfg.save_root) / "logs" / run_cfg.exp_name  # type: ignore
        os.makedirs(self.exp_logdir, exist_ok=True)
        self.checkpointer = CheckpointManager(self.exp_logdir)

        # Load checkpoint if resume requested and set params accordingly
        self.ckpt = None
        self.agents: Dict[str, LLMAgent | HumanAgent] = {}
        self.last_spawn_idx = -1
        if resume:
            self._load_state()
        else:
            self._init_state()

        # Initialize LLM router
        self.llm_router = LLMRouter(
            model_short=self.params.agent.model,
            ports=self.params.run.ports,
            instances=self.params.run.max_parallel_workers,
        )
        self.last_refresh = datetime.now()
        self.refresh_interval = timedelta(hours=1)

        self.terminate = False
        with open(self.exp_logdir / "params.json", "w") as f:
            json.dump(self.params.to_json(), f, indent=4)

        print("###### Launching with Parameters ######")
        pprint(self.params.to_json())
        print("#######################################")

    def _get_genome_cls(self):
        """Returns the genome class based on parameters."""
        if self.params.agent.genome == "ocean_5":
            return Ocean5Genome
        elif self.params.agent.genome == "no_traits":
            return NoTraitsGenome
        elif self.params.agent.genome == "sentence_directed":
            return SentenceDirectedGenome
        else:
            raise ValueError(f"Unsupported genome type: {self.params.agent.genome}")

    def _make_env(self):
        """Creates the environment instance."""
        self.env = OpenGridWorld(
            grid_size=self.params.env.grid_size,
            vision_radius=self.params.env.vision_radius,
            init_agent_energy=self.params.env.init_agent_energy,
            init_food=self.params.env.init_food,
            food_decay_rate=self.params.env.food_decay_rate,
            food_spawn_rate=self.params.env.food_spawn_rate,
            log_path=self.exp_logdir,
            use_inventory=self.params.agent.use_inventory,
            use_colors=self.params.agent.use_colors,
            reproduction_cost=self.params.env.reproduction_cost,
            artifact_creation_cost=self.params.env.artifact_creation_cost,
            artifact_creation=self.params.env.artifact_creation,
            reproduction_allowed=self.params.env.reproduction_allowed,
            lifespan=self.params.env.agent_lifespan,
            static_food=self.params.env.static_food,
            food_zones=self.params.env.food_zones,
            food_mechanism=self.params.env.food_mechanism,
            dead_agent_food=self.params.env.dead_agent_food,
            inert_artifacts=self.params.env.inert_artifacts,
            viral_init_infected=self.params.env.viral_init_infected,
            viral_outbreak_step=self.params.env.viral_outbreak_step,
            viral_lifespan=self.params.env.viral_lifespan,
            viral_incubation_min=self.params.env.viral_incubation_min,
            viral_incubation_max=self.params.env.viral_incubation_max,
            viral_dropped_lifespan=self.params.env.viral_dropped_lifespan,
            viral_infection_radius=self.params.env.viral_infection_radius,
            viral_infection_probability=self.params.env.viral_infection_probability,
            viral_contact_multiplier=self.params.env.viral_contact_multiplier,
            viral_mobile_days=self.params.env.viral_mobile_days,
            viral_mobile_infectiousness=self.params.env.viral_mobile_infectiousness,
            viral_energy_multiplier=self.params.env.viral_energy_multiplier,
            viral_death_probability=self.params.env.viral_death_probability,
            ppe_protection=self.params.env.ppe_protection,
            burials=self.params.env.burials,
            burial_infection_multiplier=self.params.env.burial_infection_multiplier,
            funeral_announcements=self.params.env.funeral_announcements,
            max_message_size=self.params.env.max_message_size,
            max_text_artifact_size=self.params.env.max_text_artifact_size,
            init_artifacts=self.params.env.init_artifacts,
            verbose=self.params.env.verbose,
            parent_authored_genome=self.params.agent.genome == "sentence_directed",
            log_world_state=self.params.run.log_world_state,
        )

    def _init_state(self):
        """Initializes state and environment from scratch."""
        self._make_env()

        init_agents = {
            f"{self.params.agent.agents_name_prefix}{i}": "text"
            for i in range(self.params.env.init_agents)
        }
        init_human_agents = {
            f"human{i}": "human" for i in range(self.params.env.init_human_agents)
        }
        init_agents.update(init_human_agents)

        positions = {"being0": (10, 10), "being1": (12, 10)}
        self.last_spawn_idx = self.params.env.init_agents - 1

        personas = load_personas(self.params.agent.personas)
        if len(personas) > self.params.env.init_agents:
            print(
                f"⚠️ {len(personas)} personas but only "
                f"{self.params.env.init_agents} initial agents; extras unused. ⚠️"
            )

        genome_cls = self._get_genome_cls()
        persona_idx = 0
        # Every being gets a human name: explicit persona names win, the rest
        # draw unique Faker names.
        explicit = {p["name"] for p in personas if p["name"]}
        n_text = sum(1 for t in init_agents.values() if t == "text")
        name_pool = _draw_names(max(0, n_text - len(explicit)), explicit)
        for agent_tag, agent_type in init_agents.items():
            if agent_type == "text":
                persona, agent_name, agent_role = "", None, None
                if persona_idx < len(personas):
                    persona = personas[persona_idx]["persona"]
                    agent_name = personas[persona_idx]["name"]
                    agent_role = personas[persona_idx].get("role")
                persona_idx += 1
                agent_name = agent_name or name_pool.pop()
                self.agents[agent_tag] = LLMAgent(
                    agent_tag=agent_tag,
                    agent_name=agent_name,
                    log_dir=self.exp_logdir,
                    max_history=self.params.agent.max_history,
                    obs_style=self.params.agent.obs_style,
                    use_internal_memory=self.params.agent.use_internal_memory,
                    use_inventory=self.params.agent.use_inventory,
                    food_mechanism=self.params.env.food_mechanism,
                    genome=genome_cls().random(),
                    exogenous_motivation=self.params.agent.exogenous_motivation,
                    persona=persona,
                )
                self.env.add_agent(
                    agent_tag=agent_tag,
                    agent_name=agent_name,
                    agent_type=init_agents[agent_tag],
                    agent_role=agent_role,
                )
            elif agent_type == "human":
                agent_name = input(f"Enter a name for human agent ({agent_tag}): ").strip()
                if not agent_name:
                    agent_name = agent_tag
                self.agents[agent_tag] = HumanAgent(
                    agent_name=agent_name,
                    agent_tag=agent_tag,
                    log_dir=self.exp_logdir,
                    max_history=self.params.agent.max_history,
                    use_internal_memory=self.params.agent.use_internal_memory,
                    use_inventory=self.params.agent.use_inventory,
                    food_mechanism=self.params.env.food_mechanism,
                    genome=genome_cls().random(),
                )
                self.env.add_agent(
                    agent_tag=agent_tag,
                    agent_name=agent_name,
                    agent_type="text",  # HumanAgent interacts through text interface
                )
            else:
                raise NotImplementedError(
                    f"{agent_tag} is non-text agent. Not implemented yet."
                )

        self.obs, self.infos = self.env.restart_env(agent_poses=positions)
        self.start_ts = 0
        self.rewards = {}
        self.dones = {a: False for a in self.agents.keys()}

    def _load_state(self):
        """Loads state and environment from checkpoint."""
        self.ckpt = self.checkpointer.load_checkpoint()
        self.params = self.checkpointer.update_parameters()

        self._make_env()
        self.last_spawn_idx = self.ckpt["last_spawn_idx"]

        genome_cls = self._get_genome_cls()
        for agent_tag, agent_ckpt in self.ckpt["agents"].items():
            if agent_ckpt["type"] == "LLMAgent":
                agent = LLMAgent(
                    agent_tag=agent_ckpt["tag"],
                    agent_name=agent_ckpt["name"],
                    log_dir=self.exp_logdir,
                    max_history=self.params.agent.max_history,
                    obs_style=self.params.agent.obs_style,
                    use_internal_memory=self.params.agent.use_internal_memory,
                    use_inventory=self.params.agent.use_inventory,
                    food_mechanism=self.params.env.food_mechanism,
                    genome=genome_cls().random(),  # Temporary genome, will be loaded
                    exogenous_motivation=self.params.agent.exogenous_motivation,
                )
            elif agent_ckpt["type"] == "HumanAgent":
                agent = HumanAgent(
                    agent_name=agent_tag,
                    agent_tag=agent_tag,
                    log_dir=self.exp_logdir,
                    max_history=self.params.agent.max_history,
                    use_internal_memory=self.params.agent.use_internal_memory,
                    use_inventory=self.params.agent.use_inventory,
                    food_mechanism=self.params.env.food_mechanism,
                    genome=genome_cls().random(),  # Temporary genome, will be loaded
                )
            else:
                raise NotImplementedError(
                    f"Unsupported agent type in checkpoint: {agent_ckpt['type']}"
                )
            agent.set_state_ckpt(agent_ckpt)
            self.agents[agent_tag] = agent
        self.env.set_state_ckpt(self.ckpt["env"])
        self.start_ts = self.ckpt["ts"]
        self.obs = self.ckpt["env_outs"]["obs"]
        self.infos = self.ckpt["env_outs"]["infos"]
        self.rewards = self.ckpt["env_outs"]["rewards"]
        self.dones = self.ckpt["env_outs"]["dones"]
        print(
            "Resumed environment and agents from checkpoint. Current timestep:",
            self.start_ts,
        )

    def _render(self, ts: int):
        """Renders the current frame and saves it as an image."""
        if self.params.run.live_render:
            try:
                self.env.render(mode="human")
            except Exception as e:
                print(f"Error during live rendering: {e}")

        if self.params.run.save_video:
            try:
                frame = self.env.render(mode="rgb_array")
            except Exception as e:
                print(f"Error during frame rendering: {e}")
                frame = None

            if frame is not None:
                try:
                    out_dir = self.exp_logdir / "frames"
                    out_dir.mkdir(exist_ok=True, parents=True)
                    img = Image.fromarray(frame.astype(np.uint8))
                    img.save(out_dir / f"{ts:05d}.png")
                except Exception as e:
                    print(f"Error during saving frame: {e}")

        time.sleep(0.1)

    def _handle_term(self, *_):
        self.terminate = True
        print("⚠️ Caught termination signal. Will terminate after current step. ⚠️")

    def _watch_stdin(self):
        """Watches stdin for EOF (Ctrl+D) and force kills the process immediately."""
        try:
            while not self.terminate:
                if sys.stdin.read(1) == "":
                    print(
                        "\n💀 Force kill (Ctrl+D). Terminating immediately... 💀",
                        flush=True,
                    )
                    os._exit(1)
        except Exception:
            pass

    def _save_checkpoint(self, ts: int):
        self.checkpointer.save_checkpoint(
            ts=ts,
            env=self.env,
            agents=self.agents,
            last_spawn_idx=self.last_spawn_idx,
            env_outs={
                "obs": self.obs,
                "infos": self.infos,
                "rewards": self.rewards,
                "dones": self.dones,
            },
        )

    def _handle_reproduction(self):
        """Creates agents based on environment reproduction infos."""
        for agent_tag, info in self.infos.items():
            if "reproduction" in info:
                reproduced = info["reproduction"].get("status", "failed")
                if reproduced == "successful":
                    parent = self.agents[agent_tag]
                    child_name = info["reproduction"]["child_name"]
                    child_tag = info["reproduction"]["child_tag"]
                    parent_b_tag = info["reproduction"].get("parent_b_tag")

                    # If reproduction is two-parent, crossover the genomes of both parents to create the child genome.
                    if parent_b_tag is not None and parent_b_tag in self.agents:
                        parent_b = self.agents[parent_b_tag]
                        child_genome = parent.genome.crossover(parent_b.genome)  # type: ignore

                    # If reproduction is solo, mutate the genome of the parent.
                    # In the case of sentence_directed, it's the parent that provides the genome
                    elif parent.genome.genome_type == "sentence_directed":
                        offspring_genome_str = str(
                            info["reproduction"].get("offspring_genome", "").strip()
                        )
                        child_genome = SentenceDirectedGenome(
                            sentence=offspring_genome_str or parent.genome.sentence  # type: ignore
                        )
                    else:
                        child_genome = parent.genome.mutate(
                            rate=self.params.env.genome_mutation_rate
                        )

                    # TODO add ability to reproduce human agents
                    if info["reproduction"]["child_type"] == "text":
                        self.agents[child_tag] = LLMAgent(
                            agent_name=child_name,
                            agent_tag=child_tag,
                            genome=child_genome,
                            log_dir=self.exp_logdir,
                            obs_style=self.params.agent.obs_style,
                            max_history=self.params.agent.max_history,
                            use_internal_memory=self.params.agent.use_internal_memory,
                            use_inventory=self.params.agent.use_inventory,
                            food_mechanism=self.params.env.food_mechanism,
                            exogenous_motivation=self.params.agent.exogenous_motivation,
                            # getattr: HumanAgent parents have no persona
                            persona=getattr(parent, "persona", "")
                            if self.params.agent.hereditary_persona
                            else "",
                        )
                    print(f"🍼 Agent {agent_tag} reproduced. New agent: {child_tag} 🍼")
                    # No need to register agent in environment, as env already did so.

                    # infos renders verbatim into the parent's next prompt:
                    # strip runner plumbing so it only ever sees names, not tags
                    plumbing = (
                        "child_tag", "child_type", "offspring_genome", "parent_b_tag"
                    )
                    for key in plumbing:
                        info["reproduction"].pop(key, None)

    def _cleanup_dead(self):
        """Cleans up agents that are done/dead."""
        dead_agents = [k for k, v in self.dones.items() if v]
        if dead_agents:
            print(f"💀 Cleaning up dead agents: {dead_agents} 💀")
        for dead in dead_agents:
            dead_agent = self.agents.pop(dead)
            dead_agent.close()

    def _respawn_if_needed(self):
        """Respawns new agents if number of alive agents is below minimum"""
        while len(self.agents) < self.params.env.min_agents:
            # Ensure unique agent tag
            while True:
                self.last_spawn_idx += 1
                new_agent_tag = (
                    f"{self.params.agent.agents_name_prefix}{self.last_spawn_idx}"
                )
                # We check env.agent_names so to verify against dead agents as well.
                if new_agent_tag not in self.env.agent_names:
                    break
            new_agent_name = _draw_names(1, set(self.env.agent_names.values()))[0]
            self.agents[new_agent_tag] = LLMAgent(
                agent_tag=new_agent_tag,
                agent_name=new_agent_name,
                log_dir=self.exp_logdir,
                max_history=self.params.agent.max_history,
                obs_style=self.params.agent.obs_style,
                use_internal_memory=self.params.agent.use_internal_memory,
                use_inventory=self.params.agent.use_inventory,
                food_mechanism=self.params.env.food_mechanism,
            )
            # Register in environment
            obs, infos = self.env.add_agent(
                agent_tag=new_agent_tag,
                agent_name=new_agent_name,
                agent_type="text",
            )
            self.obs[new_agent_tag] = obs
            self.infos[new_agent_tag] = infos
            self.dones[new_agent_tag] = False

    def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_term)

        # Only watch stdin on a real terminal: in headless runs (nohup, CI,
        # stdin redirected) stdin is at EOF immediately and the watcher would
        # force-kill the run at startup.
        if sys.stdin.isatty():
            threading.Thread(target=self._watch_stdin, daemon=True).start()
            print(
                "ℹ️  Press Ctrl+D to force kill immediately, Ctrl+C to stop after current step."
            )
        else:
            print("ℹ️  Press Ctrl+C to stop after current step.")

        max_ts = self.params.run.max_ts
        ckpt_interval = self.params.run.ckpt_interval
        empty_countdown = self.params.run.empty_countdown
        # If the loop body never runs (e.g. resuming a finished run), the final
        # checkpoint below must still have a valid ts.
        ts = self.start_ts - 1
        try:
            with ThreadPoolExecutor(
                max_workers=self.params.run.max_parallel_workers
            ) as executor:
                for ts in range(self.start_ts, max_ts):
                    print(f"\n=== Timestep {ts} ===")
                    # Here so it starts by rendering even at ts=0
                    self._render(ts=ts)

                    if (
                        ckpt_interval
                        and ts % ckpt_interval == 0
                        and ts != self.start_ts
                    ):
                        self._save_checkpoint(ts=ts)

                    if datetime.now() - self.last_refresh > self.refresh_interval:
                        print("🔄 Refreshing LLM clients... 🔄")
                        self.llm_router.refresh(
                            ports=self.params.run.ports,
                            instances=self.params.run.max_parallel_workers,
                        )
                        self.last_refresh = datetime.now()

                    # Get actions
                    # ---------------------------
                    available_actions = {
                        agent_tag: self.infos[agent_tag].pop("available_actions")
                        for agent_tag in self.agents.keys()
                    }
                    actions = {}
                    if self.params.run.max_parallel_workers > 1:
                        # Launch all agent decisions concurrently
                        futures = {
                            executor.submit(
                                select_with_retry,
                                agent,
                                self.obs[agent_tag],
                                available_actions[agent_tag],
                                self.rewards.get(agent_tag, 0),
                                self.infos.get(agent_tag, None),
                                ts,
                                *self.llm_router.next(),
                            ): agent_tag
                            for agent_tag, agent in self.agents.items()
                            if agent_tag in self.obs
                        }
                        for future in as_completed(futures):
                            agent_tag = futures[future]
                            actions[agent_tag], need_refresh = future.result()
                            if need_refresh:
                                print(
                                    "🔄 Attempting to refresh LLM_CLIENTS due to connection error..."
                                )
                                self.llm_router.refresh(
                                    self.params.run.ports,
                                    self.params.run.max_parallel_workers,
                                )
                    else:
                        # Sequential agent decisions
                        for agent_tag, agent in self.agents.items():
                            if agent_tag in self.obs:
                                actions[agent_tag], need_refresh = select_with_retry(
                                    agent,
                                    self.obs[agent_tag],
                                    available_actions[agent_tag],
                                    self.rewards.get(agent_tag, 0),
                                    self.infos.get(agent_tag, None),
                                    ts,
                                    *self.llm_router.next(),
                                )
                                if need_refresh:
                                    print(
                                        "🔄 Attempting to refresh LLM_CLIENTS due to connection error..."
                                    )
                                    self.llm_router.refresh(
                                        self.params.run.ports,
                                        self.params.run.max_parallel_workers,
                                    )
                    # ---------------------------

                    # Step the environment
                    self.obs, self.rewards, self.dones, _, self.infos = self.env.step(
                        actions
                    )

                    self._handle_reproduction()
                    self._cleanup_dead()
                    self._respawn_if_needed()

                    if len(self.agents) == 0:
                        empty_countdown -= 1

                    if empty_countdown == 0:
                        break

                    if self.terminate:
                        print(
                            "⚠️ Terminating as requested. Saving checkpoint and exiting... ⚠️"
                        )
                        break

        except KeyboardInterrupt:
            print("⚠️ User interruption ⚠️")

        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()

        # Final cleanup and saves
        # ---------------------------
        self._save_checkpoint(ts=ts + 1)

        # Render last frame
        self._render(ts=ts + 1)

        self.env.close()
        for agent in self.agents.values():
            agent.close()

        if self.params.run.save_video:
            try:
                create_video(
                    str(self.exp_logdir / "frames" / "%05d.png"),
                    output_file=str(self.exp_logdir / "video.mp4"),
                    fps=self.params.run.video_fps,
                )
            except Exception as e:
                print(f"Error during video creation: {e}")
        # ---------------------------
