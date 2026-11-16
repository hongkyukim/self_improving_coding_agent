"""
The state of an experiment on disk: its directory, its metadata, and the
settings that are stored with it.
"""

import os
import sys
import json
import shutil
import random
import asyncio
import logging
import subprocess

from uuid import uuid4
from pathlib import Path
from datetime import datetime

from base_agent.src.benchmarks import benchmark_registry
from base_agent.src.utils.lineage import (
    STATUS_OK,
    FINISHED_STATUSES,
    read_lineage,
    write_lineage,
)
from search.evaluation import EvaluationConfig

logger = logging.getLogger(__name__)


def get_next_dir_number(base_dir: str | Path) -> int:
    """
    Returns the next free directory name of the form `run_{i}` in some base_dir.
    """
    base_path = Path(base_dir)
    numbered_dirs = []
    for path in base_path.iterdir():
        if path.is_dir():
            try:
                if path.name.startswith("run_"):
                    numbered_dirs.append(int(path.name.lstrip("run_")))
            except ValueError:
                continue
    # If no numbered directories exist, start with 1
    if not numbered_dirs:
        return 1
    return max(numbered_dirs) + 1


def write_json_atomic(path: Path, data) -> None:
    """Write a JSON file so that it is either the old or the new version in
    full: a crash while writing cannot leave a truncated file behind."""
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_metadata(exp_dir: Path) -> dict:
    """Load the experiment metadata, creating if doesn't exist"""
    metadata_file = exp_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            "experiment_start_timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "executable": sys.executable,
            # Number of completed generations (evaluate, then create children)
            "generation": 0,
            # Agents that exist but still have to be evaluated
            "pending_agents": [0],
            "git_commit": None,
            # NOTE: the current_benchmark_idx field will be updated later, and
            # is set to -1 for improvement tasks, and 0, 1, 2, ... otherwise.
            # "current_benchmark_idx": None,
        }
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            pass

        write_json_atomic(metadata_file, metadata)
    else:
        with open(metadata_file) as f:
            metadata = json.load(f)
    return metadata


def update_metadata(exp_dir: Path, **kwargs) -> None:
    """Update specific metadata fields"""
    metadata = load_metadata(exp_dir)
    metadata.update(kwargs)
    write_json_atomic(exp_dir / "metadata.json", metadata)


def resolve_selection_seed(exp_dir: Path, seed: int | None = None) -> int:
    """The seed for parent selection in this experiment.

    An explicitly given seed wins. Otherwise the experiment keeps the seed it
    already has, so that resuming it does not change its random choices; a
    new experiment gets a random one. The seed in use is stored in the
    experiment metadata.
    """
    stored = load_metadata(exp_dir).get("selection_seed")
    if seed is None:
        seed = stored if stored is not None else random.SystemRandom().randrange(2**32)
    elif stored is not None and stored != seed:
        logger.warning(f"Selection seed changed from {stored} to {seed}")
    if seed != stored:
        update_metadata(exp_dir, selection_seed=seed)
    return seed


# Evaluation settings that decide which problems agents are run on
PROBLEM_CHOICE_SETTINGS = ("subset_size", "benchmark_seed", "holdout_fraction")


def resolve_evaluation_config(
    exp_dir: Path, seed: int, **given
) -> EvaluationConfig:
    """How agents are evaluated in this experiment.

    `given` holds evaluation settings (fields of EvaluationConfig) asked for
    in this invocation; None or absent means not asked for. Those keep the
    value stored with the experiment, or the default for a new experiment.

    The choice of problems is fixed once agents have been evaluated: with
    other problems, results would no longer be comparable between agents,
    and held-out problems could end up in the archive that agents see.

    Raises:
        ValueError: for invalid settings, unknown benchmarks, or an attempt
            to change the choice of problems
    """
    stored = load_metadata(exp_dir).get("evaluation")
    current = EvaluationConfig(**stored) if stored else EvaluationConfig(split_seed=seed)
    asked = {key: value for key, value in given.items() if value is not None}
    config = EvaluationConfig(**{**current.to_dict(), **asked})

    unknown = [name for name in config.benchmarks if name not in benchmark_registry]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) {', '.join(unknown)}; available: {', '.join(benchmark_registry)}"
        )

    if any(exp_dir.glob("agent_*/benchmarks")):
        for name in PROBLEM_CHOICE_SETTINGS:
            old, new = getattr(current, name), getattr(config, name)
            if old != new:
                raise ValueError(
                    f"{name} of this experiment is {old} and cannot be changed "
                    f"(to {new}) now that agents have been evaluated"
                )
        if config.benchmarks != current.benchmarks:
            logger.warning(
                f"Benchmarks changed from {list(current.benchmarks)} to {list(config.benchmarks)}: "
                "overall scores of agents evaluated before and after are not comparable"
            )
    if config.to_dict() != stored:
        update_metadata(exp_dir, evaluation=config.to_dict())
    return config


async def update_benchmark_progress(
    exp_dir: Path, benchmark_name: str, problem_id: str, total_problems: int
) -> None:
    """Update the metadata to mark a problem as completed"""
    metadata_file = exp_dir / "metadata.json"

    # Use a file lock to ensure thread safety
    async with asyncio.Lock():
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Initialize benchmark progress if not exists
        if "benchmark_progress" not in metadata:
            metadata["benchmark_progress"] = {}

        if benchmark_name not in metadata["benchmark_progress"]:
            metadata["benchmark_progress"][benchmark_name] = {
                "total": total_problems,
                "completed": 0,
                "problems_completed": [],
            }

        # Ensure total is up-to-date
        metadata["benchmark_progress"][benchmark_name]["total"] = total_problems

        # Update the progress
        if (
            problem_id
            not in metadata["benchmark_progress"][benchmark_name]["problems_completed"]
        ):
            metadata["benchmark_progress"][benchmark_name]["problems_completed"].append(
                problem_id
            )
            metadata["benchmark_progress"][benchmark_name]["completed"] += 1

        # Save metadata
        write_json_atomic(metadata_file, metadata)


def load_search_state(exp_dir: Path) -> dict:
    """The experiment metadata, with the search state fields present.

    Experiments from before generations only stored `agent_iteration`: the
    agent that was being evaluated, all agents up to it being complete. Those
    agents may have no lineage record, and get one here (of unknown ancestry).
    """
    metadata = load_metadata(exp_dir)
    if "pending_agents" not in metadata:
        current = metadata.get("agent_iteration", 0)
        for agent_id in get_agent_ids(exp_dir):
            agent_dir = exp_dir / f"agent_{agent_id}"
            if agent_id <= current and read_lineage(agent_dir) is None:
                write_lineage(agent_dir, id=agent_id, status=STATUS_OK)
        update_metadata(exp_dir, generation=current, pending_agents=[current])
        metadata = load_metadata(exp_dir)
    return metadata


def get_agent_ids(exp_dir: Path) -> list[int]:
    """The ids of all agent directories in an experiment"""
    ids = []
    for agent_dir in exp_dir.glob("agent_*"):
        try:
            ids.append(int(agent_dir.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def set_aside(exp_dir: Path, agent_dir: Path) -> None:
    """Move the remains of an interrupted meta-improvement out of the way
    (to `_interrupted/`), so that the agent can be created afresh."""
    target_root = exp_dir / "_interrupted"
    target_root.mkdir(exist_ok=True)
    target = target_root / f"{agent_dir.name}_{uuid4().hex[:8]}"
    logger.warning(f"Setting aside incomplete {agent_dir} as {target}")
    shutil.move(str(agent_dir), str(target))


def set_aside_incomplete_agents(exp_dir: Path) -> None:
    """Set aside agent directories left behind by an interrupted
    meta-improvement step: those never recorded as finished."""
    for agent_id in get_agent_ids(exp_dir):
        agent_dir = exp_dir / f"agent_{agent_id}"
        record = read_lineage(agent_dir) or {}
        if record.get("status") not in FINISHED_STATUSES:
            set_aside(exp_dir, agent_dir)


def initialise_experiment(exp_dir: Path) -> None:
    """Create a new experiment directory, with the base agent as agent_0"""
    exp_dir.mkdir(parents=True)
    load_metadata(exp_dir)  # Creates if doesn't exist

    # Copy the initial agent
    agent_0_dir = exp_dir / "agent_0"
    agent_0_dir.mkdir()
    shutil.copytree("base_agent", agent_0_dir / "agent_code")
    write_lineage(
        agent_0_dir,
        id=0,
        parent_id=None,
        meta_agent_id=None,
        generation=0,
        status=STATUS_OK,
    )
