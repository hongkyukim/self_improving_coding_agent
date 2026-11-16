"""
The self-improvement loop.
"""

import asyncio
import logging

from pathlib import Path

from base_agent.src.utils.lineage import FINISHED_STATUSES, read_lineage
from search.gate import GateConfig
from search.selection import DEFAULT_SELECTION_STRATEGY
from search.evaluation import EvaluationConfig

from . import creation
from .creation import check_initial_agent, plan_generation
from .evaluation import evaluate_pending_agents
from .state import (
    load_metadata,
    load_search_state,
    set_aside_incomplete_agents,
    update_metadata,
)

logger = logging.getLogger(__name__)


async def run_search(
    exp_id: int,
    exp_dir: Path,
    iterations: int,
    workers: int,
    n_children: int = 1,
    selection: str = DEFAULT_SELECTION_STRATEGY,
    seed: int = 0,
    config: EvaluationConfig = EvaluationConfig(),
    meta_agent: str | int = "best",
    meta_model: str | None = None,
    gate: GateConfig = GateConfig(),
) -> None:
    """The self-improvement loop. In each generation: evaluate the agents that
    have not been evaluated yet, then create `n_children` new agents."""
    start_generation = load_search_state(exp_dir)["generation"]
    logger.info(f"Starting from generation {start_generation}")
    if start_generation < iterations:
        await check_initial_agent(exp_id, exp_dir, gate)

    for generation in range(start_generation, iterations):
        await evaluate_pending_agents(exp_id, exp_dir, workers, config)

        # Plan the new agents, unless resuming a generation already planned
        plan = load_metadata(exp_dir).get("planned_children")
        if not plan:
            set_aside_incomplete_agents(exp_dir)
            plan = plan_generation(
                exp_dir, generation, n_children, selection, seed, meta_agent, meta_model
            )
            update_metadata(exp_dir, planned_children=plan)

        def is_created(child: dict) -> bool:
            record = read_lineage(exp_dir / f"agent_{child['id']}") or {}
            return record.get("status") in FINISHED_STATUSES

        todo = [child for child in plan if not is_created(child)]
        logger.info(
            f"Generation {generation}: creating agents {[c['id'] for c in todo]} "
            f"from parents {[c['parent_id'] for c in todo]}"
        )
        results = await asyncio.gather(
            *(creation.run_meta_agent_benchmark(exp_id, exp_dir, child, gate) for child in todo),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        for child, result in zip(todo, results):
            if isinstance(result, BaseException):
                logger.error(f"Creating agent {child['id']} raised {result!r}")
        interruption = next((e for e in errors if not isinstance(e, Exception)), None)
        if interruption is not None:
            raise interruption
        if errors and len(errors) == len(todo):
            # Nothing worked: more likely a broken setup than bad agents
            raise errors[0]

        update_metadata(
            exp_dir,
            generation=generation + 1,
            pending_agents=[child["id"] for child in plan],
            planned_children=None,
            benchmark_progress=dict(),
        )
        logger.info(f"Completed generation {generation}")
