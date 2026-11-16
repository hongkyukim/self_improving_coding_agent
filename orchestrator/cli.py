"""
The command line of the runner.
"""

import os
import logging
import argparse

from pathlib import Path

from base_agent.src.benchmarks import benchmark_registry
from base_agent.src.types.llm_types import Model
from search.selection import DEFAULT_SELECTION_STRATEGY, SELECTION_STRATEGIES
from search.evaluation import EvaluationConfig
from runner_config import META_AGENT_CHOICES, RunnerSettings, resolve_settings

from . import evaluation
from .containers import SANDBOX_ENV_VARS, resolve_agent_image
from .creation import check_meta_agent
from .loop import run_search
from .state import (
    get_next_dir_number,
    initialise_experiment,
    resolve_evaluation_config,
    resolve_selection_seed,
    update_metadata,
)

logger = logging.getLogger(__name__)


def parse_meta_agent(value: str) -> str | int:
    """A --meta-agent value: one of META_AGENT_CHOICES, or an agent id"""
    if value in META_AGENT_CHOICES:
        return value
    try:
        agent_id = int(value)
    except ValueError:
        agent_id = -1
    if agent_id < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not one of {', '.join(META_AGENT_CHOICES)} or an agent id"
        )
    return agent_id


def setup_argparse() -> argparse.ArgumentParser:
    """The command line. Options have no defaults here: an option that is not
    given takes its value from the config file, or else its default, both of
    which are in runner_config.RunnerSettings (see settings_from_args)."""
    parser = argparse.ArgumentParser(
        description="Self referential agent",
        epilog="Every option can also be set in a YAML file given with --config "
        "(see run.example.yaml); options on the command line take precedence.",
    )
    parser.add_argument(
        "--experiment-id", "-id", type=int, help="ID of experiment to resume"
    )
    parser.add_argument("--config", type=Path, help="YAML file with settings")

    search = parser.add_argument_group("search")
    search.add_argument(
        "--iterations", "-n", type=int, help="Number of iterations to run (default: 20)"
    )
    search.add_argument(
        "--workers", type=int, help="Number of parallel problem workers (default: 8)"
    )
    search.add_argument(
        "--children",
        type=int,
        help="Number of new agents to create per iteration, each from its own selected parent (default: 1)",
    )
    search.add_argument(
        "--selection",
        choices=SELECTION_STRATEGIES,
        help=f"How to select the agent that the next agent is built from (default: {DEFAULT_SELECTION_STRATEGY})",
    )
    search.add_argument(
        "--seed",
        type=int,
        help="Seed for parent selection (default: the experiment's stored seed, or a new random one)",
    )
    search.add_argument(
        "--meta-agent",
        type=parse_meta_agent,
        metavar="{best,parent,seed,<id>}",
        help="Which agent carries out the improvement of a new agent: the best-scoring "
        "agent (default), the new agent's parent, the initial agent, or a fixed agent id",
    )
    search.add_argument(
        "--meta-model",
        choices=[model.name for model in Model],
        metavar="MODEL",
        help="Model that the meta agent uses while improving an agent (default: the "
        "meta agent's own configured model). Benchmarks are unaffected.",
    )

    gate = parser.add_argument_group("validation gate")
    gate.add_argument(
        "--gate-checks",
        metavar="CHECKS",
        help="Checks that a new agent must pass before it is evaluated, comma separated: "
        "import (its code imports), tests (its unit tests pass, unless its parent's "
        "did not either), smoke (it solves a trivial problem). 'none' switches the "
        "gate off (default: import,tests,smoke)",
    )
    gate.add_argument(
        "--gate-retries",
        type=int,
        help="How often the meta agent is asked to fix a new agent that fails the gate (default: 0)",
    )
    gate.add_argument(
        "--gate-model",
        choices=[model.name for model in Model],
        metavar="MODEL",
        help="Model for the gate's smoke run (default: the agent's own configured model)",
    )

    evaluation = parser.add_argument_group(
        "evaluation",
        "Stored with the experiment: when resuming, options that are not given keep "
        "their stored value. The choice of problems (--subset-size, --benchmark-seed, "
        "--holdout-fraction) cannot be changed once agents have been evaluated.",
    )
    evaluation.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        help="Benchmarks to run (default: gsm8k). Available: "
        + ", ".join(benchmark_registry),
    )
    evaluation.add_argument(
        "--subset-size",
        type=int,
        help="Number of problems to take from each benchmark; 0 for all (default: 18)",
    )
    evaluation.add_argument(
        "--benchmark-seed", type=int, help="Seed of the choice of problems (default: 1)"
    )
    evaluation.add_argument(
        "--timeout-s", type=int, help="Time limit per problem, in seconds (default: 600)"
    )
    evaluation.add_argument(
        "--cost-limit-usd", type=float, help="Cost limit per problem, in USD (default: 3)"
    )
    evaluation.add_argument(
        "--repeats",
        type=int,
        help="Number of times to run each benchmark problem (default: 1)",
    )
    evaluation.add_argument(
        "--holdout-fraction",
        type=float,
        help="Share of each benchmark's problems to keep out of the archive that agents see; "
        "can only be set when an experiment is started (default: 0)",
    )
    evaluation.add_argument(
        "--stage1-size",
        type=int,
        help="Evaluate new agents on this many problems per benchmark first, and in full "
        "only if they are not clearly worse than their parent (default: 0, always in full)",
    )
    evaluation.add_argument(
        "--promote-threshold",
        type=float,
        help="With --stage1-size: evaluate in full unless the agent is worse than its "
        "parent by more than this, at 95%% confidence (default: 0)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to perform")

    # Just run through a single benchmark for testing
    test_parser = subparsers.add_parser(
        "test", help="Just run the benchmark on the latest agent iteration"
    )
    test_parser.add_argument("--name", default="gsm8k", help="Benchmark ID to run")

    return parser


def settings_from_args(args: argparse.Namespace) -> RunnerSettings:
    """The settings of this invocation: defaults, overridden by the config
    file, overridden by the command line.

    Raises:
        ValueError: if a setting is unknown or invalid
    """
    cli = {
        name: getattr(args, name, None) for name in RunnerSettings.model_fields
    }
    return resolve_settings(args.config, cli)


async def main():
    """Main entry point for the agent runner"""
    parser = setup_argparse()
    args = parser.parse_args()
    logger.debug(f"Setup parser: {args}")
    try:
        settings = settings_from_args(args)
    except ValueError as e:
        parser.error(str(e))

    # Either get the specified experiment id, or get the next one
    exp_id = args.experiment_id or get_next_dir_number("results")
    exp_dir = Path("results") / f"run_{exp_id}"

    # Setup experiment directory
    if not exp_dir.exists():
        initialise_experiment(exp_dir)
        logger.info(f"Initialized experiment {exp_id} in {exp_dir}")

    selection_seed = resolve_selection_seed(exp_dir, settings.seed)
    try:
        config = resolve_evaluation_config(
            exp_dir,
            selection_seed,
            **{
                name: getattr(settings, name)
                for name in EvaluationConfig.__dataclass_fields__
                if name in RunnerSettings.model_fields
            },
        )
        check_meta_agent(exp_dir, settings.meta_agent)
        gate = settings.gate
    except ValueError as e:
        parser.error(str(e))

    # Handle test mode
    if args.command == "test":
        # Find the latest agent iteration
        latest_agent = max(
            (d for d in exp_dir.iterdir() if d.name.startswith("agent_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        # Get the benchmark class
        benchmark_cls = benchmark_registry.get(args.name)
        if not benchmark_cls:
            logger.error(f"Benchmark {args.name} not found")
            return

        # Run the specified benchmark using the job queue approach
        await evaluation.run_benchmarks_with_job_queue(
            exp_dir,
            latest_agent,
            [benchmark_cls],
            settings.workers,
            image=await resolve_agent_image(
                exp_id, int(latest_agent.name.split("_")[1])
            ),
            repeats=config.repeats,
            subset_size=config.subset_size,
            benchmark_seed=config.benchmark_seed,
            timeout_s=config.timeout_s,
            cost_limit_usd=config.cost_limit_usd,
        )
        return

    passed_on = [name for name in SANDBOX_ENV_VARS if os.environ.get(name)]
    if passed_on:
        logger.info(f"Passed on to the sandbox: {', '.join(passed_on)}")
    else:
        logger.warning(
            "None of the provider API keys are set in this shell, so agents will "
            f"not be able to call any model ({', '.join(SANDBOX_ENV_VARS)})"
        )

    # Keep a record of what this invocation was run with
    update_metadata(
        exp_dir, runner_settings={**settings.model_dump(), "seed": selection_seed}
    )
    logger.info(
        f"Parent selection: {settings.selection} (seed {selection_seed}), "
        f"{settings.children} new agent(s) per iteration"
    )
    logger.info(f"Evaluation: {config}")
    logger.info(
        f"Meta agent: {settings.meta_agent}, model: {settings.meta_model or 'its own default'}"
    )
    logger.info(f"Validation gate: {gate}")
    await run_search(
        exp_id,
        exp_dir,
        settings.iterations,
        settings.workers,
        n_children=settings.children,
        selection=settings.selection,
        seed=selection_seed,
        config=config,
        meta_agent=settings.meta_agent,
        meta_model=settings.meta_model,
        gate=gate,
    )
