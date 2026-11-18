"""
Evaluating agents: running benchmark problems in containers, and deciding
which problems an agent is run on.
"""

import re
import json
import shutil
import random
import asyncio
import logging

from uuid import uuid4
from typing import Type
from pathlib import Path

from base_agent.src.benchmarks import benchmark_registry
from base_agent.src.benchmarks.base import BaseBenchmark, BenchmarkTracker, Problem
from base_agent.src.utils.archive_analysis import (
    ArchiveAnalyzer,
    base_problem_id,
    result_id,
)
from base_agent.src.utils.usage import USAGE_FILENAME, read_usage
from base_agent.src.utils.lineage import (
    read_lineage,
    write_lineage,
    get_failed_iterations,
)
from base_agent.src.llm.api import create_completion
from base_agent.src.llm.base import Message
from base_agent.src.types.llm_types import Model, TextContent
from search.evaluation import (
    EvaluationConfig,
    holdout_dir,
    should_promote,
    split_problems,
    stage1_problems,
)

from . import containers
from .containers import (
    MONITOR_PORT_MAPPING,
    SANDBOX_IMAGE,
    get_container_host_port,
    resolve_agent_image,
    sandbox_secret_args,
)
from .state import load_metadata, update_benchmark_progress, update_metadata

logger = logging.getLogger(__name__)


# Global locks
benchmark_trackers = {}


benchmark_locks = {}


# How long a benchmark run may go over its time limit before it is cut off
TIMEOUT_GRACE_SECONDS = 10


def read_agent_usage(logdir: Path, stdout: str) -> dict | None:
    """What an agent run consumed: from the usage report in its log directory
    or, for agents from before that existed, the last line of its output."""
    return read_usage(logdir) or parse_usage(stdout)


def parse_usage(stdout: str) -> dict | None:
    """The usage that an agent reports on the last line of its output, as
    'tokens|cached tokens|cost|duration', or None if it is not there."""
    try:
        tokens, cached, cost, time = stdout.splitlines()[-1].split("|")
        return {
            "tokens": int(tokens),
            "cached_tokens": int(cached),
            "cost": float(cost),
            "time": float(time),
        }
    except (IndexError, ValueError):
        return None


async def generate_contextual_summary(
    problem_statement: str,
    llm_answer: str,
    trace: str,
    score: float,
    parse_errors: str | None = None,
    answer_discussion: str | None = None,
) -> str:
    """
    Generates a summary of the long agent trace for archival and indexing purposes.

    Incorporates the original problem statement, the agent's final answer,
    score, any parsing errors and answer discussion to help better
    contextualise the trace summary.
    """
    scoring_context = (
        f"The agent's answer got a score of {score}"
        "(For binary correct/incorrect answers 0 represents a wrong answer and 1 represents a correct answer. "
        "For other problem types, a higher score is better.)"
    )

    if parse_errors:
        scoring_context += f"\nThere were issues parsing the answer: {parse_errors}"

    if answer_discussion:
        scoring_context += f"\nHere is some additional information about the answer to this problem:\n{answer_discussion}\n"

    summary_prompt = f"""Below is a trace of an agent's execution on the following problem:

Problem:
{problem_statement}

Agent's Answer:
{llm_answer}

Execution Result:
{scoring_context}

Trace:
{trace}

Please write a critical analysis of the agent's performance, taking into account both the solution process and the final outcome. Consider:
- Did the agent follow a logical approach?
- Were there unnecessary steps or inefficiencies?
- If the answer was wrong, where did the agent's reasoning fail?
- If there were parsing errors, what caused them?
- What specific improvements could make the agent more effective?

Keep your analysis concise (no more than 1-2 paragraphs max) but thorough.
"""

    summary = await create_completion(
        messages=[
            Message(
                role="system",
                content=[
                    TextContent(
                        text="You are a critical yet constructive and creative evaluator of AI agent performance."
                    )
                ],
            ),
            Message(role="user", content=[TextContent(text=summary_prompt)]),
        ],
        model=Model.HAIKU_55,
    )

    # Assume we'll get a single text completion
    return summary.content[0].text


class Job:
    """Represents a benchmark problem to be processed"""

    def __init__(
        self,
        benchmark_name: str,
        problem: Problem,
        benchmark: BaseBenchmark,
        repeat: int = 0,
        total: int | None = None,
    ):
        self.benchmark_name = benchmark_name
        self.problem = problem
        self.benchmark = benchmark
        # Which run of the problem this is, and the id its result is stored under
        self.repeat = repeat
        self.result_id = result_id(problem.problem_id, repeat)
        # Number of jobs of this benchmark in the current run
        self.total = total if total is not None else len(benchmark.problems)
        self.id = f"{benchmark_name}:{self.result_id}"

    def __str__(self):
        return self.id


async def process_job(
    exp_dir: Path,
    agent_dir: Path,
    job: Job,
    worker_id: int,
    image: str = SANDBOX_IMAGE,
    results_exp_dir: Path | None = None,
    timeout_s: int = 600,
    cost_limit_usd: float = 3.0,
) -> bool:
    """Process a single benchmark problem job

    The results go to the agent's directory in `results_exp_dir`, which
    mirrors the layout of the experiment directory (and defaults to it).
    `timeout_s` and `cost_limit_usd` are the limits for this one problem.
    """
    TIMEOUT_SECONDS = timeout_s
    COST_THRESHOLD_USD = cost_limit_usd

    benchmark = job.benchmark
    problem = job.problem

    results_exp_dir = results_exp_dir or exp_dir
    results_agent_dir = results_exp_dir / agent_dir.name
    benchmark_dir = results_agent_dir / "benchmarks" / benchmark.name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_lock = benchmark_locks[job.benchmark_name]
    results_tracker = benchmark_trackers[job.benchmark_name]

    # Check if the problem is already completed
    async with benchmark_lock:
        if job.result_id in results_tracker.results:
            if results_tracker.results[job.result_id].is_complete():
                logger.info(f"Problem {job.result_id} already completed, skipping")
                return True
            else:
                logger.info(f"Re-doing incomplete problem {job.result_id}")
        else:
            results_tracker.start_problem(job.result_id, repeat=job.repeat)

    logger.info(f"Starting {job.result_id} (worker {worker_id})")

    logdir = benchmark_dir / "traces" / job.result_id

    # Create problem data directory
    dirname = re.sub(r"[^a-zA-Z0-9_]", "", f"{benchmark.name}_{job.result_id}")
    problem_data_dir = results_exp_dir / "problem_data" / dirname
    if problem_data_dir.exists():
        shutil.rmtree(problem_data_dir)
    problem_data_dir.mkdir(parents=True)

    timed_out = False
    cost_threshold_exceeded = False

    async def record_failure(reason: str, **fields) -> None:
        """Keep the reason with the problem's result. Its score stays unset,
        so that it is attempted again when the experiment is resumed."""
        async with benchmark_lock:
            results_tracker.update_problem(
                job.result_id, error=reason[-2000:], **fields
            )

    # Create unique container name
    local_id = f"agent_{uuid4().hex[:8]}"
    container_name = f"sica_{benchmark.name}_{job.result_id}_{local_id}"
    container_name = re.sub(f"[^a-zA-Z0-9_]", "", container_name)

    try:
        # Ensure answer directory exists
        logdir.mkdir(parents=True, exist_ok=True)

        # Start the container
        container_cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
        ]
        container_cmd += ["-p", MONITOR_PORT_MAPPING]
        container_cmd += sandbox_secret_args()
        # Mount the experiment directory (results/run_{i})
        container_cmd += [
            "-v",
            f"{exp_dir.absolute()}:/home/agent/{exp_dir}:rw",
        ]
        if results_exp_dir != exp_dir:
            # Results are kept outside of the experiment directory
            container_cmd += [
                "-v",
                f"{results_exp_dir.absolute()}:/home/agent/{results_exp_dir}:rw",
            ]
        # Mount the problem data directory to the workdir
        container_cmd += [
            "-v",
            f"{problem_data_dir.absolute()}:/home/agent/workdir:rw",
        ]
        container_cmd += [image, "tail", "-f", "/dev/null"]
        success, _, stderr = await containers.run_docker_command(*container_cmd)
        if not success:
            logger.error(
                f"Failed to start container for {job.result_id}: {stderr}"
            )
            await record_failure(f"Could not start container: {stderr}")
            return False

        await containers.wait_for_container_ready(container_name)
        monitor_port = await get_container_host_port(container_name)
        logger.info(
            f"{job.result_id} (worker {worker_id}) monitor: http://localhost:{monitor_port}"
        )

        try:
            # Setup the problem environment
            await benchmark.setup_problem(problem, problem_data_dir, container_name)

            # Write content directly using echo
            container_prompt_file = "/tmp/prompt.txt"

            # Create the file in the container with explicit error checking
            create_file_cmd = [
                "docker",
                "exec",
                "-u",
                "agent:agent",
                container_name,
                "bash",
                "-c",
                f"cat > {container_prompt_file} << 'EOF'\n{problem.statement}\nEOF",
            ]
            success, stdout, stderr = await containers.run_docker_command(*create_file_cmd)
            if not success:
                logger.error(f"Failed to create prompt file: {stderr}")

            await asyncio.sleep(
                random.random() * 2
            )  # stagger the start of the agents to avoid API rate limit issues

            # Execute the agent command and get stdout
            agent_module = str(agent_dir / "agent_code").replace("/", ".")
            agent_cmd = ["python", "-m", agent_module, "benchmark"]
            agent_cmd += ["--prompt-file", container_prompt_file]
            agent_cmd += ["--logdir", str(logdir)]
            agent_cmd += ["--timeout", str(TIMEOUT_SECONDS)]
            agent_cmd += ["--cost-threshold", str(COST_THRESHOLD_USD)]

            # A report left by an earlier attempt must not be taken for this one's
            (logdir / USAGE_FILENAME).unlink(missing_ok=True)

            try:
                # An additional safety feature; if timeout cancellation
                # doesn't happen at the application layer, we cancel
                # the docker container
                async with asyncio.timeout(TIMEOUT_SECONDS + TIMEOUT_GRACE_SECONDS):
                    success, stdout, stderr = await containers.run_docker_command(
                        "docker", "exec", container_name, *agent_cmd
                    )
            except asyncio.TimeoutError:
                # Timeout exceeded the grace period
                logger.warning(
                    f"Problem {job.result_id} timed out after {TIMEOUT_SECONDS} seconds"
                )
                timed_out = True
                success = False
                stderr = f"Execution timed out after {TIMEOUT_SECONDS} seconds"

                # Write timeout information to trace file
                timeout_msg = (
                    f"\n\nEXECUTION TIMED OUT AFTER {TIMEOUT_SECONDS} SECONDS\n"
                )
                trace_path = logdir / "trace.txt"
                if trace_path.exists():
                    with open(trace_path, "a") as f:
                        f.write(timeout_msg)
                else:
                    trace_path.write_text(timeout_msg)

            # A run that had to be cut off is not a failed attempt to be
            # repeated: it is scored on whatever answer it left, below.
            if not success and not timed_out:
                logger.error(f"Problem {job.result_id} failed: {stderr}")
                await record_failure(stderr or "The agent exited with an error")
                return False

            if not timed_out:
                usage = read_agent_usage(logdir, stdout)
                if usage is None:
                    raise ValueError("The agent did not report its usage")
                tokens, cached, cost, time = (
                    usage["tokens"], usage["cached_tokens"], usage["cost"], usage["time"]
                )  # fmt: skip
                percent_cached = (
                    int(cached) / int(tokens) * 100 if int(tokens) > 0 else 0
                )

                if float(cost) >= COST_THRESHOLD_USD:
                    cost_threshold_exceeded = True

                if float(time) >= TIMEOUT_SECONDS - 0.5:
                    timed_out = True
                    logger.info(
                        f"Execution of {job.result_id} gracefully timed out after {TIMEOUT_SECONDS} seconds"
                    )

                logger.info(
                    f"{job.result_id} | tokens: {tokens} (cached: {percent_cached:.2f}%), cost: ${float(cost):.4f}, duration: {float(time):.2f}s"
                )
            else:
                # The run was cut off, so it did not report what it used: the
                # time is known, tokens and cost are not
                tokens, cached, cost, time = 0, 0, 0, str(TIMEOUT_SECONDS)

            # Score the answer
            score, parse_errors, answer_discussion = await benchmark.score_problem(
                problem,
                str(problem_data_dir.absolute()),
                str(logdir.absolute()),
                container_name,
            )

            score_path = logdir / "score.txt"
            score_path.write_text(str(score))

            async with benchmark_lock:
                results_tracker.update_problem(
                    job.result_id,
                    tokens_used=int(tokens),
                    num_cached_tokens=int(cached),
                    cost_estimate=float(cost),
                    wall_time=float(time),
                    score=float(score),
                    timed_out=timed_out,
                    cost_threshold_exceeded=cost_threshold_exceeded,
                    error=None,
                )

            answer_path = logdir / "answer.txt"
            if answer_path.exists():
                llm_answer = answer_path.read_text()
            else:
                llm_answer = "No LLM answer provided"
            trace_path = logdir / "trace.txt"
            if trace_path.exists():
                trace = trace_path.read_text()
            else:
                trace = "No trace was available"

            # Add timeout information to the summary context
            timeout_context = ""
            if timed_out:
                timeout_context = f"\nNOTE: This execution was terminated after reaching the {TIMEOUT_SECONDS} second time budget limit."

            # The summary is auxiliary: the problem has already been scored and
            # recorded, so an LLM failure here must not turn it into a failure.
            try:
                summary = await generate_contextual_summary(
                    problem.statement,
                    llm_answer,
                    trace,
                    score,
                    parse_errors,
                    (
                        answer_discussion + timeout_context
                        if answer_discussion
                        else timeout_context
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"Summary generation failed for {job.result_id} "
                    f"(score {score} is kept): {type(e).__name__}: {e}"
                )
                summary = (
                    f"(Summary unavailable: {type(e).__name__}: {e})\n"
                    f"Score: {score}"
                )

            summary_path = logdir / "summary.txt"
            summary_path.write_text(summary)

            # Update the benchmark progress in metadata
            await update_benchmark_progress(
                results_exp_dir, benchmark.name, job.result_id, job.total
            )

            return True

        except Exception as e:
            logger.error(f"Error processing problem {job.result_id}: {e}")
            await record_failure(f"{type(e).__name__}: {e}")
            return False
    finally:
        # Cleanup running docker container
        await containers.run_docker_command("docker", "rm", "-f", container_name)

        # Cleanup problem data directory
        shutil.rmtree(problem_data_dir)
        logger.info(f"Completed {job.result_id}")


async def generate_benchmark_statistics(agent_dir: Path, benchmark_name: str) -> None:
    """Generate and save benchmark statistics"""
    benchmark_dir = agent_dir / "benchmarks" / benchmark_name
    results_tracker = BenchmarkTracker(benchmark_dir / "results.jsonl")

    # Aggregate scores and metrics across all problems to calculate overall metrics
    scores_by_problem: dict[str, list[float]] = {}
    tokens, cached, cost, time = 0, 0, 0, 0
    for result in results_tracker.results.values():
        # Default invalid score to 0
        score = result.score if result.score is not None else 0
        problem_id = base_problem_id(result.problem_id, result.repeat)
        scores_by_problem.setdefault(problem_id, []).append(score)
        if result.tokens_used:
            tokens += result.tokens_used
        if result.num_cached_tokens:
            cached += result.num_cached_tokens
        if result.cost_estimate:
            cost += result.cost_estimate
        if result.wall_time:
            time += result.wall_time

    # Each problem counts once, however often it was run
    scores = [sum(runs) / len(runs) for runs in scores_by_problem.values()]
    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
    aggregates = dict(
        avg_score=avg_score,
        tokens=tokens,
        num_cached_tokens=cached,
        cost=cost,
        time=time,
    )
    with open(benchmark_dir / "perf.json", "w") as f:
        f.write(json.dumps(aggregates))

    logger.info(
        f"Generated performance statistics for {benchmark_name}: avg_score={avg_score:.4f}"
    )


async def run_benchmarks_with_job_queue(
    exp_dir: Path,
    agent_dir: Path,
    benchmarks: list[Type[BaseBenchmark]],
    max_workers: int = 6,
    image: str = SANDBOX_IMAGE,
    results_exp_dir: Path | None = None,
    select_problems=None,
    repeats: int = 1,
    subset_size: int = 18,
    benchmark_seed: int = 1,
    timeout_s: int = 600,
    cost_limit_usd: float = 3.0,
) -> None:
    """Run multiple benchmarks concurrently using a job queue approach

    Args:
        results_exp_dir: Where results go (see process_job)
        select_problems: Given a benchmark, returns the problems of it to run
            (default: all of them)
        repeats: Number of times to run each problem
        subset_size: Number of problems to take from each benchmark (0: all)
        benchmark_seed: Seed of that choice of problems
        timeout_s: Time limit for a problem
        cost_limit_usd: Cost limit for a problem
    """
    results_exp_dir = results_exp_dir or exp_dir
    results_agent_dir = results_exp_dir / agent_dir.name
    results_exp_dir.mkdir(parents=True, exist_ok=True)

    # Create worker ID queue (used for port assignment)
    id_queue = asyncio.Queue()
    for i in range(max_workers):
        id_queue.put_nowait(i)

    # Job queue
    job_queue = asyncio.Queue()

    # Load benchmark progress from metadata
    metadata = load_metadata(results_exp_dir)
    benchmark_progress = metadata.get("benchmark_progress", {})

    # Initialize benchmarks and enqueue jobs
    benchmarks_dict = {}

    for benchmark_cls in benchmarks:
        benchmark = benchmark_cls(seed=benchmark_seed, subset_size=subset_size or None)
        logger.info(f"Initializing benchmark: {benchmark.name}")

        benchmark_dir = results_agent_dir / "benchmarks" / benchmark.name
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Create a global results tracker and lock for this benchmark
        benchmark_locks[benchmark.name] = asyncio.Lock()
        benchmark_trackers[benchmark.name] = BenchmarkTracker(
            benchmark_dir / "results.jsonl"
        )

        # Get previously completed problems
        completed_problems = set(
            benchmark_progress.get(benchmark.name, {}).get("problems_completed", [])
        )

        # Add problems to the queue
        problems = select_problems(benchmark) if select_problems else benchmark.problems
        total = len(problems) * repeats
        problems_to_process = 0
        for repeat in range(repeats):
            for problem in problems:
                job = Job(benchmark.name, problem, benchmark, repeat, total)
                if job.result_id not in completed_problems:
                    await job_queue.put(job)
                    problems_to_process += 1

        logger.info(
            f"Added {problems_to_process} problems from {benchmark.name} to queue"
        )
        benchmarks_dict[benchmark.name] = benchmark

    # If no jobs to process, we're done
    if job_queue.empty():
        logger.info("No problems to process, all benchmarks are complete")
        # Generate final perf reports for each benchmark
        for benchmark_name, benchmark in benchmarks_dict.items():
            await generate_benchmark_statistics(results_agent_dir, benchmark_name)
        return

    async def worker():
        worker_id = await id_queue.get()
        try:
            while True:
                try:
                    job = await job_queue.get()
                except asyncio.QueueEmpty:
                    # No more jobs
                    break

                try:
                    # Process the job
                    logger.info(f"Worker {worker_id} processing job {job}")
                    await process_job(
                        exp_dir,
                        agent_dir,
                        job,
                        worker_id,
                        image,
                        results_exp_dir,
                        timeout_s,
                        cost_limit_usd,
                    )
                except Exception as e:
                    logger.error(f"Error in worker {worker_id} processing {job}: {e}")
                finally:
                    job_queue.task_done()
        finally:
            id_queue.put_nowait(worker_id)

    # Start workers
    logger.info(f"Starting {max_workers} workers")
    workers = []
    for _ in range(max_workers):
        task = asyncio.create_task(worker())
        workers.append(task)

    # Wait for all jobs to complete
    await job_queue.join()

    # Cancel workers
    for task in workers:
        task.cancel()

    # Wait for workers to be cancelled
    await asyncio.gather(*workers, return_exceptions=True)

    # Generate final perf reports for each benchmark
    for benchmark_name, benchmark in benchmarks_dict.items():
        await generate_benchmark_statistics(results_agent_dir, benchmark_name)


async def evaluate_agent(
    exp_id: int,
    exp_dir: Path,
    agent_id: int,
    workers: int,
    config: EvaluationConfig,
) -> None:
    """Run the benchmarks for one agent.

    With stages, the agent is first run on a few problems per benchmark, and
    evaluated in full only if that does not show it to be clearly worse than
    its parent. With a holdout fraction, the held-out problems are run last,
    with their results kept outside of the experiment directory.
    """
    agent_dir = exp_dir / f"agent_{agent_id}"
    unknown = [name for name in config.benchmarks if name not in benchmark_registry]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    benchmarks = [benchmark_registry[name] for name in config.benchmarks]
    image = await resolve_agent_image(exp_id, agent_id)
    seed = config.split_seed
    # How every benchmark run of this agent is made
    run_settings = dict(
        image=image,
        repeats=config.repeats,
        subset_size=config.subset_size,
        benchmark_seed=config.benchmark_seed,
        timeout_s=config.timeout_s,
        cost_limit_usd=config.cost_limit_usd,
    )

    def dev(benchmark: BaseBenchmark) -> list[Problem]:
        return split_problems(
            benchmark.problems, benchmark.name, config.holdout_fraction, seed
        )[0]

    def holdout(benchmark: BaseBenchmark) -> list[Problem]:
        return split_problems(
            benchmark.problems, benchmark.name, config.holdout_fraction, seed
        )[1]

    def stage1(benchmark: BaseBenchmark) -> list[Problem]:
        return stage1_problems(dev(benchmark), benchmark.name, config.stage1_size, seed)

    record = read_lineage(agent_dir) or {}
    parent_id = record.get("parent_id")
    evaluation = record.get("evaluation") or {}

    # First stage, unless its outcome was already decided (when resuming)
    if config.stage1_size > 0 and parent_id is not None and "promoted" not in evaluation:
        logger.info(f"Running first-stage benchmarks for agent {agent_id}")
        await run_benchmarks_with_job_queue(
            exp_dir,
            agent_dir,
            benchmarks,
            workers,
            select_problems=stage1,
            **run_settings,
        )
        delta = ArchiveAnalyzer(exp_dir).paired_delta(agent_id, parent_id)
        evaluation = {
            "promoted": should_promote(delta, config.promote_threshold),
            "promote_threshold": config.promote_threshold,
            "stage1_delta": (
                None
                if delta is None
                else {
                    "mean": delta.mean,
                    "ci_lower": delta.ci_lower,
                    "ci_upper": delta.ci_upper,
                    "n_problems": delta.n_problems,
                }
            ),
        }
        write_lineage(agent_dir, evaluation=evaluation)
        logger.info(
            f"Agent {agent_id} vs parent {parent_id} after the first stage: {delta}"
        )

    if evaluation.get("promoted") is False:
        logger.warning(
            f"Agent {agent_id} is clearly worse than its parent; not evaluating it in full"
        )
        return

    logger.info(f"Running benchmarks for agent {agent_id}")
    await run_benchmarks_with_job_queue(
        exp_dir,
        agent_dir,
        benchmarks,
        workers,
        select_problems=dev,
        **run_settings,
    )
    if config.holdout_fraction > 0:
        logger.info(f"Running held-out benchmarks for agent {agent_id}")
        await run_benchmarks_with_job_queue(
            exp_dir,
            agent_dir,
            benchmarks,
            workers,
            results_exp_dir=holdout_dir(exp_dir),
            select_problems=holdout,
            **run_settings,
        )


async def evaluate_pending_agents(
    exp_id: int,
    exp_dir: Path,
    workers: int,
    config: EvaluationConfig = EvaluationConfig(),
) -> None:
    """Run the benchmarks for every agent that has not been evaluated yet"""
    pending: list[int] = list(load_metadata(exp_dir).get("pending_agents", []))
    while pending:
        agent_id = pending[0]
        if agent_id in get_failed_iterations(exp_dir):
            # Its meta-improvement step failed, so the code is in an unknown
            # state: don't spend benchmark budget on it.
            logger.warning(f"Agent {agent_id} is marked as failed; skipping its benchmarks")
        else:
            await evaluate_agent(exp_id, exp_dir, agent_id, workers, config)
        pending = pending[1:]
        # Important: reset benchmark progress for the next agent
        update_metadata(exp_dir, pending_agents=pending, benchmark_progress=dict())
        if config.holdout_fraction > 0 and holdout_dir(exp_dir).exists():
            update_metadata(holdout_dir(exp_dir), benchmark_progress=dict())
