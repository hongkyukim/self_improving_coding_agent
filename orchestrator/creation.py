"""
Creating agents: planning a generation, running the meta-improvement step
that makes a new agent, and validating what it made.
"""

import re
import shutil
import asyncio
import logging

from uuid import uuid4
from pathlib import Path
from contextlib import asynccontextmanager

from base_agent.src.utils.archive_analysis import ArchiveAnalyzer
from base_agent.src.utils.usage import read_usage
from base_agent.src.utils.lineage import (
    STATUS_OK,
    STATUS_FAILED,
    STATUS_INVALID,
    STATUS_PENDING,
    FINISHED_STATUSES,
    read_lineage,
    write_lineage,
    child_generation,
    get_failed_iterations,
    get_pruned_iterations,
)
from search.selection import (
    DEFAULT_SELECTION_STRATEGY,
    ArchiveView,
    get_selector,
    selection_rng,
)
from search.changelog import render_other_branches, update_change_log
from search.gate import (
    AGENT_CODE_DIR,
    GateConfig,
    GateResult,
    check_tests,
    fix_prompt,
    run_gate,
)
from runner_config import META_AGENT_CHOICES

from . import containers, evaluation
from .containers import (
    MONITOR_PORT_MAPPING,
    agent_image_tag,
    build_agent_image,
    get_container_host_port,
    resolve_agent_image,
    sandbox_secret_args,
)
from .evaluation import read_agent_usage
from .state import get_agent_ids, set_aside

logger = logging.getLogger(__name__)


# Limits for the meta agent when it is asked to fix an agent that failed the
# validation gate
GATE_FIX_TIMEOUT_SECONDS = 30 * 60


GATE_FIX_COST_THRESHOLD_USD = 5.0


@asynccontextmanager
async def gate_container(exp_id: int, agent_id: int, image: str, agent_code_dir: Path):
    """A throwaway container holding a copy of an agent's code at
    AGENT_CODE_DIR. Yields a function that runs a command in it."""
    name = f"sica_run_{exp_id}_agent_{agent_id}_gate_{uuid4().hex[:8]}"
    try:
        # The code is copied rather than mounted read-write: the checks leave
        # caches and other files behind, which must not end up in the agent
        success, _, stderr = await containers.run_docker_command(
            "docker", "run", "--rm", "-d", "--name", name,
            *sandbox_secret_args(),
            "-v", f"{agent_code_dir.absolute()}:/mnt/agent_code:ro",
            image, "tail", "-f", "/dev/null",
        )  # fmt: skip
        if not success:
            raise RuntimeError(f"Could not start the validation container: {stderr}")
        await containers.wait_for_container_ready(name)
        success, _, stderr = await containers.run_docker_command(
            "docker", "exec", name, "cp", "-r", "/mnt/agent_code", AGENT_CODE_DIR
        )
        if not success:
            raise RuntimeError(f"Could not copy the agent code for validation: {stderr}")

        async def exec(
            *cmd: str, env: dict[str, str] | None = None, timeout: float | None = None
        ) -> tuple[bool, str, str]:
            docker_cmd = ["docker", "exec"]
            for key, value in (env or {}).items():
                docker_cmd += ["-e", f"{key}={value}"]
            try:
                async with asyncio.timeout(timeout):
                    return await containers.run_docker_command(*docker_cmd, name, *cmd)
            except asyncio.TimeoutError:
                return False, "", f"Timed out after {timeout} seconds"

        yield exec
    finally:
        await containers.run_docker_command("docker", "rm", "-f", name)


async def agent_tests_pass(exp_id: int, exp_dir: Path, agent_id: int) -> bool | None:
    """Whether an agent's own unit tests pass in its sandbox. Taken from its
    lineage record if known, and otherwise run now and recorded."""
    agent_dir = exp_dir / f"agent_{agent_id}"
    known = (read_lineage(agent_dir) or {}).get("tests_passed")
    if known is not None:
        return known
    image = await resolve_agent_image(exp_id, agent_id)
    async with gate_container(exp_id, agent_id, image, agent_dir / "agent_code") as exec:
        check = await check_tests(exec)
    write_lineage(agent_dir, tests_passed=check.raw_passed)
    return check.raw_passed


async def validate_agent(
    exp_id: int, exp_dir: Path, agent_id: int, gate: GateConfig, image: str
) -> GateResult:
    """Run the validation gate on an agent, in the given image."""
    agent_dir = exp_dir / f"agent_{agent_id}"
    parent_id = (read_lineage(agent_dir) or {}).get("parent_id")

    async def parent_tests_pass() -> bool | None:
        if parent_id is None:
            return None
        return await agent_tests_pass(exp_id, exp_dir, parent_id)

    async with gate_container(exp_id, agent_id, image, agent_dir / "agent_code") as exec:
        return await run_gate(exec, gate, parent_tests_pass)


async def check_initial_agent(exp_id: int, exp_dir: Path, gate: GateConfig) -> None:
    """Run the validation gate on the initial agent, once per experiment.

    If even the initial agent does not get through, the setup is at fault
    (image, API keys) and every agent made from it would be thrown out, each
    at the cost of a meta-improvement step. Its unit tests are the exception:
    they set the standard for its descendants rather than having to pass.

    Raises:
        RuntimeError: if the initial agent does not pass the gate
    """
    agent_dir = exp_dir / "agent_0"
    if not gate.enabled or (read_lineage(agent_dir) or {}).get("gate") is not None:
        return
    logger.info("Checking that the initial agent passes the validation gate")

    async def no_standard_yet() -> bool:
        return False

    image = await resolve_agent_image(exp_id, 0)
    async with gate_container(exp_id, 0, image, agent_dir / "agent_code") as exec:
        result = await run_gate(exec, gate, no_standard_yet)
    if not result.passed:
        raise RuntimeError(
            "The initial agent does not pass the validation gate, so no agent made "
            "from it would. Check the setup (is the sandbox image built, are the API "
            "keys exported?), or switch the gate off with --gate-checks none.\n"
            f"{result.failure_reason()}"
        )
    write_lineage(agent_dir, gate=result.to_dict(), tests_passed=result.tests_passed)
    if result.tests_passed is False:
        logger.warning(
            "The unit tests of the initial agent fail in the sandbox; new agents "
            "will not be required to pass theirs"
        )


async def run_meta_agent_benchmark(
    exp_id: int, exp_dir: Path, child: dict, gate: GateConfig = GateConfig()
) -> None:
    """Create a new agent version from an archived one

    We can conceptually think of this as a 'meta agent improvement' benchmark.

    Args:
        exp_id: The experiment id
        exp_dir: The experiment directory (results/run_{exp_id})
        child: The planned agent (see plan_generation): its `id`, the
            `parent_id` whose code it starts from, the `meta_agent_id` that
            makes the changes (using `meta_model`, if given), and the
            `selection` record of the parent choice
        gate: The validation that the new agent has to pass to be usable
    """
    child_id: int = child["id"]
    parent_id: int = child["parent_id"]
    meta_agent_id: int = child["meta_agent_id"]
    meta_model: str | None = child.get("meta_model")
    child_dir = exp_dir / f"agent_{child_id}"
    child_code_dir = child_dir / "agent_code"
    parent_code_dir = exp_dir / f"agent_{parent_id}" / "agent_code"
    logger.info(
        f"Creating agent {child_id} from agent {parent_id} (meta agent: {meta_agent_id})"
    )

    if child_dir.exists():
        existing = read_lineage(child_dir) or {}
        if existing.get("status") in FINISHED_STATUSES:
            raise ValueError(f"Agent {child_id} has already been created")
        set_aside(exp_dir, child_dir)
    child_dir.mkdir(parents=True)

    # Record where the new agent comes from. It stays 'pending' until the meta
    # agent has run, and is marked 'failed' if it does not complete, or
    # 'invalid' if what it made cannot run.
    write_lineage(
        child_dir,
        id=child_id,
        parent_id=parent_id,
        meta_agent_id=meta_agent_id,
        meta_model=meta_model,
        generation=child_generation(exp_dir, parent_id),
        selection=child.get("selection"),
        status=STATUS_PENDING,
    )
    status = STATUS_FAILED
    failure_reason: str | None = "Meta-improvement did not complete"
    interrupted = False

    # Create a unique container name
    local_id = f"agent_{uuid4().hex[:8]}"
    container_name = f"sica_run_{exp_id}_agent_{child_id}_improvement_{local_id}"
    container_name = re.sub(f"[^a-zA-Z0-9_]", "", container_name)

    try:
        # 1. Copy the parent's code as the starting point. Its change log
        # comes along, and describes exactly the changes present in this code.
        shutil.copytree(parent_code_dir, child_code_dir)

        # 2. What was tried on other branches goes in a generated section
        update_change_log(
            child_code_dir / "agent_change_log.md",
            render_other_branches(ArchiveAnalyzer(exp_dir), parent_id),
        )

        # Setup meta-improvement logs directory
        meta_log_dir = child_dir / "meta_improvement_logs"
        container_meta_log_path = f"/home/agent/meta_improvement_logs_{child_id}"
        meta_log_dir.mkdir(parents=True, exist_ok=True)

        # 3. Start the container, from the image of the agent that will run
        image = await resolve_agent_image(exp_id, meta_agent_id)
        container_cmd = ["docker", "run", "--rm", "-d", "--name", container_name]
        # Publish the monitoring server on a free host port
        container_cmd += ["-p", MONITOR_PORT_MAPPING]
        container_cmd += sandbox_secret_args()
        # Mount the full agent archive as read-only
        container_cmd += ["-v", f"{exp_dir.absolute()}:/home/agent/archive:ro"]
        # Mount the new agent directory as the workdir
        container_cmd += ["-v", f"{child_code_dir.absolute()}:/home/agent/workdir:rw"]
        # Mount meta agent log directory
        container_cmd += ["-v", f"{meta_log_dir.absolute()}:{container_meta_log_path}:rw"]
        container_cmd += [image, "tail", "-f", "/dev/null"]
        success, _, stderr = await containers.run_docker_command(*container_cmd)
        if not success:
            logger.error(f"Could not start container: {stderr}")
            failure_reason = f"Could not start container: {stderr[-2000:]}"
            return

        await containers.wait_for_container_ready(container_name)
        monitor_port = await get_container_host_port(container_name)
        logger.info(
            f"Meta-improvement monitor (agent {child_id}): http://localhost:{monitor_port}"
        )

        # 4. Execute the (meta) agent command and get stdout
        agent_module = f"archive.agent_{meta_agent_id}.agent_code"
        agent_cmd = ["python", "-m", agent_module, "improve"]
        agent_cmd += ["--workdir", "/home/agent/workdir"]
        agent_cmd += ["--logdir", container_meta_log_path]
        # The agent whose code is in the workdir, and the agent being created
        agent_cmd += ["--best-iter", str(parent_id)]
        agent_cmd += ["--current-iter", str(child_id)]
        exec_cmd = ["docker", "exec"]
        if meta_model:
            # Read by the agent's settings (AGENT_ prefix), for this run only
            exec_cmd += ["-e", f"AGENT_MODEL={meta_model}"]
        success, stdout, stderr = await containers.run_docker_command(
            *exec_cmd, container_name, *agent_cmd
        )

        if not success:
            logger.error(f"Meta agent failed creating agent {child_id}: {stderr}")
            logger.error(stdout)
            failure_reason = f"Meta agent exited with an error: {stderr[-2000:]}"
            return

        # What it took to create this agent, for cost accounting
        usage = read_agent_usage(meta_log_dir, stdout)
        if usage is None:
            logger.warning(f"Meta agent reported no usage for agent {child_id}")
            usage = {"tokens": 0, "cached_tokens": 0, "cost": 0.0, "time": 0.0}
        else:
            write_lineage(child_dir, usage=usage)
        tokens, cached, cost, time = (
            usage["tokens"], usage["cached_tokens"], usage["cost"], usage["time"]
        )  # fmt: skip
        percent_cached = cached / tokens * 100 if tokens > 0 else 0
        logger.info(
            f"meta-improvement (agent {child_id}) || tokens: {tokens} (cached: {percent_cached:.2f}%), cost: ${cost:.4f}, duration: {time:.2f}s"
        )

        fix_attempts = 0
        while True:
            # 5. Build the new agent's image, with its (possibly changed)
            # requirements.txt. An agent whose image does not build cannot run.
            image_tag = agent_image_tag(exp_id, child_id)
            built, build_stderr = await build_agent_image(image_tag, child_code_dir)
            if not built:
                logger.error(f"Image build failed for agent {child_id}: {build_stderr}")
                failure_reason = f"Image build failed: {build_stderr[-2000:]}"
                return

            # 6. Validate it: an agent that cannot run is not worth evaluating
            if not gate.enabled:
                break
            result = await validate_agent(exp_id, exp_dir, child_id, gate, image_tag)
            write_lineage(
                child_dir,
                gate={**result.to_dict(), "fix_attempts": fix_attempts},
                tests_passed=result.tests_passed,
            )
            if result.passed:
                break
            logger.error(f"Agent {child_id}: {result.failure_reason()}")
            if fix_attempts >= gate.retries:
                status = STATUS_INVALID
                failure_reason = result.failure_reason()
                return

            # Ask the meta agent to repair what it made, then build and
            # validate again
            fix_attempts += 1
            logger.info(
                f"Asking the meta agent to fix agent {child_id} (attempt {fix_attempts} of {gate.retries})"
            )
            fix_cmd = ["python", "-m", f"{agent_module}.agent"]
            fix_cmd += ["--prompt", fix_prompt(result)]
            fix_cmd += ["--workdir", "/home/agent/workdir"]
            fix_cmd += ["--logdir", f"{container_meta_log_path}/gate_fix_{fix_attempts}"]
            fix_cmd += ["--timeout", str(GATE_FIX_TIMEOUT_SECONDS)]
            fix_cmd += ["--cost-threshold", str(GATE_FIX_COST_THRESHOLD_USD)]
            fixed, _, fix_stderr = await containers.run_docker_command(
                *exec_cmd, container_name, *fix_cmd
            )
            if not fixed:
                logger.warning(f"The fix attempt for agent {child_id} errored: {fix_stderr}")
            # The fix is part of what it took to create this agent
            fix_usage = read_usage(meta_log_dir / f"gate_fix_{fix_attempts}")
            if fix_usage is not None:
                usage = {key: usage[key] + fix_usage[key] for key in usage}
                write_lineage(child_dir, usage=usage)

        # The agent is usable; anything that goes wrong from here on
        # (summaries) does not invalidate it.
        status, failure_reason = STATUS_OK, None

        # 7. Generate and save a summary for the meta-improvement logs
        trace_path = meta_log_dir / "execution_tree.txt"
        if trace_path.exists():
            trace = trace_path.read_text()
        else:
            trace = "No trace was available"

        # Define a simple problem statement and answer for context
        problem_statement = (
            f"Improve the agent from iteration {parent_id} to create iteration {child_id}"
        )
        llm_answer = "Agent code updated"
        score = 1.0  # Binary success/failure

        # Additional context for the summary
        meta_context = (
            f"The 'answer' actually pertains to whether the agent was successful (1) or errored out (0) during the meta-improvement task"
            f"Meta-improvement process used {tokens} tokens ({percent_cached:.2f}% cached), "
            f"cost ${cost:.4f}, and took {time:.2f} seconds."
        )
        if stderr:
            meta_context += f"\nErrors encountered: {stderr}"

        try:
            summary = await evaluation.generate_contextual_summary(
                problem_statement=problem_statement,
                llm_answer=llm_answer,
                trace=trace,
                score=score,
                parse_errors=None,
                answer_discussion=meta_context,
            )
        except Exception as e:
            logger.warning(
                f"Summary generation failed for agent {child_id} "
                f"(the agent is kept): {type(e).__name__}: {e}"
            )
            summary = f"(Summary unavailable: {type(e).__name__}: {e})"

        summary_path = meta_log_dir / "summary.txt"
        summary_path.write_text(summary)
        logger.info(f"Meta-improvement summary saved to {summary_path}")

    except Exception as e:
        if status != STATUS_OK:
            failure_reason = f"{type(e).__name__}: {e}"
        raise

    except BaseException:
        # The runner itself is being stopped (Ctrl-C, cancellation)
        interrupted = True
        raise

    finally:

        # Cleanup running docker container
        await containers.run_docker_command("docker", "rm", "-f", container_name)

        if status == STATUS_OK:
            write_lineage(child_dir, status=status)
        elif interrupted:
            # Left 'pending', so that it is created afresh on resumption
            logger.warning(f"Creation of agent {child_id} was interrupted")
        else:
            logger.error(f"Marking agent {child_id} as {status}: {failure_reason}")
            write_lineage(child_dir, status=status, failure_reason=failure_reason)


def check_meta_agent(exp_dir: Path, meta_agent: str | int) -> None:
    """Raises ValueError if a fixed meta agent cannot do the job: it has to
    exist, and to have been created successfully."""
    if isinstance(meta_agent, str):
        if meta_agent not in META_AGENT_CHOICES:
            raise ValueError(f"Unknown meta agent {meta_agent!r}")
        return
    if meta_agent not in get_agent_ids(exp_dir):
        raise ValueError(f"Meta agent {meta_agent} does not exist in {exp_dir}")
    if meta_agent in get_failed_iterations(exp_dir):
        raise ValueError(f"Meta agent {meta_agent} was not created successfully")


def plan_generation(
    exp_dir: Path,
    generation: int,
    n_children: int,
    selection: str = DEFAULT_SELECTION_STRATEGY,
    seed: int = 0,
    meta_agent: str | int = "best",
    meta_model: str | None = None,
) -> list[dict]:
    """Decide which agents to create in a generation, and from which parents.

    `meta_agent` says which agent carries out each improvement: 'best' (the
    best-scoring agent), 'parent' (the new agent's own parent), 'seed' (the
    initial agent) or a fixed agent id. `meta_model` optionally overrides the
    model that it uses.

    The plan is made once per generation and stored, so that a resumed
    experiment creates the same agents as an uninterrupted one.
    """
    # Agents whose own meta-improvement step failed are never built upon, nor
    # used as the meta agent. Nor are agents that were dropped after the first
    # evaluation stage: they were clearly worse than their parent, and their
    # scores on a subset of the problems are not comparable with the others'.
    failed = get_failed_iterations(exp_dir) | get_pruned_iterations(exp_dir)
    analyzer = ArchiveAnalyzer(exp_dir)
    agent_ids = get_agent_ids(exp_dir) or [0]

    archive_view = ArchiveView.from_analyzer(
        analyzer, max(agent_ids), score_type="mean_score", excluded=failed
    )
    selected = get_selector(selection).select(
        archive_view, k=n_children, rng=selection_rng(seed, generation)
    )

    check_meta_agent(exp_dir, meta_agent)
    try:
        best_agent_id = analyzer.get_best_agent_iteration(exclude=failed)
    except Exception:
        best_agent_id = 0

    def meta_agent_for(parent_id: int) -> int:
        if meta_agent == "best":
            return best_agent_id
        if meta_agent == "parent":
            return parent_id
        if meta_agent == "seed":
            return 0
        return int(meta_agent)

    weights = {str(i): w for i, w in selected.weights.items()}
    next_id = max(agent_ids) + 1
    return [
        {
            "id": next_id + index,
            "parent_id": parent_id,
            "meta_agent_id": meta_agent_for(parent_id),
            "meta_model": meta_model,
            # Enough to reproduce the choice of parent: it is
            # select(k=n_children, rng=selection_rng(seed, generation)).parents[index]
            "selection": {
                "strategy": selection,
                "seed": seed,
                "generation": generation,
                "n_children": n_children,
                "index": index,
                "weights": weights,
            },
        }
        for index, parent_id in enumerate(selected.parents)
    ]
