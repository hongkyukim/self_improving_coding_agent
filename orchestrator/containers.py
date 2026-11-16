"""
Docker: the sandbox image, and running commands in containers.
"""

import os
import json
import asyncio
import logging
import platform

from pathlib import Path
from asyncio.subprocess import Process

logger = logging.getLogger(__name__)


# The image that agents run in. Agents made during an experiment get an image
# of their own, tagged SANDBOX_IMAGE:run{exp_id}_agent{agent_id}.
SANDBOX_IMAGE = "sica_sandbox"


# Credentials are not built into the sandbox image, but handed to each
# container when it is started. These variables are passed on if set:
SANDBOX_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "VERTEX_PROJECT_ID",
    "VERTEX_REGION",
)


# and this file, if it exists, is mounted read-only:
GOOGLE_CREDENTIALS_FILE = Path("sandbox/GOOGLE_APPLICATION_CREDENTIALS.json")


CONTAINER_GOOGLE_CREDENTIALS = "/tmp/GOOGLE_APPLICATION_CREDENTIALS.json"


# The agent's monitoring web server listens on this port inside the container
MONITOR_CONTAINER_PORT = 8080


# Publish it on a free host port chosen by docker (loopback only), so that any
# number of containers can run concurrently without port collisions.
MONITOR_PORT_MAPPING = f"127.0.0.1::{MONITOR_CONTAINER_PORT}"


def sandbox_secret_args() -> list[str]:
    """`docker run` arguments that give a container its credentials.

    Variables are passed by name only (`-e NAME`): docker takes the value
    from this process's environment, so that it never appears on a command
    line, in a log, or in a file. Commands run later with `docker exec`
    inherit them.
    """
    args: list[str] = []
    for name in SANDBOX_ENV_VARS:
        if os.environ.get(name):
            args += ["-e", name]
    if GOOGLE_CREDENTIALS_FILE.is_file():
        args += [
            "-v",
            f"{GOOGLE_CREDENTIALS_FILE.absolute()}:{CONTAINER_GOOGLE_CREDENTIALS}:ro",
            "-e",
            f"GOOGLE_APPLICATION_CREDENTIALS={CONTAINER_GOOGLE_CREDENTIALS}",
        ]
    return args


async def run_docker_command(*args) -> tuple[bool, str, str]:
    """Run a command inside a docker container"""
    logger.debug("Running docker command:")
    logger.debug(" ".join(args))
    proc: Process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    success = True
    if proc.returncode != 0:
        logger.debug(f"Command {args[0]} failed: {stderr.decode()}")
        success = False
    return success, stdout.decode().strip(), stderr.decode().strip()


async def get_container_host_port(
    container_name: str, container_port: int = MONITOR_CONTAINER_PORT
) -> int | None:
    """Returns the host port docker assigned to a published container port,
    or None if it could not be determined."""
    success, stdout, _ = await run_docker_command(
        "docker", "port", container_name, f"{container_port}/tcp"
    )
    if not success or not stdout:
        return None
    # One line per address family, e.g. '127.0.0.1:49153' or '[::]:49153'
    try:
        return int(stdout.splitlines()[0].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


async def wait_for_container_ready(container_name: str, timeout: float = 30):
    """Waits until a container is ready after starting it up"""
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            # Get container status in JSON format
            _, stdout, _ = await run_docker_command(
                "docker",
                "inspect",
                "--format",
                "{{json .State.Status}}",
                container_name,
            )
            status = json.loads(stdout)

            if status == "running":
                # Additional health check - try a basic command
                try:
                    await run_docker_command("docker", "exec", container_name, "ps")
                    return  # Container is truly ready
                except Exception:
                    pass  # Container not quite ready yet

        except Exception:
            pass  # Container might not exist yet

        await asyncio.sleep(0.1)  # Short delay before retry

    raise TimeoutError(
        f"Container {container_name} did not become ready within {timeout} seconds"
    )


def agent_image_tag(exp_id: int, agent_id: int) -> str:
    return f"{SANDBOX_IMAGE}:run{exp_id}_agent{agent_id}"


async def resolve_agent_image(exp_id: int, agent_id: int) -> str:
    """The image to run an agent in: its own if one was built, else the base
    image (the initial agent, and agents from before per-agent images)."""
    tag = agent_image_tag(exp_id, agent_id)
    exists, _, _ = await run_docker_command("docker", "image", "inspect", tag)
    return tag if exists else SANDBOX_IMAGE


async def build_agent_image(tag: str, agent_code_dir: Path) -> tuple[bool, str]:
    """Build the sandbox image for an agent, installing the requirements.txt
    of that agent's code. Returns (success, stderr).

    No credentials go into the image: see sandbox_secret_args.
    """
    target_arch = "x86_64"  # Default architecture
    # Detect if the script is running on a Mac with Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        target_arch = "aarch64"
    build_cmd = ["docker", "buildx", "build"]
    build_cmd += ["--build-context", f"base_agent={agent_code_dir.absolute()}"]
    build_cmd += ["--build-arg", f"TARGET_ARCH={target_arch}"]
    build_cmd += ["-f", "sandbox/Dockerfile", "-t", tag, "--load", "sandbox"]
    success, _, stderr = await run_docker_command(*build_cmd)
    return success, stderr
