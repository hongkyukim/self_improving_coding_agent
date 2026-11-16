"""Live checks of the Anthropic provider against the real API.

These send a handful of small requests, which costs a few cents. They are
skipped unless asked for, and need ANTHROPIC_API_KEY:

    cd base_agent && pytest tests/llm/test_anthropic_live.py --run-llm

The model is the agent's configured one (AGENT_MODEL, default SONNET_55).
What they establish is that the requests the provider builds are accepted,
which the offline tests in test_anthropic_provider.py cannot.
"""
import os

import pytest

from src.config import settings
from src.llm.api import ModelProvider
from src.llm.base import Message
from src.types.llm_types import Provider, ReasoningEffort, StopReason, TextContent

pytestmark = [
    pytest.mark.uses_llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"
    ),
]

MODEL = settings.MODEL


def text(role, value):
    return Message(role=role, content=[TextContent(text=value)])


def reply_text(completion):
    return "".join(b.text for b in completion.content if isinstance(b, TextContent))


async def complete(messages, **kwargs):
    assert MODEL.provider == Provider.ANTHROPIC, f"{MODEL.name} is not an Anthropic model"
    provider = ModelProvider.get_provider(Provider.ANTHROPIC)
    kwargs.setdefault("max_tokens", 2000)
    return await provider.create_completion(messages=messages, model=MODEL, **kwargs)


async def test_first_step_is_accepted():
    completion = await complete(
        [text("system", "You answer in one short sentence."), text("user", "What is 2 + 2?")]
    )

    assert completion.stop_reason == StopReason.COMPLETE
    assert "4" in reply_text(completion)
    assert completion.usage.completion_tokens > 0
    assert completion.calculate_cost() > 0


async def test_step_that_continues_earlier_work_is_accepted():
    # What an agent sends on every step after its first: its work so far, as
    # an assistant turn. Current models reject a request that ends in one.
    completion = await complete(
        [
            text("system", "You are working through a task step by step."),
            text("user", "Count from 1 to 6, one number per line."),
            text("assistant", "1\n2\n3"),
        ]
    )

    assert completion.stop_reason == StopReason.COMPLETE
    reply = reply_text(completion)
    assert "4" in reply and "6" in reply


async def test_stop_sequences_are_honoured():
    # The agents' own tool calling depends on generation stopping at a marker
    completion = await complete(
        [text("user", "Write exactly this and nothing else: ALPHA </STOP_HERE> BETA")],
        stop=["</STOP_HERE>"],
    )

    assert completion.stop_reason == StopReason.STOP_TOKEN
    assert completion.stop_sequence == "</STOP_HERE>"
    reply = reply_text(completion)
    assert "ALPHA" in reply and "BETA" not in reply


async def test_reasoning_effort_is_accepted():
    completion = await complete(
        [text("user", "What is 17 * 3? Reply with the number only.")],
        reasoning_effort=ReasoningEffort.LOW,
    )

    assert completion.stop_reason == StopReason.COMPLETE
    assert "51" in reply_text(completion)


async def test_second_request_reads_the_cache():
    # The agent resends a growing prefix on every step; cached input is what
    # keeps that affordable. Needs a prefix long enough to be cacheable.
    background = "Reference notes. " + " ".join(
        f"Item {n}: the value of entry {n} is {n * 7}." for n in range(1500)
    )
    messages = [text("system", background), text("user", "What is the value of entry 12?")]

    await complete(messages)
    again = await complete(messages)

    assert again.usage.cached_prompt_tokens > 0
