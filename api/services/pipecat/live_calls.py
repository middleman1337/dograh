"""Registry of live calls that a supervisor can speak into.

A running telephony call lives inside one pipeline task. The supervisor
console needs a way to make the agent's mouth say something on that call
without waiting for the model to ask. This module keeps a small handle per
live workflow run so an HTTP request can queue speech into it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from loguru import logger


class LiveCall:
    """A handle to one running call."""

    def __init__(
        self,
        workflow_run_id: int,
        workflow_id: int,
        queue_speech: Callable[..., Awaitable[Any]],
    ) -> None:
        self.workflow_run_id = workflow_run_id
        self.workflow_id = workflow_id
        self._queue_speech = queue_speech

    async def say(self, text: str, *, mute_user: bool = True) -> dict[str, Any]:
        playback = await self._queue_speech(
            text,
            mute_user=mute_user,
            persist_to_logs=True,
            append_to_context=False,
        )
        return {"queued": True, "speech_id": getattr(playback, "id", None)}


_live_calls: dict[int, LiveCall] = {}


def register_live_call(
    workflow_run_id: int,
    workflow_id: int,
    queue_speech: Callable[..., Awaitable[Any]],
) -> LiveCall:
    handle = LiveCall(workflow_run_id, workflow_id, queue_speech)
    _live_calls[workflow_run_id] = handle
    logger.info("live call {} registered", workflow_run_id)
    return handle


def unregister_live_call(workflow_run_id: int) -> None:
    if _live_calls.pop(workflow_run_id, None) is not None:
        logger.info("live call {} unregistered", workflow_run_id)


def get_live_call(workflow_run_id: int) -> LiveCall | None:
    return _live_calls.get(workflow_run_id)


def live_call_ids() -> list[int]:
    return sorted(_live_calls)
