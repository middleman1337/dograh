"""Live call events for the supervisor console.

The browser console already receives real-time feedback over a WebSocket, but
telephony calls have no browser attached, so the supervisor could only read
a transcript after the call ended. This module streams the same events to the
supervisor service (the cockpit) so it can follow a call while it happens.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from loguru import logger


class LiveEventForwarder:
    """Forward pipeline events to the supervisor service."""

    def __init__(
        self,
        endpoint: str,
        *,
        workflow_run_id: int,
        workflow_id: int,
        call_id: str | None = None,
        caller: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._workflow_run_id = workflow_run_id
        self._workflow_id = workflow_id
        self._call_id = call_id
        self._caller = caller
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: dict[str, Any]) -> None:
        if not self._endpoint:
            return
        payload = {
            "workflow_run_id": self._workflow_run_id,
            "workflow_id": self._workflow_id,
            "call_id": self._call_id,
            "caller": self._caller,
            "event": message,
        }
        try:
            client = await self._http()
            response = await client.post(self._endpoint, json=payload)
            if response.status_code >= 400:
                logger.debug(
                    "live event rejected: {} {}", response.status_code, response.text[:120]
                )
        except httpx.HTTPError as exc:
            logger.debug("live event not delivered: {}", exc)


_forwarders: dict[int, LiveEventForwarder] = {}


async def close_forwarder(workflow_run_id: int) -> None:
    forwarder = _forwarders.pop(workflow_run_id, None)
    if forwarder is not None:
        await forwarder.close()


def live_event_forwarder(
    *,
    workflow_run_id: int,
    workflow_id: int,
    call_context: dict,
) -> LiveEventForwarder | None:
    """Build the forwarder when the supervisor service is configured."""
    endpoint = os.getenv("SUPERVISOR_LIVE_URL", "").strip()
    if not endpoint:
        return None
    forwarder = LiveEventForwarder(
        endpoint,
        workflow_run_id=workflow_run_id,
        workflow_id=workflow_id,
        call_id=call_context.get("call_id"),
        caller=call_context.get("caller_number") or call_context.get("caller_id"),
    )
    _forwarders[workflow_run_id] = forwarder
    return forwarder
