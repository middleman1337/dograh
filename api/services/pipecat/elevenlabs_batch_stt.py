"""File based ElevenLabs speech to text for deployments without realtime access.

The realtime websocket endpoint needs a plan that allows it. The file based
endpoint is available on every plan, detects the caller's language on its own
and answers in well under a second for a spoken phrase. This service wraps
pipecat's implementation and owns the HTTP session it needs, so the caller
does not have to hand one in.
"""

from __future__ import annotations

import aiohttp
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.stt import ElevenLabsSTTSettings as BatchSettings


class ElevenLabsBatchSTTService(ElevenLabsSTTService):
    """ElevenLabs transcription over the file endpoint, with automatic language detection."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "scribe_v2",
        language: str | None = None,
        **kwargs,
    ) -> None:
        self._owned_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30.0, connect=10.0)
        )
        super().__init__(
            api_key=api_key,
            aiohttp_session=self._owned_session,
            base_url=base_url.rstrip("/"),
            settings=BatchSettings(model=model, language=language),
            **kwargs,
        )

    async def stop(self, *args, **kwargs) -> None:
        await super().stop(*args, **kwargs)
        if not self._owned_session.closed:
            await self._owned_session.close()
