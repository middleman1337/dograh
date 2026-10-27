"""Supervisor escalation watchdog.

The agent is supposed to call the supervisor-decision tool when a caller asks
for a person, a manager, or a human being. A model sometimes skips that call,
which leaves the caller waiting for a supervisor who was never told.

This observer watches the live transcription stream and raises the decision
card itself when the caller asks for a human. The cockpit de-duplicates the
card against any card the agent already raised for the same call, so the
supervisor never sees two cards for one request.

The watchdog is disabled unless ``SUPERVISOR_ESCALATION_URL`` is set.
"""

import asyncio
import os
import re
import unicodedata
from typing import Optional

import httpx
from loguru import logger
from pipecat.frames.frames import TranscriptionFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

DEFAULT_PATTERNS: tuple[str, ...] = (
    # English
    r"\bspeak (?:to|with) (?:a |an |the )?(?:real |actual |human |live )?(?:person|human|human being|somebody|someone|agent|representative|manager|supervisor|advisor|adviser|operator|team|staff|consultant|specialist)\b",
    r"\b(?:talk|speak|call|connect|put|transfer) (?:me )?(?:to|with|through to) (?:a |an |the )?(?:real |actual |human |live )?(?:person|human|human being|somebody|someone|agent|representative|manager|supervisor|advisor|adviser|operator|team|staff)\b",
    r"\bi (?:want|need|would like) (?:to )?(?:speak|talk|connect) (?:to|with) (?:a |an |the )?(?:real |actual |human |live )?(?:person|human|human being|somebody|someone|agent|representative|manager|supervisor|advisor|adviser|operator)\b",
    r"\b(?:let me |can i )?(?:speak|talk) (?:to|with) (?:a |an |the )?(?:real |actual |human |live )?(?:person|human|human being|somebody|someone|agent|representative|manager|supervisor)\b",
    r"\bis (?:there|anyone) (?:a |any )?(?:real |actual |human |live )?(?:person|human|human being|somebody|someone|agent|representative|manager|supervisor)\b",
    r"\bcan i speak (?:to|with) (?:someone|somebody|anyone|anybody|a human|a person|a real person|a manager|a supervisor)\b",
    r"\b(?:human|real person|live person|actual person)\b",
    r"\b(?:your|his|her|their|the) (?:manager|supervisor|agent|representative|advisor|adviser|consultant)\b",
    r"\b(?:escalate|transfer) (?:this |my |the )?(?:call|case|issue|request|conversation)\b",
    # Swahili
    r"\b(?:nataka|nahitaji|ningependa) (?:kuongea|kutafuta|kufanya mazungumzo) (?:na|kwa) (?:mtu|mzee|mtu wa kweli|msimamizi|mwakilishi)\b",
    r"\bmtu wa kweli\b",
    r"\b(?:msimamizi|mwalimu|mwakilishi|kiongozi)\b",
    # French
    r"\b(?:je veux|voudrais|je voudrais|je dois) (?:parler|être mis en relation|être transféré) (?:à|avec|un|une|le|la)\b.{0,30}\b(?:personne|humain|vrai personne|conseiller|responsable|manager|agent|vrai personne)\b",
    r"\b(?:parler|être transféré) (?:à|avec) (?:un|une|le|la) (?:vraie |vrai )?(?:personne|humain|conseiller|responsable|manager|agent)\b",
    r"\b(?:est-ce que|ya) (?:quelqu'un|quelqu un|un humain|une personne)\b",
    # Spanish
    r"\b(?:quiero|necesito|quisiera|me gustaría) (?:hablar|hablar con|conectar|transferir|que me pasen) (?:con|un|una|al|el)\b.{0,30}\b(?:persona|humano|persona real|agente|representante|gerente|supervisor)\b",
    r"\b(?:pásame|pasame|conéctame|conectame) con (?:una |un |el |la )?(?:persona|humano|agente|representante|gerente|supervisor)\b",
    # Portuguese
    r"\b(?:quero|preciso|gostaria) (?:falar|conversar|transferir) (?:com|com um|com uma|ao|a) (?:uma |um |a |o )?(?:pessoa|humano|pessoa real|atendente|representante|gerente|supervisor)\b",
    # German
    r"\bich (?:möchte|will|muss) (?:mit|einem|einer) (?:menschen|person|einer person|echten person|sachbearbeiter|berater|supervisor)\b",
    r"\bmit (?:einem|einer) (?:menschen|person|einer person)\b",
    # Italian
    r"\b(?:voglio|vorrei|ho bisogno di) (?:parlare|chattare|passare) (?:con|un|una|al|alla)\b.{0,30}\b(?:persona|umano|operatore|responsabile|supervisor)\b",
    # Arabic
    r"\b(?:أريد|اريد|أرغب|ابغى|ابي) (?:أن )?(?:أتحدث|اتحدث|أتكلم|اتكلم) (?:مع|إلى|الى)\b.{0,30}\b(?:شخص|إنسان|انسان|موظف|مسؤول)\b",
    # Hindi
    r"\b(?:मैं )?(?:किसी|इंसान|व्यक्ति|मैनेजर|प्रतिनिधि) (?:से )?(?:बात|बातचीत)\b",
)

def _fold(value: str) -> str:
    """Drop diacritics so an accentless transcript still matches."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_PATTERNS_CACHE: Optional[list[re.Pattern[str]]] = None


def _patterns() -> list[re.Pattern[str]]:
    global _PATTERNS_CACHE
    if _PATTERNS_CACHE is not None:
        return _PATTERNS_CACHE
    raw = os.getenv("SUPERVISOR_ESCALATION_PATTERNS", "").strip()
    source = [raw] if raw else list(DEFAULT_PATTERNS)
    compiled: list[re.Pattern[str]] = []
    for item in source:
        try:
            compiled.append(re.compile(_fold(item), re.IGNORECASE))
        except re.error as exc:
            logger.warning("invalid supervisor escalation pattern ignored: {}", exc)
    _PATTERNS_CACHE = compiled
    return compiled


def detect_person_request(text: str) -> Optional[str]:
    """Return the matched phrase when the caller asks for a human."""
    if not text:
        return None
    folded = _fold(text)
    for pattern in _patterns():
        match = pattern.search(folded)
        if match:
            return match.group(0)
    return None


class SupervisorEscalationObserver(BaseObserver):
    """Raise a supervisor decision card when the caller asks for a person."""

    def __init__(
        self,
        *,
        endpoint: str,
        run_id: int,
        call_id: str | None = None,
        caller: str | None = None,
        api_key: str | None = None,
        intent: str = "speak_to_person",
        delay_seconds: float = 8.0,
        timeout_seconds: float = 15.0,
    ) -> None:
        super().__init__()
        self._endpoint = endpoint
        self._run_id = run_id
        self._call_id = call_id
        self._caller = caller
        self._api_key = api_key or ""
        self._intent = intent
        self._delay_seconds = delay_seconds
        self._timeout_seconds = timeout_seconds
        self._frames_seen: set[int] = set()
        self._escalated = False
        self._tasks: set[asyncio.Task] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, TranscriptionFrame):
            return
        if frame.id in self._frames_seen:
            return
        self._frames_seen.add(frame.id)
        text = (frame.text or "").strip()
        if not text:
            return
        match = detect_person_request(text)
        if not match:
            return
        if self._escalated:
            return
        self._escalated = True
        logger.info(
            "supervisor escalation armed for run {} call {}: {}", self._run_id, self._call_id, match
        )
        task = asyncio.create_task(self._escalate(text, match))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _escalate(self, text: str, match: str) -> None:
        await asyncio.sleep(self._delay_seconds)
        payload = {
            "intent": self._intent,
            "summary": f"The caller asked for a person: “{text[:280]}”",
            "question": "The caller asked to speak to a person. How should the agent handle it?",
            "caller": self._caller,
            "call_id": self._call_id,
            "source": "watchdog",
            "context": {
                "source": "watchdog",
                "workflow_run_id": self._run_id,
                "transcript": text[:1000],
                "matched_phrase": match[:200],
            },
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-cockpit-key"] = self._api_key
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(self._endpoint, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("supervisor escalation failed: {}", exc)
            return
        if response.status_code >= 400:
            logger.warning(
                "supervisor escalation rejected: {} {}", response.status_code, response.text[:200]
            )
            return
        logger.info("supervisor escalation card raised for run {}", self._run_id)


def escalation_observer(
    *,
    run_id: int,
    call_context: dict,
) -> SupervisorEscalationObserver | None:
    """Build the watchdog observer when escalation is configured for this call."""
    endpoint = os.getenv("SUPERVISOR_ESCALATION_URL", "").strip()
    if not endpoint:
        return None
    caller = call_context.get("caller_number") or call_context.get("caller_id")
    call_id = call_context.get("call_id")
    try:
        delay = float(os.getenv("SUPERVISOR_ESCALATION_DELAY_SECONDS", "8"))
    except ValueError:
        delay = 8.0
    try:
        timeout = float(os.getenv("SUPERVISOR_ESCALATION_TIMEOUT_SECONDS", "15"))
    except ValueError:
        timeout = 15.0
    return SupervisorEscalationObserver(
        endpoint=endpoint,
        run_id=run_id,
        call_id=str(call_id) if call_id else None,
        caller=str(caller) if caller else None,
        api_key=os.getenv("SUPERVISOR_ESCALATION_API_KEY", "").strip() or None,
        intent=os.getenv("SUPERVISOR_ESCALATION_INTENT", "speak_to_person"),
        delay_seconds=delay,
        timeout_seconds=timeout,
    )
