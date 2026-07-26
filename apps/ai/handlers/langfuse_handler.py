from __future__ import annotations

import os
import time
from typing import Any

from utils.logger import logger, redact_sensitive

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - langfuse is an optional dependency
    Langfuse = None


def langfuse_enabled() -> bool:
    """Langfuse tracing only turns on when both API keys are configured.

    This keeps the integration fully optional: a developer who has not set
    up Langfuse yet, or a CI/test environment without secrets, must never
    have a voice call fail or slow down because of this handler.
    """
    return bool(
        Langfuse is not None
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def build_langfuse_client() -> "Langfuse | None":
    if not langfuse_enabled():
        return None
    try:
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("[langfuse] client init failed: {}", redact_sensitive(str(error)))
        return None


class LangfuseCallTracer:
    """Sends one Langfuse trace per voice call, with one generation per
    user -> agent turn, so a call's full conversation and per-turn timing
    is visible in the Langfuse dashboard.

    Mirrors the structure of TranscriptCollector: `.attach(session)` wires
    into the same AgentSession events, and every method is defensive so a
    Langfuse outage or misconfiguration can never break a live call.
    """

    def __init__(
        self,
        *,
        call_context: dict[str, Any],
        config: dict[str, Any],
        client: "Langfuse | None" = None,
    ) -> None:
        self._call_context = call_context or {}
        self._config = config or {}
        self._client = client if client is not None else build_langfuse_client()
        self._pending_user_text: str | None = None
        self._pending_user_started_at: float | None = None
        self._trace = None

        if self._client is not None:
            trace_id = str(
                self._call_context.get("callId")
                or self._call_context.get("roomName")
                or f"call-{int(time.time())}"
            )
            try:
                self._trace = self._client.trace(
                    id=trace_id,
                    name="voice-call",
                    metadata={
                        "agent_id": self._call_context.get("agent_id"),
                        "direction": self._call_context.get("direction"),
                        "llm_model": self._config.get("llm_model"),
                        "stt_model": self._config.get("stt_model"),
                        "tts_model": self._config.get("tts_model"),
                    },
                )
            except Exception as error:  # pragma: no cover - defensive
                logger.warning(
                    "[langfuse] failed to start trace: {}", redact_sensitive(str(error))
                )
                self._trace = None

    @property
    def enabled(self) -> bool:
        return self._trace is not None

    def attach(self, session: Any) -> "LangfuseCallTracer":
        if self.enabled:
            session.on("conversation_item_added", self.on_conversation_item_added)
        return self

    def on_conversation_item_added(self, event: Any) -> None:
        if not self.enabled:
            return
        try:
            item = getattr(event, "item", None)
            role = getattr(item, "role", None)
            if role == "assistant":
                role = "agent"
            if role not in ("user", "agent"):
                return

            content = getattr(item, "text_content", None)
            if callable(content):
                content = content()
            if content is None:
                content = getattr(item, "content", "")
            text = str(content or "").strip()
            if not text:
                return

            if role == "user":
                self._pending_user_text = text
                self._pending_user_started_at = time.time()
                return

            # role == "agent": close out the turn as one generation, so
            # each row in Langfuse is a full user -> agent exchange.
            started_at = self._pending_user_started_at or time.time()
            self._trace.generation(
                name="voice-turn",
                model=self._config.get("llm_model", "unknown"),
                input=self._pending_user_text or "",
                output=text,
                start_time=None,  # left to Langfuse SDK default if not tracked
                metadata={"latency_seconds": round(time.time() - started_at, 3)},
            )
            self._pending_user_text = None
            self._pending_user_started_at = None
        except Exception as error:
            # Tracing must never break the live call.
            logger.warning(
                "[langfuse] failed to record turn: {}", redact_sensitive(str(error))
            )

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("[langfuse] flush failed: {}", redact_sensitive(str(error)))
