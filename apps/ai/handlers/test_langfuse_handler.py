import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from handlers.langfuse_handler import LangfuseCallTracer


class FakeSession:
    def __init__(self):
        self.handlers = {}

    def on(self, event, callback):
        self.handlers[event] = callback


def make_item(role, text, item_id):
    return SimpleNamespace(id=item_id, role=role, text_content=text, content=text)


class LangfuseCallTracerTests(unittest.TestCase):
    def test_disabled_when_no_client_is_available(self):
        # No LANGFUSE_* env vars set and no client injected -> tracer is a
        # safe no-op, so a call still runs fine without Langfuse configured.
        tracer = LangfuseCallTracer(call_context={}, config={}, client=None)
        session = FakeSession()
        tracer.attach(session)

        self.assertFalse(tracer.enabled)
        self.assertNotIn("conversation_item_added", session.handlers)
        tracer.flush()  # must not raise

    def test_records_one_generation_per_user_agent_turn(self):
        fake_trace = MagicMock()
        fake_client = MagicMock()
        fake_client.trace.return_value = fake_trace

        tracer = LangfuseCallTracer(
            call_context={"callId": "call-123", "agent_id": "agent-1"},
            config={"llm_model": "google/gemini-2.5-flash"},
            client=fake_client,
        )
        session = FakeSession()
        tracer.attach(session)

        self.assertTrue(tracer.enabled)
        fake_client.trace.assert_called_once()
        self.assertEqual(fake_client.trace.call_args.kwargs["id"], "call-123")

        session.handlers["conversation_item_added"](
            SimpleNamespace(item=make_item("user", "What are your hours?", "u1"))
        )
        session.handlers["conversation_item_added"](
            SimpleNamespace(item=make_item("assistant", "We're open 9 to 5.", "a1"))
        )

        fake_trace.generation.assert_called_once()
        call_kwargs = fake_trace.generation.call_args.kwargs
        self.assertEqual(call_kwargs["input"], "What are your hours?")
        self.assertEqual(call_kwargs["output"], "We're open 9 to 5.")
        self.assertEqual(call_kwargs["model"], "google/gemini-2.5-flash")

        tracer.flush()
        fake_client.flush.assert_called_once()

    def test_a_client_error_during_a_turn_never_raises(self):
        fake_trace = MagicMock()
        fake_trace.generation.side_effect = RuntimeError("network down")
        fake_client = MagicMock()
        fake_client.trace.return_value = fake_trace

        tracer = LangfuseCallTracer(call_context={}, config={}, client=fake_client)
        session = FakeSession()
        tracer.attach(session)

        session.handlers["conversation_item_added"](
            SimpleNamespace(item=make_item("user", "Hello", "u1"))
        )
        # Must not raise even though generation() will fail below.
        session.handlers["conversation_item_added"](
            SimpleNamespace(item=make_item("assistant", "Hi there", "a1"))
        )


if __name__ == "__main__":
    unittest.main()
