"""Optional LLM-backed features (schema understanding, content synthesis, conversational editing). All have offline fallbacks."""

from sdp.ai.llm import LLMError, get_client, set_client

__all__ = ["LLMError", "get_client", "set_client"]
