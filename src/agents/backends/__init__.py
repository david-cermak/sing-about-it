"""Agent backends for different LLM interaction approaches."""

from .pydantic_backend import PydanticAIBackend
from .baremetal_backend import BaremetalBackend

__all__ = ['PydanticAIBackend', 'BaremetalBackend']
