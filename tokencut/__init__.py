"""Local, inspectable optimizations for agent context."""

from .core import BudgetExceeded, OptimizationResult, optimize
from .model import Chunk, ContextPacket
from .request import RequestResult, optimize_request
from .tokens import TokenCounter

__version__ = "0.2.1"
__all__ = [
    "BudgetExceeded",
    "Chunk",
    "ContextPacket",
    "OptimizationResult",
    "RequestResult",
    "TokenCounter",
    "optimize",
    "optimize_request",
]
