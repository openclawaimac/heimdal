"""Model backends for Heimdal.

The runtime is backend-agnostic: it talks to a :class:`ModelBackend`. Ollama is
the initial real backend; the offline backend keeps the full pipeline runnable
(demo, run, eval) on machines without a model server.
"""

from heimdal.models.base import GenerationResult, ModelBackend
from heimdal.models.offline import OfflineBackend
from heimdal.models.ollama import OllamaBackend
from heimdal.models.base import select_backend

__all__ = [
    "GenerationResult",
    "ModelBackend",
    "OfflineBackend",
    "OllamaBackend",
    "select_backend",
]
