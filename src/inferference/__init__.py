"""Shared boundary types and runtime facts for the llama.cpp teaching core."""

__version__ = "0.4.0"

from .benchmark import BenchmarkRecord, BenchmarkTimer, write_benchmark_record
from .capabilities import (
    AdapterInfo,
    Capabilities,
    CapabilityError,
    ConstraintKind,
    ModelInfo,
)
from .client import LLAMA_SERVER_CONSTRAINTS, InferferenceClient
from .config_paths import ConfigError, config_dir, resolve_router_url
from .diagnostics import RuntimeDiagnostics, collect_runtime_diagnostics
from .fingerprint import (
    ArtifactIdentity,
    LlamaEngineFingerprint,
    file_identity,
    register_artifact,
)
from .grammar import GrammarCacheKey, GrammarCompilerCache, JsonSchemaGrammar
from .models import EngineConfig, GenerationRequest, GenerationResult
from .seq_adapter_coverage import SeqRoutedAdapterRefused
from .seq_routing import (
    LOOP_ROUTING_SYMBOLS,
    NO_ADAPTER,
    NO_LOOP_ADAPTER,
    LoopRoutingBinding,
    LoopRoutingUnavailable,
    SeqRoutingBinding,
    SeqRoutingUnavailable,
    library_supports_loop_routing,
    library_supports_seq_routing,
)
from .tied_head_adapter import TiedHeadAdapterRefused

# NOTE: the router (``router.py``) and owned decoder (``decode.py``) are heavy,
# native-backed modules; like ``decode``, ``router`` is imported directly by
# callers (``from ...llama_core.router import MultiLoRARouter``) so this shared
# import path stays lightweight (no numpy/llama_cpp import on ``import llama_core``).

__all__ = [
    "AdapterInfo",
    "ArtifactIdentity",
    "BenchmarkRecord",
    "BenchmarkTimer",
    "Capabilities",
    "CapabilityError",
    "ConfigError",
    "ConstraintKind",
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "GrammarCacheKey",
    "GrammarCompilerCache",
    "InferferenceClient",
    "JsonSchemaGrammar",
    "LLAMA_SERVER_CONSTRAINTS",
    "LlamaEngineFingerprint",
    "LOOP_ROUTING_SYMBOLS",
    "LoopRoutingBinding",
    "LoopRoutingUnavailable",
    "ModelInfo",
    "NO_ADAPTER",
    "NO_LOOP_ADAPTER",
    "RuntimeDiagnostics",
    "SeqRoutedAdapterRefused",
    "SeqRoutingBinding",
    "SeqRoutingUnavailable",
    "TiedHeadAdapterRefused",
    "collect_runtime_diagnostics",
    "config_dir",
    "file_identity",
    "library_supports_loop_routing",
    "library_supports_seq_routing",
    "register_artifact",
    "resolve_router_url",
    "write_benchmark_record",
]
