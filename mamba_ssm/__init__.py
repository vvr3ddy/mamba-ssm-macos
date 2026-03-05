from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mamba-ssm-macos")
except PackageNotFoundError:
    __version__ = "0.0.0"

import platform
import sys

if sys.platform == "darwin" and platform.machine() == "arm64":
    print("Mamba SSM macOS: Running on Apple Silicon with MPS acceleration")

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.utils.macos import (
    create_tokenizer,
    generate_text_with_model,
    get_device,
    get_optimal_dtype,
    load_and_prepare_model,
)
from mamba_ssm.utils.profiling import (
    mps_memory_tracker,
    get_mps_memory_info,
    profile_forward_pass,
    benchmark_sequence_lengths,
    mps_profiler,
)
from mamba_ssm.utils.torch import (
    autocast_context,
    custom_bwd,
    custom_fwd,
    get_amp_dtype,
    get_autocast_device_type,
)
from mamba_ssm.utils.profiling import (
    mps_memory_tracker,
    get_mps_memory_info,
    profile_forward_pass,
    benchmark_sequence_lengths,
    mps_profiler,
)

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "MambaLMHeadModel",
    "InferenceParams",
    "get_device",
    "get_optimal_dtype",
    "create_tokenizer",
    "generate_text_with_model",
    "load_and_prepare_model",
    # Torch utilities
    "autocast_context",
    "custom_bwd",
    "custom_fwd",
    "get_amp_dtype",
    "get_autocast_device_type",
    # Profiling utilities
    "mps_memory_tracker",
    "get_mps_memory_info",
    "profile_forward_pass",
    "benchmark_sequence_lengths",
    "mps_profiler",
]
