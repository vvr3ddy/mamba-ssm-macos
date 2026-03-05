# Copyright (c) 2023, Albert Gu, Tri Dao.

"""
MPS Memory Profiling Utilities for Apple Silicon.

This module provides profiling utilities specifically designed for the MPS
(Metal Performance Shaders) backend on Apple Silicon devices.
"""

import time
import statistics
from contextlib import contextmanager
from typing import Optional, List

import torch


@contextmanager
def mps_memory_tracker(label: str = ""):
    """
    Context manager to track MPS memory usage.
    
    Usage:
        with mps_memory_tracker("forward_pass"):
            output = model(input)
    
    Args:
        label: Description of the operation being tracked
        
    Yields:
        None - this is a context manager for tracking memory
    """
    if not torch.backends.mps.is_available():
        yield
        return
    
    # Clear MPS cache and get baseline
    torch.mps.empty_cache()
    start_allocated = torch.mps.current_allocated_memory()
    start_driver = torch.mps.driver_allocated_memory()
    
    yield
    
    # Get final memory stats
    end_allocated = torch.mps.current_allocated_memory()
    end_driver = torch.mps.driver_allocated_memory()
    peak_driver = torch.mps.driver_allocated_memory()
    
    # Calculate deltas
    delta_allocated = end_allocated - start_allocated
    delta_driver = end_driver - start_driver
    
    print(f"[{label}] MPS Memory: "
          f"Δallocated={delta_allocated / 1e6:.1f}MB, "
          f"Δdriver={delta_driver / 1e6:.1f}MB, "
          f"peak_driver={peak_driver / 1e9:.2f}GB")


def get_mps_memory_info() -> dict:
    """
    Get current MPS memory statistics.
    
    Returns:
        Dictionary with memory information:
        - current_allocated: Current tensor allocation in bytes
        - driver_allocated: Total Metal memory in bytes
        - is_available: Whether MPS is available
    """
    if not torch.backends.mps.is_available():
        return {
            "is_available": False,
            "current_allocated": 0,
            "driver_allocated": 0,
        }
    
    return {
        "is_available": True,
        "current_allocated": torch.mps.current_allocated_memory(),
        "driver_allocated": torch.mps.driver_allocated_memory(),
    }


def profile_forward_pass(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: str = "mps",
    n_warmup: int = 3,
    n_runs: int = 10,
    use_gradient: bool = False,
) -> List[float]:
    """
    Benchmark model forward pass with timing and memory.
    
    Args:
        model: The model to benchmark
        input_ids: Input tensor
        device: Device to run on ("mps" or "cpu")
        n_warmup: Number of warmup runs before measurement
        n_runs: Number of benchmark runs
        use_gradient: Whether to compute gradients (for training benchmark)
        
    Returns:
        List of timing measurements in seconds
    """
    model = model.to(device)
    input_ids = input_ids.to(device)
    model.eval()
    
    # Warmup runs
    for _ in range(n_warmup):
        if use_gradient:
            output = model(input_ids)
            output.sum().backward()
        else:
            with torch.no_grad():
                model(input_ids)
    
    # Synchronize to ensure all operations complete
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark runs
    times: List[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        
        if use_gradient:
            model.zero_grad()
            output = model(input_ids)
            output.sum().backward()
        else:
            with torch.no_grad():
                model(input_ids)
        
        # Synchronize for accurate timing
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # Print results
    mean_time = statistics.mean(times) * 1000
    std_time = statistics.stdev(times) * 1000 if len(times) > 1 else 0
    min_time = min(times) * 1000
    max_time = max(times) * 1000
    
    print(f"Forward pass ({device}): "
          f"mean={mean_time:.1f}ms ± {std_time:.1f}ms "
          f"(min={min_time:.1f}ms, max={max_time:.1f}ms)")
    
    # Memory info
    if device == "mps":
        mem_info = get_mps_memory_info()
        print(f"MPS Memory: allocated={mem_info['current_allocated'] / 1e6:.1f}MB, "
              f"driver={mem_info['driver_allocated'] / 1e9:.2f}GB")
    
    return times


def benchmark_sequence_lengths(
    model: torch.nn.Module,
    batch_size: int,
    d_model: int,
    sequence_lengths: List[int],
    device: str = "mps",
    n_runs: int = 5,
) -> dict:
    """
    Benchmark model performance across different sequence lengths.
    
    Args:
        model: The model to benchmark
        batch_size: Batch size for inputs
        d_model: Model dimension
        sequence_lengths: List of sequence lengths to test
        device: Device to run on
        n_runs: Number of runs per sequence length
        
    Returns:
        Dictionary mapping sequence length to timing results
    """
    results = {}
    
    for seq_len in sequence_lengths:
        # Create input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        # Run benchmark
        times = profile_forward_pass(
            model, input_ids, device=device, n_runs=n_runs, use_gradient=False
        )
        
        results[seq_len] = {
            "mean_ms": statistics.mean(times) * 1000,
            "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }
    
    return results


@contextmanager
def mps_profiler(
    wait_steps: int = 1,
    warmup_steps: int = 1,
    active_steps: int = 3,
    repeat: int = 1,
):
    """
    Context manager for MPS profiling using PyTorch profiler.
    
    Note: This is a simplified version. For full profiling, use
    torch.profiler directly with appropriate configuration.
    
    Args:
        wait_steps: Number of steps to skip at the beginning
        warmup_steps: Number of warmup steps
        active_steps: Number of steps to profile
        repeat: Number of times to repeat the profiling cycle
        
    Usage:
        with mps_profiler() as prof:
            for step in range(100):
                train_step()
                prof.step()
        
        # Print results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping profiler")
        yield None
        return
    
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MPS,
            ],
            wait_steps=wait_steps,
            warmup_steps=warmup_steps,
            active_steps=active_steps,
            repeat=repeat,
            with_stack=True,
        ) as prof:
            yield prof
    except Exception as e:
        print(f"MPS profiler not available: {e}")
        yield None
