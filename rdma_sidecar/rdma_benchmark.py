# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to measure vLLM inference impact during RDMA transfers.

This script:
1. Starts a vLLM engine with the RDMA sidecar extension
2. Runs baseline inference measurements
3. Initiates RDMA transfers while inference is running
4. Compares inference performance metrics

Usage:
    python rdma_benchmark.py --model <model_name> --remote-host <ip>
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rdma_benchmark")


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkPhase:
    """Metrics for a benchmark phase."""
    name: str
    duration_seconds: float
    metrics: list[InferenceMetrics] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.time_to_first_token_ms for m in self.metrics)

    @property
    def p99_ttft_ms(self) -> float:
        if not self.metrics:
            return 0.0
        sorted_ttft = sorted(m.time_to_first_token_ms for m in self.metrics)
        idx = int(len(sorted_ttft) * 0.99)
        return sorted_ttft[min(idx, len(sorted_ttft) - 1)]

    @property
    def avg_itl_ms(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.inter_token_latency_ms for m in self.metrics)

    @property
    def avg_tps(self) -> float:
        if not self.metrics:
            return 0.0
        return statistics.mean(m.tokens_per_second for m in self.metrics)

    @property
    def num_requests(self) -> int:
        return len(self.metrics)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    model: str
    baseline: BenchmarkPhase
    during_transfer: BenchmarkPhase
    after_transfer: BenchmarkPhase
    transfer_stats: dict = field(default_factory=dict)

    def ttft_impact_percent(self) -> float:
        """Calculate TTFT impact as percentage increase."""
        if self.baseline.avg_ttft_ms == 0:
            return 0.0
        return ((self.during_transfer.avg_ttft_ms - self.baseline.avg_ttft_ms)
                / self.baseline.avg_ttft_ms * 100)

    def itl_impact_percent(self) -> float:
        """Calculate ITL impact as percentage increase."""
        if self.baseline.avg_itl_ms == 0:
            return 0.0
        return ((self.during_transfer.avg_itl_ms - self.baseline.avg_itl_ms)
                / self.baseline.avg_itl_ms * 100)

    def tps_impact_percent(self) -> float:
        """Calculate TPS impact as percentage decrease."""
        if self.baseline.avg_tps == 0:
            return 0.0
        return ((self.baseline.avg_tps - self.during_transfer.avg_tps)
                / self.baseline.avg_tps * 100)


class InferenceBenchmarker:
    """Runs inference benchmarks on vLLM."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int | None = None,
        zmq_address: str = "ipc:///tmp/rdma-sidecar.sock",
    ):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.zmq_address = zmq_address
        self.llm = None

    def initialize(self) -> None:
        """Initialize vLLM with RDMA sidecar extension."""
        from vllm import LLM

        logger.info(f"Initializing vLLM with model {self.model}")

        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            worker_extension_cls=(
                "rdma_sidecar.rdma_sidecar_extension.RDMASidecarExtension"
            ),
            enforce_eager=True,  # Disable CUDA graphs for easier debugging
        )

        logger.info("vLLM initialized successfully")

    def start_weight_server(self) -> None:
        """Start the weight server on all workers."""
        if self.llm is None:
            raise RuntimeError("LLM not initialized")

        self.llm.collective_rpc(
            "start_weight_server",
            args=(self.zmq_address,),
        )
        logger.info(f"Weight server started at {self.zmq_address}")

    def stop_weight_server(self) -> None:
        """Stop the weight server on all workers."""
        if self.llm is not None:
            self.llm.collective_rpc("stop_weight_server", args=tuple())
            logger.info("Weight server stopped")

    def get_weight_metadata(self) -> list[dict]:
        """Get metadata for all model weights."""
        if self.llm is None:
            raise RuntimeError("LLM not initialized")

        metadata_list = self.llm.collective_rpc(
            "get_model_weight_metadata",
            args=tuple(),
        )
        # Return metadata from first worker (all should be same for TP)
        return metadata_list[0] if metadata_list else []

    def run_single_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> InferenceMetrics:
        """Run a single inference and measure metrics."""
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("LLM not initialized")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for benchmarking
        )

        start_time = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        end_time = time.perf_counter()

        output = outputs[0]
        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        total_latency_ms = (end_time - start_time) * 1000

        # Approximate TTFT and ITL (vLLM doesn't expose per-token timing directly)
        # For more accurate measurements, use the streaming API
        ttft_ms = total_latency_ms / (output_tokens + 1)  # Rough approximation
        itl_ms = total_latency_ms / max(output_tokens, 1)
        tps = output_tokens / (total_latency_ms / 1000)

        return InferenceMetrics(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            time_to_first_token_ms=ttft_ms,
            inter_token_latency_ms=itl_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tps,
        )

    def run_benchmark_phase(
        self,
        phase_name: str,
        prompts: list[str],
        max_tokens: int = 100,
        duration_seconds: float = 30.0,
    ) -> BenchmarkPhase:
        """Run a benchmark phase for a specified duration."""
        phase = BenchmarkPhase(name=phase_name, duration_seconds=duration_seconds)
        start_time = time.time()
        prompt_idx = 0

        logger.info(f"Starting benchmark phase: {phase_name} ({duration_seconds}s)")

        while time.time() - start_time < duration_seconds:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            try:
                metrics = self.run_single_inference(prompt, max_tokens)
                phase.metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Inference error: {e}")

        logger.info(
            f"Phase {phase_name} complete: {phase.num_requests} requests, "
            f"avg TTFT: {phase.avg_ttft_ms:.2f}ms, avg TPS: {phase.avg_tps:.2f}"
        )

        return phase


def run_rdma_transfer_process(
    zmq_address: str,
    master_addr: str,
    master_port: int,
    local_device_id: int,
    chunk_size_mb: int,
    delay_ms: float,
    result_queue: mp.Queue,
) -> None:
    """Run RDMA transfer in a separate process."""
    from rdma_sidecar import run_sidecar_sender

    try:
        session = run_sidecar_sender(
            zmq_address=zmq_address,
            master_addr=master_addr,
            master_port=master_port,
            local_device_id=local_device_id,
            chunk_size_mb=chunk_size_mb,
            delay_ms=delay_ms,
        )
        result_queue.put({
            "success": True,
            "total_bytes": session.total_bytes,
            "duration": session.duration,
            "bandwidth_gbps": session.aggregate_bandwidth_gbps,
            "total_tensors": session.total_tensors,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
        })


def run_benchmark(
    model: str,
    remote_host: str,
    remote_port: int = 29500,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int | None = None,
    baseline_duration: float = 30.0,
    transfer_duration: float = 60.0,
    post_transfer_duration: float = 30.0,
    chunk_size_mb: int = 256,
    delay_ms: float = 0,
    prompts: list[str] | None = None,
    max_tokens: int = 100,
    output_file: str | None = None,
) -> BenchmarkResult:
    """
    Run the complete RDMA impact benchmark.

    Args:
        model: Model name or path
        remote_host: Remote host for RDMA transfer
        remote_port: Port for NCCL rendezvous
        tensor_parallel_size: Tensor parallelism degree
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model context length
        baseline_duration: Duration for baseline measurement
        transfer_duration: Duration for measurement during transfer
        post_transfer_duration: Duration for post-transfer measurement
        chunk_size_mb: RDMA transfer chunk size
        delay_ms: Delay between transfer chunks
        prompts: List of prompts for inference
        max_tokens: Max tokens to generate per request
        output_file: File to save results

    Returns:
        BenchmarkResult with all measurements
    """
    zmq_address = "ipc:///tmp/rdma-sidecar.sock"

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Write a short story about a robot learning to paint.",
            "Explain the theory of relativity in simple terms.",
            "What are the key differences between Python and JavaScript?",
            "Describe the process of photosynthesis step by step.",
            "Write a poem about the ocean at sunset.",
        ]

    benchmarker = InferenceBenchmarker(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        zmq_address=zmq_address,
    )

    try:
        # Initialize vLLM
        benchmarker.initialize()
        benchmarker.start_weight_server()

        # Save metadata for receiver
        metadata = benchmarker.get_weight_metadata()
        metadata_file = "/tmp/rdma_weight_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        logger.info(f"Saved weight metadata to {metadata_file}")

        # Phase 1: Baseline
        baseline = benchmarker.run_benchmark_phase(
            "baseline",
            prompts=prompts,
            max_tokens=max_tokens,
            duration_seconds=baseline_duration,
        )

        # Phase 2: Start RDMA transfer and measure
        logger.info("Starting RDMA transfer in background...")

        result_queue = mp.Queue()
        transfer_process = mp.Process(
            target=run_rdma_transfer_process,
            args=(
                zmq_address,
                remote_host,
                remote_port,
                0,  # local_device_id
                chunk_size_mb,
                delay_ms,
                result_queue,
            ),
        )
        transfer_process.start()

        # Run inference during transfer
        during_transfer = benchmarker.run_benchmark_phase(
            "during_transfer",
            prompts=prompts,
            max_tokens=max_tokens,
            duration_seconds=transfer_duration,
        )

        # Wait for transfer to complete
        transfer_process.join(timeout=120)
        if transfer_process.is_alive():
            transfer_process.terminate()
            transfer_stats = {"error": "Transfer process timed out"}
        else:
            transfer_stats = result_queue.get_nowait() if not result_queue.empty() else {}

        # Phase 3: Post-transfer
        after_transfer = benchmarker.run_benchmark_phase(
            "after_transfer",
            prompts=prompts,
            max_tokens=max_tokens,
            duration_seconds=post_transfer_duration,
        )

        result = BenchmarkResult(
            model=model,
            baseline=baseline,
            during_transfer=during_transfer,
            after_transfer=after_transfer,
            transfer_stats=transfer_stats,
        )

        # Print summary
        print_benchmark_summary(result)

        # Save results
        if output_file:
            save_benchmark_results(result, output_file)

        return result

    finally:
        benchmarker.stop_weight_server()


def print_benchmark_summary(result: BenchmarkResult) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 70)
    print("RDMA TRANSFER IMPACT BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {result.model}")
    print()

    print("Phase Comparison:")
    print("-" * 70)
    print(f"{'Phase':<20} {'Requests':<10} {'Avg TTFT':<12} {'Avg ITL':<12} {'Avg TPS':<10}")
    print("-" * 70)

    for phase in [result.baseline, result.during_transfer, result.after_transfer]:
        print(
            f"{phase.name:<20} "
            f"{phase.num_requests:<10} "
            f"{phase.avg_ttft_ms:<12.2f} "
            f"{phase.avg_itl_ms:<12.2f} "
            f"{phase.avg_tps:<10.2f}"
        )

    print("-" * 70)
    print()

    print("Impact Analysis (during transfer vs baseline):")
    print("-" * 70)
    print(f"  TTFT increase:    {result.ttft_impact_percent():+.2f}%")
    print(f"  ITL increase:     {result.itl_impact_percent():+.2f}%")
    print(f"  TPS decrease:     {result.tps_impact_percent():+.2f}%")
    print()

    if result.transfer_stats:
        print("Transfer Statistics:")
        print("-" * 70)
        if result.transfer_stats.get("success"):
            print(f"  Total bytes:      {result.transfer_stats['total_bytes'] / (1024**3):.2f} GB")
            print(f"  Duration:         {result.transfer_stats['duration']:.2f} s")
            print(f"  Bandwidth:        {result.transfer_stats['bandwidth_gbps']:.2f} Gbps")
        else:
            print(f"  Error: {result.transfer_stats.get('error', 'Unknown error')}")

    print("=" * 70)


def save_benchmark_results(result: BenchmarkResult, output_file: str) -> None:
    """Save benchmark results to JSON file."""
    output = {
        "model": result.model,
        "baseline": {
            "name": result.baseline.name,
            "duration_seconds": result.baseline.duration_seconds,
            "num_requests": result.baseline.num_requests,
            "avg_ttft_ms": result.baseline.avg_ttft_ms,
            "p99_ttft_ms": result.baseline.p99_ttft_ms,
            "avg_itl_ms": result.baseline.avg_itl_ms,
            "avg_tps": result.baseline.avg_tps,
        },
        "during_transfer": {
            "name": result.during_transfer.name,
            "duration_seconds": result.during_transfer.duration_seconds,
            "num_requests": result.during_transfer.num_requests,
            "avg_ttft_ms": result.during_transfer.avg_ttft_ms,
            "p99_ttft_ms": result.during_transfer.p99_ttft_ms,
            "avg_itl_ms": result.during_transfer.avg_itl_ms,
            "avg_tps": result.during_transfer.avg_tps,
        },
        "after_transfer": {
            "name": result.after_transfer.name,
            "duration_seconds": result.after_transfer.duration_seconds,
            "num_requests": result.after_transfer.num_requests,
            "avg_ttft_ms": result.after_transfer.avg_ttft_ms,
            "p99_ttft_ms": result.after_transfer.p99_ttft_ms,
            "avg_itl_ms": result.after_transfer.avg_itl_ms,
            "avg_tps": result.after_transfer.avg_tps,
        },
        "impact": {
            "ttft_increase_percent": result.ttft_impact_percent(),
            "itl_increase_percent": result.itl_impact_percent(),
            "tps_decrease_percent": result.tps_impact_percent(),
        },
        "transfer_stats": result.transfer_stats,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM inference impact during RDMA transfers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name or path",
    )
    parser.add_argument(
        "--remote-host",
        type=str,
        required=True,
        help="Remote host IP for RDMA transfer",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=29500,
        help="Remote port for NCCL rendezvous",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--baseline-duration",
        type=float,
        default=30.0,
        help="Duration for baseline measurement (seconds)",
    )
    parser.add_argument(
        "--transfer-duration",
        type=float,
        default=60.0,
        help="Duration for measurement during transfer (seconds)",
    )
    parser.add_argument(
        "--post-transfer-duration",
        type=float,
        default=30.0,
        help="Duration for post-transfer measurement (seconds)",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=256,
        help="RDMA transfer chunk size in MB",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0,
        help="Delay between transfer chunks in milliseconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save benchmark results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_benchmark(
        model=args.model,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        baseline_duration=args.baseline_duration,
        transfer_duration=args.transfer_duration,
        post_transfer_duration=args.post_transfer_duration,
        chunk_size_mb=args.chunk_size_mb,
        delay_ms=args.delay_ms,
        max_tokens=args.max_tokens,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

