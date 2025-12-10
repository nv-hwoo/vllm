#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example script demonstrating the RDMA sidecar with vLLM.

This script shows how to:
1. Start vLLM with the RDMA sidecar extension
2. Run inference while the sidecar is active
3. Access model weight information

For the full benchmark with RDMA transfer, use rdma_benchmark.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def run_example(
    model: str = "facebook/opt-125m",
    gpu_memory_utilization: float = 0.5,
    max_model_len: int = 512,
    export_metadata: bool = True,
) -> None:
    """
    Run the RDMA sidecar example.

    Args:
        model: Model name or path
        gpu_memory_utilization: GPU memory utilization fraction
        max_model_len: Maximum model context length
        export_metadata: Whether to export weight metadata to file
    """
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM with model: {model}")
    print("Loading with RDMA sidecar extension...")

    # Initialize vLLM with the RDMA sidecar extension
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        worker_extension_cls="rdma_sidecar_extension.RDMASidecarExtension",
        enforce_eager=True,
    )

    print("\n" + "=" * 60)
    print("vLLM initialized successfully!")
    print("=" * 60)

    # Get model weight metadata
    print("\nFetching model weight metadata...")
    metadata_list = llm.collective_rpc("get_model_weight_metadata", args=tuple())

    if metadata_list and len(metadata_list) > 0:
        metadata = metadata_list[0]  # Get from first worker
        total_size = sum(m["nbytes"] for m in metadata)

        print(f"\nModel Weight Summary:")
        print("-" * 60)
        print(f"Total weight tensors: {len(metadata)}")
        print(f"Total weight size:    {total_size / (1024**3):.2f} GB")
        print()

        # Group by location
        location_counts: dict[str, tuple[int, int]] = {}
        for m in metadata:
            loc = m["location"]
            if loc not in location_counts:
                location_counts[loc] = (0, 0)
            count, size = location_counts[loc]
            location_counts[loc] = (count + 1, size + m["nbytes"])

        print("Weights by location:")
        for loc, (count, size) in sorted(location_counts.items()):
            print(f"  {loc:<12}: {count:>4} tensors, {size / (1024**2):>8.2f} MB")

        # Export metadata if requested
        if export_metadata:
            metadata_file = "/tmp/rdma_weight_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"\nMetadata exported to: {metadata_file}")

    # Start the weight server for sidecar connections
    zmq_address = "ipc:///tmp/rdma-sidecar.sock"
    print(f"\nStarting weight server at: {zmq_address}")
    llm.collective_rpc("start_weight_server", args=(zmq_address,))

    print("\n" + "=" * 60)
    print("Weight server is running!")
    print("=" * 60)
    print()
    print("To connect a sidecar, run in another terminal:")
    print()
    print(f"  python rdma_sidecar.py \\")
    print(f"      --zmq-address {zmq_address} \\")
    print(f"      --master-addr <REMOTE_IP> \\")
    print(f"      --master-port 29500")
    print()
    print("On the remote node, run:")
    print()
    print(f"  python rdma_remote_receiver.py \\")
    print(f"      --master-addr <THIS_NODE_IP> \\")
    print(f"      --master-port 29500 \\")
    print(f"      --metadata-file /tmp/rdma_weight_metadata.json")
    print()
    print("=" * 60)

    # Run some inference to demonstrate the model is working
    print("\nRunning sample inference while weight server is active...")

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The best programming language is",
    ]

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.8,
        top_p=0.95,
    )

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        generated = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)

        print(f"Output: {generated.strip()}")
        print(f"Generated {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.1f} tok/s)")

    print("\n" + "=" * 60)
    print("Example complete!")
    print()
    print("The weight server is still running. Press Ctrl+C to exit.")
    print("=" * 60)

    # Keep running for sidecar connections
    try:
        while True:
            # Process any sidecar requests
            # In a real scenario, you'd have a more sophisticated event loop
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        llm.collective_rpc("stop_weight_server", args=tuple())
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="RDMA Sidecar Example with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name or path (default: facebook/opt-125m)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization (default: 0.5)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model context length (default: 512)",
    )
    parser.add_argument(
        "--no-export-metadata",
        action="store_true",
        help="Don't export weight metadata to file",
    )

    args = parser.parse_args()

    run_example(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        export_metadata=not args.no_export_metadata,
    )


if __name__ == "__main__":
    main()

