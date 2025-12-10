# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Remote RDMA Receiver for vLLM Model Weights.

This process runs on a remote node and receives model weights
transferred via RDMA from the sidecar sender.

Usage:
    python rdma_remote_receiver.py --master-addr <ip> --master-port <port>
"""

import argparse
import gc
import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rdma_receiver")


@dataclass
class ReceivedTensor:
    """Information about a received tensor."""
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    receive_time: float


@dataclass
class ReceiveSession:
    """Statistics for a receive session."""
    tensors: list[ReceivedTensor] = field(default_factory=list)
    total_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def bandwidth_gbps(self) -> float:
        if self.duration > 0:
            return (self.total_bytes * 8) / (self.duration * 1e9)
        return 0.0


class RDMAReceiver:
    """
    Remote receiver for RDMA model weight transfers.

    This class runs on the destination node and receives weights
    transferred from the sidecar sender.
    """

    def __init__(
        self,
        local_device_id: int = 0,
        store_received: bool = False,
    ):
        self.local_device_id = local_device_id
        self.store_received = store_received
        self.device = torch.device(f"cuda:{local_device_id}")
        torch.cuda.set_device(self.device)

        self._received_weights: dict[str, torch.Tensor] = {}

        logger.info(f"Receiver initialized on device {self.device}")

    def init_process_group(
        self,
        master_addr: str,
        master_port: int,
        rank: int = 1,
        world_size: int = 2,
        backend: str = "nccl",
    ) -> None:
        """
        Initialize distributed process group.

        Args:
            master_addr: IP address of the master (sender) node
            master_port: Port for rendezvous
            rank: This process's rank (1 for receiver)
            world_size: Total number of processes
            backend: Communication backend (nccl for GPU)
        """
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
            logger.info(
                f"Initialized {backend} process group: rank={rank}, world_size={world_size}"
            )

    def receive_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        name: str = "",
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Receive a single tensor from the sender.

        Args:
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            name: Tensor name (for logging)
            src_rank: Source rank

        Returns:
            Received tensor
        """
        # Pre-allocate buffer
        tensor = torch.empty(shape, dtype=dtype, device=self.device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        dist.recv(tensor, src=src_rank)

        torch.cuda.synchronize()
        end = time.perf_counter()

        size_bytes = tensor.numel() * tensor.element_size()
        bandwidth = (size_bytes * 8) / ((end - start) * 1e9)

        logger.debug(
            f"Received {name}: shape={shape}, dtype={dtype}, "
            f"{size_bytes / (1024**2):.2f} MB, {bandwidth:.2f} Gbps"
        )

        if self.store_received:
            self._received_weights[name] = tensor

        return tensor

    def receive_weights_from_metadata(
        self,
        metadata_list: list[dict],
        src_rank: int = 0,
    ) -> ReceiveSession:
        """
        Receive all weights based on metadata from sender.

        The metadata should be obtained from the sender beforehand
        (e.g., via ZMQ or shared config).

        Args:
            metadata_list: List of tensor metadata dicts
            src_rank: Source rank

        Returns:
            Receive session statistics
        """
        session = ReceiveSession()
        session.start_time = time.perf_counter()

        # Sort by size (must match sender order)
        sorted_metadata = sorted(
            metadata_list,
            key=lambda x: x["nbytes"],
            reverse=True,
        )

        logger.info(f"Expecting {len(sorted_metadata)} tensors")

        for meta in sorted_metadata:
            name = meta["name"]
            shape = tuple(meta["shape"])
            dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
            size_bytes = meta["nbytes"]

            tensor = self.receive_tensor(
                shape=shape,
                dtype=dtype,
                name=name,
                src_rank=src_rank,
            )

            received = ReceivedTensor(
                name=name,
                shape=shape,
                dtype=dtype,
                size_bytes=size_bytes,
                receive_time=time.perf_counter(),
            )
            session.tensors.append(received)
            session.total_bytes += size_bytes

        session.end_time = time.perf_counter()

        logger.info(
            f"Receive complete: {len(session.tensors)} tensors, "
            f"{session.total_bytes / (1024**3):.2f} GB, "
            f"{session.duration:.2f}s, "
            f"{session.bandwidth_gbps:.2f} Gbps"
        )

        return session

    def receive_weights_dynamic(
        self,
        src_rank: int = 0,
        timeout_seconds: float = 300,
    ) -> ReceiveSession:
        """
        Dynamically receive weights with metadata sent inline.

        Protocol:
        1. Receive number of tensors (int64)
        2. For each tensor:
           a. Receive metadata (name_len, name, shape_len, shape, dtype, nbytes)
           b. Receive tensor data

        Args:
            src_rank: Source rank
            timeout_seconds: Timeout for entire operation

        Returns:
            Receive session statistics
        """
        session = ReceiveSession()
        session.start_time = time.perf_counter()

        # Receive number of tensors
        num_tensors_t = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(num_tensors_t, src=src_rank)
        num_tensors = int(num_tensors_t.item())

        logger.info(f"Expecting {num_tensors} tensors")

        for i in range(num_tensors):
            if time.perf_counter() - session.start_time > timeout_seconds:
                raise TimeoutError("Receive operation timed out")

            # Receive tensor metadata
            meta_size = torch.zeros(1, dtype=torch.int64, device=self.device)
            dist.recv(meta_size, src=src_rank)

            meta_buffer = torch.zeros(
                int(meta_size.item()), dtype=torch.uint8, device=self.device
            )
            dist.recv(meta_buffer, src=src_rank)

            # Decode metadata (simple: nbytes as int64, then shape dims, then dtype code)
            meta_cpu = meta_buffer.cpu().numpy().tobytes()
            # This is a simplified protocol - in production, use proper serialization

            # For now, just receive the tensor with pre-known metadata
            # (This is a placeholder - the actual metadata decoding depends on protocol)

        session.end_time = time.perf_counter()
        return session

    def get_received_weights(self) -> dict[str, torch.Tensor]:
        """Get all received weight tensors (if store_received=True)."""
        return self._received_weights

    def clear_received(self) -> None:
        """Clear stored received weights to free GPU memory."""
        self._received_weights.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Shutdown receiver and clean up."""
        self.clear_received()

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed")


def run_receiver(
    master_addr: str,
    master_port: int,
    metadata_file: str | None = None,
    local_device_id: int = 0,
    store_received: bool = False,
) -> ReceiveSession:
    """
    Run the receiver process.

    Args:
        master_addr: Master address for NCCL
        master_port: Master port for NCCL
        metadata_file: Optional file containing tensor metadata JSON
        local_device_id: Local GPU device ID
        store_received: Whether to store received tensors

    Returns:
        Receive session statistics
    """
    receiver = RDMAReceiver(
        local_device_id=local_device_id,
        store_received=store_received,
    )

    try:
        receiver.init_process_group(
            master_addr=master_addr,
            master_port=master_port,
            rank=1,
            world_size=2,
        )

        if metadata_file:
            import json
            with open(metadata_file) as f:
                metadata_list = json.load(f)
            session = receiver.receive_weights_from_metadata(
                metadata_list=metadata_list,
                src_rank=0,
            )
        else:
            # If no metadata provided, expect sender to send it first
            logger.warning(
                "No metadata file provided. Using dynamic receive "
                "(requires sender to send metadata first)."
            )
            session = receiver.receive_weights_dynamic(src_rank=0)

        return session

    finally:
        receiver.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Remote RDMA Receiver for vLLM Model Weights"
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        required=True,
        help="Master address (sender's IP) for NCCL rendezvous",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for NCCL rendezvous",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="JSON file containing tensor metadata",
    )
    parser.add_argument(
        "--local-device-id",
        type=int,
        default=0,
        help="Local GPU device ID",
    )
    parser.add_argument(
        "--store-received",
        action="store_true",
        help="Store received tensors in GPU memory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    session = run_receiver(
        master_addr=args.master_addr,
        master_port=args.master_port,
        metadata_file=args.metadata_file,
        local_device_id=args.local_device_id,
        store_received=args.store_received,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Receive Summary")
    print("=" * 60)
    print(f"Total tensors:      {len(session.tensors)}")
    print(f"Total size:         {session.total_bytes / (1024**3):.2f} GB")
    print(f"Total duration:     {session.duration:.2f} s")
    print(f"Aggregate bandwidth: {session.bandwidth_gbps:.2f} Gbps")
    print("=" * 60)


if __name__ == "__main__":
    main()

