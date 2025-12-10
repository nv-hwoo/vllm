# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RDMA Sidecar Process for vLLM Model Weight Transfer.

This sidecar process:
1. Connects to vLLM workers via CUDA IPC to access model weights on GPU
2. Initiates RDMA/NCCL transfers to remote nodes
3. Measures transfer performance and impact on inference

Usage:
    python rdma_sidecar.py --local-rank 0 --remote-host <ip> --remote-port <port>
"""

import argparse
import gc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

import torch
import torch.distributed as dist
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rdma_sidecar")


class TransferBackend(StrEnum):
    """Backend for RDMA transfer."""
    NCCL = auto()  # Use NCCL with GPUDirect RDMA
    GLOO = auto()  # Use Gloo (CPU, for testing)


@dataclass
class TransferStats:
    """Statistics for a single transfer operation."""
    tensor_name: str
    size_bytes: int
    start_time: float
    end_time: float
    bandwidth_gbps: float = field(init=False)

    def __post_init__(self):
        duration = self.end_time - self.start_time
        if duration > 0:
            self.bandwidth_gbps = (self.size_bytes * 8) / (duration * 1e9)
        else:
            self.bandwidth_gbps = 0.0


@dataclass
class TransferSession:
    """Aggregated statistics for a transfer session."""
    total_bytes: int = 0
    total_tensors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    tensor_stats: list[TransferStats] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def aggregate_bandwidth_gbps(self) -> float:
        if self.duration > 0:
            return (self.total_bytes * 8) / (self.duration * 1e9)
        return 0.0


def rebuild_ipc_tensor(
    ipc_handle: tuple[Callable, tuple],
    device_id: int | None = None,
) -> torch.Tensor:
    """
    Rebuild a GPU tensor from its CUDA IPC handle.

    Args:
        ipc_handle: Tuple of (rebuild_function, args) from reduce_tensor
        device_id: Override device ID (for multi-GPU with different CUDA_VISIBLE_DEVICES)

    Returns:
        Reconstructed GPU tensor
    """
    func, args = ipc_handle
    list_args = list(args)
    if device_id is not None:
        # Override device ID (position 6 in the args tuple)
        list_args[6] = device_id
    return func(*list_args)


class RDMASidecar:
    """
    Sidecar process that accesses vLLM model weights and transfers via RDMA.

    This class runs alongside a vLLM engine process and can:
    1. Access model weights on GPU via CUDA IPC
    2. Transfer weights to remote nodes using NCCL with GPUDirect RDMA
    3. Track and report transfer statistics
    """

    def __init__(
        self,
        local_device_id: int = 0,
        backend: TransferBackend = TransferBackend.NCCL,
    ):
        self.local_device_id = local_device_id
        self.backend = backend
        self.device = torch.device(f"cuda:{local_device_id}")
        torch.cuda.set_device(self.device)

        self._zmq_context: zmq.Context | None = None
        self._zmq_socket: zmq.Socket | None = None
        self._weight_handles: dict[str, dict[str, Any]] = {}
        self._reconstructed_tensors: dict[str, torch.Tensor] = {}
        self._transfer_group: dist.ProcessGroup | None = None

        logger.info(f"Sidecar initialized on device {self.device}")

    def connect_to_worker(self, zmq_address: str, timeout_ms: int = 30000) -> None:
        """
        Connect to vLLM worker's weight server via ZMQ.

        Args:
            zmq_address: ZMQ IPC address of the worker's weight server
            timeout_ms: Connection timeout in milliseconds
        """
        if self._zmq_context is None:
            self._zmq_context = zmq.Context()

        self._zmq_socket = self._zmq_context.socket(zmq.REQ)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._zmq_socket.connect(zmq_address)
        logger.info(f"Connected to worker at {zmq_address}")

    def fetch_weight_handles(self) -> dict[str, dict[str, Any]]:
        """
        Fetch all weight IPC handles from the connected worker.

        Returns:
            Dictionary mapping weight names to metadata and IPC handles.
        """
        if self._zmq_socket is None:
            raise RuntimeError("Not connected to worker. Call connect_to_worker first.")

        self._zmq_socket.send_string("get_all")
        self._weight_handles = self._zmq_socket.recv_pyobj()

        total_size = sum(
            h["metadata"]["nbytes"] for h in self._weight_handles.values()
        )
        logger.info(
            f"Fetched {len(self._weight_handles)} weight handles, "
            f"total size: {total_size / (1024**3):.2f} GB"
        )
        return self._weight_handles

    def reconstruct_tensor(self, name: str) -> torch.Tensor:
        """
        Reconstruct a GPU tensor from its IPC handle.

        Args:
            name: Name of the weight tensor

        Returns:
            Reconstructed GPU tensor
        """
        if name in self._reconstructed_tensors:
            return self._reconstructed_tensors[name]

        if name not in self._weight_handles:
            raise KeyError(f"Weight '{name}' not found in handles")

        handle_info = self._weight_handles[name]
        tensor = rebuild_ipc_tensor(
            handle_info["ipc_handle"],
            device_id=self.local_device_id,
        )

        self._reconstructed_tensors[name] = tensor
        return tensor

    def reconstruct_all_tensors(self) -> dict[str, torch.Tensor]:
        """
        Reconstruct all weight tensors from IPC handles.

        Returns:
            Dictionary mapping weight names to GPU tensors.
        """
        for name in self._weight_handles:
            self.reconstruct_tensor(name)
        logger.info(f"Reconstructed {len(self._reconstructed_tensors)} tensors")
        return self._reconstructed_tensors

    def init_transfer_group(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
    ) -> None:
        """
        Initialize distributed process group for RDMA transfer.

        Args:
            master_addr: IP address of the master node
            master_port: Port for rendezvous
            rank: This process's rank (0 = sender, 1 = receiver typically)
            world_size: Total number of processes
        """
        import os

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        backend = "nccl" if self.backend == TransferBackend.NCCL else "gloo"

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
            logger.info(
                f"Initialized {backend} process group: rank={rank}, world_size={world_size}"
            )
        else:
            logger.info("Process group already initialized")

        self._transfer_group = dist.group.WORLD

    def transfer_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        dst_rank: int = 1,
    ) -> TransferStats:
        """
        Transfer a single tensor to a remote rank via RDMA.

        Args:
            tensor: GPU tensor to transfer
            name: Name of the tensor (for logging)
            dst_rank: Destination rank

        Returns:
            Transfer statistics
        """
        if self._transfer_group is None:
            raise RuntimeError("Transfer group not initialized")

        # Ensure tensor is contiguous
        tensor = tensor.contiguous()

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        dist.send(tensor, dst=dst_rank, group=self._transfer_group)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        return TransferStats(
            tensor_name=name,
            size_bytes=tensor.numel() * tensor.element_size(),
            start_time=start_time,
            end_time=end_time,
        )

    def transfer_all_weights(
        self,
        dst_rank: int = 1,
        layer_filter: Callable[[str], bool] | None = None,
    ) -> TransferSession:
        """
        Transfer all model weights to a remote rank.

        Args:
            dst_rank: Destination rank
            layer_filter: Optional filter function for tensor names

        Returns:
            Transfer session statistics
        """
        session = TransferSession()
        session.start_time = time.perf_counter()

        # Get all tensors to transfer
        tensors_to_transfer = []
        for name, handle_info in self._weight_handles.items():
            if layer_filter is not None and not layer_filter(name):
                continue
            tensors_to_transfer.append((name, handle_info))

        # Sort by size for better scheduling (largest first)
        tensors_to_transfer.sort(
            key=lambda x: x[1]["metadata"]["nbytes"],
            reverse=True,
        )

        logger.info(f"Starting transfer of {len(tensors_to_transfer)} tensors")

        for name, handle_info in tensors_to_transfer:
            tensor = self.reconstruct_tensor(name)
            stats = self.transfer_tensor(tensor, name, dst_rank)
            session.tensor_stats.append(stats)
            session.total_bytes += stats.size_bytes
            session.total_tensors += 1

            logger.debug(
                f"Transferred {name}: {stats.size_bytes / (1024**2):.2f} MB, "
                f"{stats.bandwidth_gbps:.2f} Gbps"
            )

        session.end_time = time.perf_counter()

        logger.info(
            f"Transfer complete: {session.total_tensors} tensors, "
            f"{session.total_bytes / (1024**3):.2f} GB, "
            f"{session.duration:.2f}s, "
            f"{session.aggregate_bandwidth_gbps:.2f} Gbps aggregate"
        )

        return session

    def transfer_weights_chunked(
        self,
        dst_rank: int = 1,
        chunk_size_mb: int = 256,
        delay_between_chunks_ms: float = 0,
    ) -> TransferSession:
        """
        Transfer weights in chunks with optional delays.

        This method allows testing different transfer patterns to
        measure impact on inference performance.

        Args:
            dst_rank: Destination rank
            chunk_size_mb: Size of each transfer chunk in MB
            delay_between_chunks_ms: Delay between chunks in milliseconds

        Returns:
            Transfer session statistics
        """
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        session = TransferSession()
        session.start_time = time.perf_counter()

        # Collect all tensor data
        all_data = []
        for name, handle_info in self._weight_handles.items():
            tensor = self.reconstruct_tensor(name)
            all_data.append((name, tensor, handle_info["metadata"]["nbytes"]))

        # Sort by size
        all_data.sort(key=lambda x: x[2], reverse=True)

        current_chunk = []
        current_chunk_size = 0

        for name, tensor, size in all_data:
            if current_chunk_size + size > chunk_size_bytes and current_chunk:
                # Transfer current chunk
                self._transfer_chunk(current_chunk, dst_rank, session)

                if delay_between_chunks_ms > 0:
                    time.sleep(delay_between_chunks_ms / 1000)

                current_chunk = []
                current_chunk_size = 0

            current_chunk.append((name, tensor, size))
            current_chunk_size += size

        # Transfer remaining chunk
        if current_chunk:
            self._transfer_chunk(current_chunk, dst_rank, session)

        session.end_time = time.perf_counter()

        logger.info(
            f"Chunked transfer complete: {session.total_tensors} tensors, "
            f"{session.total_bytes / (1024**3):.2f} GB, "
            f"{session.duration:.2f}s"
        )

        return session

    def _transfer_chunk(
        self,
        chunk: list[tuple[str, torch.Tensor, int]],
        dst_rank: int,
        session: TransferSession,
    ) -> None:
        """Transfer a chunk of tensors."""
        for name, tensor, size in chunk:
            stats = self.transfer_tensor(tensor, name, dst_rank)
            session.tensor_stats.append(stats)
            session.total_bytes += stats.size_bytes
            session.total_tensors += 1

    def disconnect(self) -> None:
        """Clean up connections and resources."""
        # Close ZMQ
        if self._zmq_socket is not None:
            try:
                self._zmq_socket.send_string("done")
                self._zmq_socket.recv_string()
            except zmq.ZMQError:
                pass
            self._zmq_socket.close()
            self._zmq_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        # Clear tensors
        self._reconstructed_tensors.clear()
        self._weight_handles.clear()

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Sidecar disconnected and cleaned up")

    def shutdown(self) -> None:
        """Shutdown sidecar and distributed group."""
        self.disconnect()

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed")


def run_sidecar_sender(
    zmq_address: str,
    master_addr: str,
    master_port: int,
    local_device_id: int = 0,
    chunk_size_mb: int = 256,
    delay_ms: float = 0,
) -> TransferSession:
    """
    Run the sidecar as a sender (rank 0).

    Args:
        zmq_address: ZMQ address to connect to vLLM worker
        master_addr: Master address for NCCL
        master_port: Master port for NCCL
        local_device_id: Local GPU device ID
        chunk_size_mb: Transfer chunk size
        delay_ms: Delay between chunks

    Returns:
        Transfer session statistics
    """
    sidecar = RDMASidecar(local_device_id=local_device_id)

    try:
        # Connect and fetch handles
        sidecar.connect_to_worker(zmq_address)
        sidecar.fetch_weight_handles()
        sidecar.reconstruct_all_tensors()

        # Initialize transfer group
        sidecar.init_transfer_group(
            master_addr=master_addr,
            master_port=master_port,
            rank=0,
            world_size=2,
        )

        # Perform transfer
        if delay_ms > 0 or chunk_size_mb < 1024:
            session = sidecar.transfer_weights_chunked(
                dst_rank=1,
                chunk_size_mb=chunk_size_mb,
                delay_between_chunks_ms=delay_ms,
            )
        else:
            session = sidecar.transfer_all_weights(dst_rank=1)

        return session

    finally:
        sidecar.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="RDMA Sidecar for vLLM Model Weight Transfer"
    )
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="ipc:///tmp/rdma-sidecar.sock",
        help="ZMQ address to connect to vLLM worker",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Master address for NCCL rendezvous",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for NCCL rendezvous",
    )
    parser.add_argument(
        "--local-device-id",
        type=int,
        default=0,
        help="Local GPU device ID",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=256,
        help="Transfer chunk size in MB",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0,
        help="Delay between chunk transfers in milliseconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    session = run_sidecar_sender(
        zmq_address=args.zmq_address,
        master_addr=args.master_addr,
        master_port=args.master_port,
        local_device_id=args.local_device_id,
        chunk_size_mb=args.chunk_size_mb,
        delay_ms=args.delay_ms,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Transfer Summary")
    print("=" * 60)
    print(f"Total tensors:      {session.total_tensors}")
    print(f"Total size:         {session.total_bytes / (1024**3):.2f} GB")
    print(f"Total duration:     {session.duration:.2f} s")
    print(f"Aggregate bandwidth: {session.aggregate_bandwidth_gbps:.2f} Gbps")
    print("=" * 60)


if __name__ == "__main__":
    main()

