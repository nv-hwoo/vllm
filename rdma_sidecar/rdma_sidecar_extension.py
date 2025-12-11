# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Worker Extension for RDMA Sidecar.

This extension exposes model weight tensors via CUDA IPC handles,
allowing a sidecar process to access and transfer them using RDMA.

Usage:
    Pass `worker_extension_cls="rdma_sidecar.rdma_sidecar_extension.RDMASidecarExtension"`
    when initializing vLLM.

Auto-start weight server:
    Set RDMA_ZMQ_ADDRESS env var to auto-start the weight server on model load.
    Example: RDMA_ZMQ_ADDRESS=ipc:///tmp/rdma.sock
"""
import json
import os
import threading
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import torch
import zmq
from pydantic import BaseModel
from torch.multiprocessing.reductions import reduce_tensor


class TensorLocation(StrEnum):
    """Location of tensor in model hierarchy."""
    EMBEDDING = auto()
    ATTENTION = auto()
    MLP = auto()
    NORM = auto()
    LM_HEAD = auto()
    OTHER = auto()


class TensorMetadata(BaseModel):
    """Metadata for a model weight tensor."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    location: TensorLocation
    layer_idx: int | None = None

    class Config:
        frozen = True


@dataclass
class WeightInfo:
    """Complete information about a model weight."""
    metadata: TensorMetadata
    data_ptr: int  # GPU memory address
    ipc_handle: tuple  # CUDA IPC handle from reduce_tensor


def classify_tensor_location(name: str) -> tuple[TensorLocation, int | None]:
    """Classify a tensor's location in the model architecture."""
    name_lower = name.lower()
    layer_idx = None

    # Try to extract layer index
    parts = name.split(".")
    for part in parts:
        if part.isdigit():
            layer_idx = int(part)
            break
        if part.startswith("layers") or part.startswith("layer"):
            # Try next part
            continue

    if "embed" in name_lower or "wte" in name_lower or "wpe" in name_lower:
        return TensorLocation.EMBEDDING, layer_idx
    if "attn" in name_lower or "attention" in name_lower or "self_attn" in name_lower:
        return TensorLocation.ATTENTION, layer_idx
    if "mlp" in name_lower or "ffn" in name_lower or "feed_forward" in name_lower:
        return TensorLocation.MLP, layer_idx
    if "norm" in name_lower or "ln_" in name_lower or "layernorm" in name_lower:
        return TensorLocation.NORM, layer_idx
    if "lm_head" in name_lower or "output" in name_lower:
        return TensorLocation.LM_HEAD, layer_idx

    return TensorLocation.OTHER, layer_idx


class RDMASidecarExtension:
    """
    vLLM Worker Extension that exposes model weights for RDMA transfer.

    This class is dynamically loaded into vLLM workers when specified via
    the `worker_extension_cls` parameter. It provides methods to:
    1. Report device information
    2. Export CUDA IPC handles for all model weight tensors
    3. Serve weight data to a sidecar process via ZMQ

    NOTE: This class is designed to be mixed into a vLLM Worker class,
    so it assumes access to `self.model_runner`, `self.device`, etc.

    NOTE: Since this is a mixin injected via __bases__, __init__ is never
    called. All attributes use lazy initialization via getattr().
    """

    def get_device_uuid(self) -> str:
        """Get the UUID of the GPU device this worker is using."""
        device_uuid = getattr(self, "_device_uuid", None)
        if device_uuid is None:
            from vllm.platforms import current_platform
            device_uuid = current_platform.get_device_uuid(self.device.index)
            self._device_uuid = device_uuid
        return device_uuid

    def get_model_weight_metadata(self) -> list[dict[str, Any]]:
        """
        Get metadata for all model weight tensors.

        Returns:
            List of dictionaries containing tensor metadata.
        """
        self._ensure_weight_infos()
        return [
            info.metadata.model_dump()
            for info in self._weight_infos.values()
        ]

    def get_total_weight_size(self) -> int:
        """Get total size of all model weights in bytes."""
        self._ensure_weight_infos()
        return sum(info.metadata.nbytes for info in self._weight_infos.values())

    def _ensure_weight_infos(self) -> None:
        """Lazily build weight info cache and auto-start weight server if configured."""
        if getattr(self, "_weight_infos", None) is not None:
            return

        self._weight_infos: dict[str, WeightInfo] = {}
        model = self.model_runner.model

        for name, param in model.named_parameters():
            if not param.is_cuda:
                continue

            # Ensure tensor is contiguous for IPC
            tensor = param.data.contiguous()
            location, layer_idx = classify_tensor_location(name)

            metadata = TensorMetadata(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                nbytes=tensor.numel() * tensor.element_size(),
                location=location,
                layer_idx=layer_idx,
            )

            # Get CUDA IPC handle
            ipc_handle = reduce_tensor(tensor)

            self._weight_infos[name] = WeightInfo(
                metadata=metadata,
                data_ptr=tensor.data_ptr(),
                ipc_handle=ipc_handle,
            )

        # Auto-start weight server if RDMA_ZMQ_ADDRESS is set
        zmq_address = os.environ.get("RDMA_ZMQ_ADDRESS")
        if zmq_address and getattr(self, "_server_thread", None) is None:
            # Ensure socket directory exists
            if zmq_address.startswith("ipc://"):
                Path(zmq_address[6:]).parent.mkdir(parents=True, exist_ok=True)
            
            # Export metadata to file for receiver
            metadata_path = os.environ.get("RDMA_METADATA_PATH", "/tmp/rdma_weight_metadata.json")
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            metadata_list = [info.metadata.model_dump() for info in self._weight_infos.values()]
            Path(metadata_path).write_text(json.dumps(metadata_list))
            
            self.start_weight_server(zmq_address)
            total_gb = sum(info.metadata.nbytes for info in self._weight_infos.values()) / 1e9
            print(f"[RDMA] Weight server auto-started: {zmq_address}")
            print(f"[RDMA] {len(self._weight_infos)} tensors, {total_gb:.2f} GB total")
            print(f"[RDMA] Metadata exported to: {metadata_path}")

    def start_weight_server(self, zmq_address: str) -> None:
        """
        Start a ZMQ server to serve weight tensor IPC handles to sidecar.

        Args:
            zmq_address: ZMQ IPC address (e.g., "ipc:///tmp/rdma-sidecar.sock")
        """
        # Guard against starting twice
        if getattr(self, "_server_thread", None) is not None:
            return

        if getattr(self, "_zmq_context", None) is None:
            self._zmq_context = zmq.Context()

        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        self._zmq_socket.bind(zmq_address)

        # Create stop event BEFORE starting thread
        self._stop_event = threading.Event()

        # Background thread that listens
        self._server_thread = threading.Thread(
            target=self._serve_loop,
            daemon=True,
            name="zmq-weight-server",
        )
        self._server_thread.start()

    def _serve_loop(self) -> None:
        """Background loop to serve ZMQ requests."""
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

        while not self._stop_event.is_set():
            try:
                done = self.serve_weights_once()
                if done:
                    break
            except zmq.Again:
                # Timeout - check stop flag and continue
                continue
            except zmq.ZMQError:
                # Socket closed, exit loop
                break

    def serve_weights_once(self) -> bool:
        """
        Handle one request from the sidecar process.

        Protocol:
            - "list": Return list of all weight names
            - "metadata": Return all weight metadata
            - "get:<name>": Return IPC handle for specific weight
            - "get_all": Return all IPC handles
            - "done": Signal completion, return True to stop

        Returns:
            True if "done" received, False otherwise.
        """
        zmq_socket = getattr(self, "_zmq_socket", None)
        if zmq_socket is None:
            raise RuntimeError("Weight server not started. Call start_weight_server first.")
        self._zmq_socket = zmq_socket  # For type checker

        self._ensure_weight_infos()

        request = self._zmq_socket.recv_string()

        if request == "list":
            self._zmq_socket.send_pyobj(list(self._weight_infos.keys()))
            return False

        if request == "metadata":
            metadata_list = [
                info.metadata.model_dump()
                for info in self._weight_infos.values()
            ]
            self._zmq_socket.send_pyobj(metadata_list)
            return False

        if request.startswith("get:"):
            name = request[4:]
            if name not in self._weight_infos:
                self._zmq_socket.send_pyobj({"error": f"Weight '{name}' not found"})
                return False

            info = self._weight_infos[name]
            self._zmq_socket.send_pyobj({
                "metadata": info.metadata.model_dump(),
                "ipc_handle": info.ipc_handle,
            })
            return False

        if request == "get_all":
            # Send all IPC handles in batches to avoid memory issues
            all_handles = {
                name: {
                    "metadata": info.metadata.model_dump(),
                    "ipc_handle": info.ipc_handle,
                }
                for name, info in self._weight_infos.items()
            }
            self._zmq_socket.send_pyobj(all_handles)
            return False

        if request == "done":
            self._zmq_socket.send_string("ok")
            self._stop_event.set()
            return True

        self._zmq_socket.send_pyobj({"error": f"Unknown request: {request}"})
        return False

    def stop_weight_server(self) -> None:
        """Stop the ZMQ weight server and background thread."""
        # 1. Signal thread to stop
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        # 2. Wait for thread to finish
        server_thread = getattr(self, "_server_thread", None)
        if server_thread is not None:
            server_thread.join(timeout=2.0)
            self._server_thread = None

        # 3. Close socket
        zmq_socket = getattr(self, "_zmq_socket", None)
        if zmq_socket is not None:
            zmq_socket.close()
            self._zmq_socket = None

        # 4. Terminate context
        zmq_context = getattr(self, "_zmq_context", None)
        if zmq_context is not None:
            zmq_context.term()
            self._zmq_context = None

    def get_weight_ipc_handles_for_sidecar(self) -> dict[str, tuple]:
        """
        Get all weight IPC handles for direct access by sidecar.

        This is an alternative to ZMQ - call this via collective_rpc
        and the handles can be used to reconstruct tensors in another process.

        Returns:
            Dictionary mapping weight names to (metadata_dict, ipc_handle) tuples.
        """
        self._ensure_weight_infos()
        return {
            name: (info.metadata.model_dump(), info.ipc_handle)
            for name, info in self._weight_infos.items()
        }

