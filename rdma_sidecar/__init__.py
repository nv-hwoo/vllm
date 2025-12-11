# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RDMA Sidecar for vLLM Model Weight Transfer.

This package provides utilities for:
1. Exposing vLLM model weights via CUDA IPC
2. Transferring weights to remote nodes using RDMA/NCCL
3. Benchmarking inference performance impact during transfers
"""

from rdma_sidecar.rdma_sidecar import (
    RDMASidecar,
    TransferBackend,
    TransferSession,
    TransferStats,
    run_sidecar_sender,
)
from rdma_sidecar.rdma_sidecar_extension import (
    RDMASidecarExtension,
    TensorLocation,
    TensorMetadata,
)

__all__ = [
    "RDMASidecar",
    "RDMASidecarExtension",
    "TensorLocation",
    "TensorMetadata",
    "TransferBackend",
    "TransferSession",
    "TransferStats",
    "run_sidecar_sender",
]

