# RDMA Sidecar for vLLM Model Weight Transfer

This module provides a sidecar process that can access vLLM's model weights on GPU
and transfer them to remote nodes using RDMA (via NCCL with GPUDirect RDMA support).

## Purpose

The primary use case is to test whether RDMA transfers of model weights impact
vLLM's inference serving performance. This is useful for:

- **Model migration**: Moving models between nodes without stopping inference
- **Checkpointing**: Saving model state to remote storage during serving
- **Distributed training-inference co-location**: Sharing weights between training and inference processes

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Local Node                                     │
│                                                                          │
│  ┌─────────────────────┐       ┌─────────────────────────────────────┐ │
│  │    vLLM Engine      │       │        RDMA Sidecar                 │ │
│  │                     │       │                                     │ │
│  │  ┌──────────────┐   │  ZMQ  │  ┌──────────────────────────────┐  │ │
│  │  │ Worker +     │◄──┼──────►┼──│ Receives IPC handles         │  │ │
│  │  │ Extension    │   │       │  │ Reconstructs GPU tensors     │  │ │
│  │  └──────┬───────┘   │       │  │ Initiates NCCL transfers     │  │ │
│  │         │           │       │  └──────────────┬───────────────┘  │ │
│  │         │           │       │                 │                   │ │
│  │  ┌──────▼───────┐   │       │                 │ NCCL/RDMA         │ │
│  │  │ GPU Memory   │◄──┼───────┼─────────────────┘                   │ │
│  │  │ (Weights)    │   │       │                                     │ │
│  │  └──────────────┘   │       │                                     │ │
│  └─────────────────────┘       └─────────────────────────────────────┘ │
│                                              │                          │
└──────────────────────────────────────────────┼──────────────────────────┘
                                               │
                                          RDMA Network
                                               │
┌──────────────────────────────────────────────┼──────────────────────────┐
│                           Remote Node        │                          │
│                                              │                          │
│  ┌───────────────────────────────────────────▼───────────────────────┐ │
│  │                    RDMA Receiver                                   │ │
│  │                                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │ Receives tensors via NCCL                                    │ │ │
│  │  │ Stores in GPU memory                                         │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. `rdma_sidecar_extension.py`

A vLLM worker extension that exposes model weights via CUDA IPC handles.

```python
# Used automatically when specified in vLLM config
llm = LLM(
    model="facebook/opt-125m",
    worker_extension_cls="rdma_sidecar.rdma_sidecar_extension.RDMASidecarExtension",
)
```

### 2. `rdma_sidecar.py`

The main sidecar process that:
- Connects to vLLM workers via ZMQ
- Reconstructs GPU tensors from IPC handles
- Transfers weights to remote nodes using NCCL

```bash
# Run on the sender node
python rdma_sidecar.py \
    --zmq-address ipc:///tmp/rdma-sidecar.sock \
    --master-addr <remote_ip> \
    --master-port 29500 \
    --chunk-size-mb 256
```

### 3. `rdma_remote_receiver.py`

The receiver process running on the remote node.

```bash
# Run on the receiver node
python rdma_remote_receiver.py \
    --master-addr <sender_ip> \
    --master-port 29500 \
    --metadata-file /tmp/rdma_weight_metadata.json
```

### 4. `rdma_benchmark.py`

Benchmarking script that measures inference performance impact during RDMA transfers.

```bash
python rdma_benchmark.py \
    --model facebook/opt-125m \
    --remote-host <remote_ip> \
    --baseline-duration 30 \
    --transfer-duration 60 \
    --output-file results.json
```

## Requirements

### Hardware
- NVIDIA GPUs with CUDA support
- InfiniBand or RoCE network for RDMA
- GPUDirect RDMA support (MLNX_OFED driver)

### Software
- PyTorch with NCCL support
- vLLM
- PyZMQ

### System Configuration

For GPUDirect RDMA to work, ensure:

```bash
# Check GPUDirect RDMA kernel module
lsmod | grep nvidia_peermem

# If not loaded
modprobe nvidia_peermem
```

Docker configuration:
```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    --cap-add=IPC_LOCK \
    vllm/vllm-openai
```

## Usage Example

### Single-Node Test (Simulated)

```python
import multiprocessing as mp
from vllm import LLM

# Start vLLM with extension
llm = LLM(
    model="facebook/opt-125m",
    worker_extension_cls="rdma_sidecar.rdma_sidecar_extension.RDMASidecarExtension",
)

# Get weight metadata
metadata = llm.collective_rpc("get_model_weight_metadata", args=tuple())
print(f"Model has {len(metadata[0])} weight tensors")

# Start weight server
llm.collective_rpc("start_weight_server", args=("ipc:///tmp/rdma-sidecar.sock",))

# In another process, run the sidecar
# python rdma_sidecar.py --zmq-address ipc:///tmp/rdma-sidecar.sock ...
```

### Multi-Node Transfer

**Node 1 (Sender with vLLM):**
```bash
# Terminal 1: Start vLLM
python -c "
from vllm import LLM
llm = LLM(
    model='meta-llama/Llama-2-7b-hf',
    worker_extension_cls='rdma_sidecar.rdma_sidecar_extension.RDMASidecarExtension',
)
llm.collective_rpc('start_weight_server', args=('ipc:///tmp/rdma.sock',))
# Keep running for inference...
"

# Terminal 2: Start sidecar
python rdma_sidecar.py \
    --zmq-address ipc:///tmp/rdma.sock \
    --master-addr 10.0.0.1 \
    --master-port 29500
```

**Node 2 (Receiver):**
```bash
# Copy metadata file from Node 1 first
python rdma_remote_receiver.py \
    --master-addr 10.0.0.1 \
    --master-port 29500 \
    --metadata-file /tmp/rdma_weight_metadata.json
```

### Running the Benchmark

```bash
# On sender node (with vLLM)
python rdma_benchmark.py \
    --model meta-llama/Llama-2-7b-hf \
    --remote-host 10.0.0.2 \
    --remote-port 29500 \
    --baseline-duration 60 \
    --transfer-duration 120 \
    --post-transfer-duration 60 \
    --chunk-size-mb 512 \
    --output-file benchmark_results.json

# On receiver node (start before sender)
python rdma_remote_receiver.py \
    --master-addr 10.0.0.1 \
    --master-port 29500 \
    --metadata-file weight_metadata.json
```

## Benchmark Output

The benchmark produces output like:

```
======================================================================
RDMA TRANSFER IMPACT BENCHMARK RESULTS
======================================================================
Model: meta-llama/Llama-2-7b-hf

Phase Comparison:
----------------------------------------------------------------------
Phase                Requests   Avg TTFT     Avg ITL      Avg TPS   
----------------------------------------------------------------------
baseline             150        45.23        12.34        81.05     
during_transfer      120        52.18        14.67        68.12     
after_transfer       145        45.89        12.45        80.23     
----------------------------------------------------------------------

Impact Analysis (during transfer vs baseline):
----------------------------------------------------------------------
  TTFT increase:    +15.37%
  ITL increase:     +18.88%
  TPS decrease:     +15.94%

Transfer Statistics:
----------------------------------------------------------------------
  Total bytes:      13.50 GB
  Duration:         45.23 s
  Bandwidth:        2.39 Gbps
======================================================================
```

## Tuning Parameters

### Transfer Chunk Size (`--chunk-size-mb`)
- Smaller chunks: More granular control, potentially less interference
- Larger chunks: Better bandwidth efficiency, potentially more interference

### Delay Between Chunks (`--delay-ms`)
- Adding delay allows inference to "catch up" between transfer batches
- Useful for testing gradual transfer scenarios

## Troubleshooting

### NCCL Timeout
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### IPC Handle Issues
Ensure both processes can access the same GPU:
```bash
# Check CUDA_VISIBLE_DEVICES is set correctly
echo $CUDA_VISIBLE_DEVICES
```

### Permission Issues
```bash
# Ensure IPC_LOCK capability
docker run --cap-add=IPC_LOCK ...
```

## Limitations

1. **Same-GPU Only**: CUDA IPC requires both processes to have access to the same physical GPU
2. **Contiguous Tensors**: Only contiguous tensors can be shared via IPC
3. **NCCL Streams**: RDMA transfers use NCCL which may compete with inference CUDA streams

## Future Improvements

- [ ] Support for UCX backend (more flexible RDMA control)
- [ ] Per-layer transfer scheduling to minimize interference
- [ ] Integration with vLLM's async streaming API for accurate TTFT measurement
- [ ] Support for quantized model transfers

