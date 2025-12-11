# RDMA Sidecar K8s Experiment

Run RDMA weight transfer experiment on Kubernetes with two InfiniBand-connected GPU nodes.

## Prerequisites

- 2 GPU nodes with InfiniBand connectivity
- NVIDIA device plugin installed
- GPUDirect RDMA enabled (`modprobe nvidia_peermem` on each node)

## Deploy

```bash
./deploy.sh <sender-node> <receiver-node> [model]

# Example
./deploy.sh gpu-node-1 gpu-node-2 facebook/opt-125m
```

## Monitor

```bash
kubectl get pods -n rdma-exp -w                    # Pod status
kubectl logs -n rdma-exp sender -c vllm -f         # vLLM OpenAI server
kubectl logs -n rdma-exp sender -c sidecar -f      # RDMA sidecar (starts weight server via collective_rpc)
kubectl logs -n rdma-exp receiver -f               # RDMA receiver
```

You should see in sidecar logs:
```
vLLM ready!
Starting weight server via collective_rpc...
start_weight_server response: 200
Exported metadata: X tensors
```

## Send Inference Requests

```bash
# Port-forward to access vLLM API
kubectl port-forward -n rdma-exp svc/vllm 8000:8000

# List models
curl http://localhost:8000/v1/models

# Completion request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The meaning of life is","max_tokens":50}'

# Chat request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

## Run Your Benchmark

With port-forward active, point your benchmark tool at `http://localhost:8000`.

The RDMA transfer will happen automatically ~30s after the receiver pod starts.

## Cleanup

```bash
kubectl delete ns rdma-exp
kubectl label node <sender-node> rdma-
kubectl label node <receiver-node> rdma-
```

## Architecture

```
┌───────────────────────────────┐     RDMA/NCCL      ┌─────────────────────┐
│  Sender Node                  │◄──────────────────►│  Receiver Node      │
│                               │                    │                     │
│  ┌─────────────────────────┐  │                    │  ┌───────────────┐  │
│  │ Pod: sender             │  │                    │  │ Pod: receiver │  │
│  │  ├─ vllm (HTTP :8000)   │  │                    │  │  └─ receiver  │  │
│  │  │   builtin OpenAI     │  │                    │  └───────────────┘  │
│  │  │   server + extension │  │                    │                     │
│  │  │        ▲             │  │                    │                     │
│  │  │        │ collective_rpc (HTTP)               │                     │
│  │  │        │             │  │                    │                     │
│  │  └─ sidecar (triggers   │  │                    │                     │
│  │     weight server,      │  │                    │                     │
│  │     initiates transfer) │  │                    │                     │
│  └─────────────────────────┘  │                    │                     │
└───────────────────────────────┘                    └─────────────────────┘
```

The sidecar calls `POST /collective_rpc` to invoke extension methods:
- `start_weight_server` - starts ZMQ server for IPC handles
- `get_model_weight_metadata` - exports tensor metadata

## Troubleshooting

**Pods stuck pending**: Check node labels (`kubectl get nodes --show-labels | grep rdma`)

**NCCL timeout**: Enable debug logging - already set to `NCCL_DEBUG=INFO`

**GPU not found**: Verify NVIDIA device plugin (`kubectl get pods -n kube-system | grep nvidia`)

