#!/bin/bash
# Minimal RDMA experiment deployment
# Usage: ./deploy.sh <sender-node> <receiver-node> [model]
set -e

SENDER=${1:?Usage: $0 <sender-node> <receiver-node> [model]}
RECEIVER=${2:?Usage: $0 <sender-node> <receiver-node> [model]}
MODEL=${3:-facebook/opt-125m}
DIR=$(dirname "$0")

echo "=== RDMA Experiment ==="
echo "Sender:   $SENDER"
echo "Receiver: $RECEIVER"
echo "Model:    $MODEL"

# Label nodes
kubectl label node "$SENDER" rdma=sender --overwrite
kubectl label node "$RECEIVER" rdma=receiver --overwrite

# Create namespace
kubectl create namespace rdma-exp --dry-run=client -o yaml | kubectl apply -f -

# Create code configmap
kubectl create configmap rdma-code -n rdma-exp \
    --from-file="$DIR/../rdma_sidecar.py" \
    --from-file="$DIR/../rdma_sidecar_extension.py" \
    --from-file="$DIR/../rdma_remote_receiver.py" \
    --from-file="$DIR/../__init__.py" \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply manifest with model override
sed "s|facebook/opt-125m|$MODEL|g" "$DIR/rdma-experiment.yaml" | kubectl apply -f -

echo ""
echo "=== Deployed ==="
echo "Watch:  kubectl get pods -n rdma-exp -w"
echo "Logs:   kubectl logs -n rdma-exp sender -c vllm -f"
echo "API:    kubectl port-forward -n rdma-exp svc/vllm 8000:8000"
echo "Clean:  kubectl delete ns rdma-exp && kubectl label node $SENDER rdma- && kubectl label node $RECEIVER rdma-"

