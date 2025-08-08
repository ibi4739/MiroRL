#!/bin/bash
# Copyright 2025 MiroMind Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# set -ex

export JOB_NAME=mirorl-odr-mcp

# Setup Node.js for MCP
export NODEJS_HOME=/your/path/to/nodejs
export PATH=$NODEJS_HOME/bin:$PATH
export NODE_SHARED=$NODEJS_HOME/node-shared/node_modules
export PATH=$NODE_SHARED/.bin:$PATH

# Setup proxies
export HTTP_PROXY=xxx
export HTTPS_PROXY=xxx
export NO_PROXY=localhost,127.0.0.1

# Check for singlenode flag
SCRIPT=$1
SINGLENODE=${SINGLENODE:-false}

export LOG_DIR=$(pwd)/outputs/$MLP_TASK_ID/logs
mkdir -p $LOG_DIR

if [ "$SINGLENODE" == "true" ]; then
    bash $SCRIPT 2>&1 | tee $LOG_DIR/main_ppo.log
    exit 0
fi

#==============================================================================#

export NCCL_IB_TIMEOUT=80
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_AR_THRESHOLD=0

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-10086}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "HTTP_PROXY: $HTTP_PROXY, HTTPS_PROXY: $HTTPS_PROXY"

export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=$PWD:$PYTHONPATH

export LOG_DIR=$(pwd)/outputs/$JOB_NAME/logs
mkdir -p $LOG_DIR

# num_nodes has to be at least 1
if [ $NNODES -lt 1 ]; then
    echo "Number of nodes must be at least 1"
    exit 1
fi

# if HOST contains "master", then this is the head node
if [[ $NODE_RANK -eq 0 ]]; then
    node_role="master"
else
    node_role="worker"
fi
head_node_ip=${MASTER_ADDR:-127.0.0.1}

wait_time=30
if [ "$node_role" == "master" ]; then
    echo "Starting Ray head node..."
    # Start Ray on this node as the head node and extract its address
    # The `ray start --head` command outputs information that includes the address,
    # but here we're assuming it's known or statically assigned for simplicity.
    ray start --head --node-ip-address=$head_node_ip --include-dashboard=True --dashboard-host $head_node_ip --port=6379 --min-worker-port 15000 --max-worker-port 19999
    sleep $wait_time
elif [ "$node_role" == "worker" ]; then
    sleep $wait_time
    attempt=1
    echo "Starting Ray worker node and attempting to connect to the head node at $head_node_ip:6379"
    while true; do
        # Attempt to start Ray and connect to the head node
        ray start --address="$head_node_ip:6379" --min-worker-port 15000 --max-worker-port 19999 && break || {
            if [ $attempt -le 5 ]; then
                echo "Ray worker start attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        }
    done
fi

# run the training script once Ray has been started on all nodes
sleep $wait_time
if [ "$node_role" == "master" ]; then
    num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
    echo "Number of active Ray nodes: $num_active_ray_nodes"
    if [ $num_active_ray_nodes -lt $NNODES ]; then
        echo "Waiting for all Ray nodes to start..."
        attempt=1
        while true; do
            num_active_ray_nodes=$(ray list nodes | grep ALIVE | wc -l)
            if [ $num_active_ray_nodes -eq $NNODES ]; then
                break
            elif [ $attempt -le 5 ]; then
                echo "python command attempt $attempt failed. Retrying in $wait_time seconds..."
                ((attempt++))
                sleep $wait_time
            else
                echo "Failed to connect to the head node after $wait_time attempts. Exiting."
                exit 1
            fi
        done
    fi
    echo "End starting"
    bash ${SCRIPT} 2>&1 | tee $LOG_DIR/main_ppo.log
else
    echo "End starting"
    # Continuously check the health of the Ray cluster by pinging the head node.
    # If the health check fails, break the loop and proceed.
    while true; do
        ray health-check --address $head_node_ip:6379 &>/dev/null
        if [ $? -ne 0 ]; then
            break
        fi
        sleep 60
    done
fi
