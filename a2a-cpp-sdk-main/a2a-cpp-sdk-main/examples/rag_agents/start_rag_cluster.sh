#!/bin/bash
# start_rag_cluster.sh — 一键启动 RAG 分布式集群
# 用法: ./start_rag_cluster.sh

REGISTRY_PORT=8500
RETRIEVER_BASE=9001
ORCHESTRATOR_PORT=9000
GENERATOR_PORT=9004

echo "=== RAG Agent Cluster ==="

# 1. 启动注册中心
echo "[1/5] Starting Registry Center..."
./registry_server $REGISTRY_PORT &
REGISTRY_PID=$!
sleep 2

# 2. 启动 3 个 Retriever Agent
echo "[2/5] Starting Retriever Agents..."
./retriever_agent retriever-1 9001 &
PID_R1=$!
./retriever_agent retriever-2 9002 &
PID_R2=$!
./retriever_agent retriever-3 9003 &
PID_R3=$!
sleep 1

# 3. 启动 Generator Agent
echo "[3/5] Starting Generator Agent..."
./generator_agent generator-1 $GENERATOR_PORT "your-api-key" &
PID_GEN=$!
sleep 1

# 4. 启动 Orchestrator
echo "[4/5] Starting Orchestrator..."
./orchestrator_agent orchestrator-1 $ORCHESTRATOR_PORT &
PID_ORC=$!
sleep 2

# 5. 测试查询
echo "[5/5] Sending test query..."
curl -X POST http://localhost:$ORCHESTRATOR_PORT/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"query","params":{"query":"What is C++ concurrency?"},"id":1}'

echo ""
echo "=== Cluster running ==="
echo "Orchestrator: http://localhost:$ORCHESTRATOR_PORT"
echo "Retrievers:   9001 9002 9003"
echo "Generator:    $GENERATOR_PORT"
echo ""
echo "Press Ctrl+C to stop all agents"

# 等待
wait $PID_ORC
kill $PID_R1 $PID_R2 $PID_R3 $PID_GEN $REGISTRY_PID 2>/dev/null
echo "Cluster stopped."
