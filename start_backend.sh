#!/bin/bash

# Local Embed Engine 后端启动脚本
# 不使用 faiss.index，仅用 NumPy 精确检索

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 配置环境变量（可选，使用默认值时可省略）
export LOCAL_AEF_SERVICE_NAME="Ningbo Local Embed Engine"
export LOCAL_AEF_HOST="${LOCAL_AEF_HOST:-0.0.0.0}"
export LOCAL_AEF_PORT="${LOCAL_AEF_PORT:-8010}"
export LOCAL_AEF_YEAR="${LOCAL_AEF_YEAR:-2024}"
export LOCAL_AEF_DATA_DIR="${LOCAL_AEF_DATA_DIR:-/mnt_llm_A100_V1/aef-zhejiang/2024/51N}"
export LOCAL_AEF_BOUNDARY_KML="${LOCAL_AEF_BOUNDARY_KML:-$PROJECT_ROOT/data/宁波市.kml}"
export LOCAL_AEF_DERIVED_DIR="${LOCAL_AEF_DERIVED_DIR:-$PROJECT_ROOT/data/derived}"

echo "=========================================="
echo "Local Embed Engine - Backend Server"
echo "=========================================="
echo "Host: $LOCAL_AEF_HOST"
echo "Port: $LOCAL_AEF_PORT"
echo "Year: $LOCAL_AEF_YEAR"
echo "Data Dir: $LOCAL_AEF_DATA_DIR"
echo "Derived Dir: $LOCAL_AEF_DERIVED_DIR"
echo "=========================================="
echo ""

# 启动后端服务
uvicorn app.main:app \
  --host "$LOCAL_AEF_HOST" \
  --port "$LOCAL_AEF_PORT" \
  --reload
