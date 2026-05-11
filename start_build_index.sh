#!/bin/bash

#========================================
# Build Index 启动脚本
#========================================

#----------------------------------------
# 环境变量配置（按需修改）
#----------------------------------------
export LOCAL_AEF_YEAR="${LOCAL_AEF_YEAR:-2024}"
export LOCAL_AEF_DATA_DIR="${LOCAL_AEF_DATA_DIR:-/mnt_llm_A100_V1/aef-zhejiang/2024/51N}"
export LOCAL_AEF_DERIVED_DIR="${LOCAL_AEF_DERIVED_DIR:-./data/derived}"
export LOCAL_AEF_BOUNDARY_KML="${LOCAL_AEF_BOUNDARY_KML:-./data/beilunqu.kml}"

# build_index.py 参数
export LOCAL_AEF_BLOCK_SIZE="${LOCAL_AEF_BLOCK_SIZE:-512}"
export LOCAL_AEF_FLUSH_ROWS="${LOCAL_AEF_FLUSH_ROWS:-250000}"
export LOCAL_AEF_WITH_FAISS="${LOCAL_AEF_WITH_FAISS:-false}"

# build_coarse_index.py 参数
export LOCAL_AEF_COARSE_BLOCK_ROWS="${LOCAL_AEF_COARSE_BLOCK_ROWS:-250000}"
export LOCAL_AEF_COARSE_STRIDE="${LOCAL_AEF_COARSE_STRIDE:-1}"
export LOCAL_AEF_COARSE_REDUCED_DIM="${LOCAL_AEF_COARSE_REDUCED_DIM:-128}"
export LOCAL_AEF_PROJECTION_SEED="${LOCAL_AEF_PROJECTION_SEED:-20240424}"

# 获取当前时间作为日志文件名
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/build_index_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "========================================" | tee ${LOG_FILE}
echo "开始构建索引" | tee -a ${LOG_FILE}
echo "日志文件: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "启动时间: $(date)" | tee -a ${LOG_FILE}
echo "DATA_DIR: ${LOCAL_AEF_DATA_DIR}" | tee -a ${LOG_FILE}
echo "DERIVED_DIR: ${LOCAL_AEF_DERIVED_DIR}" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Step 1: 构建主索引
echo "--- Step 1/2: 构建主索引 ---" | tee -a ${LOG_FILE}
python scripts/build_index.py \
    --data-dir "${LOCAL_AEF_DATA_DIR}" \
    --boundary-kml "${LOCAL_AEF_BOUNDARY_KML}" \
    --output-dir "${LOCAL_AEF_DERIVED_DIR}" \
    --year "${LOCAL_AEF_YEAR}" \
    --block-size "${LOCAL_AEF_BLOCK_SIZE}" \
    --flush-rows "${LOCAL_AEF_FLUSH_ROWS}" \
    $([ "${LOCAL_AEF_WITH_FAISS}" = "true" ] && echo "--with-faiss") \
    >> ${LOG_FILE} 2>&1
EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "" | tee -a ${LOG_FILE}
    echo "主索引构建失败 (exit code: ${EXIT_CODE})，终止后续流程。" | tee -a ${LOG_FILE}
    exit ${EXIT_CODE}
fi

echo "" | tee -a ${LOG_FILE}
echo "--- Step 2/2: 构建粗糙 Embedding 索引 ---" | tee -a ${LOG_FILE}
python scripts/build_coarse_index.py \
    --derived-dir "${LOCAL_AEF_DERIVED_DIR}" \
    --block-rows "${LOCAL_AEF_COARSE_BLOCK_ROWS}" \
    --stride "${LOCAL_AEF_COARSE_STRIDE}" \
    --reduced-dim "${LOCAL_AEF_COARSE_REDUCED_DIM}" \
    --projection-seed "${LOCAL_AEF_PROJECTION_SEED}" \
    >> ${LOG_FILE} 2>&1
EXIT_CODE=$?

echo "" | tee -a ${LOG_FILE}
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "全部构建完成。完成时间: $(date)" | tee -a ${LOG_FILE}
else
    echo "粗糙索引构建失败 (exit code: ${EXIT_CODE})。完成时间: $(date)" | tee -a ${LOG_FILE}
fi

echo "日志文件: ${LOG_FILE}"
