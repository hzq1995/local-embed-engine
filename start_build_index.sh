#!/bin/bash

#========================================
# Build Index 启动脚本
#========================================

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
echo "========================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# 使用 nohup 后台运行，并将输出追加到日志文件
nohup python scripts/build_index.py >> ${LOG_FILE} 2>&1 &

PID=$!
echo "进程已启动，PID: ${PID}" | tee -a ${LOG_FILE}
echo "实时日志查看命令: tail -f ${LOG_FILE}"

# 保存 PID 到文件
echo ${PID} > ${LOG_DIR}/build_index.pid
