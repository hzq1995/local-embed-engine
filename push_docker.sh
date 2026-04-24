#!/bin/bash

#========================================
# Docker镜像一键构建和推送脚本
#========================================

set -e  # 任何命令失败时退出

# ========== 配置部分 ==========
REGISTRY="10.200.53.208"  # 实际仓库地址
DOCKER_USER="zrzyb_003062"
DOCKER_PASSWORD="Zrzyb123"
IMAGE_NAME="local-embed-engine"
IMAGE_TAG="v7-tif-api"
FULL_IMAGE_NAME="${REGISTRY}/${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

# 获取当前时间作为版本号
BUILD_TAG=$(date +%Y%m%d_%H%M%S)
VERSIONED_IMAGE="${REGISTRY}/${DOCKER_USER}/${IMAGE_NAME}:${BUILD_TAG}"

# ========== 颜色输出 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========== Docker镜像构建和推送 ==========${NC}"

# ========== 步骤1: 构建镜像 ==========
echo -e "${YELLOW}[1/5] 构建Docker镜像...${NC}"
if docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .; then
    echo -e "${GREEN}✓ 镜像构建成功${NC}"
else
    echo -e "${RED}✗ 镜像构建失败${NC}"
    exit 1
fi

# ========== 步骤2: 给镜像打标签（latest） ==========
echo -e "${YELLOW}[2/5] 给镜像打标签 (latest)...${NC}"
if docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${FULL_IMAGE_NAME}; then
    echo -e "${GREEN}✓ 标签打标成功: ${FULL_IMAGE_NAME}${NC}"
else
    echo -e "${RED}✗ 标签打标失败${NC}"
    exit 1
fi

# ========== 步骤3: 给镜像打标签（版本号） ==========
echo -e "${YELLOW}[3/5] 给镜像打标签 (${BUILD_TAG})...${NC}"
if docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${VERSIONED_IMAGE}; then
    echo -e "${GREEN}✓ 版本标签打标成功: ${VERSIONED_IMAGE}${NC}"
else
    echo -e "${RED}✗ 版本标签打标失败${NC}"
    exit 1
fi

# ========== 步骤4: 推送镜像 ==========
echo -e "${YELLOW}[4/5] 推送镜像到仓库...${NC}"

echo -e "${YELLOW}推送 ${FULL_IMAGE_NAME}...${NC}"
if docker push ${FULL_IMAGE_NAME}; then
    echo -e "${GREEN}✓ latest版本推送成功${NC}"
else
    echo -e "${RED}✗ latest版本推送失败${NC}"
    exit 1
fi

echo -e "${YELLOW}推送 ${VERSIONED_IMAGE}...${NC}"
if docker push ${VERSIONED_IMAGE}; then
    echo -e "${GREEN}✓ 版本镜像推送成功${NC}"
else
    echo -e "${RED}✗ 版本镜像推送失败${NC}"
    exit 1
fi

# ========== [5/5]清理别名镜像 ==========
echo -e "${YELLOW}[5/5][清理] 删除别名镜像...${NC}"
docker rmi ${FULL_IMAGE_NAME}
docker rmi ${VERSIONED_IMAGE}
echo -e "${GREEN}✓ 别名镜像已删除${NC}"

# ========== 完成 ==========
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 所有步骤执行成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "镜像信息："
echo "  Latest: ${FULL_IMAGE_NAME}"
echo "  Versioned: ${VERSIONED_IMAGE}"
echo ""
echo "使用命令查看推送的镜像："
echo "  docker pull ${FULL_IMAGE_NAME}"
echo "  docker pull ${VERSIONED_IMAGE}"
