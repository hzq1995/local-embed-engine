# 使用现有的synrs3d_env镜像作为基础
FROM local-embed-engine:v2

# 设置工作目录
WORKDIR /app

# 复制项目文件（排除data、embed-demo-ui、release）
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY requirements.txt ./
COPY start_backend.sh ./

# 安装Python依赖（使用阿里云镜像加速）
RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt
