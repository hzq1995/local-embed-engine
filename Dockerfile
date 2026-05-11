# 使用现有的local-embed-engine镜像作为基础
FROM local-embed-engine:v20

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY app/ ./app/
COPY data/*.kml ./data/
COPY scripts/ ./scripts/
COPY requirements.txt ./
COPY start_backend.sh ./
COPY start_build_index.sh ./

# 安装Python依赖（使用阿里云镜像加速）
RUN pip install -i  https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt
