# Local Embed Engine

本项目是一个面向宁波区域影像嵌入数据的本地检索服务，提供离线索引构建能力，以及基于 FastAPI 的查询 API。仓库内同时附带一个前端演示页面，便于联调“点选取 embedding”“按 embedding 相似检索”“区域采样/聚类展示”等能力。

项目核心目标：

- 将多波段 GeoTIFF 中的 64 维 embedding 像元整理为可检索索引
- 使用宁波行政区边界裁剪有效区域，过滤边界外数据
- 提供点查 embedding、embedding 相似检索、按 bbox 采样等接口
- 支持前端地图交互和本地可视化调试

## 1. 项目结构

```text
local-embed-engine/
├── app/                    # FastAPI 应用与核心服务
│   ├── main.py             # API 入口
│   ├── config.py           # 环境变量与路径配置
│   ├── schemas.py          # 请求/响应模型
│   └── services/           # 边界、瓦片、索引、检索、构建逻辑
├── scripts/
│   └── build_index.py      # 离线索引构建脚本
├── data/
│   ├── 宁波市.kml          # 宁波边界源文件
│   └── derived/            # 构建输出目录（元数据、embedding、索引等）
├── embed-demo-ui/          # Vite + React + Leaflet 演示前端
├── tests/                  # 单元测试与接口测试
├── requirements.txt        # Python 依赖
└── start_ui.sh             # 前端本地启动脚本
```

## 2. 适用数据约定

当前实现默认假设输入数据满足以下条件：

- 输入目录下为 `*.tif` 或 `*.tiff` 文件
- 每个像元包含 64 个 band，对应一个 64 维 embedding
- nodata 像元会被过滤
- 栅格具备合法 CRS，代码会在读取时转换到 WGS84
- 最终仅保留落在宁波行政边界内的像元

默认数据目录是：

```bash
/mnt_llm_A100_V1/aef-zhejiang/2024/51N
```

如果你的数据不在这个路径，需要通过环境变量或命令行参数覆盖。

## 3. 安装依赖

建议使用 Python 3.11+。

```bash
cd local-embed-engine
pip install -r requirements.txt
```

依赖说明：

- `fastapi`、`uvicorn`：服务运行
- `numpy`、`pandas`、`pyarrow`：向量与元数据存储
- `rasterio`、`pyproj`、`shapely`：栅格读取、坐标转换、空间裁剪
- `faiss-cpu`：可选的 ANN 索引加速
- `tqdm`：构建过程进度条

注意：

- 即使 `faiss-cpu` 不可用，服务仍可运行，此时会退化为 NumPy 精确检索
- 如果构建环境安装 `faiss-cpu`，会额外生成 `faiss.index`

## 4. 快速开始

### 4.1 构建离线索引

```bash
cd local-embed-engine
python scripts/build_index.py
```

默认参数：

- 数据目录：`/mnt_llm_A100_V1/aef-zhejiang/2024/51N`
- 边界文件：`./宁波市.kml` 或代码根目录下的同名文件
- 输出目录：`./data/derived`
- `block_size`：`512`
- `flush_rows`：`250000`

也可以显式指定：

```bash
python scripts/build_index.py \
  --data-dir /path/to/tifs \
  --boundary-kml ./data/宁波市.kml \
  --output-dir ./data/derived \
  --year 2024 \
  --block-size 512 \
  --flush-rows 250000
```

构建完成后会输出类似信息：

```bash
Built index in data/derived with <vector_count> vectors from <tile_count> tiles using <index_type>.
```

### 4.2 启动后端服务

```bash
cd local-embed-engine
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

启动后可访问：

- `GET /health`
- `GET /index/info`

### 4.3 启动前端演示页

```bash
cd local-embed-engine/embed-demo-ui
npm install
npm run dev
```

前端默认地址通常为：

```text
http://127.0.0.1:5173
```

仓库根目录还提供了一个便捷脚本：

```bash
./start_ui.sh
```

该脚本会以 `0.0.0.0:5200` 启动 Vite，并保持严格端口。

前端通过 `/api/*` 代理到后端，默认代理目标为：

```text
http://127.0.0.1:8010
```

如果后端端口不同，可以这样启动：

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8011 npm run dev
```

## 5. 构建产物说明

`data/derived/` 目录中的主要文件如下：

- `ningbo_boundary.geojson`
  - 由 `宁波市.kml` 解析后缓存得到
  - 用于避免每次启动都重复解析 KML
- `metadata.parquet`
  - 每条向量对应一条元数据
  - 字段包括：`id`、`lon`、`lat`、`tile_path`、`row`、`col`
- `embeddings.npy`
  - 形状为 `N x 64` 的浮点数组
  - 存储归一化后的 embedding
- `faiss.index`
  - 可选
  - 当安装了 `faiss-cpu` 且向量数大于 0 时生成
- `build_info.json`
  - 记录构建时间、数据源、索引类型、向量数、瓦片数等信息

## 6. 服务行为说明

### 6.1 点查 embedding

输入经纬度后，服务会：

1. 检查点是否位于宁波边界内
2. 定位落在哪个 GeoTIFF 瓦片
3. 读取对应像元的 64 维 embedding
4. 返回像元行列号和源瓦片路径

如果点在边界外、瓦片外、或落在 nodata 像元上，会返回 `422`。

### 6.2 按 embedding 检索

检索逻辑支持：

- `top_k` 最近邻
- `bbox` 空间范围裁剪
- `min_distance_m` 结果去重，避免返回过于密集的邻近点
- `min_score` 相似度阈值

说明：

- embedding 会先做 L2 归一化
- 如果满足条件，会优先使用 FAISS HNSW 内积索引
- 如果启用了 `bbox`、`min_distance_m` 或 `min_score`，会转为精确筛选路径

### 6.3 按 bbox 采样 embedding

该接口会：

1. 先将输入 bbox 裁剪到宁波边界范围内
2. 在裁剪后的范围内构造近似 `total_samples` 的规则网格
3. 为每个网格单元选择一个代表像元
4. 返回 embedding、经纬度和网格位置索引

这个接口更适合前端做区域聚类、热力展示或网格分析，而不是严格的最近邻搜索。

## 7. API 接口

接口已单独整理到 [API_README.md](./API_README.md)。

该文档适合直接发给：

- 前端对接人员
- 第三方调用方
- 需要“单独做一个”兼容实现的人

当前 API 一览：

- `GET /health`
- `GET /index/info`
- `POST /embedding/by-point`
- `POST /search/by-embedding`
- `POST /embedding/by-bbox`

如果需要在线调试，也可以在服务启动后访问：

- `/docs`
- `/redoc`

## 8. 环境变量

后端配置定义在 `app/config.py`，支持以下环境变量：

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `LOCAL_AEF_SERVICE_NAME` | `Ningbo Local Embed Engine` | 服务名称 |
| `LOCAL_AEF_HOST` | `0.0.0.0` | 服务监听地址 |
| `LOCAL_AEF_PORT` | `8010` | 服务端口 |
| `LOCAL_AEF_YEAR` | `2024` | 数据年份标识 |
| `LOCAL_AEF_DATA_DIR` | `/mnt_llm_A100_V1/aef-zhejiang/2024/51N` | GeoTIFF 数据目录 |
| `LOCAL_AEF_BOUNDARY_KML` | `./宁波市.kml` | 宁波边界 KML 路径 |
| `LOCAL_AEF_DERIVED_DIR` | `./data/derived` | 构建产物目录 |
| `LOCAL_AEF_BOUNDARY_CACHE` | `derived/ningbo_boundary.geojson` | 边界缓存文件 |
| `LOCAL_AEF_METADATA_PATH` | `derived/metadata.parquet` | 元数据文件 |
| `LOCAL_AEF_EMBEDDINGS_PATH` | `derived/embeddings.npy` | embedding 文件 |
| `LOCAL_AEF_INDEX_PATH` | `derived/faiss.index` | FAISS 索引文件 |
| `LOCAL_AEF_BUILD_INFO_PATH` | `derived/build_info.json` | 构建信息文件 |

示例：

```bash
export LOCAL_AEF_DATA_DIR=/data/ningbo_tiles
export LOCAL_AEF_DERIVED_DIR=./data/derived
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

## 9. 测试

运行测试：

```bash
cd local-embed-engine
python -m unittest discover -s tests
```

当前测试覆盖：

- 宁波边界 KML 解析与点包含判断
- 构建流程是否生成预期产物
- `/health`、`/embedding/by-point`、`/search/by-embedding` 等接口行为
- 参数非法场景
- `min_distance_m` 结果去重行为

## 10. 前端演示页说明

`embed-demo-ui/` 是一个独立前端工程，当前技术栈是：

- Vite
- React 18
- Leaflet
- React Leaflet

主要场景：

- `选点查询`
  - 地图点击后调用 `POST /embedding/by-point`
  - 再用返回向量调用 `POST /search/by-embedding`
- `区域聚类`
  - 基于 `POST /embedding/by-bbox` 做区域采样和前端聚类展示
- `互花米草`
  - 独立图层/专题展示

注意：仓库里的前端实现使用的是 Leaflet，不是 Google Maps。

## 11. 常见问题

### 11.1 启动后 `index_loaded=false`

说明后端正常启动了，但尚未找到有效索引文件。检查：

- 是否已经执行过 `python scripts/build_index.py`
- `LOCAL_AEF_DERIVED_DIR` 是否指向正确目录
- `metadata.parquet`、`embeddings.npy`、`build_info.json` 是否都存在

### 11.2 没有生成 `faiss.index`

可能原因：

- 当前环境没有成功安装 `faiss-cpu`
- 构建结果向量数为 0

即使没有 `faiss.index`，服务仍可回退到 NumPy 精确检索。

### 11.3 点查返回 422

常见原因：

- 点不在宁波边界内
- 点不在任何 GeoTIFF 范围内
- 点落在 nodata 像元上

### 11.4 bbox 查询没有结果

常见原因：

- bbox 完全落在宁波边界外
- bbox 与边界裁剪后范围过小
- 对应区域没有有效 embedding 像元

## 12. 开发建议

- 大规模数据构建时优先调节 `block_size` 和 `flush_rows`
- 如果主要使用全局 Top-K 检索，建议安装 `faiss-cpu`
- 如果经常切换数据目录，建议将环境变量写入启动脚本
- 如果要扩展到其他城市，至少需要替换边界 KML，并确认输入栅格坐标系与 band 维度约定一致

## 13. 当前仓库已包含的内容

仓库中已经带有一份示例构建产物：

- `data/derived/metadata.parquet`
- `data/derived/embeddings.npy`
- `data/derived/build_info.json`
- `data/derived/faiss.index`
- `data/derived/ningbo_boundary.geojson`

因此在默认配置不变的情况下，可以直接启动后端查看现有索引信息；如果替换原始数据或边界文件，建议重新构建。
