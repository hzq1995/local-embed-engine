# Local Embed Engine API README

本文档只描述后端 API 接口，给前端、第三方调用方或独立实现方使用。

如果有人需要“单独做一个”兼容实现，这份文档就是接口契约，至少应保持：

- 路径一致
- 请求方法一致
- 请求/响应 JSON 字段一致
- 字段含义和约束一致
- 主要错误码语义一致

## 1. 基本信息

- 服务类型：HTTP JSON API
- 默认监听地址：`http://127.0.0.1:8010`
- 数据格式：`application/json`
- 字符编码：`UTF-8`

当前接口列表：

- `GET /health`
- `GET /index/info`
- `POST /embedding/by-point`
- `POST /search/by-embedding`
- `POST /embedding/by-bbox`

## 2. 通用约定

### 2.1 坐标系

所有接口中的经纬度都使用 WGS84：

- `lon`：经度，范围 `[-180, 180]`
- `lat`：纬度，范围 `[-90, 90]`

### 2.2 embedding 约定

- `embedding` 固定为 64 维浮点数组
- 响应中的 `embedding` 也是 64 维数组
- 调用方不应假设返回值已经做过额外压缩或截断

### 2.3 bbox 约定

`bbox` 格式固定为：

```json
[minLon, minLat, maxLon, maxLat]
```

并满足：

- 长度必须为 4
- `minLon <= maxLon`
- `minLat <= maxLat`

### 2.4 错误约定

常见返回码：

- `200`：成功
- `422`：参数非法，或业务条件不满足

`422` 常见场景：

- 点不在宁波边界内
- 点落在无效像元 / nodata
- `embedding` 维度不是 64
- `bbox` 格式不合法
- `top_k`、`total_samples` 超出允许范围

FastAPI 默认错误体示例：

```json
{
  "detail": "Point is outside Ningbo boundary."
}
```

或参数校验错误：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "embedding"],
      "msg": "Value error, embedding must contain exactly 64 values.",
      "input": [1, 2, 3]
    }
  ]
}
```

独立实现时不要求逐字一致，但建议保持：

- 非法请求返回 `422`
- 错误体包含可读的 `detail`

## 3. 接口明细

### 3.1 `GET /health`

用途：检查服务是否启动，以及索引是否已加载。

请求示例：

```bash
curl http://127.0.0.1:8010/health
```

响应示例：

```json
{
  "status": "ok",
  "service": "Ningbo Local Embed Engine",
  "year": 2024,
  "index_loaded": true,
  "vector_count": 123456,
  "build_time": "2026-04-20T00:00:00+00:00",
  "index_type": "faiss_hnsw_ip"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | `string` | 固定为健康状态标记，当前为 `ok` |
| `service` | `string` | 服务名称 |
| `year` | `int` | 当前数据年份标识 |
| `index_loaded` | `bool` | 是否成功加载索引数据 |
| `vector_count` | `int` | 当前索引中的向量数量 |
| `build_time` | `string \| null` | 构建时间，可能为空 |
| `index_type` | `string \| null` | 索引类型，可能为空 |

### 3.2 `GET /index/info`

用途：获取索引文件路径、边界文件路径、向量数量、构建信息。

请求示例：

```bash
curl http://127.0.0.1:8010/index/info
```

响应示例：

```json
{
  "year": 2024,
  "data_dir": "/mnt_llm_A100_V1/aef-zhejiang/2024/51N",
  "boundary_kml_path": "/mnt_llm_A100_V1/hzq/ShowAEF/local-embed-engine/data/宁波市.kml",
  "boundary_cache_path": "/mnt_llm_A100_V1/hzq/ShowAEF/local-embed-engine/data/derived/ningbo_boundary.geojson",
  "metadata_path": "/mnt_llm_A100_V1/hzq/ShowAEF/local-embed-engine/data/derived/metadata.parquet",
  "embeddings_path": "/mnt_llm_A100_V1/hzq/ShowAEF/local-embed-engine/data/derived/embeddings.npy",
  "index_path": "/mnt_llm_A100_V1/hzq/ShowAEF/local-embed-engine/data/derived/faiss.index",
  "tile_count": 321,
  "vector_count": 123456,
  "build_time": "2026-04-20T00:00:00+00:00",
  "index_type": "faiss_hnsw_ip"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `year` | `int` | 当前数据年份标识 |
| `data_dir` | `string` | 原始 GeoTIFF 数据目录 |
| `boundary_kml_path` | `string` | 行政边界 KML 路径 |
| `boundary_cache_path` | `string` | 边界缓存文件路径 |
| `metadata_path` | `string` | 元数据文件路径 |
| `embeddings_path` | `string` | embedding 数据文件路径 |
| `index_path` | `string` | 向量索引文件路径 |
| `tile_count` | `int` | 参与构建的瓦片数量 |
| `vector_count` | `int` | 当前向量数量 |
| `build_time` | `string \| null` | 构建时间 |
| `index_type` | `string \| null` | 索引类型 |

### 3.3 `POST /embedding/by-point`

用途：根据经纬度查询该点对应像元的 embedding。

请求体：

```json
{
  "lon": 121.544,
  "lat": 29.8683
}
```

参数约束：

| 字段 | 类型 | 必填 | 约束 |
| --- | --- | --- | --- |
| `lon` | `float` | 是 | `-180 <= lon <= 180` |
| `lat` | `float` | 是 | `-90 <= lat <= 90` |

请求示例：

```bash
curl -X POST http://127.0.0.1:8010/embedding/by-point \
  -H 'Content-Type: application/json' \
  -d '{"lon":121.544,"lat":29.8683}'
```

成功响应示例：

```json
{
  "year": 2024,
  "lon": 121.544,
  "lat": 29.8683,
  "embedding": [0.1, 0.2, 0.3],
  "tile_path": "/path/to/tile.tif",
  "row": 120,
  "col": 256
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `year` | `int` | 当前数据年份标识 |
| `lon` | `float` | 查询点经度 |
| `lat` | `float` | 查询点纬度 |
| `embedding` | `float[]` | 64 维 embedding |
| `tile_path` | `string` | 命中的源栅格文件路径 |
| `row` | `int` | 像元行号 |
| `col` | `int` | 像元列号 |

业务说明：

- 点必须位于宁波边界内
- 点必须命中有效瓦片
- 像元不能是 nodata

否则返回 `422`。

### 3.4 `POST /search/by-embedding`

用途：以一个 64 维 embedding 作为查询向量，返回相似结果列表。

请求体：

```json
{
  "embedding": [0.1, 0.2, 0.3],
  "top_k": 10,
  "bbox": [121.54, 29.86, 121.56, 29.88],
  "min_distance_m": 0,
  "min_score": 0.0
}
```

注意：示例中的 `embedding` 仅为简写，真实请求必须传 64 维。

参数约束：

| 字段 | 类型 | 必填 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `embedding` | `float[]` | 是 | 无 | 长度必须为 `64` |
| `top_k` | `int` | 否 | `10` | `1 <= top_k <= 1000` |
| `min_distance_m` | `float` | 否 | `0` | `>= 0` |
| `min_score` | `float` | 否 | `0.0` | `0 <= min_score <= 1` |
| `bbox` | `float[] \| null` | 否 | `null` | 长度必须为 `4`，格式为 `[minLon, minLat, maxLon, maxLat]` |

请求示例：

```bash
curl -X POST http://127.0.0.1:8010/search/by-embedding \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding":[0.1,0.2,0.3],
    "top_k":10,
    "bbox":[121.54,29.86,121.56,29.88],
    "min_distance_m":0,
    "min_score":0.0
  }'
```

成功响应示例：

```json
{
  "top_k": 10,
  "result_count": 2,
  "results": [
    {
      "rank": 1,
      "score": 0.998,
      "lon": 121.545,
      "lat": 29.868,
      "embedding": [0.1, 0.2, 0.3],
      "tile_path": "/path/to/tile.tif",
      "row": 100,
      "col": 200
    }
  ]
}
```

响应字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `top_k` | `int` | 请求中的 `top_k` |
| `result_count` | `int` | 实际返回结果数 |
| `results` | `object[]` | 检索结果列表 |

`results[i]` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `rank` | `int` | 排名，从 1 开始 |
| `score` | `float` | 相似度得分 |
| `lon` | `float` | 结果点经度 |
| `lat` | `float` | 结果点纬度 |
| `embedding` | `float[]` | 64 维 embedding |
| `tile_path` | `string` | 源栅格路径 |
| `row` | `int` | 像元行号 |
| `col` | `int` | 像元列号 |

行为约定：

- `bbox` 传入后，只在该空间范围内筛选结果
- `min_distance_m` 可用于结果去重，避免返回过密点位
- `min_score` 可用于过滤低相似度结果
- `result_count` 可能小于 `top_k`

独立实现时，建议保持以上过滤语义不变。

### 3.5 `POST /embedding/by-bbox`

用途：在给定范围内采样一批 embedding，供前端做区域聚类、热力或网格展示。

请求体：

```json
{
  "bbox": [121.54, 29.86, 121.56, 29.88],
  "total_samples": 5000
}
```

参数约束：

| 字段 | 类型 | 必填 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `bbox` | `float[]` | 是 | 无 | 长度必须为 `4` |
| `total_samples` | `int` | 否 | `5000` | `100 <= total_samples <= 50000` |

请求示例：

```bash
curl -X POST http://127.0.0.1:8010/embedding/by-bbox \
  -H 'Content-Type: application/json' \
  -d '{"bbox":[121.54,29.86,121.56,29.88],"total_samples":5000}'
```

成功响应示例：

```json
{
  "count": 3,
  "lons": [121.541, 121.545, 121.552],
  "lats": [29.861, 29.867, 29.878],
  "embeddings": [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
  ],
  "grid_rows": 2,
  "grid_cols": 2,
  "grid_row_indices": [0, 0, 1],
  "grid_col_indices": [0, 1, 1],
  "effective_bbox": [121.54, 29.86, 121.56, 29.88]
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `count` | `int` | 实际采样返回数量 |
| `lons` | `float[]` | 每个样本点的经度数组 |
| `lats` | `float[]` | 每个样本点的纬度数组 |
| `embeddings` | `float[][]` | 每个样本点的 64 维 embedding |
| `grid_rows` | `int` | 采样网格总行数 |
| `grid_cols` | `int` | 采样网格总列数 |
| `grid_row_indices` | `int[]` | 每个样本所在网格行索引 |
| `grid_col_indices` | `int[]` | 每个样本所在网格列索引 |
| `effective_bbox` | `float[]` | 实际生效的 bbox，可能是裁剪到宁波边界后的范围 |

行为说明：

- 输入 bbox 会先和宁波边界求交
- 服务会在有效范围内构造近似 `total_samples` 的规则网格
- 每个网格单元最多返回一个代表样本
- `count` 可能小于 `total_samples`

## 4. 对接建议

如果你是前端或第三方服务调用方，推荐调用顺序如下：

### 4.1 点选查询场景

1. 调用 `POST /embedding/by-point`
2. 取返回中的 `embedding`
3. 调用 `POST /search/by-embedding`
4. 渲染相似点结果

### 4.2 区域聚类场景

1. 调用 `POST /embedding/by-bbox`
2. 使用 `lons`、`lats`、`embeddings` 做聚类或可视化

## 5. 独立实现兼容要求

如果别人要单独做一个兼容后端，建议至少满足以下要求：

- 保留完全一致的 URL 路径和 HTTP 方法
- 请求字段名保持一致，不改大小写
- 返回字段名保持一致，不额外改名
- `embedding` 维度仍固定为 64
- 参数非法时返回 `422`
- 能区分“参数不合法”和“查询无有效结果”的场景

如果需要扩展字段，建议：

- 仅新增字段，不删除现有字段
- 不改变现有字段类型
- 保持旧客户端不改代码也能继续工作

## 6. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

启动后可直接访问：

- `http://127.0.0.1:8010/health`
- `http://127.0.0.1:8010/docs`
- `http://127.0.0.1:8010/redoc`

其中 `/docs` 是 FastAPI 自动生成的 Swagger 文档，适合在线调试；本文件更适合作为交付给他人的接口说明。
