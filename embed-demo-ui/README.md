# Embed Demo UI

独立前端调试页，使用 Vite + React + Google Maps JavaScript API。

## 启动

```bash
cd local-embed-engine/embed-demo-ui
npm install
npm run dev
```

默认会把前端的 `/api/*` 请求代理到 `http://127.0.0.1:8010/*`。

地图脚本通过 `VITE_GOOGLE_MAPS_API_KEY` 加载。当前目录已放置 `.env`。

如果后端地址不是这个，可以这样启动：

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:8011 npm run dev
```

## 用途

- 点击 Google 地图调用 `POST /embedding/by-point`
- 用返回的 embedding 调用 `POST /search/by-embedding`
- 支持 `Top K`
- 支持“仅搜索当前视野” bbox 检索
- 支持 `/health` 连通性检查
