import { useMemo, useState } from "react";
import { apiRequest, getErrorMessage } from "../utils/api";
import { useLocalStorage } from "../utils/useLocalStorage";
import StatusPanel from "../components/StatusPanel";

/**
 * Sidebar panel for the "Point Query" scene.
 * Supports multi-point selection: each map click appends a query point,
 * and the average embedding is used for similarity search.
 */
export default function PointQueryPanel({ mapReady, apiStatus, onCheckHealth, sceneState }) {
  const {
    requestStatus,
    errorMessage,
    topK, setTopK,
    minDistanceM, setMinDistanceM,
    minScore, setMinScore,
    searchRadiusKm, setSearchRadiusKm,
    useCoarseSearch, setUseCoarseSearch,
    selectedPoints,
    avgEmbedding,
    searchResults,
    activeRank,
    isSearching,
    savedCategories,
    handleSearch,
    handleSelectResult,
    handleSaveCategory,
    handleLoadCategory,
    handleDeleteCategory,
    removeLastPoint,
    clearAll,
  } = sceneState;

  const embeddingPreview = useMemo(() => {
    if (!avgEmbedding?.length) return "";
    return avgEmbedding
      .slice(0, 12)
      .map((value, index) => `${String(index).padStart(2, "0")}: ${Number(value).toFixed(6)}`)
      .join("\n");
  }, [avgEmbedding]);

  return (
    <>
      <section className="panel controls">
        <p className="desc">
          以地图上点击的位置作为参考，快速定位与查询。
        </p>
        <div className="control-row">
          <div className="field">
            <label htmlFor="pq-topK">Top K</label>
            <input
              id="pq-topK"
              type="number"
              min="1"
              max="1000"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label htmlFor="pq-radius">搜索半径（km）</label>
            <input
              id="pq-radius"
              type="number"
              min="0.1"
              max="500"
              step="0.5"
              value={searchRadiusKm}
              onChange={(e) => setSearchRadiusKm(Number(e.target.value))}
            />
          </div>
        </div>

        <div className="field">
          <label htmlFor="pq-minDist">最小结果间距（米）</label>
          <input
            id="pq-minDist"
            type="number"
            min="0"
            step="10"
            value={minDistanceM}
            onChange={(e) => setMinDistanceM(Number(e.target.value))}
          />
        </div>

        <div className="field">
          <label htmlFor="pq-minScore">相似度阈值（0–1）</label>
          <input
            id="pq-minScore"
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
          />
        </div>

        <label className="checkbox" htmlFor="pq-useCoarseSearch">
          <input
            id="pq-useCoarseSearch"
            type="checkbox"
            checked={useCoarseSearch}
            onChange={(e) => setUseCoarseSearch(e.target.checked)}
          />
          使用粗糙索引
        </label>

        <div className="button-row">
          <button
            type="button"
            className="primary"
            onClick={handleSearch}
            disabled={!avgEmbedding || !mapReady || isSearching}
            style={isSearching ? { cursor: "not-allowed" } : undefined}
          >
            {isSearching ? "搜索中…" : "搜索相似位置"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={removeLastPoint}
            disabled={selectedPoints.length === 0}
          >
            移除末点
          </button>
          <button type="button" className="secondary" onClick={clearAll}>
            清空
          </button>
        </div>
      </section>

      {savedCategories.length > 0 && (
        <section className="panel saved-categories-panel">
          <h2>已保存的查询类别</h2>
          <div className="category-list">
            {savedCategories.map((cat) => (
              <div key={cat.id} className="category-item">
                <button
                  type="button"
                  className="category-load-btn"
                  onClick={() => handleLoadCategory(cat.id)}
                >
                  <span className="category-name">{cat.name}</span>
                  <span className="category-count">{cat.points.length} 个点</span>
                </button>
                <button
                  type="button"
                  className="category-del-btn"
                  onClick={() => handleDeleteCategory(cat.id)}
                  title="删除此类别"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel info-panel">
        <h2>查询点 {selectedPoints.length > 0 ? `（${selectedPoints.length} 个，取平均向量）` : ""}</h2>
        {selectedPoints.length === 0 ? (
          <div className="empty">点击地图添加查询点，可多次点击累积，检索使用全部查询点的平均 embedding。</div>
        ) : (
          <>
            <SaveCategoryForm onSave={handleSaveCategory} />
            <div className="point-list">
              {selectedPoints.map((p, idx) => (
                <div key={idx} className="point-list-item">
                  <span className="point-badge" style={p.pending ? { opacity: 0.5 } : undefined}>
                    Q{idx + 1}
                  </span>
                  <span className="mono" style={p.pending ? { opacity: 0.5 } : undefined}>
                    {p.lat.toFixed(5)}, {p.lng.toFixed(5)}
                    {p.pending && <span style={{ marginLeft: 6, fontSize: "0.85em" }}>获取中…</span>}
                  </span>
                </div>
              ))}
            </div>
            {avgEmbedding && (
              <>
                <div className="info-card" style={{ marginTop: 8 }}>
                  <span>平均 Embedding</span>
                  <strong>{avgEmbedding.length} dims</strong>
                </div>
                <pre className="embed-preview">{embeddingPreview}</pre>
              </>
            )}
          </>
        )}
      </section>

      <section className="panel results-panel">
        <div className="results-summary">
          {searchResults.length
            ? `共 ${searchResults.length} 个结果，搜索半径 ${searchRadiusKm} km，${useCoarseSearch ? "粗糙索引" : "精细索引"}`
            : "暂无结果"}
        </div>
        <ResultList results={searchResults} activeRank={activeRank} onSelect={handleSelectResult} />
      </section>

      <StatusPanel
        mapReady={mapReady}
        apiStatus={apiStatus}
        requestStatus={requestStatus}
        errorMessage={errorMessage}
        onCheckHealth={onCheckHealth}
      />
    </>
  );
}

function ResultList({ results, activeRank, onSelect }) {
  if (!results.length) return <div className="empty">暂无结果</div>;

  return (
    <div className="result-list">
      {results.map((item) => (
        <button
          key={`${item.rank}-${item.lon}-${item.lat}`}
          type="button"
          className={`result-card${activeRank === item.rank ? " active" : ""}`}
          onClick={() => onSelect(item)}
        >
          <div className="result-head">
            <strong>#{item.rank}</strong>
            <span>{item.score.toFixed(6)}</span>
          </div>
          <div className="mono">lon={item.lon.toFixed(6)}, lat={item.lat.toFixed(6)}</div>
          <div className="mono">tile={String(item.tile_path ?? "")}</div>
          <div className="mono">row={item.row}, col={item.col}</div>
        </button>
      ))}
    </div>
  );
}

// ── SaveCategoryForm ─────────────────────────────────────────────────────────

function SaveCategoryForm({ onSave }) {
  const [name, setName] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onSave(trimmed);
    setName("");
  }

  return (
    <form className="save-category-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="输入类别名称…"
        value={name}
        maxLength={40}
        onChange={(e) => setName(e.target.value)}
      />
      <button type="submit" className="primary" disabled={!name.trim()}>
        保存为类别
      </button>
    </form>
  );
}

// ── scene state hook ──────────────────────────────────────────────────────────

/**
 * Encapsulates all state and handlers for the multi-point Point Query scene.
 * selectedPoints: Array<{ lat, lng, embedding }>
 * avgEmbedding: average of all selected point embeddings (or null)
 */
export function usePointQueryState(mapInstanceRef) {
  const [requestStatus, setRequestStatus] = useState({ kind: "warn", text: "等待操作" });
  const [errorMessage, setErrorMessage] = useState("无");
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useLocalStorage("pq.topK", 20);
  const [minDistanceM, setMinDistanceM] = useLocalStorage("pq.minDistanceM", 100);
  const [minScore, setMinScore] = useLocalStorage("pq.minScore", 0.9);
  const [searchRadiusKm, setSearchRadiusKm] = useLocalStorage("pq.searchRadiusKm", 5);
  const [useCoarseSearch, setUseCoarseSearch] = useLocalStorage("pq.useCoarseSearch", false);
  const [savedCategories, setSavedCategories] = useLocalStorage("pq.savedCategories", []);
  const [selectedPoints, setSelectedPoints] = useState([]); // [{ lat, lng, embedding }]
  const [searchResults, setSearchResults] = useState([]);
  const [activeRank, setActiveRank] = useState(null);
  const [activePopup, setActivePopup] = useState(null);
  const [queryPointPopup, setQueryPointPopup] = useState(null); // { idx, lat, lng }

  // Compute average embedding using only ready (non-pending) points
  const avgEmbedding = useMemo(() => {
    const ready = selectedPoints.filter((p) => !p.pending);
    if (!ready.length) return null;
    const dim = ready[0].embedding.length;
    const sum = new Float64Array(dim);
    for (const p of ready) {
      for (let i = 0; i < dim; i++) sum[i] += p.embedding[i];
    }
    return Array.from(sum, (v) => v / ready.length);
  }, [selectedPoints]);

  const activeResult = useMemo(
    () => searchResults.find((r) => r.rank === activeRank) ?? null,
    [searchResults, activeRank],
  );

  function clearResultsOnly() {
    setSearchResults([]);
    setActiveRank(null);
    setActivePopup(null);
  }

  function clearAll() {
    setSelectedPoints([]);
    clearResultsOnly();
    setRequestStatus({ kind: "warn", text: "已清空" });
    setErrorMessage("无");
  }

  function removeLastPoint() {
    setSelectedPoints((prev) => {
      const next = prev.slice(0, -1);
      if (next.length === 0) clearResultsOnly();
      return next;
    });
    setRequestStatus({ kind: "warn", text: "已移除末点" });
  }

  function handleSaveCategory(name) {
    const ready = selectedPoints.filter((p) => !p.pending);
    if (!name || !ready.length) return;
    const newCat = { id: Date.now().toString(), name, points: ready };
    setSavedCategories((prev) => [...prev, newCat]);
    setRequestStatus({ kind: "ok", text: `已保存类别「${name}」，共 ${ready.length} 个点` });
  }

  function handleLoadCategory(id) {
    const cat = savedCategories.find((c) => c.id === id);
    if (!cat) return;
    setSelectedPoints(cat.points);
    clearResultsOnly();
    setQueryPointPopup(null);
    setRequestStatus({ kind: "ok", text: `已加载类别「${cat.name}」，共 ${cat.points.length} 个点` });
    setErrorMessage("无");

    // 将地图视角移至该分类的点位范围
    const map = mapInstanceRef.current;
    if (map && cat.points.length > 0) {
      const latlngs = cat.points.map((p) => [p.lat, p.lng]);
      if (latlngs.length === 1) {
        map.setView(latlngs[0], Math.max(map.getZoom(), 14), { animate: true });
      } else {
        map.fitBounds(latlngs, { padding: [60, 60], animate: true });
      }
    }
  }

  function handleDeleteCategory(id) {
    setSavedCategories((prev) => prev.filter((c) => c.id !== id));
  }

  function handleRemovePoint(idx) {
    setSelectedPoints((prev) => {
      const next = [...prev];
      next.splice(idx, 1);
      if (next.length === 0) clearResultsOnly();
      return next;
    });
    setQueryPointPopup(null);
    setRequestStatus({ kind: "warn", text: "已删除查询点" });
  }

  function openQueryPointPopup(idx, lat, lng) {
    setQueryPointPopup({ idx, lat, lng });
  }

  async function handlePickPoint(latlng) {
    setRequestStatus({ kind: "warn", text: "查询点位 embedding..." });
    setErrorMessage("无");
    clearResultsOnly();

    // 立即将点位置添加到地图（pending 状态），给用户即时反馈
    setSelectedPoints((prev) => [
      ...prev,
      { lat: latlng.lat, lng: latlng.lng, embedding: null, pending: true },
    ]);

    try {
      const payload = await apiRequest("/embedding/by-point", {
        lon: latlng.lng,
        lat: latlng.lat,
      });
      setSelectedPoints((prev) => {
        const idx = prev.findLastIndex(
          (p) => p.pending && p.lat === latlng.lat && p.lng === latlng.lng,
        );
        if (idx === -1) return prev; // 点已被清除
        const next = [...prev];
        next[idx] = { lat: latlng.lat, lng: latlng.lng, embedding: payload.embedding };
        const readyCount = next.filter((p) => !p.pending).length;
        setRequestStatus({
          kind: "ok",
          text: `已添加第 ${readyCount} 个查询点，共 ${payload.embedding.length} 维`,
        });
        return next;
      });
    } catch (error) {
      // 失败则移除挂起的点
      setSelectedPoints((prev) => {
        const idx = prev.findLastIndex(
          (p) => p.pending && p.lat === latlng.lat && p.lng === latlng.lng,
        );
        if (idx === -1) return prev;
        const next = [...prev];
        next.splice(idx, 1);
        return next;
      });
      setRequestStatus({ kind: "error", text: "embedding 查询失败" });
      setErrorMessage(getErrorMessage(error));
    }
  }

  async function handleSearch() {
    if (!avgEmbedding) return;

    if (!Number.isInteger(topK) || topK < 1 || topK > 1000) {
      setRequestStatus({ kind: "error", text: "搜索参数无效" });
      setErrorMessage("Top K 必须是 1 到 1000 之间的整数。");
      return;
    }
    if (!Number.isFinite(minDistanceM) || minDistanceM < 0) {
      setRequestStatus({ kind: "error", text: "搜索参数无效" });
      setErrorMessage("最小距离必须大于等于 0。");
      return;
    }
    if (!Number.isFinite(minScore) || minScore < 0 || minScore > 1) {
      setRequestStatus({ kind: "error", text: "搜索参数无效" });
      setErrorMessage("相似度阈值必须在 0 到 1 之间。");
      return;
    }
    if (!Number.isFinite(searchRadiusKm) || searchRadiusKm <= 0) {
      setRequestStatus({ kind: "error", text: "搜索参数无效" });
      setErrorMessage("搜索半径必须大于 0。");
      return;
    }

    // 以所有 ready 点的中心为原点，构建正方形 bbox（边长 = 2 × 半径）
    const readyPoints = selectedPoints.filter((p) => !p.pending);
    const centerLat = readyPoints.reduce((s, p) => s + p.lat, 0) / readyPoints.length;
    const centerLng = readyPoints.reduce((s, p) => s + p.lng, 0) / readyPoints.length;
    const halfM = searchRadiusKm * 1000;
    const dLat = halfM / 111320;
    const dLng = halfM / (111320 * Math.cos((centerLat * Math.PI) / 180));
    const bbox = [
      Number((centerLng - dLng).toFixed(6)),
      Number((centerLat - dLat).toFixed(6)),
      Number((centerLng + dLng).toFixed(6)),
      Number((centerLat + dLat).toFixed(6)),
    ];

    const requestBody = {
      embedding: avgEmbedding,
      search_mode: useCoarseSearch ? "coarse" : "fine",
      top_k: topK,
      min_distance_m: minDistanceM,
      min_score: minScore,
      bbox,
    };

    setRequestStatus({ kind: "warn", text: "检索相似位置中..." });
    setErrorMessage("无");
    setIsSearching(true);

    try {
      const payload = await apiRequest("/search/by-embedding", requestBody);
      setSearchResults(payload.results || []);
      setRequestStatus({
        kind: "ok",
        text: `检索完成，${selectedPoints.length} 个查询点均值，${useCoarseSearch ? "粗糙索引" : "精细索引"}返回 ${payload.result_count} 个结果`,
      });
    } catch (error) {
      clearResultsOnly();
      setRequestStatus({ kind: "error", text: "检索失败" });
      setErrorMessage(getErrorMessage(error));
    } finally {
      setIsSearching(false);
    }
  }

  function handleSelectResult(item, openPopup = false) {
    setActiveRank(item.rank);
    if (openPopup) {
      setActivePopup({ lat: item.lat, lon: item.lon, rank: item.rank, score: item.score });
    } else {
      setActivePopup(null);
    }
  }

  return {
    requestStatus,
    errorMessage,
    topK, setTopK,
    minDistanceM, setMinDistanceM,
    minScore, setMinScore,
    searchRadiusKm, setSearchRadiusKm,
    useCoarseSearch, setUseCoarseSearch,
    selectedPoints,
    avgEmbedding,
    searchResults,
    activeRank,
    activeResult,
    activePopup,
    queryPointPopup,
    isSearching,
    savedCategories,
    handlePickPoint,
    handleSearch,
    handleSelectResult,
    handleSaveCategory,
    handleLoadCategory,
    handleDeleteCategory,
    handleRemovePoint,
    openQueryPointPopup,
    removeLastPoint,
    clearAll,
  };
}
