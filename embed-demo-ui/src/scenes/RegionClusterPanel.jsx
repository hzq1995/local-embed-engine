import { useState } from "react";
import { apiRequest, getErrorMessage } from "../utils/api";
import { kMeans, assignToCentroids } from "../utils/kmeans";
import StatusPanel from "../components/StatusPanel";
import { useLocalStorage } from "../utils/useLocalStorage";

// 10-color palette for cluster labels
export const CLUSTER_PALETTE = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
];

/**
 * Sidebar panel for the "Region Cluster" scene.
 *
 * Props:
 *   mapInstanceRef, mapReady, apiStatus, onCheckHealth, sceneState
 */
export default function RegionClusterPanel({ mapReady, apiStatus, onCheckHealth, sceneState }) {
  const {
    kValue, setKValue,
    totalSamples, setTotalSamples,
    requestStatus, errorMessage,
    isAnalyzing,
    clusterResult,
    handleAnalyze,
    clearResult,
  } = sceneState;

  return (
    <>
      <section className="panel controls">
        <p className="desc">
          为感兴趣区域快速构建Embedding聚类图层，支持一键矢量化。
        </p>

        <div className="control-row">
          <div className="field">
            <label htmlFor="rc-k">聚类数 K</label>
            <input
              id="rc-k"
              type="number"
              min="2"
              max="20"
              value={kValue}
              onChange={(e) => setKValue(Math.max(2, Math.min(20, Number(e.target.value))))}
            />
          </div>
          <div className="field">
            <label htmlFor="rc-samples">采样数上限</label>
            <input
              id="rc-samples"
              type="number"
              min="100"
              max="50000"
              step="500"
              value={totalSamples}
              onChange={(e) =>
                setTotalSamples(Math.max(100, Math.min(50000, Number(e.target.value))))
              }
            />
          </div>
        </div>

        <div className="button-row">
          <button
            type="button"
            className="primary"
            onClick={handleAnalyze}
            disabled={!mapReady || isAnalyzing}
            style={isAnalyzing ? { cursor: "not-allowed" } : undefined}
          >
            {isAnalyzing ? "分析中…" : "分析当前视野"}
          </button>
          <button type="button" className="secondary" onClick={clearResult} disabled={!clusterResult}>
            清空
          </button>
        </div>
      </section>

      <StatusPanel
        mapReady={mapReady}
        apiStatus={apiStatus}
        requestStatus={requestStatus}
        errorMessage={errorMessage}
        onCheckHealth={onCheckHealth}
      />

      {clusterResult && (
        <section className="panel legend-panel">
          <h2>聚类图例</h2>
          <div className="legend-info">
            网格点总数：{clusterResult.count}（其中 {clusterResult.clusterSampleSize} 个用于训练），聚类数：{kValue}
          </div>
          <div className="legend-list">
            {Array.from({ length: kValue }, (_, i) => {
              const count = clusterResult.labels.filter((l) => l === i).length;
              return (
                <div key={i} className="legend-item">
                  <span
                    className="legend-dot"
                    style={{ background: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length] }}
                  />
                  <span>
                    类 {i + 1}：{count} 个点
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </>
  );
}

// ── scene state hook ──────────────────────────────────────────────────────────

/**
 * Encapsulates all state and handlers for the Region Cluster scene.
 */
export function useRegionClusterState(mapInstanceRef) {
  const [kValue, setKValue] = useLocalStorage("rc.kValue", 5);
  const [totalSamples, setTotalSamples] = useLocalStorage("rc.totalSamples", 5000);
  const [requestStatus, setRequestStatus] = useState({ kind: "warn", text: "等待操作" });
  const [errorMessage, setErrorMessage] = useState("无");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [clusterResult, setClusterResult] = useState(null);

  function clearResult() {
    setClusterResult(null);
    setRequestStatus({ kind: "warn", text: "已清空" });
    setErrorMessage("无");
  }

  async function handleAnalyze() {
    if (!mapInstanceRef.current) return;

    const bounds = mapInstanceRef.current.getBounds();
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    const bbox = [
      Number(sw.lng.toFixed(6)),
      Number(sw.lat.toFixed(6)),
      Number(ne.lng.toFixed(6)),
      Number(ne.lat.toFixed(6)),
    ];

    setRequestStatus({ kind: "warn", text: `从后端网格采样（上限 ${totalSamples}）...` });
    setErrorMessage("无");
    setClusterResult(null);
    setIsAnalyzing(true);

    let bboxData;
    try {
      bboxData = await apiRequest("/embedding/by-bbox", { bbox, total_samples: totalSamples });
    } catch (error) {
      setRequestStatus({ kind: "error", text: "获取区域 embedding 失败" });
      setErrorMessage(getErrorMessage(error));
      setIsAnalyzing(false);
      return;
    }

    if (!bboxData.count || bboxData.count === 0) {
      setRequestStatus({ kind: "warn", text: "当前视野内无有效数据点" });
      setErrorMessage("无");
      setIsAnalyzing(false);
      return;
    }

    // Use up to 1000 evenly-spaced grid points for k-means training,
    // then assign labels to ALL points using the resulting centroids.
    const CLUSTER_BUDGET = 1000;
    const n = bboxData.count;
    let clusterEmbeddings;
    let clusterSampleSize;
    if (n <= CLUSTER_BUDGET) {
      clusterEmbeddings = bboxData.embeddings;
      clusterSampleSize = n;
    } else {
      // Evenly-spaced indices → good spatial coverage across the grid
      const step = n / CLUSTER_BUDGET;
      clusterEmbeddings = Array.from({ length: CLUSTER_BUDGET }, (_, i) =>
        bboxData.embeddings[Math.floor(i * step)],
      );
      clusterSampleSize = CLUSTER_BUDGET;
    }

    setRequestStatus({
      kind: "warn",
      text: `正在聚类（训练集 ${clusterSampleSize} 个，K=${kValue}）...`,
    });

    const { centroids } = kMeans(clusterEmbeddings, kValue);
    // Assign ALL grid points to nearest centroid
    const labels = assignToCentroids(bboxData.embeddings, centroids);

    setClusterResult({
      lons: bboxData.lons,
      lats: bboxData.lats,
      labels,
      centroids,
      count: n,
      clusterSampleSize,
      gridRows: bboxData.grid_rows,
      gridCols: bboxData.grid_cols,
      gridRowIndices: bboxData.grid_row_indices,
      gridColIndices: bboxData.grid_col_indices,
      effectiveBbox: bboxData.effective_bbox,
    });
    setRequestStatus({
      kind: "ok",
      text: `完成：${n} 个格点，${kValue} 个聚类`,
    });
    setIsAnalyzing(false);
  }

  return {
    kValue, setKValue,
    totalSamples, setTotalSamples,
    requestStatus, errorMessage,
    isAnalyzing,
    clusterResult,
    handleAnalyze,
    clearResult,
  };
}
