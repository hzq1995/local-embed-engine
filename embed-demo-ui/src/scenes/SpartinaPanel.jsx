import { useState } from "react";
import { statusClass } from "../utils/geo";
import spartinaImg from "../../data/互花米草-v3.png";

/**
 * Hook – manages spartina scene state.
 */
export function useSpartinaState() {
  const [isExtracting, setIsExtracting] = useState(false);
  const [maskVisible, setMaskVisible] = useState(false);

  function handleExtract() {
    if (isExtracting) return;
    setIsExtracting(true);
    setTimeout(() => {
      setMaskVisible(true);
      setIsExtracting(false);
    }, 1000);
  }

  function clearMask() {
    setMaskVisible(false);
  }

  return { isExtracting, maskVisible, handleExtract, clearMask };
}

/**
 * Sidebar panel for the "Spartina" (互花米草) scene.
 *
 * Props:
 *   mapReady, apiStatus, onCheckHealth, sceneState
 */
export default function SpartinaPanel({ mapReady, apiStatus, onCheckHealth, sceneState }) {
  const { isExtracting, maskVisible, handleExtract, clearMask } = sceneState;

  return (
    <>
      <section className="panel controls">
        <p className="desc">
          互花米草（Spartina alterniflora）入侵分布提取。加载遥感影像并点击「提取」，系统将自动识别并高亮显示互花米草分布区域。
        </p>

        <div className="spartina-diagram">
          <img
            src={spartinaImg}
            alt="互花米草分布示意图"
            style={{ width: "100%", borderRadius: "6px", display: "block" }}
          />
          <p style={{ fontSize: "0.75rem", color: "#888", textAlign: "center", marginTop: "4px" }}>
            互花米草分布示意图
          </p>
        </div>

        <div className="button-row">
          <button
            type="button"
            className="primary"
            onClick={handleExtract}
            disabled={!mapReady || isExtracting}
            style={isExtracting ? { cursor: "not-allowed" } : undefined}
          >
            {isExtracting ? "提取中…" : "提取"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={clearMask}
            disabled={!maskVisible}
          >
            清空
          </button>
        </div>

        {maskVisible && (
          <div className="spartina-result-hint">
            <span className="spartina-result-dot" />
            已识别互花米草分布区域
          </div>
        )}
      </section>

      <section className="panel status-panel">
        <h2>状态</h2>
        <div className="status-grid">
          <div>
            <span className="label">API 连通性</span>
            <div className={statusClass(apiStatus?.kind ?? "warn")}>
              {apiStatus?.text ?? "未检查"}
            </div>
          </div>
        </div>
        <button type="button" className="secondary" onClick={onCheckHealth}>
          检查 API
        </button>
      </section>
    </>
  );
}
