import { statusClass } from "../utils/geo";

/**
 * Shared status panel shown in every scene's sidebar.
 *
 * Props:
 *   mapReady: boolean
 *   apiStatus: { kind: string, text: string }
 *   requestStatus: { kind: string, text: string }
 *   errorMessage: string
 *   onCheckHealth: () => void
 */
export default function StatusPanel({ mapReady, apiStatus, requestStatus, errorMessage, onCheckHealth }) {
  return (
    <section className="panel status-panel">
      <h2>状态</h2>
      <div className="status-grid">
        <div>
          <span className="label">地图状态</span>
          <div className={statusClass(mapReady ? "ok" : "warn")}>{mapReady ? "已就绪" : "加载中"}</div>
        </div>
        <div>
          <span className="label">API 连通性</span>
          <div className={statusClass(apiStatus.kind)}>{apiStatus.text}</div>
        </div>
        <div>
          <span className="label">最近请求</span>
          <div className={statusClass(requestStatus.kind)}>{requestStatus.text}</div>
        </div>
        <div>
          <span className="label">错误信息</span>
          <div className="mono error-box">{errorMessage}</div>
        </div>
      </div>
      <button type="button" className="secondary" onClick={onCheckHealth}>
        检查 API
      </button>
    </section>
  );
}
