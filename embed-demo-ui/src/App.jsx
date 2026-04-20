import { useCallback, useRef, useState } from "react";
import { MapContainer, TileLayer, ZoomControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";

import { checkHealth as apiCheckHealth, getErrorMessage } from "./utils/api";
import TabBar from "./components/TabBar";
import { MapController, SceneViewRestorer } from "./components/MapController";

import PointQueryPanel, { usePointQueryState } from "./scenes/PointQueryPanel";
import PointQueryMapLayers from "./scenes/PointQueryMapLayers";
import RegionClusterPanel, { useRegionClusterState } from "./scenes/RegionClusterPanel";
import RegionClusterMapLayers from "./scenes/RegionClusterMapLayers";
import SpartinaPanel, { useSpartinaState } from "./scenes/SpartinaPanel";
import SpartinaMapOverlay from "./scenes/SpartinaMapOverlay";

const TIANDITU_URL =
  "http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=fd7d577a54bb54358f1f243e3c0646e0";

const defaultCenter = [29.8683, 121.544];
const defaultZoom = 17;

const SCENES = [
  { id: "point-query", label: "选点查询" },
  { id: "region-cluster", label: "区域聚类" },
  { id: "spartina", label: "互花米草" },
];

// ── per-scene map view persistence helpers ────────────────────────────────────
function sceneViewKey(sceneId) {
  return `mapView.${sceneId}`;
}

function readStoredView(sceneId) {
  try {
    const stored = localStorage.getItem(sceneViewKey(sceneId));
    if (stored) return JSON.parse(stored);
  } catch { /* ignore */ }
  return null;
}

function saveCurrentView(map, sceneId) {
  if (!map) return;
  const center = map.getCenter();
  const zoom = map.getZoom();
  try {
    localStorage.setItem(
      sceneViewKey(sceneId),
      JSON.stringify({ lat: center.lat, lng: center.lng, zoom }),
    );
  } catch { /* ignore quota errors */ }
}
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const mapInstanceRef = useRef(null);
  const mapClickHandlerRef = useRef(null);

  const [mapReady, setMapReady] = useState(false);
  const [activeScene, setActiveScene] = useState("point-query");
  const [apiStatus, setApiStatus] = useState({ kind: "warn", text: "未检查" });
  // targetView drives SceneViewRestorer; each scene switch sets a new object to trigger useEffect
  const [targetView, setTargetView] = useState(null);

  const pointQueryState = usePointQueryState(mapInstanceRef);
  const regionClusterState = useRegionClusterState(mapInstanceRef);
  const spartinaState = useSpartinaState();

  async function handleCheckHealth() {
    setApiStatus({ kind: "warn", text: "检查中..." });
    try {
      const data = await apiCheckHealth();
      const suffix = data.index_loaded ? `，index=${data.index_type || "unknown"}` : "";
      setApiStatus({ kind: "ok", text: `可用，vectors=${data.vector_count}${suffix}` });
    } catch (error) {
      setApiStatus({ kind: "error", text: "不可达：" + getErrorMessage(error) });
    }
  }

  const handleMapReady = useCallback(() => {
    setMapReady(true);
    handleCheckHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleMapClick = useCallback((latlng) => {
    mapClickHandlerRef.current?.(latlng);
  }, []);

  mapClickHandlerRef.current =
    activeScene === "point-query" ? pointQueryState.handlePickPoint : null;
  // spartina and region-cluster scenes don't use map click

  function handleSceneSwitch(id) {
    // Save current map view for the scene we're leaving
    saveCurrentView(mapInstanceRef.current, activeScene);
    // Restore (or use default) view for the scene we're entering
    const stored = readStoredView(id);
    setTargetView(stored ?? { lat: defaultCenter[0], lng: defaultCenter[1], zoom: defaultZoom });
    setActiveScene(id);
  }

  const sharedPanelProps = { mapReady, apiStatus, onCheckHealth: handleCheckHealth };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="panel intro">
          <h1>影像嵌入分析平台</h1>
          <p>自然资源部</p>
        </header>

        <TabBar scenes={SCENES} activeScene={activeScene} onSwitch={handleSceneSwitch} />

        {activeScene === "point-query" && (
          <PointQueryPanel
            {...sharedPanelProps}
            sceneState={pointQueryState}
          />
        )}
        {activeScene === "region-cluster" && (
          <RegionClusterPanel
            {...sharedPanelProps}
            sceneState={regionClusterState}
          />
        )}
        {activeScene === "spartina" && (
          <SpartinaPanel
            {...sharedPanelProps}
            sceneState={spartinaState}
          />
        )}
      </aside>

      <main className="map-wrap">
        <MapContainer
          center={readStoredView("point-query")?.lat
            ? [readStoredView("point-query").lat, readStoredView("point-query").lng]
            : defaultCenter}
          zoom={readStoredView("point-query")?.zoom ?? defaultZoom}
          className="map"
          zoomControl={false}
          attributionControl={true}
        >
          <TileLayer
            url={TIANDITU_URL}
            attribution='&copy; <a href="https://www.tianditu.gov.cn" target="_blank">天地图</a>'
            maxZoom={18}
          />
          <ZoomControl position="topright" />

          <SceneViewRestorer targetView={targetView} />
          <MapController
            mapInstanceRef={mapInstanceRef}
            onMapReady={handleMapReady}
            onMapClick={handleMapClick}
          />

          {activeScene === "point-query" && (
            <PointQueryMapLayers sceneState={pointQueryState} />
          )}
          {activeScene === "region-cluster" && (
            <RegionClusterMapLayers clusterResult={regionClusterState.clusterResult} />
          )}
        </MapContainer>

        {activeScene === "spartina" && (
          <SpartinaMapOverlay maskVisible={spartinaState.maskVisible} />
        )}
      </main>
    </div>
  );
}
