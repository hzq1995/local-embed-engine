import { useMemo } from "react";
import { Marker } from "react-leaflet";
import { MapMover, PopupManager, QueryPointPopupManager } from "../components/MapController";
import { makeCircleIcon } from "../utils/geo";

/**
 * Map layer for the Point Query scene.
 * Renders one numbered Q marker per selected point, result markers, and camera helpers.
 *
 * Props: sceneState (from usePointQueryState)
 */
export default function PointQueryMapLayers({ sceneState }) {
  const {
    selectedPoints, searchResults, activeRank, activeResult,
    activePopup, handleSelectResult,
    queryPointPopup, handleRemovePoint, openQueryPointPopup,
  } = sceneState;

  return (
    <>
      <MapMover searchResults={searchResults} activeResult={activeResult} />
      <PopupManager activePopup={activePopup} />
      <QueryPointPopupManager queryPointPopup={queryPointPopup} onRemove={handleRemovePoint} />

      {/* One marker per query point, labeled Q1, Q2, Q3 … */}
      {selectedPoints.map((p, idx) => (
        <Marker
          key={`qpt-${idx}-${p.lat}-${p.lng}`}
          position={[p.lat, p.lng]}
          icon={
            p.pending
              ? makeCircleIcon(`Q${idx + 1}`, "#9e9e9e", "#666666", 10)
              : makeCircleIcon(`Q${idx + 1}`, "#e07b39", "#b85c1a", 10)
          }
          eventHandlers={
            p.pending
              ? undefined
              : { click: () => openQueryPointPopup(idx, p.lat, p.lng) }
          }
          zIndexOffset={1000 + idx}
        />
      ))}

      {searchResults.map((item) => (
        <Marker
          key={`result-${item.rank}-${item.lon}-${item.lat}`}
          position={[item.lat, item.lon]}
          icon={makeCircleIcon(
            String(item.rank),
            activeRank === item.rank ? "#1a72c7" : "#5aabf5",
            activeRank === item.rank ? "#0d4890" : "#2d7ac7",
            activeRank === item.rank ? 11 : 9,
          )}
          eventHandlers={{
            click: () => handleSelectResult(item, true),
          }}
          zIndexOffset={activeRank === item.rank ? 500 : 0}
        />
      ))}
    </>
  );
}
