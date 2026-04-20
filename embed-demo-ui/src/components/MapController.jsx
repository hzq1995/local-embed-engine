import { useEffect, useRef } from "react";
import L from "leaflet";
import { useMap, useMapEvents } from "react-leaflet";

/**
 * Registers the map click handler and exposes the Leaflet map instance.
 * Must be rendered inside a <MapContainer>.
 *
 * Props:
 *   mapInstanceRef: React.MutableRefObject
 *   onMapReady: () => void
 *   onMapClick: ({ lat, lng }) => void
 */
export function MapController({ mapInstanceRef, onMapReady, onMapClick }) {
  const map = useMap();

  useEffect(() => {
    mapInstanceRef.current = map;
    onMapReady();
  }, [map, mapInstanceRef, onMapReady]);

  useMapEvents({
    click(e) {
      onMapClick({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });

  return null;
}

/**
 * Restores the map view (center + zoom) when targetView changes.
 * Used to persist and recover per-scene map positions.
 *
 * Props:
 *   targetView: { lat, lng, zoom } | null
 */
export function SceneViewRestorer({ targetView }) {
  const map = useMap();

  useEffect(() => {
    if (!targetView) return;
    map.setView([targetView.lat, targetView.lng], targetView.zoom, { animate: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetView]);

  return null;
}

/**
 * Pans/zooms the map in response to search result changes.
 *
 * Props:
 *   searchResults: Array<{ lat, lon }>
 *   activeResult: { lat, lon } | null
 */
export function MapMover({ searchResults, activeResult }) {
  const map = useMap();

  useEffect(() => {
    if (!searchResults.length) return;
    const latlngs = searchResults.map((item) => [item.lat, item.lon]);
    map.fitBounds(latlngs, { padding: [40, 40] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchResults]);

  useEffect(() => {
    if (!activeResult) return;
    map.panTo([activeResult.lat, activeResult.lon]);
    if (map.getZoom() < 14) map.setZoom(14);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeResult]);

  return null;
}

/**
 * Manages an L.popup imperatively, driven by the activePopup state object.
 *
 * Props:
 *   activePopup: { lat, lon, rank, score } | null
 */
export function PopupManager({ activePopup }) {
  const map = useMap();
  const popupRef = useRef(null);

  useEffect(() => {
    if (!activePopup) {
      popupRef.current?.close();
      return;
    }

    if (!popupRef.current) {
      popupRef.current = L.popup({ offset: [0, -8] });
    }

    const { lat, lon, rank, score } = activePopup;
    const div = document.createElement("div");
    div.style.minWidth = "160px";
    div.innerHTML = `<strong>#${rank}</strong><br/>score: ${score.toFixed(6)}<br/>lon: ${lon.toFixed(6)}<br/>lat: ${lat.toFixed(6)}`;
    popupRef.current.setLatLng([lat, lon]).setContent(div).openOn(map);
  }, [activePopup, map]);

  return null;
}

/**
 * Manages a Leaflet popup for query point deletion.
 *
 * Props:
 *   queryPointPopup: { idx, lat, lng } | null
 *   onRemove: (idx) => void
 */
export function QueryPointPopupManager({ queryPointPopup, onRemove }) {
  const map = useMap();
  const popupRef = useRef(null);
  const onRemoveRef = useRef(onRemove);
  onRemoveRef.current = onRemove;

  useEffect(() => {
    if (!queryPointPopup) {
      popupRef.current?.close();
      return;
    }

    if (!popupRef.current) {
      popupRef.current = L.popup({ offset: [0, -12] });
    }

    const { idx, lat, lng } = queryPointPopup;
    const container = document.createElement("div");
    container.style.cssText = "min-width:150px;text-align:center;padding:2px 0;";

    const title = document.createElement("div");
    title.style.cssText = "font-weight:700;margin-bottom:4px;font-size:14px;";
    title.textContent = `Q${idx + 1}`;
    container.appendChild(title);

    const coords = document.createElement("div");
    coords.style.cssText = "font-size:12px;color:#777;margin-bottom:10px;";
    coords.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    container.appendChild(coords);

    const btn = document.createElement("button");
    btn.textContent = "删除此点";
    btn.style.cssText =
      "width:100%;padding:6px 12px;background:#c0392b;color:#fff;border:0;border-radius:8px;cursor:pointer;font-weight:700;font-size:13px;";
    btn.onclick = () => onRemoveRef.current(idx);
    container.appendChild(btn);

    popupRef.current.setLatLng([lat, lng]).setContent(container).openOn(map);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryPointPopup, map]);

  return null;
}
