import L from "leaflet";

/** Creates a circular L.divIcon with a centered label. */
export function makeCircleIcon(label, fillColor, strokeColor, radius) {
  const size = radius * 2;
  return L.divIcon({
    className: "",
    html: `<div style="width:${size}px;height:${size}px;border-radius:50%;background:${fillColor};border:2px solid ${strokeColor};display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,0.4);cursor:pointer;">${label}</div>`,
    iconSize: [size, size],
    iconAnchor: [radius, radius],
    popupAnchor: [0, -radius - 4],
  });
}

export function statusClass(kind) {
  if (kind === "ok") return "status ok";
  if (kind === "error") return "status error";
  return "status warn";
}
