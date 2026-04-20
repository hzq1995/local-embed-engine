import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import { CLUSTER_PALETTE } from "./RegionClusterPanel";

/**
 * Renders cluster results as a seamless grid mask on the Leaflet map.
 *
 * Each backend grid cell is drawn as a filled rectangle.  The pixel position
 * and size of each cell is derived from the effective_bbox corners and the
 * grid dimensions, so the cells always line up with the actual data extent
 * regardless of current zoom/pan.
 *
 * Props:
 *   clusterResult: {
 *     labels, gridRows, gridCols, gridRowIndices, gridColIndices,
 *     effectiveBbox: [minLon, minLat, maxLon, maxLat]
 *   } | null
 */
export default function RegionClusterMapLayers({ clusterResult }) {
  const map = useMap();
  const canvasRef = useRef(null);

  useEffect(() => {
    // Mount on the map container element (never transformed by Leaflet).
    if (!canvasRef.current) {
      const canvas = document.createElement("canvas");
      canvas.style.position = "absolute";
      canvas.style.top = "0";
      canvas.style.left = "0";
      canvas.style.pointerEvents = "none";
      canvas.style.zIndex = "500";
      map.getContainer().appendChild(canvas);
      canvasRef.current = canvas;
    }

    const canvas = canvasRef.current;

    function redraw() {
      const size = map.getSize();
      const dpr = window.devicePixelRatio || 1;
      const newW = size.x * dpr;
      const newH = size.y * dpr;

      if (canvas.width !== newW || canvas.height !== newH) {
        canvas.width = newW;
        canvas.height = newH;
        canvas.style.width = `${size.x}px`;
        canvas.style.height = `${size.y}px`;
      }

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.x, size.y);

      if (!clusterResult) return;

      const { labels, gridRows, gridCols, gridRowIndices, gridColIndices, effectiveBbox } =
        clusterResult;

      if (!gridRows || !gridCols || !effectiveBbox?.length) return;

      const [minLon, minLat, maxLon, maxLat] = effectiveBbox;

      // Pixel coordinates of the data extent's NW and SE corners.
      // Using latLngToContainerPoint gives coordinates relative to the map
      // container element — exactly where our canvas is anchored.
      const nw = map.latLngToContainerPoint([maxLat, minLon]);
      const se = map.latLngToContainerPoint([minLat, maxLon]);

      const totalW = se.x - nw.x;
      const totalH = se.y - nw.y;
      const cellW = totalW / gridCols;
      const cellH = totalH / gridRows;

      // +0.5px overlap prevents hairline gaps between adjacent cells at
      // sub-pixel boundaries on high-DPR screens.
      const drawW = cellW + 0.5;
      const drawH = cellH + 0.5;

      ctx.globalAlpha = 0.72;
      for (let i = 0; i < labels.length; i++) {
        ctx.fillStyle = CLUSTER_PALETTE[labels[i] % CLUSTER_PALETTE.length];
        ctx.fillRect(
          nw.x + gridColIndices[i] * cellW,
          nw.y + gridRowIndices[i] * cellH,
          drawW,
          drawH,
        );
      }
      ctx.globalAlpha = 1;
    }

    redraw();

    map.on("move zoom resize viewreset", redraw);
    return () => {
      map.off("move zoom resize viewreset", redraw);
    };
  }, [map, clusterResult]);

  // Cleanup canvas on unmount
  useEffect(() => {
    return () => {
      if (canvasRef.current) {
        canvasRef.current.remove();
        canvasRef.current = null;
      }
    };
  }, []);

  return null;
}
