import { useEffect, useRef, useState } from "react";
import { fromUrl } from "geotiff";
import tifUrl from "../../data/spartina_v1.tif?url";
import maskUrl from "../../data/spartina_mask_v1_removed_box_component.png";

/**
 * Full-map overlay for the Spartina scene.
 * Covers the map container (inset: 5%) and renders the GeoTIFF on a <canvas>.
 * When maskVisible=true, overlays the mask PNG as a red-tinted canvas layer
 * with identical intrinsic dimensions to the TIF canvas → pixel-perfect alignment.
 *
 * Props:
 *   maskVisible: boolean
 */
export default function SpartinaMapOverlay({ maskVisible }) {
  const canvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const [tifError, setTifError] = useState(null);
  const [tifLoaded, setTifLoaded] = useState(false);
  const [tifDims, setTifDims] = useState({ w: 0, h: 0 });

  // Load and render TIF
  useEffect(() => {
    let cancelled = false;

    async function loadTif() {
      try {
        const tiff = await fromUrl(tifUrl);
        const image = await tiff.getImage();
        const width = image.getWidth();
        const height = image.getHeight();
        const samplesPerPixel = image.getSamplesPerPixel();

        const rasters = await image.readRasters({ interleave: true });

        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(width, height);
        const pixelCount = width * height;

        if (samplesPerPixel >= 3) {
          for (let i = 0; i < pixelCount; i++) {
            const base = i * 4;
            const rBase = i * samplesPerPixel;
            imgData.data[base]     = clamp(rasters[rBase]);
            imgData.data[base + 1] = clamp(rasters[rBase + 1]);
            imgData.data[base + 2] = clamp(rasters[rBase + 2]);
            imgData.data[base + 3] = samplesPerPixel >= 4 ? clamp(rasters[rBase + 3]) : 255;
          }
        } else {
          for (let i = 0; i < pixelCount; i++) {
            const base = i * 4;
            const v = clamp(rasters[i]);
            imgData.data[base] = imgData.data[base + 1] = imgData.data[base + 2] = v;
            imgData.data[base + 3] = 255;
          }
        }

        ctx.putImageData(imgData, 0, 0);
        setTifDims({ w: width, h: height });
        setTifLoaded(true);
      } catch (err) {
        if (!cancelled) {
          console.error("[SpartinaMapOverlay] Failed to load TIF:", err);
          setTifError(err.message ?? String(err));
        }
      }
    }

    loadTif();
    return () => { cancelled = true; };
  }, []);

  // Draw red-tinted mask onto mask canvas when maskVisible switches on
  useEffect(() => {
    if (!maskVisible || !tifLoaded) return;
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const img = new Image();
    img.onload = () => {
      const ctx = maskCanvas.getContext("2d");
      ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      // Stretch mask PNG to exact TIF pixel dimensions so they align
      ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);

      const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const a = data[i + 3];
        if (a > 0) {
          data[i]     = 255;                      // R – red
          data[i + 1] = 70;                       // G
          data[i + 2] = 70;                       // B
          data[i + 3] = Math.round(a * 0.6);      // 60% opacity
        }
      }
      ctx.putImageData(imageData, 0, 0);
    };
    img.src = maskUrl;
  }, [maskVisible, tifLoaded]);

  return (
    <div className="spartina-overlay-wrap">
      {tifError ? (
        <div className="spartina-tif-error">
          <span>影像加载失败：{tifError}</span>
        </div>
      ) : (
        <div className="spartina-canvas-wrap">
          {/* TIF base layer */}
          <canvas
            ref={canvasRef}
            className="spartina-canvas"
            title="互花米草遥感影像"
          />
          {/* Mask layer – same intrinsic size as TIF canvas → same CSS layout → aligned */}
          {tifLoaded && (
            <canvas
              ref={maskCanvasRef}
              className="spartina-canvas spartina-mask-canvas"
              width={tifDims.w}
              height={tifDims.h}
              style={{ display: maskVisible ? "block" : "none" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

function clamp(v) {
  if (v === undefined || v === null || Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(255, Math.round(Number(v))));
}
