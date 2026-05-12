"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CameraEntry } from "@/hooks/useCameraStore";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";

interface Props {
  camera: CameraEntry;
  onRemove: (id: string) => void;
  onDetection: (entry: Omit<EvidenceEntry, "id">) => void;
  onAlert: (msg: string) => void;
}

export default function CameraFeed({ camera, onRemove, onDetection, onAlert }: Props) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const streamRef  = useRef<MediaStream | null>(null);
  const frameRef   = useRef(0);
  const rafRef     = useRef(0);
  const [ready, setReady]       = useState(false);
  const [error, setError]       = useState("");
  const [alerting, setAlerting] = useState(false);
  const alertCooldown           = useRef(false);

  // Start webcam
  useEffect(() => {
    if (camera.type !== "webcam") return;
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then(stream => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      })
      .catch(err => setError(`Camera denied: ${err.message}`));
    return () => {
      streamRef.current?.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(rafRef.current);
    };
  }, [camera]);

  // Motion-based detection loop for webcam
  const lastFrameData = useRef<Uint8ClampedArray | null>(null);
  useEffect(() => {
    if (camera.type !== "webcam" || !ready) return;
    const video  = videoRef.current!;
    const canvas = canvasRef.current!;
    canvas.width  = 160;
    canvas.height = 120;
    const ctx = canvas.getContext("2d")!;

    let detected = false;

    function analyse() {
      if (!video.readyState || video.readyState < 2) { rafRef.current = requestAnimationFrame(analyse); return; }
      ctx.drawImage(video, 0, 0, 160, 120);
      const curr = ctx.getImageData(0, 0, 160, 120).data;

      if (lastFrameData.current) {
        let diff = 0;
        for (let i = 0; i < curr.length; i += 16) diff += Math.abs(curr[i] - lastFrameData.current[i]);
        const score = diff / (curr.length / 16);
        // score > 18 = significant motion (simulates a model detecting violence)
        // In production replace this with real ONNX model inference
        if (score > 18 && !alertCooldown.current) {
          alertCooldown.current = true;
          const confidence = Math.min(0.99, 0.80 + score / 300);
          triggerDetection(confidence);
          setTimeout(() => { alertCooldown.current = false; }, 20000); // 20s cooldown
        }
      }

      lastFrameData.current = new Uint8ClampedArray(curr);
      frameRef.current++;
      rafRef.current = requestAnimationFrame(analyse);
    }

    rafRef.current = requestAnimationFrame(analyse);
    return () => cancelAnimationFrame(rafRef.current);
  }, [ready, camera]);

  const triggerDetection = useCallback((confidence: number) => {
    setAlerting(true);
    setTimeout(() => setAlerting(false), 8000);

    // Capture thumbnail
    const video  = videoRef.current;
    const canvas = document.createElement("canvas");
    if (video) {
      canvas.width = 320; canvas.height = 240;
      canvas.getContext("2d")?.drawImage(video, 0, 0, 320, 240);
    }
    const thumbnail = canvas.toDataURL("image/jpeg", 0.6);

    const now = new Date();
    onDetection({
      cameraId: camera.id,
      cameraLabel: camera.label,
      timestamp: now.toLocaleString("en-IN"),
      isoTime: now.toISOString(),
      confidence,
      type: confidence > 0.88 ? "VIOLENCE" : "HARASSMENT",
      thumbnail,
    });
    onAlert(`[${camera.label}] Violence detected · ${(confidence * 100).toFixed(0)}% confidence`);
  }, [camera, onDetection, onAlert]);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  return (
    <div style={{
      position: "relative", background: "var(--bg3)", overflow: "hidden",
      outline: alerting ? "2px solid var(--danger)" : "1px solid var(--border)",
      animation: alerting ? "cam-alert-outline 0.5s infinite" : "none",
      minHeight: 180,
    }}>
      {/* Webcam video */}
      {camera.type === "webcam" && (
        <>
          <video ref={videoRef} autoPlay muted playsInline
            style={{ width: "100%", height: "100%", objectFit: "cover", display: error ? "none" : "block", minHeight: 180 }}/>
          <canvas ref={canvasRef} style={{ display: "none" }}/>
        </>
      )}

      {/* CCTV stream via img (MJPEG) */}
      {camera.type === "cctv" && camera.url && (
        <img src={camera.url} alt={camera.label} onLoad={() => setReady(true)} onError={() => setError("Stream unavailable")}
          style={{ width: "100%", height: "100%", objectFit: "cover", display: error ? "none" : "block", minHeight: 180 }}/>
      )}

      {/* Error state */}
      {error && (
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8 }}>
          <div style={{ fontSize: 28 }}>⚠️</div>
          <div style={{ ...mono, fontSize: 10, color: "var(--danger)", textAlign: "center", padding: "0 16px" }}>{error}</div>
        </div>
      )}

      {/* Alert flash */}
      {alerting && (
        <div style={{ position: "absolute", inset: 0, background: "rgba(255,34,68,0.12)", pointerEvents: "none" }}/>
      )}

      {/* Overlay */}
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", justifyContent: "space-between", padding: 8, pointerEvents: "none" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
            <div style={{ ...mono, fontSize: 9, background: "rgba(5,8,16,0.8)", color: "var(--text)", padding: "2px 7px", letterSpacing: 1 }}>
              {camera.label}
            </div>
            <div style={{ ...mono, fontSize: 8, background: "rgba(5,8,16,0.8)", color: camera.type === "webcam" ? "var(--accent)" : "var(--warning)", padding: "2px 6px", letterSpacing: 1, border: "1px solid currentColor", opacity: 0.8 }}>
              {camera.type.toUpperCase()}
            </div>
          </div>
          {alerting && (
            <div style={{ ...mono, fontSize: 8, background: "rgba(255,34,68,0.85)", color: "#fff", padding: "2px 8px", letterSpacing: 1, animation: "blink 0.5s infinite" }}>⚠ VIOLENCE DETECTED</div>
          )}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          {camera.lat && camera.lng && (
            <div style={{ ...mono, fontSize: 8, color: "rgba(200,218,240,0.6)", background: "rgba(5,8,16,0.7)", padding: "2px 5px" }}>
              {camera.lat.toFixed(4)}°N · {camera.lng.toFixed(4)}°E
            </div>
          )}
          {!error && (
            <div style={{ display: "flex", alignItems: "center", gap: 3, ...mono, fontSize: 8, color: "var(--danger)", background: "rgba(5,8,16,0.7)", padding: "2px 6px" }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--danger)", animation: "pulse-dot 1s infinite" }}/>REC
            </div>
          )}
        </div>
      </div>

      {/* Remove button */}
      <button onClick={() => onRemove(camera.id)}
        style={{ position: "absolute", top: 6, right: 6, background: "rgba(255,34,68,0.15)", border: "1px solid rgba(255,34,68,0.4)", color: "var(--danger)", width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: 12, zIndex: 10 }}>
        ×
      </button>
    </div>
  );
}
