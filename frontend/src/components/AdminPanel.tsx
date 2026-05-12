"use client";
import { useState } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";

interface Props {
  cameras: CameraEntry[];
  onRemoveCamera: (id: string) => void;
  currentUser: string;
}

const MODULE_CONTROLS = [
  { id: "VisionGuard",  label: "VisionGuard — Object & Weapon Detection", modelFile: "yolov8n_weapons_v3.onnx",   modelSize: "22.4 MB" },
  { id: "BehaviorNet",  label: "BehaviorNet — Violence & Harassment",     modelFile: "behaviornet_v2_int8.onnx",  modelSize: "18.1 MB" },
  { id: "CrowdSense",   label: "CrowdSense — Crowd Density & Flow",       modelFile: "crowdsense_csrnet_v1.onnx", modelSize: "14.3 MB" },
];

const ENABLED_KEY = "sd_module_enabled";

function loadEnabled(): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(ENABLED_KEY) || "{}"); } catch { return {}; }
}
function saveEnabled(e: Record<string, boolean>) {
  try { localStorage.setItem(ENABLED_KEY, JSON.stringify(e)); } catch {}
}

export default function AdminPanel({ cameras, onRemoveCamera, currentUser }: Props) {
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() => {
    const stored = loadEnabled();
    return Object.fromEntries(MODULE_CONTROLS.map(m => [m.id, stored[m.id] ?? true]));
  });

  function toggle(id: string) {
    setEnabled(prev => {
      const next = { ...prev, [id]: !prev[id] };
      saveEnabled(next);
      return next;
    });
  }

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  // Real browser / device info
  const browserInfo = typeof navigator !== "undefined" ? navigator.userAgent.split(" ").slice(-2).join(" ") : "Unknown";
  const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
  const camCount = cameras.length;
  const onlineCount = cameras.filter(c => c.status === "active").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", padding: 20, gap: 16, overflowY: "auto", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 16, fontWeight: 700, letterSpacing: 4, color: "var(--accent)" }}>SYSTEM ADMIN</div>
        <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)" }}>SESSION: {currentUser}</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* AI Module Controls */}
        <AdminCard title="AI MODULE CONTROLS" style={{ gridColumn: "1 / -1" }}>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 12, lineHeight: 1.5 }}>
            Toggle AI modules that run during camera monitoring. Changes take effect on the next camera session.
          </div>
          {MODULE_CONTROLS.map(mod => (
            <div key={mod.id} style={{ padding: "12px 0", borderBottom: "1px solid rgba(26,40,64,0.4)", display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: enabled[mod.id] ? "var(--text)" : "var(--text-dim)", marginBottom: 3 }}>{mod.label}</div>
                <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 1 }}>{mod.modelFile} · {mod.modelSize}</div>
              </div>
              <div onClick={() => toggle(mod.id)}
                style={{ width: 36, height: 20, background: enabled[mod.id] ? "var(--accent2)" : "var(--border)", borderRadius: 10, position: "relative", cursor: "pointer", transition: "background 0.2s", flexShrink: 0 }}>
                <div style={{ position: "absolute", width: 14, height: 14, background: "#fff", borderRadius: "50%", top: 3, left: 3, transform: enabled[mod.id] ? "translateX(16px)" : "none", transition: "transform 0.2s" }}/>
              </div>
              <div style={{ ...mono, fontSize: 10, color: enabled[mod.id] ? "var(--safe)" : "var(--text-dim)", minWidth: 50, textAlign: "right" }}>
                {enabled[mod.id] ? "ON" : "OFF"}
              </div>
            </div>
          ))}
        </AdminCard>

        {/* Active Cameras */}
        <AdminCard title={`ACTIVE CAMERAS (${camCount})`}>
          {camCount === 0 ? (
            <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)", padding: "12px 0" }}>No cameras added. Go to Dashboard to add cameras.</div>
          ) : (
            cameras.map(cam => (
              <div key={cam.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid rgba(26,40,64,0.3)" }}>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>{cam.label}</div>
                  <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)" }}>
                    {cam.type.toUpperCase()} · {cam.id}
                    {cam.lat && ` · ${cam.lat.toFixed(4)}°N`}
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ ...mono, fontSize: 9, color: "var(--safe)" }}>ACTIVE</div>
                  <button onClick={() => onRemoveCamera(cam.id)}
                    style={{ ...mono, fontSize: 9, background: "transparent", border: "1px solid rgba(255,34,68,0.3)", color: "var(--danger)", padding: "3px 8px", cursor: "pointer" }}>
                    REMOVE
                  </button>
                </div>
              </div>
            ))
          )}
        </AdminCard>

        {/* System Info */}
        <AdminCard title="SYSTEM INFO">
          {[
            ["App Version",       "SurakshaDrishti v2.4.1"],
            ["Cameras Active",    `${onlineCount} / ${camCount}`],
            ["Inference",         "In-Browser (ONNX Runtime Web)"],
            ["Connection",        isHttps ? "HTTPS ✓" : "HTTP (webcam may need HTTPS)"],
            ["Storage",           "localStorage (client-side)"],
            ["Browser",          browserInfo],
          ].map(([l, v]) => (
            <div key={l} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: "1px solid rgba(26,40,64,0.3)" }}>
              <div style={{ fontSize: 12, color: "var(--text)" }}>{l}</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", textAlign: "right", maxWidth: 180 }}>{v}</div>
            </div>
          ))}
        </AdminCard>

      </div>
    </div>
  );
}

function AdminCard({ title, children, style }: { title: string; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: "var(--panel)", border: "1px solid var(--border)", padding: 16, ...style }}>
      <div style={{ fontFamily: "monospace", fontSize: 9, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 12, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        {title}
      </div>
      {children}
    </div>
  );
}
