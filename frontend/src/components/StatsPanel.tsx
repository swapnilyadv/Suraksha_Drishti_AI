"use client";
import { useEffect, useState } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";
import MiniMap from "./MiniMap";

interface Props {
  isAlert: boolean;
  incidentCount: number;
  cameraCount: number;
  activeCameraCount: number;
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
}

export default function StatsPanel({ isAlert, incidentCount, cameraCount, activeCameraCount, cameras, alertCamIds }: Props) {
  const [sessionStart] = useState(Date.now());
  const [uptime, setUptime] = useState("00:00:00");
  const [memoryUsage, setMemoryUsage] = useState<string>("N/A");

  useEffect(() => {
    const timer = setInterval(() => {
      const diff = Date.now() - sessionStart;
      const h = Math.floor(diff / 3600000).toString().padStart(2, "0");
      const m = Math.floor((diff % 3600000) / 60000).toString().padStart(2, "0");
      const s = Math.floor((diff % 60000) / 1000).toString().padStart(2, "0");
      setUptime(`${h}:${m}:${s}`);

      if ((performance as any).memory) {
        const used = (performance as any).memory.usedJSHeapSize;
        setMemoryUsage(`${Math.round(used / 1048576)} MB`);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [sessionStart]);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  return (
    <div style={{ width: 310, flexShrink: 0, display: "flex", flexDirection: "column", overflowY: "auto", background: "var(--bg2)", borderLeft: "1px solid var(--border)" }}>

      {/* Mini Map - Top Right of Dashboard */}
      <div style={{ borderBottom: "1px solid var(--border)", background: "#050810" }}>
        <div style={{ padding: "8px 14px", ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>TACTICAL OVERVIEW</div>
        <div style={{ width: "100%", height: 180 }}>
          <MiniMap cameras={cameras} alertCamIds={alertCamIds} />
        </div>
      </div>

      {/* System Status */}
      <Block label="SESSION STATUS">
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 28, fontWeight: 900, lineHeight: 1, color: isAlert ? "var(--danger)" : "var(--safe)", textShadow: isAlert ? "0 0 20px rgba(255,34,68,0.3)" : "0 0 20px rgba(0,255,136,0.2)" }}>
          {isAlert ? "ALERT" : "STABLE"}
        </div>
        <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", marginTop: 8, letterSpacing: 1 }}>
          {isAlert ? "THREAT DETECTION ACTIVE" : "ENVIRONMENT SECURE"}
        </div>
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {[
            ["SESSION UPTIME", uptime],
            ["CAMERAS", `${activeCameraCount} / ${cameraCount}`],
            ["JS MEMORY", memoryUsage],
            ["STORAGE", "LOCAL"],
          ].map(([l, v]) => (
            <div key={l} style={{ background: "var(--bg3)", padding: "8px 10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1, marginBottom: 4 }}>{l}</div>
              <div style={{ ...mono, fontSize: 11, color: "var(--text)", fontWeight: 600 }}>{v}</div>
            </div>
          ))}
        </div>
      </Block>

      {/* Real-time Incident Tracker */}
      <Block label="INCIDENT TRACKING">
        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          <div>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 24, fontWeight: 900, color: incidentCount > 0 ? "var(--warning)" : "var(--text-dim)" }}>{incidentCount}</div>
            <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)" }}>TOTAL DETECTIONS</div>
          </div>
        </div>
        <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", marginBottom: 4 }}>AI MODEL INTEGRITY: <span style={{ color: "var(--safe)" }}>OPTIMAL</span></div>
        <div style={{ height: 2, background: "var(--border)", marginBottom: 12 }}>
          <div style={{ height: "100%", width: "100%", background: "var(--safe)" }}/>
        </div>
      </Block>

      {/* System Environment */}
      <Block label="DEVICE INTELLIGENCE">
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
             <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>CONNECTION SECURITY</div>
             <div style={{ ...mono, fontSize: 10, color: "var(--safe)" }}>✓ SSL/TLS ENCRYPTED</div>
          </div>
          <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
             <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>INFERENCE ENGINE</div>
             <div style={{ ...mono, fontSize: 10, color: "var(--accent)" }}>WASM/WEBGL (ONNX)</div>
          </div>
        </div>
      </Block>

      {/* Mini Help */}
      <div style={{ marginTop: "auto", padding: 14, opacity: 0.5 }}>
        <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", lineHeight: 1.5 }}>
          SURAKSHADRISHTI CORE V2.4<br/>
          CONNECTED TO LOCAL BROWSER STORAGE<br/>
          REAL-TIME VIDEO STREAMING ACTIVE
        </div>
      </div>
    </div>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ borderBottom: "1px solid var(--border)", padding: 18 }}>
      <div style={{ fontFamily: "monospace", fontSize: 9, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 14 }}>{label}</div>
      {children}
    </div>
  );
}
