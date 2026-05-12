"use client";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CameraEntry } from "@/hooks/useCameraStore";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";
import AddCameraModal from "./AddCameraModal";
import CameraFeed from "./CameraFeed";
import MiniMap from "./MiniMap";

interface Props {
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
  onAddCamera: (cam: Omit<CameraEntry, "id" | "addedAt">) => void;
  onRemoveCamera: (id: string) => void;
  onDetection: (entry: Omit<EvidenceEntry, "id">) => void;
  onAlert: (msg: string) => void;
}

export default function CameraGrid({ cameras, alertCamIds, onAddCamera, onRemoveCamera, onDetection, onAlert }: Props) {
  const [showModal, setShowModal] = useState(false);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  const cols = cameras.length === 0 ? 1 : cameras.length === 1 ? 1 : cameras.length <= 4 ? 2 : 3;

  return (
    <div style={{ flex: 1, borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, background: "var(--bg2)", flexShrink: 0 }}>
        <span style={{ ...mono, fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)" }}>
          CAMERA FEEDS <span style={{ color: "var(--accent)" }}>{cameras.length > 0 ? `· ${cameras.length} ACTIVE` : "· EMPTY"}</span>
        </span>
        <button onClick={() => setShowModal(true)}
          style={{ ...mono, fontSize: 10, letterSpacing: 1, background: "rgba(0,100,200,0.15)", border: "1px solid var(--accent2)", color: "var(--accent)", padding: "5px 14px", cursor: "pointer", textTransform: "uppercase" }}
          onMouseEnter={e => { (e.target as any).style.background = "rgba(0,100,200,0.3)"; }}
          onMouseLeave={e => { (e.target as any).style.background = "rgba(0,100,200,0.15)"; }}>
          + ADD CAMERA
        </button>
      </div>

      {/* Grid area */}
      <div style={{ flex: 1, overflow: "auto", background: "#020508", position: "relative" }}>
        {cameras.length === 0 ? (
          /* Empty State */
          <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 20, padding: 40 }}>
            <div style={{ opacity: 0.2 }}>
              <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                <rect x="8" y="20" width="64" height="44" rx="3" stroke="#4a6080" strokeWidth="2"/>
                <circle cx="40" cy="42" r="12" stroke="#4a6080" strokeWidth="2"/>
                <circle cx="40" cy="42" r="5" fill="#4a6080" opacity="0.5"/>
                <rect x="32" y="14" width="16" height="8" rx="1" stroke="#4a6080" strokeWidth="1.5"/>
                <line x1="8" y1="8" x2="72" y2="72" stroke="#4a6080" strokeWidth="1.5" opacity="0.4"/>
              </svg>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 16, fontWeight: 700, color: "var(--text-dim)", letterSpacing: 4, marginBottom: 10 }}>
                NO CAMERAS ADDED
              </div>
              <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)", lineHeight: 1.7, marginBottom: 20 }}>
                Click <span style={{ color: "var(--accent)" }}>+ ADD CAMERA</span> to connect your first camera.<br/>
                Supports webcams and CCTV/IP camera streams.
              </div>
              <button onClick={() => setShowModal(true)}
                style={{ ...mono, fontSize: 12, letterSpacing: 2, background: "var(--accent2)", border: "none", color: "#fff", padding: "12px 28px", cursor: "pointer", fontFamily: "Orbitron,sans-serif", fontWeight: 700 }}>
                + ADD FIRST CAMERA
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 2, padding: 2, height: "100%", alignContent: "start" }}>
            <AnimatePresence>
              {cameras.map(cam => (
                <motion.div key={cam.id} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} transition={{ duration: 0.2 }}>
                  <CameraFeed camera={cam} onRemove={onRemoveCamera} onDetection={onDetection} onAlert={onAlert}/>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Modal */}
      <AnimatePresence>
        {showModal && (
          <AddCameraModal
            onAdd={cam => onAddCamera({ ...cam, status: "active" })}
            onClose={() => setShowModal(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
