"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  hasWebcam: boolean; // only one webcam allowed
  onAdd: (cam: {
    type: "webcam" | "cctv";
    label: string;
    url?: string;
    lat?: number;
    lng?: number;
    status: "active";
  }) => void;
  onClose: () => void;
}

type Step = "choose" | "webcam-form" | "cctv-form";

export default function AddCameraModal({ hasWebcam, onAdd, onClose }: Props) {
  const [step, setStep] = useState<Step>("choose");
  const [label, setLabel] = useState("");
  const [url, setUrl] = useState("");
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");
  const [gpsLoading, setGpsLoading] = useState(false);
  const [ipLoading, setIpLoading] = useState(false);
  const [error, setError] = useState("");

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  // Auto-use device GPS (for webcam — laptop/phone location)
  function getDeviceGPS() {
    setGpsLoading(true);
    setError("");
    navigator.geolocation.getCurrentPosition(
      pos => {
        setLat(pos.coords.latitude.toFixed(6));
        setLng(pos.coords.longitude.toFixed(6));
        setGpsLoading(false);
      },
      () => { setError("GPS access denied — please enter manually"); setGpsLoading(false); },
      { timeout: 8000 }
    );
  }

  // Approximate location via IP (for CCTV — uses public IP of current network as fallback)
  async function getIPLocation() {
    setIpLoading(true);
    setError("");
    try {
      const res = await fetch("https://ip-api.com/json/?fields=lat,lon,city,status");
      const data = await res.json();
      if (data.status === "success") {
        setLat(data.lat.toFixed(6));
        setLng(data.lon.toFixed(6));
      } else {
        setError("IP location failed — enter manually");
      }
    } catch { setError("IP location failed — enter manually"); }
    setIpLoading(false);
  }

  function goWebcam() {
    if (hasWebcam) return;
    setStep("webcam-form");
    // Auto-detect GPS when webcam form opens
    setTimeout(getDeviceGPS, 100);
  }

  function submitWebcam() {
    if (!label.trim()) { setError("Label is required"); return; }
    onAdd({
      type: "webcam", label: label.trim(), status: "active",
      lat: lat ? parseFloat(lat) : undefined,
      lng: lng ? parseFloat(lng) : undefined,
    });
    onClose();
  }

  function submitCCTV() {
    if (!label.trim()) { setError("Label is required"); return; }
    if (!url.trim())   { setError("Stream URL is required"); return; }
    onAdd({
      type: "cctv", label: label.trim(), url: url.trim(), status: "active",
      lat: lat ? parseFloat(lat) : undefined,
      lng: lng ? parseFloat(lng) : undefined,
    });
    onClose();
  }

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.78)" }}
      onClick={e => e.target === e.currentTarget && onClose()}>
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
        style={{ width: 500, background: "var(--bg2)", border: "1px solid var(--border-glow)", padding: 28, position: "relative", maxHeight: "90vh", overflowY: "auto" }}>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 14, fontWeight: 700, letterSpacing: 3, color: "var(--accent)" }}>
            {step === "choose" ? "ADD CAMERA" : step === "webcam-form" ? "📷 WEBCAM SETUP" : "📡 CCTV SETUP"}
          </div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", color: "var(--text-dim)", fontSize: 20, cursor: "pointer", lineHeight: 1 }}>×</button>
        </div>

        {/* Step 1: Choose */}
        {step === "choose" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {/* Webcam card */}
            <button onClick={goWebcam} disabled={hasWebcam}
              style={{ padding: "24px 16px", background: hasWebcam ? "rgba(20,30,50,0.5)" : "var(--bg3)", border: `1px solid ${hasWebcam ? "var(--border)" : "var(--border)"}`, cursor: hasWebcam ? "not-allowed" : "pointer", textAlign: "center", opacity: hasWebcam ? 0.5 : 1, transition: "all 0.15s", position: "relative" }}
              onMouseEnter={e => { if (!hasWebcam) { e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.background = "rgba(0,100,200,0.1)"; } }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.background = hasWebcam ? "rgba(20,30,50,0.5)" : "var(--bg3)"; }}>
              <div style={{ fontSize: 32, marginBottom: 10 }}>📷</div>
              <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 13, fontWeight: 700, color: "var(--accent)", letterSpacing: 2, marginBottom: 6 }}>WEBCAM</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", lineHeight: 1.6 }}>
                {hasWebcam ? "Already active\n(only 1 webcam allowed)" : "Device camera\nvia MediaDevices API"}
              </div>
              {hasWebcam && (
                <div style={{ position: "absolute", top: 8, right: 8, ...mono, fontSize: 8, background: "rgba(255,34,68,0.2)", border: "1px solid rgba(255,34,68,0.4)", color: "var(--danger)", padding: "2px 6px", letterSpacing: 1 }}>
                  ACTIVE
                </div>
              )}
            </button>

            {/* CCTV card */}
            <button onClick={() => setStep("cctv-form")}
              style={{ padding: "24px 16px", background: "var(--bg3)", border: "1px solid var(--border)", cursor: "pointer", textAlign: "center", transition: "all 0.15s" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--warning)"; e.currentTarget.style.background = "rgba(255,170,0,0.08)"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.background = "var(--bg3)"; }}>
              <div style={{ fontSize: 32, marginBottom: 10 }}>📡</div>
              <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 13, fontWeight: 700, color: "var(--warning)", letterSpacing: 2, marginBottom: 6 }}>CCTV</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", lineHeight: 1.6 }}>
                IP/CCTV stream<br/>MJPEG or HLS URL
              </div>
            </button>
          </div>
        )}

        {/* Webcam form */}
        {step === "webcam-form" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Field label="Camera Label" value={label} onChange={setLabel} placeholder="e.g. Front Door Webcam" />

            <div>
              <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>GPS Location</div>
              {gpsLoading ? (
                <div style={{ ...mono, fontSize: 11, color: "var(--accent)", padding: "8px 0" }}>📍 Detecting your location...</div>
              ) : (
                <>
                  <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
                    <input value={lat} onChange={e => setLat(e.target.value)} placeholder="Latitude"
                      style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
                    <input value={lng} onChange={e => setLng(e.target.value)} placeholder="Longitude"
                      style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
                  </div>
                  {lat && lng && (
                    <div style={{ ...mono, fontSize: 9, color: "var(--safe)" }}>✓ Location detected: {parseFloat(lat).toFixed(4)}°N, {parseFloat(lng).toFixed(4)}°E</div>
                  )}
                  <button onClick={getDeviceGPS} style={{ ...mono, fontSize: 10, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "5px 12px", cursor: "pointer", letterSpacing: 1, marginTop: 4 }}>
                    📍 RE-DETECT MY LOCATION
                  </button>
                </>
              )}
            </div>

            {error && <div style={{ ...mono, fontSize: 10, color: "var(--danger)" }}>⚠ {error}</div>}
            <Buttons onBack={() => { setStep("choose"); setError(""); }} onSubmit={submitWebcam} />
          </div>
        )}

        {/* CCTV form */}
        {step === "cctv-form" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Field label="Camera Label" value={label} onChange={setLabel} placeholder="e.g. Platform CCTV-01" />
            <Field label="Stream URL (MJPEG / HLS)" value={url} onChange={setUrl} placeholder="http://192.168.1.100:8080/video" />

            <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", lineHeight: 1.6, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", padding: "8px 10px" }}>
              ℹ For CCTV cameras, GPS is fetched via your network's IP location (approximate). You can override manually.
            </div>

            <div>
              <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>GPS Location</div>
              <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
                <input value={lat} onChange={e => setLat(e.target.value)} placeholder="Latitude"
                  style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
                <input value={lng} onChange={e => setLng(e.target.value)} placeholder="Longitude"
                  style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={getIPLocation} disabled={ipLoading}
                  style={{ ...mono, fontSize: 10, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "5px 12px", cursor: "pointer", letterSpacing: 1 }}>
                  {ipLoading ? "FETCHING..." : "🌐 USE IP LOCATION"}
                </button>
                <button onClick={getDeviceGPS} disabled={gpsLoading}
                  style={{ ...mono, fontSize: 10, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "5px 12px", cursor: "pointer", letterSpacing: 1 }}>
                  {gpsLoading ? "DETECTING..." : "📍 USE DEVICE GPS"}
                </button>
              </div>
              {lat && lng && <div style={{ ...mono, fontSize: 9, color: "var(--safe)", marginTop: 4 }}>✓ {parseFloat(lat).toFixed(4)}°N, {parseFloat(lng).toFixed(4)}°E</div>}
            </div>

            {error && <div style={{ ...mono, fontSize: 10, color: "var(--danger)" }}>⚠ {error}</div>}
            <Buttons onBack={() => { setStep("choose"); setError(""); }} onSubmit={submitCCTV} />
          </div>
        )}
      </motion.div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder: string }) {
  const mono: React.CSSProperties = { fontFamily: "monospace" };
  return (
    <div>
      <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{ width: "100%", background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 13, padding: "9px 12px", outline: "none" }}/>
    </div>
  );
}

function Buttons({ onBack, onSubmit }: { onBack: () => void; onSubmit: () => void }) {
  const mono: React.CSSProperties = { fontFamily: "monospace" };
  return (
    <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
      <button onClick={onBack} style={{ flex: 1, padding: "10px", ...mono, fontSize: 11, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "pointer" }}>← BACK</button>
      <button onClick={onSubmit} style={{ flex: 2, padding: "10px", fontFamily: "Orbitron,sans-serif", fontSize: 12, fontWeight: 700, background: "var(--accent2)", border: "none", color: "#fff", cursor: "pointer", letterSpacing: 2 }}>ADD CAMERA</button>
    </div>
  );
}
