"use client";
// This file is dynamically imported with ssr:false — safe to use Leaflet here
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { CameraEntry } from "@/hooks/useCameraStore";

// Fix leaflet default icon paths broken by webpack
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// Custom icons
function makeIcon(color: string) {
  return L.divIcon({
    className: "",
    html: `<div style="
      width:28px;height:28px;border-radius:50% 50% 50% 0;
      background:${color};border:2px solid rgba(255,255,255,0.7);
      transform:rotate(-45deg);box-shadow:0 0 12px ${color}88;
    "></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 28],
    popupAnchor: [0, -30],
  });
}

const iconWebcam = makeIcon("#00aaff");
const iconCCTV   = makeIcon("#ffaa00");
const iconAlert  = makeIcon("#ff2244");

function FitBounds({ cameras }: { cameras: CameraEntry[] }) {
  const map = useMap();
  useEffect(() => {
    const valid = cameras.filter(c => c.lat && c.lng);
    if (valid.length === 0) return;
    if (valid.length === 1) {
      map.setView([valid[0].lat!, valid[0].lng!], 16);
    } else {
      const bounds = L.latLngBounds(valid.map(c => [c.lat!, c.lng!] as [number, number]));
      map.fitBounds(bounds, { padding: [60, 60] });
    }
  }, [cameras, map]);
  return null;
}

interface Props {
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
  showToast: (msg: string, danger?: boolean) => void;
  onMarkerClick?: (id: string) => void;
}

export default function MapClient({ cameras, alertCamIds, showToast, onMarkerClick }: Props) {
  const validCams = cameras.filter(c => c.lat && c.lng);
  // Default center: Mumbai if no cameras with GPS
  const defaultCenter: [number, number] = validCams.length > 0
    ? [validCams[0].lat!, validCams[0].lng!]
    : [19.0760, 72.8777];

  return (
    <MapContainer center={defaultCenter} zoom={15} style={{ width: "100%", height: "100%" }} zoomControl={true}>
      <TileLayer
        attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds cameras={validCams}/>

      {validCams.map(cam => {
        const isAlert = alertCamIds.has(cam.id);
        const icon = isAlert ? iconAlert : cam.type === "webcam" ? iconWebcam : iconCCTV;
        return (
          <Marker
            key={cam.id}
            position={[cam.lat!, cam.lng!]}
            icon={icon}
            eventHandlers={{
              click: () => onMarkerClick?.(cam.id),
            }}
          >
            <Popup>
              <div style={{ fontFamily: "monospace", minWidth: 180 }}>
                <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13 }}>{cam.label}</div>
                <div style={{ fontSize: 11, color: "#555", marginBottom: 2 }}>
                  Type: <strong style={{ color: cam.type === "webcam" ? "#0066cc" : "#cc7700" }}>{cam.type.toUpperCase()}</strong>
                </div>
                <div style={{ fontSize: 10, color: "#777" }}>
                  {cam.lat?.toFixed(6)}°N, {cam.lng?.toFixed(6)}°E
                </div>
                {isAlert && (
                  <div style={{ marginTop: 6, color: "#cc0022", fontSize: 11, fontWeight: 700 }}>
                    ⚠ ALERT ACTIVE
                  </div>
                )}
                {cam.url && (
                  <div style={{ fontSize: 10, color: "#777", marginTop: 4, wordBreak: "break-all" }}>
                    Stream: {cam.url}
                  </div>
                )}
                <div style={{ fontSize: 10, color: "#999", marginTop: 4 }}>
                  Added: {new Date(cam.addedAt).toLocaleDateString()}
                </div>
              </div>
            </Popup>
          </Marker>
        );
      })}
    </MapContainer>
  );
}
