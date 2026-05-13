"use client";
import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import LoginScreen      from "@/components/LoginScreen";
import TopBar           from "@/components/TopBar";
import CameraGrid       from "@/components/CameraGrid";
import StatsPanel       from "@/components/StatsPanel";
import EvidenceVault    from "@/components/EvidenceVault";
import LiveMap          from "@/components/LiveMap";
import AdminPanel       from "@/components/AdminPanel";
import Analysis         from "@/components/Analysis";
import CameraManagement from "@/components/CameraManagement";
import Toast, { useToast } from "@/components/Toast";

import { useCameraStore }   from "@/hooks/useCameraStore";
import { useEvidenceStore } from "@/hooks/useEvidenceStore";

type Tab = "dashboard" | "evidence" | "map" | "admin" | "analysis" | "manage";

export default function Home() {
  const [loggedIn,    setLoggedIn]    = useState(false);
  const [currentUser, setCurrentUser] = useState("");
  const [tab,         setTab]         = useState<Tab>("dashboard");
  const [alertMsg,    setAlertMsg]    = useState("");

  const { toast, showToast, setToast } = useToast();
  const { cameras, addCamera, removeCamera, updateCamera } = useCameraStore();
  const { evidence, addEvidence, clearAll, updateEvidence } = useEvidenceStore();

  const [alertCamIds, setAlertCamIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4500);
    return () => clearTimeout(t);
  }, [toast, setToast]);

  useEffect(() => {
    if (!alertMsg) return;
    const t = setTimeout(() => setAlertMsg(""), 12000);
    return () => clearTimeout(t);
  }, [alertMsg]);

  const handleDetection = useCallback((entry: Parameters<typeof addEvidence>[0] & { cameraId: string }) => {
    const item = addEvidence(entry);
    setAlertCamIds(prev => new Set(prev).add(entry.cameraId));
    setAlertMsg(`VIOLENCE DETECTED — ${entry.cameraLabel} · ${(entry.confidence * 100).toFixed(0)}% confidence`);
    
    setTimeout(() => {
      setAlertCamIds(prev => { const s = new Set(prev); s.delete(entry.cameraId); return s; });
    }, 15000);

    return item.id;
  }, [addEvidence]);

  const handleAlert = useCallback((msg: string) => {
    showToast(msg, true);
  }, [showToast]);

  const isAlert = alertMsg.length > 0;

  return (
    <>
      <AnimatePresence>
        {!loggedIn && (
          <motion.div key="login" exit={{ opacity: 0 }} transition={{ duration: 0.4 }} style={{ position: "fixed", inset: 0, zIndex: 100 }}>
            <LoginScreen onLogin={u => { setCurrentUser(u); setLoggedIn(true); }}/>
          </motion.div>
        )}
      </AnimatePresence>

      {loggedIn && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}>

          <TopBar currentTab={tab} onTabChange={t => setTab(t as Tab)} isAlert={isAlert}
            onLogout={() => { setLoggedIn(false); setAlertMsg(""); }} currentUser={currentUser}/>

          <AnimatePresence>
            {isAlert && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                style={{ background: "rgba(255,34,68,0.13)", borderBottom: "2px solid var(--danger)", padding: "9px 20px", fontFamily: "monospace", fontSize: 12, color: "var(--danger)", letterSpacing: 2, textTransform: "uppercase", display: "flex", alignItems: "center", gap: 14, flexShrink: 0, animation: "alert-flash-bg 0.5s infinite" }}>
                <span>⚠</span>
                <span>{alertMsg}</span>
                <button onClick={() => setAlertMsg("")}
                  style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 10, background: "transparent", border: "1px solid var(--danger)", color: "var(--danger)", padding: "3px 12px", cursor: "pointer", letterSpacing: 1 }}>
                  ACKNOWLEDGE
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <div style={{ flex: 1, overflow: "hidden" }}>
            {tab === "dashboard" && (
              <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
                <CameraGrid
                  cameras={cameras}
                  alertCamIds={alertCamIds}
                  onAddCamera={addCamera}
                  onRemoveCamera={removeCamera}
                  onDetection={handleDetection}
                  onAlert={handleAlert}
                />
                <StatsPanel 
                  isAlert={isAlert} 
                  evidence={evidence}
                  onUpdateEvidence={updateEvidence}
                  cameraCount={cameras.length}
                  activeCameraCount={cameras.filter(c => c.status === "active").length}
                  cameras={cameras}
                  alertCamIds={alertCamIds}
                />
              </div>
            )}

            {tab === "evidence" && (
              <EvidenceVault evidence={evidence} onClearAll={clearAll}/>
            )}

            {tab === "map" && (
              <div style={{ height: "100%" }}>
                <LiveMap cameras={cameras} alertCamIds={alertCamIds} showToast={showToast} onUpdateCamera={updateCamera}/>
              </div>
            )}

            {tab === "analysis" && (
              <Analysis evidence={evidence} cameras={cameras}/>
            )}

            {tab === "manage" && (
              <CameraManagement 
                cameras={cameras} 
                onAddCamera={addCamera} 
                onRemoveCamera={removeCamera} 
                onEditCamera={updateCamera}
              />
            )}

            {tab === "admin" && (
              <AdminPanel cameras={cameras} onRemoveCamera={removeCamera} currentUser={currentUser}/>
            )}
          </div>
        </motion.div>
      )}

      <Toast toast={toast}/>
    </>
  );
}
