"use client";
import { useEffect, useState } from "react";

export interface EvidenceEntry {
  id: string;
  cameraId: string;
  cameraLabel: string;
  timestamp: string;       // human-readable
  isoTime: string;         // ISO for sorting
  confidence: number;
  type: "VIOLENCE" | "HARASSMENT";
  thumbnail?: string;      // base64 data-URL of captured frame (small JPEG)
  // videoUrl is NOT in localStorage — only in-memory (cleared on reload)
}

const KEY = "sd_evidence_v2";

function load(): EvidenceEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function save(items: EvidenceEntry[]) {
  // strip thumbnails before saving to keep localStorage light
  try {
    const slim = items.map(({ thumbnail: _t, ...rest }) => rest);
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {}
}

export function useEvidenceStore() {
  const [evidence, setEvidence] = useState<EvidenceEntry[]>([]);

  useEffect(() => { setEvidence(load()); }, []);

  function addEvidence(entry: Omit<EvidenceEntry, "id">) {
    const item: EvidenceEntry = { ...entry, id: `EVD-${Date.now()}` };
    setEvidence(prev => {
      const next = [item, ...prev];
      save(next);
      return next;
    });
    return item;
  }

  function clearAll() {
    setEvidence([]);
    localStorage.removeItem(KEY);
  }

  function updateEvidence(id: string, updates: Partial<EvidenceEntry>) {
    setEvidence(prev => {
      const next = prev.map(item => item.id === id ? { ...item, ...updates } : item);
      save(next);
      return next;
    });
  }

  return { evidence, addEvidence, clearAll, updateEvidence };
}
