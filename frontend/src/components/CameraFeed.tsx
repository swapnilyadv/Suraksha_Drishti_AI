"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { surakshaAI } from "@/utils/onnx_engine";
import { CameraEntry } from "@/hooks/useCameraStore";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";

interface Props {
  camera: CameraEntry;
  onRemove: (id: string) => void;
  onDetection: (entry: Omit<EvidenceEntry, "id">) => any;
  onAlert: (msg: string) => void;
}

const MIN_RECORD_MS = 15000;
const DROP_HYSTERESIS_MS = 3000;
const FRAME_BUFFER_SIZE = 16;

export default function CameraFeed({ camera, onRemove, onDetection, onAlert }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [alerting, setAlerting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Logic state
  const recordingStartTime = useRef<number | null>(null);
  const lastDetectionTime = useRef<number | null>(null);
  const frameBuffer = useRef<Float32Array[]>([]);
  const isStopping = useRef(false);

  // Initialize AI and Camera
  useEffect(() => {
    surakshaAI.loadModel();

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
    };
  }, [camera]);

  const activeEvidenceId = useRef<string | null>(null);

  // Recording Functions
  const startRecording = useCallback((evidenceId: string) => {
    if (isRecording || !streamRef.current) return;
    setIsRecording(true);
    activeEvidenceId.current = evidenceId;
    recordingStartTime.current = Date.now();
    chunksRef.current = [];
    
    const recorder = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });
    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const videoUrl = URL.createObjectURL(blob);
      
      // Update the evidence entry with the final video URL
      if (activeEvidenceId.current) {
        // We'll need a way to update evidence here. 
        // For now, we'll log it and let the hook handle state.
        console.log(`🎬 Evidence video saved for ${activeEvidenceId.current}:`, videoUrl);
        // Dispatch custom event to update store
        window.dispatchEvent(new CustomEvent('update-evidence-video', { 
          detail: { id: activeEvidenceId.current, videoUrl } 
        }));
      }
    };
    recorder.start();
    recorderRef.current = recorder;
  }, [isRecording]);

  const stopRecording = useCallback(() => {
    if (!isRecording || !recorderRef.current) return;
    recorderRef.current.stop();
    setIsRecording(false);
    recordingStartTime.current = null;
    lastDetectionTime.current = null;
    isStopping.current = false;
  }, [isRecording]);

  // AI Processing Loop
  useEffect(() => {
    if (camera.type !== "webcam" || !ready) return;
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    canvas.width = 112; canvas.height = 112;
    const ctx = canvas.getContext("2d")!;

    let active = true;

    async function process() {
      if (!active) return;
      if (!video.readyState || video.readyState < 2) { requestAnimationFrame(process); return; }

      // 1. Capture and Pre-process frame
      ctx.drawImage(video, 0, 0, 112, 112);
      const imageData = ctx.getImageData(0, 0, 112, 112).data;
      const floatData = new Float32Array(3 * 112 * 112);
      
      // Normalize to 0-1 and CHW format
      for (let i = 0; i < 112 * 112; i++) {
        floatData[i] = imageData[i * 4] / 255.0; // R
        floatData[i + 112 * 112] = imageData[i * 4 + 1] / 255.0; // G
        floatData[i + 224 * 112] = imageData[i * 4 + 2] / 255.0; // B
      }

      frameBuffer.current.push(floatData);
      if (frameBuffer.current.length > FRAME_BUFFER_SIZE) frameBuffer.current.shift();

      // 2. Run Inference every 8 frames
      if (frameBuffer.current.length === FRAME_BUFFER_SIZE && Date.now() % 8 === 0) {
        const result = await surakshaAI.predict(frameBuffer.current);
        const isThreat = result.label === "Harassment" || result.label === "Weapon";

        if (isThreat && result.confidence > 0.6) {
          lastDetectionTime.current = Date.now();
          if (!isRecording) {
            const evidenceId = triggerSnapshot(result);
            if (evidenceId) startRecording(evidenceId);
          }
          setAlerting(true);
        } else {
          setAlerting(false);
        }
      }

      // 3. Evidence Logic: Stop Handling
      if (isRecording) {
        const now = Date.now();
        const duration = now - (recordingStartTime.current || 0);
        const timeSinceLastDetection = now - (lastDetectionTime.current || 0);

        // Conditions to stop: Min duration met AND no detection for hysteresis period
        if (duration > MIN_RECORD_MS && timeSinceLastDetection > DROP_HYSTERESIS_MS) {
          stopRecording();
        }
      }

      requestAnimationFrame(process);
    }

    process();
    return () => { active = false; };
  }, [ready, camera, isRecording, startRecording, stopRecording]);

  const triggerSnapshot = useCallback((result: any) => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, 640, 480);
    const thumbnail = canvas.toDataURL("image/jpeg", 0.8);

    return onDetection({
      cameraId: camera.id,
      cameraLabel: camera.label,
      timestamp: new Date().toLocaleString("en-IN"),
      isoTime: new Date().toISOString(),
      confidence: result.confidence,
      type: result.label.toUpperCase(),
      thumbnail,
      // Metadata fields for the new requirement
      maleCount: result.counts.male,
      femaleCount: result.counts.female,
      weaponDetected: result.counts.weapon,
      threatLevel: result.label === "Weapon" ? "Critical" : "High",
    });
  }, [camera, onDetection]);

  return (
    <div className="relative bg-zinc-900 overflow-hidden border border-zinc-800 rounded-sm" 
         style={{ outline: alerting ? "2px solid #ff2244" : "none" }}>
      <video ref={videoRef} autoPlay muted playsInline className="w-full aspect-video object-cover" />
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Overlay UI */}
      <div className="absolute inset-0 p-3 flex flex-col justify-between pointer-events-none">
        <div className="flex justify-between items-start">
          <div className="flex gap-2">
            <span className="bg-black/80 text-[10px] px-2 py-1 text-white font-mono uppercase tracking-widest">{camera.label}</span>
            <span className="bg-black/80 text-[10px] px-2 py-1 text-cyan-400 border border-cyan-900 font-mono uppercase">{camera.type}</span>
          </div>
          {isRecording && (
            <div className="flex items-center gap-2 bg-red-600/90 text-white text-[10px] px-2 py-1 font-mono animate-pulse">
              <div className="w-2 h-2 rounded-full bg-white" /> REC EV-{(Date.now() % 10000).toString().padStart(4, "0")}
            </div>
          )}
        </div>
        
        {alerting && (
          <div className="absolute inset-0 bg-red-500/10 pointer-events-none" />
        )}
      </div>

      {/* Remove button */}
      <button onClick={() => onRemove(camera.id)}
        className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center bg-red-500/20 border border-red-500/40 text-red-500 text-xs hover:bg-red-500/40 transition-colors">
        ×
      </button>
    </div>
  );
}
