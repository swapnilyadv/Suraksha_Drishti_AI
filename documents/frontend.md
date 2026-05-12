# 🖥️ SurakshaDrishti Frontend — Tactical Command Center

> **Status**: Production-Ready React Implementation
> **Framework**: Next.js 15 (App Router)
> **Refrence**: Based on `documents/suraksaprototype.html`

---

## 🚀 Tech Stack
*   **Core**: Next.js 15, TypeScript, React 19.
*   **Styling**: Tailwind CSS + Custom Design System (glowing borders, glassmorphism).
*   **Animations**: Framer Motion (page transitions, alert flashing, map pulses).
*   **AI Inference**: ONNX Runtime Web (to run the harassment model in-browser).
*   **Icons**: Lucide React.

---

## 🎨 Design System (globals.css)
The frontend uses a custom tactical theme inspired by modern surveillance HUDs:
*   **Primary Background**: `#050810` (Deep Space Blue)
*   **Accent Color**: `#00aaff` (Tactical Cyan)
*   **Danger Color**: `#ff2244` (Alert Red)
*   **Safe Color**: `#00ff88` (Status Green)
*   **Overlay**: Real-time scanline/CRT effect using CSS gradients.

---

## 🧩 Component Architecture

### 1. 🛡️ Login System (`LoginScreen.tsx`)
*   **Feature**: Multi-role authentication (Admin / Officer).
*   **Visuals**: Animated bracket boxes and grid backgrounds.
*   **Creds**: 
    *   `ADMIN_001` / `admin@123`
    *   `OFFICER_1` / `police@456`

### 2. 📡 Intelligence Dashboard (`CameraGrid.tsx` & `StatsPanel.tsx`)
*   **Live Webcam (CAM-01)**: Uses `navigator.mediaDevices` for real-time local monitoring.
*   **Simulation (CAM-02/03)**: Custom `<canvas>` engine that generates tactical noise and fake detections.
*   **Metrics**: Real-time incident counts, AI model confidence meters, and event logs.

### 3. 📹 Evidence Vault (`EvidenceVault.tsx`)
*   **Cards**: Individual cards for each recorded incident.
*   **Actions**: Verify, Dispatch Police, or Reject evidence.
*   **Indicators**: Visual "Confidence Bars" and GPS coordinate tags.

### 4. 🗺️ Tactical Live Map (`LiveMap.tsx`)
*   **Visuals**: Custom SVG floorplan with pulsing camera nodes.
*   **Dispatch**: Interactive "Dispatch Unit" feature that draws a vector line from the station to the alert camera.
*   **Registry**: Sidebar showing the status of all camera IDs in the network.

### 5. ⚙️ Admin Control (`AdminPanel.tsx`)
*   **Toggles**: Real-time control for AI modules (Violence Detection, Pose Estimation, etc.).
*   **Logs**: System-level debugging log (Auth events, boot sequences, model status).

---

## ⚠️ Alert & Notification System
*   **Alert Banner**: Flashes across the top when violence is detected.
*   **Audio Alert**: High-frequency tactical beep that triggers during an active incident.
*   **Toast System**: Minimalist notifications for user actions (e.g., "Unit Dispatched").

---

## 🏃 How to Run
1.  Navigate to the directory: `cd frontend`
2.  Install packages: `npm install`
3.  Launch: `npm run dev`
4.  URL: `http://localhost:3000`
