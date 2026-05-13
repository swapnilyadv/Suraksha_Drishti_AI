# 🖥️ SurakshaDrishti Frontend — Tactical Command Center

> **Status**: Production-Ready React Implementation (High Fidelity)
> **Framework**: Next.js 15 (App Router)
> **Platform**: Client-Side Tactical Surveillance Suite

---

## 🚀 Tech Stack
*   **Core**: Next.js 15, TypeScript, React 19.
*   **Styling**: Vanilla CSS + Tailwind (Tactical Design System).
*   **Animations**: Framer Motion (Page transitions, alert pulses, dynamic HUD).
*   **Maps**: Leaflet.js + React-Leaflet (Real-world OpenStreetMap integration).
*   **State**: Custom Hooks with `localStorage` persistence (`useCameraStore`, `useEvidenceStore`).
*   **Media**: Browser `MediaDevices` API for real-time webcam streams.

---

## 🎨 Design System
The frontend uses a custom tactical HUD theme inspired by premium surveillance systems:
*   **Colors**: 
    *   `#050810` (Deep Space Blue) — Core UI background.
    *   `#00aaff` (Tactical Cyan) — Primary active components.
    *   `#ff2244` (Alert Red) — Critical violence detections.
    *   `#00ff88` (Status Green) — Secure system state.
*   **Effects**: Glassmorphism, CSS Scanlines (CRT effect), and CSS Glowing Borders.

---

## 🧩 Core Architecture & Features

### 1. 📡 Intelligence Dashboard (`Dashboard`)
*   **Camera Grid**: Dynamic rendering of active streams. Supports **Webcam** (Real device camera) and **CCTV** (MJPEG/HLS URL streams).
*   **Stats Panel**: 100% Real-time metrics. Tracks actual **JS Memory Heap**, **Session Uptime**, and **Persistence Stats**.
*   **Tactical Mini-Map**: A permanent map preview in the sidebar showing real-time GPS locations of all connected nodes.

### 2. 📍 Live Tactical Map (`MapClient`)
*   **Real-World Tracking**: Uses **Leaflet.js** to map cameras to actual GPS coordinates.
*   **Moving Target**: Webcams use high-accuracy `watchPosition` to follow the user/laptop on the map in real-time.
*   **Big Screen Mode**: Clicking a map marker opens a full-screen high-fidelity live feed of that camera.

### 3. 🔍 Evidence Vault (`EvidenceVault`)
*   **Real Detections Only**: No simulated data. Displays actual motion/violence detections captured from live feeds.
*   **Memory Efficiency**: Thumbnails are handled in-memory; metadata is persisted in `localStorage`.
*   **Clear All**: One-click wipe to reset storage and clear system memory.

### 4. 📊 Intelligence Analysis (`Analysis`)
*   **Data Trends**: SVG-based graphing of incident detection spikes over a 7-hour rolling window.
*   **Category Split**: Percentage breakdown of Violence vs. Harassment vs. System Events.
*   **Infrastructure Health**: Real monitoring of **LocalStorage utilization** and Camera connectivity uptime.

### 5. 🛠️ Camera Management (`Manage`)
*   **Full CRUD**: Add, Edit, or Delete camera configurations.
*   **GPS Automation**: 
    - **Webcam**: Auto-detects device GPS.
    - **CCTV**: Fetches approximate coordinates via IP-Geolocation (using `ip-api`).
*   **Hardware Guard**: Logic to prevent multiple webcam registrations, protecting system resources.

---

## 💾 Data Persistence
The system is fully standalone and persists configuration via the browser's `localStorage` API:
*   `sd-cameras`: Stores the registry of all added devices and their GPS coords.
*   `sd-evidence`: Stores the log of all security incidents.
*   `sd-ai-modules`: Persists toggle states for different AI detection models.

---

## 🚦 System Roles
*   **ADMIN**: Full access to settings, camera management, and clear-all functions.
*   **OFFICER**: Monitor-only access to dashboard and live map.

---

## 🛠️ How to Run
```bash
# Install dependencies
npm install

# Run tactical command center
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** to access the interface.
