# StoreGuard 1.0.0-beta.1

Initial public beta release of **StoreGuard** — a multi-stream CCTV retail theft detection system pairing a **C# .NET 8 WPF counter & owner desktop application** (`StoreGuard.Desktop`) with a **headless Python multi-stream AI engine** (`ai_engine/`) powered natively by the **Apex Vision SDK Runtime** (`apex_vision.sdk.ApexRuntimeEngine`).

---

## Highlights in `v1.0.0-beta.1`

### 1. Apex Vision SDK Core Multi-Stream Execution
- Every camera stream executes as an independent `<ApexFlow>` procedure (`Image Source → ROI Zone Manager → Motion Gate → YOLO Pose → Context Object Detector → BoT-SORT Tracker → Behavior Gate → Async Cloud/Local VLM`) backed by `runtime_data/storeguard.apex`.
- **Cross-Camera Dynamic ONNX Micro-Batching**: Shares `models/yolo11n-pose.onnx` (17-keypoint pose) and `models/yolo11n.onnx` (person + bag/container context) across all camera workers via `SharedModelRegistry` and `DynamicBatchScheduler`.

### 2. Per-Layer Live Grid Overlays & TrueType Pill Badges
- Real-time MJPEG Live Camera Grid with Auto / 2×2 / 3×3 / 4×4 matrix layouts, per-camera visibility chips, and click-to-enlarge HD modal.
- **Granular Overlay Layer Toggles**: Toggle **Shelf**, **Aisle**, **Pose** (17-keypoint skeleton + wrists), **BoT-SORT** (person bbox, track ID, behavioral phase, and motion trails), **Bags** (handbag/backpack/suitcase context boxes), and **Motion** (active motion gate ROI) independently in real time.
- **Anti-Aliased Pill Badges**: Rendered via LRU-cached `Segoe UI Semibold` TrueType font badges.

### 3. Dual-Trigger Behavioral State Machine & VLM Verification
- Tracks shelf dwell/reach, concealment/tuck proximity, and cross-camera shelf-to-exit correlation before triggering VLM review.
- Assembles a chronological **2×3 Visual Sequence Grid** and a **10-second H.264 MP4 evidence clip** from the per-camera RAM ring buffer.
- Supports **Google Gemini**, **OpenAI**, **OpenRouter**, **Ollama**, and **LM Studio** VLM providers.

### 4. Staff Counter Dashboard & Owner Admin Settings
- **Counter Alerts View**: Live alert queue, 2×3 sequence grid & 10s MP4 playback toggle, AI audit summary, staff triage notes, and 1-click *Confirm Theft* / *False Alarm* disposition.
- **History & Audit View**: Date range and status filtering with historical evidence playback.
- **PIN-Protected Admin Settings (default PIN `1234`)**:
  - Interactive **ROI Canvas Editor** for drawing/resizing normalized `Shelf` and `Aisle` zones on live camera snapshots.
  - **Uniview (UNV) NVR Batch Channel Generator** (`s0` main stream / `s1` sub-stream).
  - **Compute, VLM & Alerts**: Hot-switch CPU/GPU execution, test VLM connection, and configure Telegram Bot notifications.
  - **Embedded Apex Flow Editor**: Inspect and edit per-module parameters in `storeguard.apex` with bidirectional XML sync.

---

## Quick Start (Windows x64 Release Bundle)

1. Download and extract **`StoreGuard-v1.0.0-beta.1-Windows-x64.zip`** alongside a sibling **`ApexVision`** folder (or set `APEX_VISION_PATH` to your `ApexVision` clone).
2. Install Python runtime dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Copy `runtime_data\cameras.example.json` to `runtime_data\cameras.json` (or configure cameras directly in **Admin Settings**, default PIN: `1234`).
4. Double-click **`Launch_StoreGuard.bat`** (or run `StoreGuard.Desktop.exe`).
