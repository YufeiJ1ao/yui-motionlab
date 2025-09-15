# Real-time Markerless Motion Capture (OAK-D → Blender Avatar)

A lightweight, real-time, **vision-based** motion capture pipeline that streams human motion from an **OAK-D S2** camera into a **VRoid/Blender** avatar.  
Features: full-body pose (upper/lower body), **hand articulation** (finger curls), **head pose** (yaw/pitch/roll), **basic facial expressions**, and **stereo-depth fusion** for better front/back disambiguation. End-to-end latency is instrumented and reported.

> Demo targets: interactive avatar control; future directions include **responsive AI avatars** and **humanoid robot** control.

---

## ✨ Features
- **Markerless** motion capture (no suits/markers).
- **Stereo depth fusion** (median pooling + 3D back-projection).
- **Stabilization stack**: OneEuro smoothing, visibility/velocity gating, **bone-length projection**, hinge stabilization, **angle clamping**, Slerp with time-constant.
- **Hands** (finger curls), **Head** (YPR), **Expressions** (blink, mouth open, smile, brow up).
- **Latency measurement**: camera timestamp vs. Blender apply time, CSV export.
- **Accessible**: consumer hardware + Python + Blender.

---

## 📦 Installation

### Option A: pip
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
