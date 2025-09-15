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
```
### Option B: conda
```bash
conda env create -f environment.yml
conda activate oakd_blender
```

## ▶️ Quick Start

### 1 Run the OAK-D Sender
```bash
cd sender
python oakd_sender.py --ip 127.0.0.1 --port 9000 --k 5 --alpha 0.2 \
  --det 0.35 --track 0.35 --flip \
  # optional toggles:
  # --no-depth    # disable StereoDepth fallback to MP pseudo-3D
  # --no-hands    # disable finger curls
  # --no-head     # disable head YPR
  # --no-expr     # disable facial expression metrics
```
### Common flags

--k: median pooling radius for depth (patch size = 2k+1). Try 5 or 7 for smoother depth.

--flip: mirror selfie-view.

--alpha: OneEuro smoothing factor for pose vectors.

### 2 Run the Blender Receiver
1. Open your VRoid avatar .blend.

2. Select the armature object (e.g., Armature) and switch to Pose Mode.

3. Text Editor → open blender/blender_receiver.py → Run Script.

4. Console should show:
```csharp
[Receiver] Listening UDP on 127.0.0.1:9000
[Receiver] Timer registered @60.0Hz. Ready ✓
```
5. Move in front of camera — avatar follows.

#### Notes

If you see missing bones, update BONE_MAP and finger bone names in blender_receiver.py to match your rig.

Port already in use? Change UDP_PORT in both sender and receiver to a free port.

## 🧪 Latency Measurement

1. Open your VRoid avatar .blend.

2. Select the armature object (e.g., Armature) and switch to Pose Mode.

3. Text Editor → open blender/blender_receiver.py → Run Script.

4. Console should show:
```csharp
[Receiver] Listening UDP on 127.0.0.1:9000
[Receiver] Timer registered @60.0Hz. Ready ✓
```

