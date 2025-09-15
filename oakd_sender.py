#!/usr/bin/env python3
# OAK-D S2 sender (Pose+Hands+Head+Expressions) with StereoDepth -> metric 3D back-projection,
# median depth pooling, per-joint OneEuro smoothing, bone-length projection,
# hinge stabilization & angle clamping. UDP JSON out (meters). HUD overlay.
# pip install depthai opencv-python mediapipe numpy

import socket, json, time, sys, argparse, math
import numpy as np, cv2
import depthai as dai
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# --------------------------------- Args ---------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--ip", default="127.0.0.1")
ap.add_argument("--port", type=int, default=9000)
ap.add_argument("--flip", action="store_true", help="mirror the image for selfie view")
ap.add_argument("--det", type=float, default=0.35, help="detection confidence threshold")
ap.add_argument("--track", type=float, default=0.35, help="tracking confidence threshold")
ap.add_argument("--alpha", type=float, default=0.2, help="smoothing factor for OneEuro filter (pose joints & metric3D)")
ap.add_argument("--no-hands", action="store_true", help="disable hands curls")
ap.add_argument("--no-head", action="store_true", help="disable head yaw/pitch/roll")
ap.add_argument("--no-expr", action="store_true", help="disable facial expression outputs")
ap.add_argument("--no-depth", action="store_true", help="disable StereoDepth (fallback to MediaPipe relative depth)")
ap.add_argument("--k", type=int, default=3, help="median pooling radius for depth (patch size 2k+1)")
ap.add_argument("--scale", type=float, default=1.0, help="output unit scale: 1.0=meters, 100.0=centimeters")
ap.add_argument("--vthr", type=float, default=0.50, help="visibility threshold for keypoints")
ap.add_argument("--velthr", type=float, default=1.0, help="per-frame max delta (normalized coords) for outlier gating")
ap.add_argument("--face-lines", choices=["off","contours","mesh","both"], default="contours",
                help="overlay face wireframe (contours/mesh/both/off)")
args = ap.parse_args()

UDP_ADDR = (args.ip, args.port)

# --------------------------------- MediaPipe ---------------------------------
mp_hol = mp.solutions.holistic
holistic = mp_hol.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    refine_face_landmarks=True,   # better eyes/iris
    min_detection_confidence=args.det,
    min_tracking_confidence=args.track
)

# === Expressions from face landmarks (0..1) ===
IDX = dict(
    eyeL_up=159, eyeL_dn=145, eyeR_up=386, eyeR_dn=374,
    mouth_up=13, mouth_dn=14, mouth_l=61, mouth_r=291,
    browL=105, browR=334, eyeL_ctr=33, eyeR_ctr=263
)
def _dist2(a,b): return math.hypot(a.x-b.x, a.y-b.y)
def face_expr(face_lm):
    lm = face_lm.landmark
    eye_base = max(1e-6, _dist2(lm[IDX['eyeL_ctr']], lm[IDX['eyeR_ctr']]))
    blinkL = 1.0 - min(1.0, _dist2(lm[IDX['eyeL_up']], lm[IDX['eyeL_dn']]) / (0.06*eye_base))
    blinkR = 1.0 - min(1.0, _dist2(lm[IDX['eyeR_up']], lm[IDX['eyeR_dn']]) / (0.06*eye_base))
    mouthOpen = min(1.0, _dist2(lm[IDX['mouth_up']], lm[IDX['mouth_dn']]) / (0.09*eye_base))
    smile_raw = _dist2(lm[IDX['mouth_l']], lm[IDX['mouth_r']]) / (0.38*eye_base)
    smile = max(0.0, min(1.0, smile_raw - 1.0))
    browL = min(1.0, _dist2(lm[IDX['browL']], lm[IDX['eyeL_up']]) / (0.25*eye_base))
    browR = min(1.0, _dist2(lm[IDX['browR']], lm[IDX['eyeR_up']]) / (0.25*eye_base))
    browUp = max(0.0, (browL+browR)*0.5 - 0.5)
    return {
        "mouthOpen": float(mouthOpen),
        "smile":     float(smile),
        "browUp":    float(browUp),
        "blinkL":    float(max(0.0, min(1.0, blinkL))),
        "blinkR":    float(max(0.0, min(1.0, blinkR))),
    }

# --------------------------------- Pose subsets ---------------------------------
LANDMARKS = {
    11:"shoulder_L",12:"shoulder_R",13:"elbow_L",14:"elbow_R",
    15:"wrist_L",16:"wrist_R",23:"hip_L",24:"hip_R",25:"knee_L",
    26:"knee_R",27:"ankle_L",28:"ankle_R",0:"nose"
}
CONNS=[("shoulder_L","elbow_L"),("elbow_L","wrist_L"),("shoulder_R","elbow_R"),
       ("elbow_R","wrist_R"),("hip_L","knee_L"),("knee_L","ankle_L"),
       ("hip_R","knee_R"),("knee_R","ankle_R"),("shoulder_L","shoulder_R"),
       ("hip_L","hip_R"),("shoulder_L","hip_L"),("shoulder_R","hip_R"),
       ("nose","shoulder_L"),("nose","shoulder_R")]

# Hands
FINGERS = {"thumb":[1,2,3,4], "index":[5,6,7,8], "middle":[9,10,11,12],
           "ring":[13,14,15,16], "pinky":[17,18,19,20]}
HAND_COLOR_L = (255,180,60)
HAND_COLOR_R = (60,200,255)
HAND_CHAINS = [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]

# Bone constraints
PARENTS = {
  "spine":"pelvis", "neck":"spine", "head":"neck",
  "elbow_L":"shoulder_L", "wrist_L":"elbow_L",
  "elbow_R":"shoulder_R", "wrist_R":"elbow_R",
  "knee_L":"hip_L", "ankle_L":"knee_L",
  "knee_R":"hip_R", "ankle_R":"knee_R"
}
ROOTS = ["pelvis","hip_L","hip_R","shoulder_L","shoulder_R"]

# --------------------------------- Filters & Draw ---------------------------------
class OneEuro:
    def __init__(self,a=0.2): self.a=a; self.prev_vec=None
    def __call__(self,x_vec):  # np.array shape (N,)
        x_vec = x_vec.astype(np.float32, copy=False)
        if self.prev_vec is None: self.prev_vec=x_vec.copy()
        self.prev_vec = self.a*x_vec + (1-self.a)*self.prev_vec
        return self.prev_vec

filt_vec = OneEuro(args.alpha)   # for pseudo-3D fallback
filter3d = {}                    # per-joint filters for metric 3D

def smooth3d(name, vec, alpha=None):
    if alpha is None: alpha = args.alpha
    f = filter3d.get(name)
    if f is None:
        f = OneEuro(alpha); filter3d[name] = f
    out = f(np.asarray(vec, np.float32).reshape(-1))
    return out.reshape(3)

def draw_pose(frame,px,conf,thr=0.4):
    for a,b in CONNS:
        if a in px and b in px:
            ok = conf.get(a,0.0)>=thr and conf.get(b,0.0)>=thr
            cv2.line(frame, px[a], px[b], (0,220,0) if ok else (0,0,255), 2, cv2.LINE_AA)
    for n,(u,v) in px.items():
        c=conf.get(n,0.0)
        cv2.circle(frame,(u,v),4,(0,220,0) if c>=thr else (0,0,255),-1, lineType=cv2.LINE_AA)

def draw_one_hand(frame, hand_lms, color):
    h, w = frame.shape[:2]
    pts = []
    for lm in hand_lms.landmark:
        u = int(np.clip(lm.x * w, 0, w - 1))
        v = int(np.clip(lm.y * h, 0, h - 1))
        pts.append((u, v))
    for chain in HAND_CHAINS:
        for a,b in zip(chain[:-1], chain[1:]):
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for (u,v) in pts:
        cv2.circle(frame, (u,v), 3, color, -1, lineType=cv2.LINE_AA)

def draw_hands_overlay(frame, res):
    if res.left_hand_landmarks:  draw_one_hand(frame, res.left_hand_landmarks, HAND_COLOR_L)
    if res.right_hand_landmarks: draw_one_hand(frame, res.right_hand_landmarks, HAND_COLOR_R)

def draw_face_lines(frame, face_landmarks, mode="contours"):
    if not face_landmarks or mode == "off": return
    if mode in ("contours", "both"):
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )
    if mode in ("mesh", "both"):
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )
    try:
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
        )
    except Exception:
        pass

# --------------------------------- Head YPR ---------------------------------
def head_ypr_from_face(face_lm):
    try:
        le = face_lm.landmark[263]; re = face_lm.landmark[33]; nose = face_lm.landmark[1]
    except Exception:
        return (0.0,0.0,0.0)
    le2 = np.array([le.x, le.y]); re2 = np.array([re.x, re.y])
    eye_vec = le2 - re2
    eye_mid = 0.5*(le2 + re2)
    nose2 = np.array([nose.x, nose.y])
    roll = math.atan2(eye_vec[1], eye_vec[0])
    yaw  = math.atan2((nose2 - eye_mid)[0], abs(eye_vec[0]) + 1e-6)
    pitch= -math.atan2((nose2 - eye_mid)[1], abs(eye_vec[0]) + 1e-6)
    return (float(yaw), float(pitch), float(roll))

# --------------------------------- Depth pipeline ---------------------------------
def build_pipeline(use_depth=True):
    p=dai.Pipeline()
    cam=p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setFps(30)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setVideoSize(1280,720)
    xrgb=p.create(dai.node.XLinkOut); xrgb.setStreamName("video")
    cam.video.link(xrgb.input)
    if not use_depth:
        return p

    monoL = p.create(dai.node.MonoCamera)
    monoR = p.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    depth = p.create(dai.node.StereoDepth)
    depth.setDepthAlign(dai.CameraBoardSocket.RGB)   # align to color
    depth.setSubpixel(True)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    # 可选：轻时域滤波（极小延迟开销）
    # depth.setTemporalFilter(True)

    monoL.out.link(depth.left)
    monoR.out.link(depth.right)

    xdep = p.create(dai.node.XLinkOut); xdep.setStreamName("depth")
    depth.depth.link(xdep.input)
    return p

# --------------------------------- Geometry helpers ---------------------------------
def ensure_pelvis(xyz):
    if "hip_L" in xyz and "hip_R" in xyz:
        xyz["pelvis"] = 0.5*(xyz["hip_L"] + xyz["hip_R"])
    return xyz

def bone_length_template_init(xyz):
    bl = {}
    for ch, pa in PARENTS.items():
        if ch in xyz and pa in xyz:
            v = xyz[ch]-xyz[pa]
            bl[(pa,ch)] = float(np.linalg.norm(v)+1e-6)
    return bl

def bone_length_project(xyz, bl):
    out = dict(xyz)
    for (pa,ch), L in bl.items():
        if pa in out and ch in out:
            v = out[ch]-out[pa]
            n = np.linalg.norm(v)
            if n>1e-6:
                out[ch] = out[pa] + v*(L/n)
    return out

def stabilize_hinge(xyz, pa, ch, tip, pole):
    if not (pa in xyz and ch in xyz and tip in xyz): return
    a = xyz[pa]; b = xyz[ch]; t = xyz[tip]
    axis = b - a
    n_axis = np.linalg.norm(axis)
    if n_axis < 1e-6: return
    axis_n = axis / n_axis
    n = np.cross(axis_n, pole)
    nn = np.linalg.norm(n)
    if nn < 1e-6: return
    n = n/nn
    t_proj = t - np.dot(t - b, n) * n
    xyz[tip] = t_proj

def clamp_hinge(xyz, pa, ch, tip, min_deg=0.0, max_deg=155.0):
    if not (pa in xyz and ch in xyz and tip in xyz): return
    a = xyz[pa]; b = xyz[ch]; t = xyz[tip]
    u = a - b; v = t - b
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: return
    cosang = np.clip(np.dot(u,v)/(nu*nv), -1, 1)
    ang = math.degrees(math.acos(cosang))
    if ang < min_deg or ang > max_deg:
        u_n = u/nu
        v_par = u_n * np.dot(v, u_n)
        v_perp = v - v_par
        nvp = np.linalg.norm(v_perp)
        if nvp < 1e-6: return
        v_perp_n = v_perp / nvp
        target = np.deg2rad(np.clip(ang, min_deg, max_deg))
        new_v = nu*np.cos(target)*(-u_n) + nv*np.sin(target)*v_perp_n
        xyz[tip] = b + new_v

# --- finger curl computation (0..1) ---
def finger_curl(hand_lms, finger):
    idx = FINGERS[finger]
    p1 = np.array([hand_lms[idx[0]].x, hand_lms[idx[0]].y, hand_lms[idx[0]].z], dtype=np.float32)
    p2 = np.array([hand_lms[idx[1]].x, hand_lms[idx[1]].y, hand_lms[idx[1]].z], dtype=np.float32)
    p3 = np.array([hand_lms[idx[2]].x, hand_lms[idx[2]].y, hand_lms[idx[2]].z], dtype=np.float32)
    v1 = p2 - p1; v2 = p3 - p2
    n1 = np.linalg.norm(v1) + 1e-6; n2 = np.linalg.norm(v2) + 1e-6
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang_deg = math.degrees(math.acos(cosang))  # ~30° straight, ~150° curled
    return float(np.clip((ang_deg - 30.0) / (150.0 - 30.0), 0.0, 1.0))

# --------------------------------- Main ---------------------------------
def main():
    if not dai.Device.getAllAvailableDevices():
        print("No OAK-D found. Please use USB3 cable/port.", file=sys.stderr); return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pipe = build_pipeline(use_depth=not args.no_depth)
    fps,n,t0 = 0.0,0,time.time()
    last_frame = None
    seq = 0

    with dai.Device(pipe) as device:
        q_rgb = device.getOutputQueue("video", maxSize=4, blocking=False)
        q_dep = device.getOutputQueue("depth", maxSize=4, blocking=False) if not args.no_depth else None

        calib = device.readCalibration()
        intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 1280, 720)
        fx, fy = intr[0][0], intr[1][1]; cx, cy = intr[0][2], intr[1][2]

        bone_len_template = None
        prev_xyz_norm = {}
        ts_dev = None

        while True:
            pkt = q_rgb.tryGet()
            if pkt is not None:
                last_frame = pkt.getCvFrame()
                try:
                    ts_dev = float(pkt.getTimestampDevice().total_seconds())
                except Exception:
                    ts_dev = float(pkt.getTimestamp().total_seconds())

            if last_frame is None:
                if (cv2.waitKey(1)&0xFF) in (27,ord('q'),ord('Q')): break
                time.sleep(0.005); continue

            depth_m = None
            if q_dep is not None:
                dpk = q_dep.tryGet()
                if dpk is not None:
                    depth_m = dpk.getFrame().astype(np.float32) / 1000.0  # meters

            frame = last_frame.copy()
            if args.flip: frame = cv2.flip(frame, 1)
            h,w = frame.shape[:2]

            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            payload={"t":time.time(),"joints":{},"conf":{}}
            px={}; detected = res.pose_landmarks is not None

            # Pose pixels & confidences
            if detected:
                lm = res.pose_landmarks.landmark
                for idx,name in LANDMARKS.items():
                    p=lm[idx]
                    u=int(np.clip(p.x*w,0,w-1)); v=int(np.clip(p.y*h,0,h-1))
                    px[name]=(u,v); payload["conf"][name]=float(p.visibility)

            # metric 3D by back-projection (+ per-joint smoothing)
            xyz_metric = {}
            if detected and (depth_m is not None):
                k = max(1, int(args.k))
                def depth_at(u, v):
                    u0,v0 = int(u), int(v)
                    x1,x2 = max(0,u0-k), min(w-1,u0+k)
                    y1,y2 = max(0,v0-k), min(h-1,v0+k)
                    patch = depth_m[y1:y2+1, x1:x2+1]
                    if patch.size==0: return 0.0
                    valid = patch[patch>0]
                    return float(np.median(valid)) if valid.size>0 else 0.0
                for name,(u,v) in px.items():
                    Z = depth_at(u,v)
                    if Z<=0: continue
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    raw = np.array([X, -Y, Z], np.float32)   # Y up
                    xyz_metric[name] = smooth3d(name, raw, alpha=args.alpha)

            # fallback: MediaPipe pseudo-3D (smoothed)
            if detected and (depth_m is None or len(xyz_metric)==0):
                lm = res.pose_landmarks.landmark
                flat=[]; order=[]
                for idx,name in LANDMARKS.items():
                    p=lm[idx]; X=(p.x-0.5); Y=(0.5-p.y); Z=-p.z*0.7
                    order.append(name); flat += [X,Y,Z]
                    payload["conf"][name]=float(p.visibility)
                sm = filt_vec(np.array(flat,np.float32))
                i=0
                for name in order:
                    xyz_metric[name]=np.array([float(sm[i]),float(sm[i+1]),float(sm[i+2])], np.float32); i+=3

            # no pose at all
            if len(xyz_metric)==0:
                n+=1
                if n%10==0:
                    t1=time.time(); fps=10.0/max(1e-3,(t1-t0)); t0=t1
                cv2.putText(frame, f"OAK-D S2 | UDP {UDP_ADDR[0]}:{UDP_ADDR[1]} | FPS {fps:.1f} | pose:False",
                            (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(frame, "NO POSE: step back, show torso, add light",
                            (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                if res.face_landmarks and args.face_lines != "off":
                    draw_face_lines(frame, res.face_landmarks, args.face_lines)
                cv2.imshow("OAK-D S2 Pose+Hands Sender", frame)
                if (cv2.waitKey(1)&0xFF) in (27,ord('q'),ord('Q')): break
                time.sleep(0.005); continue

            # synthesize pelvis
            xyz_metric = ensure_pelvis(xyz_metric)

            # --- visibility & velocity gating (drop outliers) ---
            V_THR = args.vthr; VEL_THR = args.velthr
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                xyz_norm = {}
                for idx,name in LANDMARKS.items():
                    if name in xyz_metric:
                        p=lm[idx]
                        xn=np.array([(p.x-0.5),(0.5-p.y),-p.z*0.7], np.float32)
                        xyz_norm[name]=xn
                for name in list(xyz_metric.keys()):
                    vis = payload["conf"].get(name, 1.0)
                    if vis < V_THR and name in prev_xyz_norm:
                        # keep previous position by dropping this sample (project+SLERP will smooth)
                        xyz_metric.pop(name, None)
                        continue
                    if (name in prev_xyz_norm) and (name in xyz_norm):
                        if np.linalg.norm(xyz_norm[name] - prev_xyz_norm[name]) > VEL_THR:
                            # sudden jump -> drop current joint this frame
                            xyz_metric.pop(name, None)
                prev_xyz_norm = xyz_norm

            # init bone-length template once
            if bone_len_template is None:
                bone_len_template = bone_length_template_init(xyz_metric)

            # length projection
            xyz_metric = bone_length_project(xyz_metric, bone_len_template)

            # hinge stabilize & clamp
            CAM_DIR = np.array([0,0,-1], np.float32)
            def pole_for(side):
                s = "shoulder_"+side; h = "hip_"+side
                if s in xyz_metric and h in xyz_metric:
                    p = np.cross(CAM_DIR, xyz_metric[s]-xyz_metric[h])
                    n = np.linalg.norm(p)
                    if n>1e-6: return p/n
                return np.array([0,1,0], np.float32)
            pole_L = pole_for("L"); pole_R = pole_for("R")
            stabilize_hinge(xyz_metric, "shoulder_L","elbow_L","wrist_L", pole_L)
            stabilize_hinge(xyz_metric, "shoulder_R","elbow_R","wrist_R", pole_R)
            stabilize_hinge(xyz_metric, "hip_L","knee_L","ankle_L", pole_L)
            stabilize_hinge(xyz_metric, "hip_R","knee_R","ankle_R", pole_R)

            clamp_hinge(xyz_metric, "shoulder_L","elbow_L","wrist_L", 0, 155)
            clamp_hinge(xyz_metric, "shoulder_R","elbow_R","wrist_R", 0, 155)
            clamp_hinge(xyz_metric, "hip_L","knee_L","ankle_L", 0, 160)
            clamp_hinge(xyz_metric, "hip_R","knee_R","ankle_R", 0, 160)

            # root align (pelvis at origin)
            if "pelvis" in xyz_metric:
                root = xyz_metric["pelvis"].copy()
                for k in list(xyz_metric.keys()):
                    xyz_metric[k] = (xyz_metric[k] - root)

            # units
            if abs(args.scale - 1.0) > 1e-6:
                for k in list(xyz_metric.keys()):
                    xyz_metric[k] = xyz_metric[k] * args.scale

            # Hands curls
            if not args.no_hands:
                hands = {"L":{}, "R":{}}
                if res.left_hand_landmarks:
                    lms = res.left_hand_landmarks.landmark
                    for f in FINGERS.keys(): hands["L"][f] = finger_curl(lms, f)
                if res.right_hand_landmarks:
                    lms = res.right_hand_landmarks.landmark
                    for f in FINGERS.keys(): hands["R"][f] = finger_curl(lms, f)
                payload["hands"] = hands

            # Head YPR
            if not args.no_head:
                head = {"yaw":0.0,"pitch":0.0,"roll":0.0}
                if res.face_landmarks:
                    y,p,r = head_ypr_from_face(res.face_landmarks)
                    if args.flip:  # selfie-view correction
                        y = -y; r = -r
                    head = {"yaw":y,"pitch":p,"roll":r}
                payload["head"] = head

            # Expressions
            if not args.no_expr:
                expr = {}
                if res.face_landmarks:
                    expr = face_expr(res.face_landmarks)
                payload["expr"] = expr
            else:
                payload["expr"] = {}

            # timestamps & sequence
            payload["ts"] = {"t_dev": ts_dev, "t_send": time.time(), "seq": int(seq)}
            payload["detected"] = bool(detected)
            payload["fps"] = float(fps)
            seq += 1

            # joints payload
            payload["joints"] = {k:[float(v[0]), float(v[1]), float(v[2])] for k,v in xyz_metric.items()}

            # send
            try:
                sock.sendto(json.dumps(payload).encode("utf-8"), UDP_ADDR)
            except Exception as e:
                print("UDP send error:", e, file=sys.stderr)

            # ---------------- HUD & Overlays ----------------
            n+=1
            if n%10==0:
                t1=time.time(); fps=10.0/max(1e-3,(t1-t0)); t0=t1

            cv2.putText(frame, f"OAK-D S2 | UDP {UDP_ADDR[0]}:{UDP_ADDR[1]} | FPS {fps:.1f} | pose:{bool(detected)} | depth:{'ON' if q_dep else 'OFF'}",
                        (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2, cv2.LINE_AA)

            if detected: draw_pose(frame, px, payload["conf"], thr=max(0.25, args.det*0.8))
            else:
                cv2.putText(frame, "NO POSE: step back, show torso, add light",
                            (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

            if res.face_landmarks and args.face_lines != "off":
                draw_face_lines(frame, res.face_landmarks, args.face_lines)

            draw_hands_overlay(frame, res)

            if (not args.no_expr) and res.face_landmarks:
                e = payload.get("expr", {})
                cv2.putText(frame, f"expr M:{e.get('mouthOpen',0):.2f} S:{e.get('smile',0):.2f} B:{e.get('browUp',0):.2f} L:{e.get('blinkL',0):.2f} R:{e.get('blinkR',0):.2f}",
                            (8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,230,255), 1, cv2.LINE_AA)

            if ts_dev is not None:
                cv2.putText(frame, f"dev:{ts_dev:.3f}s  seq:{seq-1}", (8,74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,255), 1, cv2.LINE_AA)

            cv2.imshow("OAK-D S2 Pose+Hands Sender", frame)
            if (cv2.waitKey(1)&0xFF) in (27,ord('q'),ord('Q')): break
            time.sleep(0.002)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
