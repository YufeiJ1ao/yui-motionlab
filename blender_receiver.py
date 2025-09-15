# =========================================================
# Blender UDP Pose Receiver — Human-Like Smooth Version (FULL)
# + Fingers (curl) and Head (yaw/pitch/roll) with smoothing & speed limits
# + Wrist/Ankle driving, seq/fps/detected fields, drop stats, axis auto-detect
# =========================================================
# Usage:
# 1) In Blender, open Scripting > Text Editor, paste this file, set ARMATURE_NAME.
# 2) Run. It starts a UDP thread and a 60Hz timer to drive your rig.
# 3) Works with the OAK-D sender you shared (with seq/fps/detected).
# ---------------------------------------------------------

import bpy, socket, json, threading, time, math
from collections import defaultdict
from mathutils import Vector, Quaternion, Euler

# ---------- Config ----------
UDP_IP, UDP_PORT = "127.0.0.1", 9000
ARMATURE_NAME     = "Armature"    # <-- change to your armature object name

APPLY_HZ   = 60.0
MAX_LAG    = 0.10          # seconds: if packet older than this, relax to rest
CONF_THR   = 0.35
ANG_SPEED_LIMIT_DPS = 720.0  # per-bone max angular speed

# Time constants (lower = snappier)
TAU_ARM   = 0.08
TAU_LEG   = 0.10
TAU_TORSO = 0.12   # also used by head filter if desired

# EMA on landmark positions (for segment dir vectors)
EMA_POS = 0.25

# Fingers config
FINGER_MAX_DEG = [30.0, 45.0, 60.0]  # 3 joints per finger
FINGER_WEIGHTS = [0.5, 0.8, 1.0]
FINGER_RELAX_DPS = 300.0             # relax speed when stale (deg/sec)

# Head smoothing/limit
HEAD_TAU = 0.10
HEAD_MAX_DPS = 360.0

# ---------- Joints list (from sender) ----------
JOINTS = [
    "shoulder_L","elbow_L","wrist_L",
    "shoulder_R","elbow_R","wrist_R",
    "hip_L","knee_L","ankle_L",
    "hip_R","knee_R","ankle_R",
    "nose"
]

# ---------- UDP name -> Blender bone ----------
BONE_MAP = {
    "shoulder_L": "J_Bip_L_UpperArm",
    "elbow_L":    "J_Bip_L_LowerArm",
    "wrist_L":    "J_Bip_L_Hand",
    "shoulder_R": "J_Bip_R_UpperArm",
    "elbow_R":    "J_Bip_R_LowerArm",
    "wrist_R":    "J_Bip_R_Hand",
    "hip_L":      "J_Bip_L_UpperLeg",
    "knee_L":     "J_Bip_L_LowerLeg",
    "ankle_L":    "J_Bip_L_Foot",
    "hip_R":      "J_Bip_R_UpperLeg",
    "knee_R":     "J_Bip_R_LowerLeg",
    "ankle_R":    "J_Bip_R_Foot",
}

# ---------- Head & Fingers mapping ----------
HEAD_BONE = "J_Head"  # change if your rig uses another name

FINGER_BONES = {
    "L": {
        "thumb":  ["J_Bip_L_Thumb1","J_Bip_L_Thumb2","J_Bip_L_Thumb3"],
        "index":  ["J_Bip_L_Index1","J_Bip_L_Index2","J_Bip_L_Index3"],
        "middle": ["J_Bip_L_Middle1","J_Bip_L_Middle2","J_Bip_L_Middle3"],
        "ring":   ["J_Bip_L_Ring1","J_Bip_L_Ring2","J_Bip_L_Ring3"],
        "pinky":  ["J_Bip_L_Little1","J_Bip_L_Little2","J_Bip_L_Little3"], 
    },
    "R": {
        "thumb":  ["J_Bip_R_Thumb1","J_Bip_R_Thumb2","J_Bip_R_Thumb3"],
        "index":  ["J_Bip_R_Index1","J_Bip_R_Index2","J_Bip_R_Index3"],
        "middle": ["J_Bip_R_Middle1","J_Bip_R_Middle2","J_Bip_R_Middle3"],
        "ring":   ["J_Bip_R_Ring1","J_Bip_R_Ring2","J_Bip_R_Ring3"],
        "pinky":  ["J_Bip_R_Little1","J_Bip_R_Little2","J_Bip_R_Little3"],
    }
}

# ---------- Helpers ----------
def mp_to_blender_vec(j):
    """Sender: (x right, y up, z forward) -> Blender: (X right, Y forward, Z up)"""
    x, y, z = j
    return Vector((x, z, y))

def quat_from_two_vectors(v_from: Vector, v_to: Vector) -> Quaternion:
    a = v_from.normalized(); b = v_to.normalized()
    dot = max(-1.0, min(1.0, a.dot(b)))
    if dot < -0.999999:
        axis = a.orthogonal().normalized()
        return Quaternion((0.0, axis.x, axis.y, axis.z))
    axis = a.cross(b)
    q = Quaternion((1.0 + dot, axis.x, axis.y, axis.z))
    return q.normalized()

def slerp_time(q_from: Quaternion, q_to: Quaternion, dt: float, tau: float) -> Quaternion:
    if tau <= 1e-6: return q_to.copy()
    alpha = 1.0 - math.exp(-dt / tau)
    return Quaternion.slerp(q_from, q_to, max(0.0, min(1.0, alpha)))

def clamp_ang_speed(q_curr: Quaternion, q_next: Quaternion, dt: float, max_dps: float) -> Quaternion:
    dq = q_curr.rotation_difference(q_next)
    angle = math.degrees(dq.angle)
    max_deg = max_dps * dt
    if angle <= max_deg or angle < 1e-3:
        return q_next
    ratio = max_deg / angle
    return Quaternion.slerp(q_curr, q_next, ratio)

# ---------- Armature / bones ----------
arm = bpy.data.objects.get(ARMATURE_NAME)
if arm is None:
    raise RuntimeError(f"Armature '{ARMATURE_NAME}' not found")

pbones = {}
rest_dir = {}
for udp, bone_name in BONE_MAP.items():
    pb = arm.pose.bones.get(bone_name)
    if not pb: continue
    pb.rotation_mode = 'QUATERNION'
    pbones[bone_name] = pb
    eb = pb.bone
    d = (eb.tail_local - eb.head_local)
    rest_dir[bone_name] = d.normalized() if d.length > 0 else Vector((0,1,0))

# Head & finger bones
_head_pb = arm.pose.bones.get(HEAD_BONE)
if _head_pb: _head_pb.rotation_mode = 'XYZ'
for side in FINGER_BONES:
    for fname, chain in FINGER_BONES[side].items():
        for bname in chain:
            pb = arm.pose.bones.get(bname)
            if pb:
                pb.rotation_mode = 'XYZ'  # easier to bend on a single axis

# ensure pose mode once
try:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
except Exception:
    pass

# ---------- Optional per-bone static offsets (fine-tuning) ----------
bone_offset = defaultdict(lambda: Quaternion((1,0,0,0)))
# Example tweak (uncomment if needed):
# bone_offset["J_Bip_L_UpperArm"] = Quaternion(Euler((0,0,math.radians(15)), 'XYZ'))

# ---------- UDP listener (upgraded) ----------
_latest = {
    "t":0.0, "seq":-1, "joints":{}, "conf":{}, "hands":{}, "head":{},
    "fps":0.0, "detected":False
}
_lock = threading.Lock()
_stats = {"last_seq":-1, "drops":0, "recv":0}

def udp_loop():
    global _latest
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1<<20)  # 1MB recv buffer
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    while True:
        try:
            data,_ = sock.recvfrom(65535)
            msg = json.loads(data.decode("utf-8"))
            t   = float(msg.get("t", time.time()))
            seq = int(msg.get("seq", -1))
            with _lock:
                prev = _latest.get("seq", -1)
                _stats["recv"] += 1
                if prev >= 0 and seq > prev + 1:
                    _stats["drops"] += (seq - prev - 1)
                _latest = {
                    "t": t,
                    "seq": seq,
                    "joints": msg.get("joints", {}),
                    "conf":   msg.get("conf",   {}),
                    "hands":  msg.get("hands",  {}),
                    "head":   msg.get("head",   {}),
                    "fps":    msg.get("fps",    0.0),
                    "detected": bool(msg.get("detected", False)),
                }
        except BlockingIOError:
            time.sleep(0.0005)
        except Exception as e:
            print("[UDP] error:", e); time.sleep(0.01)

threading.Thread(target=udp_loop, daemon=True).start()

# ---------- EMA smoothing on joint positions ----------
ema_pos = {name: None for name in JOINTS}

def ema_update(name, v3: Vector, alpha=EMA_POS):
    prev = ema_pos.get(name)
    if prev is None:
        ema_pos[name] = v3.copy()
    else:
        ema_pos[name] = prev.lerp(v3, max(0.0, min(1.0, alpha)))
    return ema_pos[name]

def get_smooth_dir(joints, conf, a, b):
    if a not in joints or b not in joints: return None
    if conf.get(a,0.0) < CONF_THR or conf.get(b,0.0) < CONF_THR: return None
    va = ema_update(a, mp_to_blender_vec(joints[a]))
    vb = ema_update(b, mp_to_blender_vec(joints[b]))
    d = vb - va
    return d if d.length > 1e-6 else None

# ---------- Per-bone state ----------
last_rot = {bn: pb.rotation_quaternion.copy() for bn, pb in pbones.items()}
last_time = time.time()

def apply_segment(bone_name, dir_vec, tau, dt):
    pb = pbones.get(bone_name)
    if not pb or dir_vec is None: return
    q_curr = pb.rotation_quaternion.copy()
    q_target = quat_from_two_vectors(rest_dir[bone_name], dir_vec) * bone_offset[bone_name]
    q_smooth = slerp_time(q_curr, q_target, dt, tau)
    q_limited = clamp_ang_speed(q_curr, q_smooth, dt, ANG_SPEED_LIMIT_DPS)
    pb.rotation_quaternion = q_limited
    last_rot[bone_name] = q_limited

# ---------- Auto-detect finger bend axis (one-time heuristic) ----------
_finger_axis = {}  # bname -> (axis 'x'|'y'|'z', sign +1|-1)

def _detect_finger_axis(pb):
    base = pb.rotation_euler.copy()
    best_axis, best_score, best_sign = 'x', -1e9, 1
    for axis in ('x','y','z'):
        for sgn in (1,-1):
            e = base.copy()
            setattr(e, axis, getattr(e, axis) + sgn*math.radians(5))
            pb.rotation_euler = e
            # crude score: how much local -Y points into palm direction after bending
            loc = pb.matrix.to_3x3() @ Vector((0, -1, 0))
            score = loc.length
            if score > best_score:
                best_score, best_axis, best_sign = score, axis, sgn
    pb.rotation_euler = base
    return best_axis, best_sign

for side in FINGER_BONES:
    for fname, chain in FINGER_BONES[side].items():
        for bname in chain:
            pb = arm.pose.bones.get(bname)
            if pb and bname not in _finger_axis:
                try:
                    ax, sgn = _detect_finger_axis(pb)
                except Exception:
                    ax, sgn = 'x', -1
                _finger_axis[bname] = (ax, sgn)

# ---------- Fingers & Head ----------
def apply_fingers(arm_obj, hands):
    """
    hands = {"L":{"thumb":0..1, ...}, "R":{...}}
    Bend each finger chain according to curl value.
    """
    if not hands: return
    for side in ("L","R"):
        if side not in hands: continue
        hdict = hands[side]
        for fname, curl in hdict.items():
            chain = FINGER_BONES.get(side, {}).get(fname, [])
            for i, bname in enumerate(chain):
                pb = arm_obj.pose.bones.get(bname)
                if not pb: continue
                deg = FINGER_MAX_DEG[i] * FINGER_WEIGHTS[i] * float(curl)
                rad = math.radians(deg)
                ax, sgn = _finger_axis.get(bname, ('x', -1))
                e = pb.rotation_euler
                setattr(e, ax, sgn * -rad)  # bend toward palm
                pb.rotation_euler = e

def relax_fingers(arm_obj, dt):
    """When stale, gradually relax finger rotations back toward 0."""
    step = math.radians(FINGER_RELAX_DPS) * dt
    for side in FINGER_BONES:
        for fname, chain in FINGER_BONES[side].items():
            for bname in chain:
                pb = arm_obj.pose.bones.get(bname)
                if not pb: continue
                e = pb.rotation_euler
                # move each axis toward 0 by 'step'
                for ax in ('x','y','z'):
                    val = getattr(e, ax)
                    if abs(val) <= step:
                        setattr(e, ax, 0.0)
                    else:
                        setattr(e, ax, val - math.copysign(step, val))
                pb.rotation_euler = e

_head_last_euler = None
def _limit_euler_delta(prev: Euler, curr: Euler, dt: float, max_dps: float) -> Euler:
    if prev is None: return curr
    max_deg = max_dps * dt
    out = Euler((0,0,0), 'XYZ')
    for i in range(3):
        # wrap to [-pi, pi]
        d = ((curr[i] - prev[i] + math.pi) % (2*math.pi)) - math.pi
        d_deg = math.degrees(abs(d))
        if d_deg > max_deg:
            d = math.radians(math.copysign(max_deg, d))
        out[i] = prev[i] + d
    return out

def apply_head(arm_obj, head):
    """head = {"yaw":rad,"pitch":rad,"roll":rad} from sender (camera-ish)"""
    global _head_last_euler
    if not head: return
    pb = arm_obj.pose.bones.get(HEAD_BONE)
    if not pb: return
    pb.rotation_mode = 'XYZ'
    yaw   = float(head.get("yaw",   0.0))
    pitch = float(head.get("pitch", 0.0))
    roll  = float(head.get("roll",  0.0))

    # Mapping: X=pitch (nod), Y=roll (tilt), Z=yaw (turn)
    curr = Euler((pitch, roll, yaw), 'XYZ')

    # Low-pass
    now = time.time()
    dt = max(1e-3, now - getattr(apply_head, "_last_t", now))
    apply_head._last_t = now
    alpha = 1.0 - math.exp(-dt / HEAD_TAU)
    if _head_last_euler is None:
        smooth = curr
    else:
        smooth = Euler((
            _head_last_euler.x + alpha*(curr.x - _head_last_euler.x),
            _head_last_euler.y + alpha*(curr.y - _head_last_euler.y),
            _head_last_euler.z + alpha*(curr.z - _head_last_euler.z),
        ), 'XYZ')

    # Speed limit
    limited = _limit_euler_delta(_head_last_euler, smooth, dt, HEAD_MAX_DPS)
    pb.rotation_euler = limited
    _head_last_euler  = limited

# ---------- Timer tick ----------
last_time = time.time()

def tick():
    global last_time
    now = time.time()
    dt = max(1e-3, now - last_time)
    last_time = now

    # snapshot latest
    with _lock:
        data = _latest.copy()

    # stale / no joints: relax head & fingers and skip bones
    if (not data["joints"]) or ((now - data["t"]) > MAX_LAG):
        relax_fingers(arm, dt)
        apply_head(arm, {"yaw":0.0, "pitch":0.0, "roll":0.0})
        return 1.0 / APPLY_HZ

    j, c = data["joints"], data["conf"]

    # Upper limbs
    d_uL = get_smooth_dir(j,c,"shoulder_L","elbow_L")
    d_lL = get_smooth_dir(j,c,"elbow_L","wrist_L")
    d_uR = get_smooth_dir(j,c,"shoulder_R","elbow_R")
    d_lR = get_smooth_dir(j,c,"elbow_R","wrist_R")
    # Wrist orientation (segment alignment)
    d_wL = get_smooth_dir(j,c,"elbow_L","wrist_L")
    d_wR = get_smooth_dir(j,c,"elbow_R","wrist_R")

    # Lower limbs
    d_tL = get_smooth_dir(j,c,"hip_L","knee_L")
    d_sL = get_smooth_dir(j,c,"knee_L","ankle_L")
    d_tR = get_smooth_dir(j,c,"hip_R","knee_R")
    d_sR = get_smooth_dir(j,c,"knee_R","ankle_R")
    # Ankle orientation
    d_aL = get_smooth_dir(j,c,"knee_L","ankle_L")
    d_aR = get_smooth_dir(j,c,"knee_R","ankle_R")

    # Apply arms
    apply_segment(BONE_MAP["shoulder_L"], d_uL, TAU_ARM, dt)
    apply_segment(BONE_MAP["elbow_L"],    d_lL, TAU_ARM, dt)
    apply_segment(BONE_MAP["wrist_L"],    d_wL, TAU_ARM, dt)

    apply_segment(BONE_MAP["shoulder_R"], d_uR, TAU_ARM, dt)
    apply_segment(BONE_MAP["elbow_R"],    d_lR, TAU_ARM, dt)
    apply_segment(BONE_MAP["wrist_R"],    d_wR, TAU_ARM, dt)

    # Apply legs
    apply_segment(BONE_MAP["hip_L"],  d_tL, TAU_LEG, dt)
    apply_segment(BONE_MAP["knee_L"], d_sL, TAU_LEG, dt)
    apply_segment(BONE_MAP["ankle_L"],d_aL, TAU_LEG, dt)

    apply_segment(BONE_MAP["hip_R"],  d_tR, TAU_LEG, dt)
    apply_segment(BONE_MAP["knee_R"], d_sR, TAU_LEG, dt)
    apply_segment(BONE_MAP["ankle_R"],d_aR, TAU_LEG, dt)

    # Hands & Head
    apply_fingers(arm, data.get("hands", {}))
    apply_head(arm, data.get("head", {}))

    return 1.0 / APPLY_HZ

# ---------- Register timer ----------
try:
    bpy.app.timers.unregister(tick)
except Exception:
    pass
bpy.app.timers.register(tick, first_interval=0.0)

print(f"✅ Smooth Receiver @ {APPLY_HZ:.0f}Hz with fingers & head.")
print(f"   UDP bind {UDP_IP}:{UDP_PORT} | lag-thr {MAX_LAG*1000:.0f} ms | conf≥{CONF_THR}")
print("   If you need bone offsets, edit 'bone_offset' above and rerun.")
