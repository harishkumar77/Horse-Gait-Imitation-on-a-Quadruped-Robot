"""
Horse video (DeepLabCut keypoints) -> Unitree A1 motion, using foot-space retargeting.

Why this replaces the old pipeline:
  - Old: keypoints -> draw skeleton image -> MobileNet -> "angles".
    The MobileNet output barely changes (about 3 deg over a gallop) and the values
    (2.6-3.2 rad) are interior angles, not A1 joint angles. They were also sent to the
    A1 *hip abduction* joint, which moves the leg sideways.
  - New: compute everything straight from the keypoints with geometry. No model needed.

Steps:
  1. Load DLC keypoints, drop low-confidence points, interpolate, smooth.
  2. For each leg, get the paw position relative to its shoulder/hip ("thai") point.
  3. Scale by (robot leg length / horse leg length) so the stride fits the robot.
  4. Body pitch comes from the front/back "thai" line.
  5. Body height is set so the lowest foot touches the ground -> natural bounce.
  6. Forward motion is set so the stance foot does not slide (no fake constant speed).
  7. Analytic 2-link IK gives A1 thigh/calf angles. Hip abduction stays at 0.
  8. Save a CSV and play it in PyBullet (optionally record an MP4).

Usage:
  python retarget_dlc_to_a1.py <dlc_file.h5|.csv> --fps 30 --out a1_motion.csv
  python retarget_dlc_to_a1.py <dlc_file.h5|.csv> --fps 30 --play
  python retarget_dlc_to_a1.py <dlc_file.h5|.csv> --fps 30 --play --record a1.mp4
  python retarget_dlc_to_a1.py --from-csv a1_motion.csv --play
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------- A1 geometry (pybullet_data a1/a1.urdf) ----------------
L_THIGH = 0.2
L_CALF = 0.2
LEG_LEN = L_THIGH + L_CALF

# Robot leg name -> (horse prefix in DLC, front?)
# A1: FR, FL, RR, RL. Horse DLC names: front_right, front_left, back_right, back_left
LEGS = {
    "FR": "front_right",
    "FL": "front_left",
    "RR": "back_right",
    "RL": "back_left",
}

# Thigh-joint position in the base frame (x forward, y left, z up).
# Overwritten from the URDF at runtime when pybullet is available.
HIP_POS = {
    "FR": np.array([0.183, -0.132, 0.0]),
    "FL": np.array([0.183, 0.132, 0.0]),
    "RR": np.array([-0.183, -0.132, 0.0]),
    "RL": np.array([-0.183, 0.132, 0.0]),
}


# ---------------- 1. Load DLC keypoints ----------------
def load_dlc(path):
    """Return dict bodypart -> array (N, 3) of x, y, likelihood."""
    if path.endswith(".h5"):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    cols = df.columns
    bp_level = cols.nlevels - 2
    out = {}
    for bp in sorted({c[bp_level] for c in cols}):
        sub = df.xs(bp, level=bp_level, axis=1)
        out[bp] = np.stack(
            [sub.xs("x", level=-1, axis=1).iloc[:, 0].values,
             sub.xs("y", level=-1, axis=1).iloc[:, 0].values,
             sub.xs("likelihood", level=-1, axis=1).iloc[:, 0].values],
            axis=1,
        ).astype(float)
    return out


def clean_track(xyp, pmin=0.6, fps=30.0):
    """Mask low-confidence points, interpolate gaps, smooth. Returns (N, 2) with y UP."""
    xy = xyp[:, :2].copy()
    xy[xyp[:, 2] < pmin] = np.nan
    xy = pd.DataFrame(xy).interpolate(limit_direction="both").values
    if np.isnan(xy).any():
        raise ValueError("A keypoint has no confident detections at all.")
    n = len(xy)
    win = int(round(fps * 0.15)) | 1  # ~150 ms window, odd
    win = min(win, n - (1 - n % 2))   # must be <= n and odd
    if win >= 5:
        xy = savgol_filter(xy, win, 3, axis=0)
    xy[:, 1] *= -1.0  # image y is down -> make it up
    return xy


# ---------------- 2. IK ----------------
def leg_ik(x, z):
    """Foot (x fwd, z up) relative to thigh joint, in sagittal plane -> (thigh, calf).
    A1 convention: foot = l1*a(t1) + l2*a(t1+t2), a(t) = (-sin t, -cos t). Calf < 0."""
    d = math.hypot(x, z)
    # A1 calf limit is [-2.7, -0.92] rad, so max reach is ~0.358 m (not the full 0.4 m).
    d = min(max(d, 0.17), 0.355)
    c2 = (d * d - L_THIGH ** 2 - L_CALF ** 2) / (2 * L_THIGH * L_CALF)
    t2 = -math.acos(max(-1.0, min(1.0, c2)))
    phi = math.atan2(-x, -z)
    t1 = phi - math.atan2(L_CALF * math.sin(t2), L_THIGH + L_CALF * math.cos(t2))
    return t1, t2


def leg_fk(t1, t2):
    x = -L_THIGH * math.sin(t1) - L_CALF * math.sin(t1 + t2)
    z = -L_THIGH * math.cos(t1) - L_CALF * math.cos(t1 + t2)
    return x, z


# ---------------- 3. Retarget ----------------
def retarget(kp, fps, pmin=0.6, stance_height=0.30, stride_gain=1.0):
    tr = {}
    for leg, h in LEGS.items():
        for part in ("thai", "knee", "paw"):
            name = f"{h}_{part}"
            if name not in kp:
                raise KeyError(f"Missing bodypart '{name}'. Found: {sorted(kp)}")
            tr[name] = clean_track(kp[name], pmin, fps)
    n = len(next(iter(tr.values())))

    front = 0.5 * (tr["front_left_thai"] + tr["front_right_thai"])
    back = 0.5 * (tr["back_left_thai"] + tr["back_right_thai"])

    # Which way is the horse facing in the video? Make "forward" = +x.
    facing = np.sign(np.median(front[:, 0] - back[:, 0])) or 1.0
    for k in tr:
        tr[k][:, 0] *= facing
    front[:, 0] *= facing
    back[:, 0] *= facing

    axis = front - back
    pitch_img = np.arctan2(axis[:, 1], axis[:, 0])  # nose up = positive
    # Remove constant camera tilt, keep the oscillation; also damp it for the small robot.
    pitch = 0.6 * (pitch_img - np.median(pitch_img))
    pitch = np.clip(pitch, -0.25, 0.25)  # stop tracking glitches from tipping the body

    # Horse leg length (pixels), per leg, median over time.
    scale = {}
    for leg, h in LEGS.items():
        l1 = np.linalg.norm(tr[f"{h}_knee"] - tr[f"{h}_thai"], axis=1)
        l2 = np.linalg.norm(tr[f"{h}_paw"] - tr[f"{h}_knee"], axis=1)
        horse_len = np.median(l1 + l2)
        # A standing horse leg is ~85% extended. Map that to the A1's normal stance
        # height under the hip (~0.30 m), so the robot keeps its own posture.
        scale[leg] = stance_height / (0.85 * horse_len)

    # Foot offset from its hip in the *body* frame (rotate out the image pitch).
    rel = {}
    for leg, h in LEGS.items():
        d = (tr[f"{h}_paw"] - tr[f"{h}_thai"]) * scale[leg]
        c, s = np.cos(-pitch_img), np.sin(-pitch_img)
        bx = c * d[:, 0] - s * d[:, 1]
        bz = s * d[:, 0] + c * d[:, 1]
        # Center the stride under the hip, then apply gain.
        bx = (bx - np.median(bx)) * stride_gain
        rel[leg] = np.stack([bx, bz], axis=1)

    # World foot heights for a base at z=0 with the body pitch applied.
    def foot_world(leg, i, base_x=0.0, base_z=0.0):
        hp = HIP_POS[leg]
        x_b = hp[0] + rel[leg][i, 0]
        z_b = hp[2] + rel[leg][i, 1]
        cp, sp = math.cos(pitch[i]), math.sin(pitch[i])
        return base_x + cp * x_b - sp * z_b, base_z + sp * x_b + cp * z_b

    # Body height: lowest foot on the ground -> natural bounce, flight phase if all feet up
    # relative to the median stance. Clip so the robot never sinks.
    low = np.array([min(foot_world(l, i)[1] for l in LEGS) for i in range(n)])
    base_z = -low
    # Contacts: a foot is in stance if it is within 1.5 cm of the ground.
    contact = {l: np.zeros(n, bool) for l in LEGS}
    for i in range(n):
        for l in LEGS:
            contact[l][i] = foot_world(l, i, 0.0, base_z[i])[1] < 0.015

    # Forward motion: keep stance feet fixed on the ground (no sliding).
    base_x = np.zeros(n)
    for i in range(1, n):
        stance = [l for l in LEGS if contact[l][i] and contact[l][i - 1]]
        if stance:
            dx = np.mean([foot_world(l, i)[0] - foot_world(l, i - 1)[0] for l in stance])
            base_x[i] = base_x[i - 1] - dx
        else:  # flight: keep previous velocity
            base_x[i] = base_x[i - 1] + (base_x[i - 1] - base_x[i - 2] if i > 1 else 0.0)

    # Make forward motion smooth and one-way. Paw noise, left/right label swaps and
    # wrong contacts can make single frames step backward; the horse never does that.
    v = np.clip(np.diff(base_x, prepend=base_x[0]), 0.0, None)
    k = max(1, int(fps * 0.5))  # average over ~half a second (about one stride)
    v = np.convolve(np.pad(v, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    base_x = np.cumsum(v)
    # Light smoothing on body height to remove frame-to-frame jitter.
    base_z = np.convolve(np.pad(base_z, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")

    # IK per leg.
    rows = []
    for i in range(n):
        r = {"t": i / fps, "base_x": base_x[i], "base_z": base_z[i], "base_pitch": pitch[i]}
        for l in LEGS:
            t1, t2 = leg_ik(rel[l][i, 0], rel[l][i, 1])
            r[f"{l}_hip"] = 0.0
            r[f"{l}_thigh"] = t1
            r[f"{l}_calf"] = t2
            r[f"{l}_contact"] = int(contact[l][i])
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------- 4. Playback ----------------
def load_a1(p, gui=True):
    import pybullet_data
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    robot = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "a1/a1.urdf"), [0, 0, 0.5], useFixedBase=False)
    names = {p.getJointInfo(robot, j)[1].decode(): j for j in range(p.getNumJoints(robot))}

    def jid(leg, part):
        alt = {"hip": ["hip_joint"], "thigh": ["upper_joint", "thigh_joint"],
               "calf": ["lower_joint", "calf_joint"]}[part]
        for a in alt:
            if f"{leg}_{a}" in names:
                return names[f"{leg}_{a}"]
        raise KeyError(f"No joint for {leg} {part}: {list(names)}")

    ids = {(l, part): jid(l, part) for l in LEGS for part in ("hip", "thigh", "calf")}
    # Read real thigh-joint positions from the URDF.
    p.resetBasePositionAndOrientation(robot, [0, 0, 0], [0, 0, 0, 1])
    for l in LEGS:
        HIP_POS[l] = np.array(p.getLinkState(robot, ids[(l, "thigh")])[4])
    return robot, ids


def play(df, fps, speed=1.0, record=None, loop=True, camera="fixed"):
    """camera='fixed': camera stays still and the robot walks across the screen.
       camera='follow': camera tracks the robot (robot looks like it stays in place)."""
    import pybullet as p
    robot, ids = load_a1(p, gui=True)
    dist_total = float(df["base_x"].iloc[-1] - df["base_x"].iloc[0])
    x_mid = float(df["base_x"].iloc[0]) + 0.5 * dist_total
    cam_dist = max(1.5, 0.9 * dist_total)
    print(f"Robot travels {dist_total:.2f} m in {len(df) / fps:.1f} s "
          f"({dist_total / (len(df) / fps):.2f} m/s)")
    if camera == "fixed":
        p.resetDebugVisualizerCamera(cam_dist, 0, -8, [x_mid, 0, 0.25])
    if record:
        log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, record)
    dt = 1.0 / fps / speed
    while True:
        for _, r in df.iterrows():
            q = p.getQuaternionFromEuler([0, -r["base_pitch"], 0])  # +pitch = nose up
            p.resetBasePositionAndOrientation(robot, [r["base_x"], 0, r["base_z"]], q)
            for l in LEGS:
                for part in ("hip", "thigh", "calf"):
                    p.resetJointState(robot, ids[(l, part)], r[f"{l}_{part}"])
            if camera == "follow":
                p.resetDebugVisualizerCamera(1.2, 0, -10, [r["base_x"], 0, 0.25])
            time.sleep(dt)
        if record:
            p.stopStateLogging(log)
            record = None
        if not loop:
            break
        if camera == "follow":
            # keep walking forward from where it stopped
            df = df.copy()
            df["base_x"] += df["base_x"].iloc[-1] - df["base_x"].iloc[0]
        # fixed camera: start again from the left side, like replaying the clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dlc", nargs="?", help="DeepLabCut .h5 or .csv")
    ap.add_argument("--from-csv", help="play an already retargeted CSV")
    ap.add_argument("--fps", type=float, default=30.0, help="video fps")
    ap.add_argument("--pmin", type=float, default=0.6, help="DLC likelihood cutoff")
    ap.add_argument("--stride-gain", type=float, default=1.0)
    ap.add_argument("--out", default="a1_motion.csv")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = slow-mo)")
    ap.add_argument("--record", help="save MP4 of the playback")
    ap.add_argument("--camera", choices=["fixed", "follow"], default="fixed",
                    help="fixed: robot walks across the screen; follow: camera tracks it")
    a = ap.parse_args()

    if a.from_csv:
        df = pd.read_csv(a.from_csv)
    else:
        if not a.dlc:
            ap.error("give a DLC file or --from-csv")
        try:  # get exact hip positions from the URDF before retargeting
            import pybullet as p
            load_a1(p, gui=False)
            p.disconnect()
        except Exception as e:
            print("[info] using default A1 hip positions:", e)
        df = retarget(load_dlc(a.dlc), a.fps, a.pmin, stride_gain=a.stride_gain)
        df.to_csv(a.out, index=False)
        print(f"Saved {len(df)} frames -> {a.out}")
        print("Contact fraction per leg:",
              {l: round(df[f'{l}_contact'].mean(), 2) for l in LEGS})
        print("Thigh range (rad):",
              {l: (round(df[f'{l}_thigh'].min(), 2), round(df[f'{l}_thigh'].max(), 2)) for l in LEGS})

    if a.play:
        play(df, a.fps, a.speed, a.record, camera=a.camera)


if __name__ == "__main__":
    main()
