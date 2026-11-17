"""
Physics replay of a retargeted horse motion on the Unitree A1 (PyBullet).

Gravity ON, free-floating body, ground contact, real motor torque limits.
The legs follow the reference with PD motor control. The robot only moves
forward if its feet push on the ground.

A virtual "harness" (spring + damper on the body) can help hold the body on
the reference path. Turn it down step by step to see what the legs can do alone:
    --assist 1.0   strong help (body almost held on the path)
    --assist 0.3   weak help
    --assist 0     no help: pure physics, real-life conditions

Input: the CSV saved by retarget_dlc_to_a1.py (e.g. trot_a1.csv).

Usage:
  python physics_replay_a1.py trot_a1.csv --assist 1.0
  python physics_replay_a1.py trot_a1.csv --assist 0
  python physics_replay_a1.py trot_a1.csv --assist 0.3 --speed 0.5 --record trot_physics.mp4
  python physics_replay_a1.py trot_a1.csv --assist 0 --direct      (no window, prints results only)
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data

LEGS = ["FR", "FL", "RR", "RL"]
PARTS = ["hip", "thigh", "calf"]

SIM_DT = 1.0 / 1000.0     # 1 kHz physics, like a real motor loop
TORQUE_LIMIT = 33.5       # A1 motor peak torque (Nm)
KP, KD = 80.0, 2.0        # joint PD gains (Nm/rad, Nm*s/rad)


def load_world(gui):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    p.changeDynamics(plane, -1, lateralFriction=1.0)
    robot = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "a1/a1.urdf"),
                       [0, 0, 0.35], useFixedBase=False)

    names = {p.getJointInfo(robot, j)[1].decode(): j for j in range(p.getNumJoints(robot))}
    alt = {"hip": ["hip_joint"], "thigh": ["upper_joint", "thigh_joint"],
           "calf": ["lower_joint", "calf_joint"]}
    ids = []
    for l in LEGS:
        for part in PARTS:
            ids.append(next(names[f"{l}_{a}"] for a in alt[part] if f"{l}_{a}" in names))

    # Feet need friction to push. Turn off the default motors so we can send torques.
    for j in range(p.getNumJoints(robot)):
        p.changeDynamics(robot, j, lateralFriction=1.0)
    for j in ids:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)
    mass = sum(p.getDynamicsInfo(robot, j)[0] for j in range(-1, p.getNumJoints(robot)))
    return robot, ids, mass


def ref_at(df, t, loop_shift):
    """Reference (base pos, pitch, joint angles) at time t, linear interpolation."""
    T = df["t"].values
    period = T[-1] + (T[1] - T[0])
    k, tt = divmod(t, period)
    i = min(int(np.searchsorted(T, tt, side="right")) - 1, len(T) - 2)
    a = (tt - T[i]) / (T[i + 1] - T[i])
    a = min(max(a, 0.0), 1.0)
    r0, r1 = df.iloc[i], df.iloc[i + 1]
    lerp = lambda c: (1 - a) * r0[c] + a * r1[c]
    x = lerp("base_x") + k * loop_shift
    q = np.array([lerp(f"{l}_{part}") for l in LEGS for part in PARTS])
    return np.array([x, 0.0, lerp("base_z")]), lerp("base_pitch"), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from retarget_dlc_to_a1.py")
    ap.add_argument("--assist", type=float, default=1.0, help="harness strength, 0 = none")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to simulate")
    ap.add_argument("--speed", type=float, default=1.0, help="display speed (0.5 = slow-mo)")
    ap.add_argument("--record", help="save MP4")
    ap.add_argument("--direct", action="store_true", help="no window, just print results")
    ap.add_argument("--camera", choices=["fixed", "follow"], default="follow")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    loop_shift = float(df["base_x"].iloc[-1] - df["base_x"].iloc[0])
    loop_shift += float(df["base_x"].iloc[-1] - df["base_x"].iloc[-2])  # step for the wrap

    robot, ids, mass = load_world(gui=not a.direct)

    # Start in the first reference pose.
    pos0, pitch0, q0 = ref_at(df, 0.0, loop_shift)
    p.resetBasePositionAndOrientation(robot, pos0 + [0, 0, 0.01],
                                      p.getQuaternionFromEuler([0, -pitch0, 0]))
    for j, qj in zip(ids, q0):
        p.resetJointState(robot, j, qj)

    # Harness gains, scaled by mass so they mean the same for any robot.
    kp_lin = a.assist * mass * np.array([30.0, 60.0, 150.0])   # x, y, z (N/m)
    kd_lin = a.assist * mass * np.array([8.0, 12.0, 20.0])
    kp_ang = a.assist * 60.0                                     # Nm/rad
    kd_ang = a.assist * 4.0

    if a.record and not a.direct:
        log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, a.record)
    if not a.direct and a.camera == "fixed":
        p.resetDebugVisualizerCamera(max(2.0, 0.6 * loop_shift * a.duration / df["t"].iloc[-1]),
                                     0, -8, [pos0[0] + 0.5 * loop_shift, 0, 0.25])

    steps = int(a.duration / SIM_DT)
    render_every = max(1, int((1 / 60) / SIM_DT))  # update the screen at ~60 Hz
    fell_at = None
    joint_err, pos_err, max_harness = [], [], 0.0
    wall0 = time.time()

    for s in range(steps):
        t = s * SIM_DT
        ref_pos, ref_pitch, q_ref = ref_at(df, t, loop_shift)

        # --- joint PD with real torque limits ---
        js = p.getJointStates(robot, ids)
        q = np.array([x[0] for x in js])
        dq = np.array([x[1] for x in js])
        tau = np.clip(KP * (q_ref - q) - KD * dq, -TORQUE_LIMIT, TORQUE_LIMIT)
        p.setJointMotorControlArray(robot, ids, p.TORQUE_CONTROL, forces=tau.tolist())

        # --- harness on the body ---
        pos, orn = p.getBasePositionAndOrientation(robot)
        vel, angvel = p.getBaseVelocity(robot)
        pos, vel, angvel = np.array(pos), np.array(vel), np.array(angvel)
        roll, pitch_b, yaw = p.getEulerFromQuaternion(orn)
        if a.assist > 0:
            f = kp_lin * (ref_pos - pos) - kd_lin * vel
            ang_err = np.array([-roll, (-ref_pitch) - pitch_b, -yaw])
            tq = kp_ang * ang_err - kd_ang * angvel
            p.applyExternalForce(robot, -1, f.tolist(), pos.tolist(), p.WORLD_FRAME)
            p.applyExternalTorque(robot, -1, tq.tolist(), p.WORLD_FRAME)
            max_harness = max(max_harness, abs(f[2]) / (mass * 9.81))

        p.stepSimulation()

        joint_err.append(np.abs(q_ref - q).mean())
        pos_err.append(abs(ref_pos[0] - pos[0]))

        # Fall check: body too low or tipped over.
        if fell_at is None and (pos[2] < 0.12 or abs(roll) > 0.9 or abs(pitch_b) > 0.9):
            fell_at = t
            print(f"[FALL] at t={t:.2f}s (height {pos[2]:.2f} m, roll {roll:.2f}, pitch {pitch_b:.2f})")

        if not a.direct and s % render_every == 0:
            if a.camera == "follow":
                p.resetDebugVisualizerCamera(1.4, 0, -10, [pos[0], 0, 0.25])
            # keep real-time pacing (adjusted by --speed)
            lag = (t / a.speed) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)

    if a.record and not a.direct:
        p.stopStateLogging(log)

    pos, _ = p.getBasePositionAndOrientation(robot)
    ref_end, _, _ = ref_at(df, a.duration, loop_shift)
    print("\n=== Result ===")
    print(f"assist            : {a.assist}")
    print(f"fell              : {'no' if fell_at is None else f'yes, at {fell_at:.2f} s'}")
    print(f"distance walked   : {pos[0] - pos0[0]:.2f} m  (reference {ref_end[0] - pos0[0]:.2f} m)")
    print(f"mean joint error  : {math.degrees(np.mean(joint_err)):.1f} deg")
    print(f"mean x lag behind : {np.mean(pos_err):.2f} m")
    if a.assist > 0:
        print(f"peak harness lift : {max_harness:.0%} of body weight")

    if not a.direct:
        print("\nClose the window or press Ctrl+C to exit.")
        try:
            while p.isConnected():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
