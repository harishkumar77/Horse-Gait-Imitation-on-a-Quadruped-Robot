"""
RL motion imitation for the Unitree A1 (PyBullet + Stable-Baselines3 PPO).

The policy learns to follow a retargeted horse motion (CSV from retarget_dlc_to_a1.py)
with gravity ON and NO harness, while keeping its balance.

Main ideas (DeepMimic / Peng et al. 2020 style):
  - Action = small correction added to the reference joint angles (residual).
  - Reward = follow the reference (joint pose, body height, tilt, forward speed,
    which feet are on the ground) + small smoothness/torque penalties.
  - Falling ends the episode early -> the policy loses all future reward.
  - Each episode starts at a random point of the clip (reference state init).
  - Light domain randomization (friction, mass, motor strength) for robustness.

Commands:
  # quick check that everything loads (a few seconds, no training)
  python rl_track_a1.py check trot_a1.csv

  # train (CPU is fine; use more envs if you have more cores)
  python rl_track_a1.py train trot_a1.csv --steps 20000000 --envs 8

  # watch the trained policy (and optionally record it)
  python rl_track_a1.py play trot_a1.csv runs/trot_a1/final_model.zip
  python rl_track_a1.py play trot_a1.csv runs/trot_a1/final_model.zip --speed 0.5 --record trot_rl.mp4

  # watch training curves
  tensorboard --logdir runs
"""

import argparse
import math
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

LEGS = ["FR", "FL", "RR", "RL"]
PARTS = ["hip", "thigh", "calf"]
JOINT_LOW = np.array([-0.8, -1.05, -2.7] * 4)
JOINT_HIGH = np.array([0.8, 4.19, -0.92] * 4)

PHYS_DT = 1.0 / 500.0
CONTROL_DT = 1.0 / 50.0
SUBSTEPS = int(round(CONTROL_DT / PHYS_DT))
TORQUE_LIMIT = 33.5
KP, KD = 80.0, 2.0
ACTION_SCALE = 0.3  # max correction (rad) the policy can add to the reference


# ---------------------------------------------------------------- reference
class Reference:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.t = df["t"].values
        self.dt = self.t[1] - self.t[0]
        self.duration = self.t[-1]
        self.base_x = df["base_x"].values
        self.base_z = df["base_z"].values
        self.pitch = df["base_pitch"].values
        self.q = np.stack([df[f"{l}_{p_}"].values for l in LEGS for p_ in PARTS], axis=1)
        self.q = np.clip(self.q, JOINT_LOW, JOINT_HIGH)
        self.contact = np.stack([df[f"{l}_contact"].values for l in LEGS], axis=1).astype(float)
        self.vx = np.gradient(self.base_x, self.dt)
        self.dq = np.gradient(self.q, self.dt, axis=0)

    def at(self, t):
        t = min(max(t, 0.0), self.duration)
        i = min(int(t / self.dt), len(self.t) - 2)
        a = (t - self.t[i]) / self.dt
        lerp = lambda arr: (1 - a) * arr[i] + a * arr[i + 1]
        return dict(z=lerp(self.base_z), pitch=lerp(self.pitch), vx=lerp(self.vx),
                    q=lerp(self.q), dq=lerp(self.dq),
                    contact=self.contact[i] if a < 0.5 else self.contact[i + 1])


# ---------------------------------------------------------------- environment
class A1ImitationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path, gui=False, randomize=True, seed=None):
        super().__init__()
        self.ref = Reference(csv_path)
        self.gui = gui
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.bc = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self._build()
        self.t = 0.0
        self.last_action = np.zeros(12)
        self.motor_scale = 1.0
        n_obs = len(self._obs())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

    # world setup
    def _build(self):
        bc = self.bc
        bc.resetSimulation()
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, -9.81)
        bc.setTimeStep(PHYS_DT)
        if self.gui:
            bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.plane = bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        self.robot = bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "a1/a1.urdf"),
                                 [0, 0, 0.35], useFixedBase=False)
        names = {bc.getJointInfo(self.robot, j)[1].decode(): j
                 for j in range(bc.getNumJoints(self.robot))}
        alt = {"hip": ["hip_joint"], "thigh": ["upper_joint", "thigh_joint"],
               "calf": ["lower_joint", "calf_joint"]}
        self.ids = [next(names[f"{l}_{a}"] for a in alt[pt] if f"{l}_{a}" in names)
                    for l in LEGS for pt in PARTS]
        self.toe_ids = [names[f"{l}_toe_fixed"] for l in LEGS]
        for j in self.ids:
            bc.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)
        self.base_mass0 = bc.getDynamicsInfo(self.robot, -1)[0]

    # helpers
    def _state(self):
        bc = self.bc
        pos, orn = bc.getBasePositionAndOrientation(self.robot)
        vel, angvel = bc.getBaseVelocity(self.robot)
        js = bc.getJointStates(self.robot, self.ids)
        q = np.array([s[0] for s in js])
        dq = np.array([s[1] for s in js])
        rot = np.array(bc.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return np.array(pos), orn, rot, np.array(vel), np.array(angvel), q, dq

    def _foot_contacts(self):
        c = np.zeros(4)
        for k, toe in enumerate(self.toe_ids):
            if self.bc.getContactPoints(self.robot, self.plane, linkIndexA=toe):
                c[k] = 1.0
        return c

    def _obs(self):
        pos, orn, rot, vel, angvel, q, dq = self._state()
        r_now = self.ref.at(self.t)
        r_next = self.ref.at(self.t + CONTROL_DT)
        gravity_body = rot.T @ np.array([0, 0, -1.0])     # tilt, as an IMU sees it
        vel_body = rot.T @ vel
        angvel_body = rot.T @ angvel
        phase = 2 * math.pi * self.t / max(self.ref.duration, 1e-6)
        return np.concatenate([
            [pos[2]], gravity_body, vel_body, angvel_body,
            q, dq * 0.1, self.last_action,
            r_next["q"] - q, [r_next["z"] - pos[2], r_next["pitch"], r_next["vx"]],
            r_now["contact"], [math.sin(phase), math.cos(phase)],
        ]).astype(np.float32)

    # gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        bc = self.bc
        # Reference state init: start at a random point in the clip.
        # (In play mode, randomize=False, always start from the beginning.)
        self.t0 = (self.rng.uniform(0, max(self.ref.duration - 1.0, 0.0))
                   if self.randomize else 0.0)
        self.t = self.t0
        r = self.ref.at(self.t)
        bc.resetBasePositionAndOrientation(
            self.robot, [0, 0, r["z"] + 0.01], bc.getQuaternionFromEuler([0, -r["pitch"], 0]))
        bc.resetBaseVelocity(self.robot, [r["vx"], 0, 0], [0, 0, 0])
        for j, qj, dqj in zip(self.ids, r["q"], r["dq"]):
            bc.resetJointState(self.robot, j, qj, dqj)

        # Domain randomization.
        if self.randomize:
            fr = self.rng.uniform(0.6, 1.2)
            bc.changeDynamics(self.plane, -1, lateralFriction=fr)
            for toe in self.toe_ids:
                bc.changeDynamics(self.robot, toe, lateralFriction=fr)
            bc.changeDynamics(self.robot, -1, mass=self.base_mass0 * self.rng.uniform(0.9, 1.15))
            self.motor_scale = self.rng.uniform(0.85, 1.1)
        else:
            bc.changeDynamics(self.plane, -1, lateralFriction=0.9)
            for toe in self.toe_ids:
                bc.changeDynamics(self.robot, toe, lateralFriction=0.9)
            self.motor_scale = 1.0

        self.last_action = np.zeros(12)
        self.x_start = 0.0
        return self._obs(), {}

    def step(self, action):
        bc = self.bc
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        r_target = self.ref.at(self.t + CONTROL_DT)
        q_des = np.clip(r_target["q"] + ACTION_SCALE * action, JOINT_LOW, JOINT_HIGH)

        torque_sq = 0.0
        for _ in range(SUBSTEPS):
            js = bc.getJointStates(self.robot, self.ids)
            q = np.array([s[0] for s in js])
            dq = np.array([s[1] for s in js])
            tau = KP * (q_des - q) - KD * dq
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT) * self.motor_scale
            bc.setJointMotorControlArray(self.robot, self.ids, p.TORQUE_CONTROL, forces=tau.tolist())
            bc.stepSimulation()
            torque_sq += float(np.sum(tau ** 2))
        self.t += CONTROL_DT

        pos, orn, rot, vel, angvel, q, dq = self._state()
        roll, pitch_b, yaw = bc.getEulerFromQuaternion(orn)
        r = self.ref.at(self.t)
        feet = self._foot_contacts()

        # ---- reward: follow the motion ----
        r_pose = math.exp(-2.0 * np.sum((q - r["q"]) ** 2))
        r_jvel = math.exp(-0.01 * np.sum((dq - r["dq"]) ** 2))
        r_height = math.exp(-80.0 * (pos[2] - r["z"]) ** 2)
        r_orient = math.exp(-10.0 * (roll ** 2 + ((-r["pitch"]) - pitch_b) ** 2 + yaw ** 2))
        r_vel = math.exp(-4.0 * ((vel[0] - r["vx"]) ** 2 + vel[1] ** 2))
        r_contact = float(np.mean(feet == r["contact"]))
        reward = (0.30 * r_pose + 0.05 * r_jvel + 0.15 * r_height + 0.15 * r_orient
                  + 0.25 * r_vel + 0.10 * r_contact)
        # Never reward going backward: the horse only moves forward.
        reward -= 1.0 * max(0.0, -vel[0])
        # ---- small penalties: smooth, low effort ----
        reward -= 0.02 * float(np.sum((action - self.last_action) ** 2))
        reward -= 2e-6 * torque_sq / SUBSTEPS
        self.last_action = action

        # ---- falling ends the episode ----
        base_hits_ground = len(bc.getContactPoints(self.robot, self.plane, linkIndexA=-1)) > 0
        fell = pos[2] < 0.15 or abs(roll) > 0.8 or abs(pitch_b) > 0.8 or base_hits_ground
        terminated = bool(fell)
        truncated = bool(self.t >= self.ref.duration - CONTROL_DT)

        if self.gui:
            bc.resetDebugVisualizerCamera(1.4, 0, -10, [pos[0], 0, 0.25])

        info = {"fell": fell, "x": pos[0], "pose_err_deg": math.degrees(np.mean(np.abs(q - r["q"])))}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.bc.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------- commands
def cmd_check(a):
    from gymnasium.utils.env_checker import check_env
    env = A1ImitationEnv(a.csv, gui=False, seed=0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset()
    total, steps, t0 = 0.0, 0, time.time()
    for _ in range(200):  # zero action = pure reference replay (no learning)
        obs, rew, term, trunc, info = env.step(np.zeros(12))
        total += rew
        steps += 1
        if term or trunc:
            break
    print(f"obs size {obs.shape[0]}, ran {steps} steps, reward/step {total / steps:.3f}, "
          f"fell={info['fell']}, speed {steps / (time.time() - t0):.0f} steps/s per env")
    print("Check OK. Zero action = blind replay, so falling here is expected.")


def make_env(csv, rank, randomize):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(A1ImitationEnv(csv, gui=False, randomize=randomize, seed=rank))
    return _f


def cmd_train(a):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(1)
    name = a.name or os.path.splitext(os.path.basename(a.csv))[0]
    out = os.path.join("runs", name)
    os.makedirs(out, exist_ok=True)

    venv = SubprocVecEnv([make_env(a.csv, i, not a.no_random) for i in range(a.envs)])
    norm_file = os.path.join(out, "vecnormalize.pkl")
    if a.resume and os.path.exists(norm_file):
        venv = VecNormalize.load(norm_file, venv)  # keep the old normalization stats
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.95)

    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", venv, device="cpu",
            n_steps=max(64, 4096 // a.envs), batch_size=1024, n_epochs=5,
            learning_rate=3e-4, gamma=0.95, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=-1.0),
            tensorboard_log="runs", verbose=1,
        )

    class SaveNorm(CheckpointCallback):
        def _on_step(self):
            ok = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                venv.save(os.path.join(out, "vecnormalize.pkl"))
            return ok

    cb = SaveNorm(save_freq=max(1, 500_000 // a.envs), save_path=out, name_prefix="ckpt")
    try:
        model.learn(total_timesteps=a.steps, callback=cb, tb_log_name=name,
                    reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:
        print("Stopped early, saving...")
    model.save(os.path.join(out, "final_model"))
    venv.save(os.path.join(out, "vecnormalize.pkl"))
    print(f"Saved to {out}/final_model.zip and {out}/vecnormalize.pkl")


def cmd_play(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    norm_path = a.norm or os.path.join(os.path.dirname(a.model), "vecnormalize.pkl")
    holder = {}

    def _f():
        holder["env"] = A1ImitationEnv(a.csv, gui=True, randomize=False, seed=0)
        return holder["env"]

    venv = VecNormalize.load(norm_path, DummyVecEnv([_f]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(a.model, device="cpu")
    env = holder["env"]

    if a.record:
        log = env.bc.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, a.record)
    obs = venv.reset()
    episodes = 0
    x0, ep_t, errs = 0.0, 0.0, []
    while episodes < a.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(act)
        info = info[0]
        errs.append(info.get("pose_err_deg", 0))
        ep_t += CONTROL_DT
        time.sleep(CONTROL_DT / a.speed)
        if done[0]:
            episodes += 1
            print(f"episode {episodes}: {'FELL' if info['fell'] else 'finished'} after "
                  f"{ep_t:.1f}s, walked {info.get('x', 0) - x0:.2f} m, "
                  f"mean joint error {np.mean(errs):.1f} deg")
            ep_t, errs = 0.0, []
            if a.record and episodes == 1:
                env.bc.stopStateLogging(log)
    venv.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("csv")

    t = sub.add_parser("train")
    t.add_argument("csv")
    t.add_argument("--steps", type=int, default=20_000_000)
    t.add_argument("--envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    t.add_argument("--name", help="run name (default: csv name)")
    t.add_argument("--resume", help="continue from a saved model .zip")
    t.add_argument("--no-random", action="store_true", help="turn off domain randomization")

    pl = sub.add_parser("play")
    pl.add_argument("csv")
    pl.add_argument("model")
    pl.add_argument("--norm", help="vecnormalize.pkl (default: next to the model)")
    pl.add_argument("--episodes", type=int, default=3)
    pl.add_argument("--speed", type=float, default=1.0)
    pl.add_argument("--record")

    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "play": cmd_play}[a.cmd](a)


if __name__ == "__main__":
    main()
