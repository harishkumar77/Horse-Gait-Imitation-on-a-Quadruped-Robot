"""
Measure how close the robot's motion is to the horse reference.

Compares, over one full clip:
  - Horse reference (trot_a1.csv: timing and foot contacts come from the horse video)
  - Baseline: blind replay of the reference with physics (zero correction from the policy)
  - RL policy (your trained model)

Metrics:
  survived         did it get through the clip without falling
  time             seconds before falling / end of clip
  speed            forward speed (m/s)
  speed ratio      robot speed / reference speed (1.0 = same)
  joint err        mean |robot joint - reference joint| (deg)
  contact match    % of time each foot is on/off the ground at the same time as the horse
  stride freq      strides per second
  duty factor      fraction of the stride each foot is on the ground
  phase offsets    timing of each leg relative to front-right (0..1 of a stride)
                   trot: FL ~0.5, RR ~0.5, RL ~0.0 (diagonal pairs together)

Also saves gait_diagram.png: footfall pattern of horse vs baseline vs RL.

Usage:
  python eval_gait.py trot_a1.csv runs/trot_a1/final_model.zip
  python eval_gait.py trot_a1.csv --baseline-only
"""

import argparse
import math
import os

import numpy as np

from rl_track_a1 import A1ImitationEnv, CONTROL_DT, LEGS


# ---------------------------------------------------------------- gait analysis
def stride_freq(contact, dt):
    """Main frequency of the footfall signal (Hz), via FFT, in 0.4-6 Hz."""
    sig = contact.mean(axis=1) if contact.ndim == 2 else contact
    # Use one leg's signal (more periodic than the average for a trot).
    sig = contact[:, 0] if contact.ndim == 2 else contact
    sig = sig - sig.mean()
    if len(sig) < 8 or np.allclose(sig, 0):
        return float("nan")
    n = 1 << int(math.ceil(math.log2(len(sig) * 4)))  # zero-pad for resolution
    spec = np.abs(np.fft.rfft(sig, n))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs > 0.4) & (freqs < 6.0)
    if not band.any():
        return float("nan")
    return float(freqs[band][np.argmax(spec[band])])


def phase_offsets(contact, dt, f):
    """Lag of each leg vs leg 0 (FR) as a fraction of the stride, 0..1."""
    if not np.isfinite(f) or f <= 0:
        return [float("nan")] * contact.shape[1]
    period = int(round(1.0 / f / dt))
    if period < 2:
        return [float("nan")] * contact.shape[1]
    a = contact[:, 0] - contact[:, 0].mean()
    out = []
    for k in range(contact.shape[1]):
        b = contact[:, k] - contact[:, k].mean()
        best, best_lag = -1e9, 0
        for lag in range(period):
            if lag >= len(a):
                break
            c = np.dot(a[: len(a) - lag], b[lag:]) if lag else np.dot(a, b)
            if c > best:
                best, best_lag = c, lag
        out.append(best_lag / period)
    return out


def gait_stats(contact, dt):
    f = stride_freq(contact, dt)
    return dict(freq=f, duty=contact.mean(axis=0), phase=phase_offsets(contact, dt, f))


# ---------------------------------------------------------------- rollout
def rollout(env, policy):
    obs, _ = env.reset(seed=0)
    log = dict(c_robot=[], c_ref=[], q=[], q_ref=[], x=[], vx_ref=[])
    fell = False
    x0 = env.bc.getBasePositionAndOrientation(env.robot)[0][0]
    while True:
        obs, _, term, trunc, info = env.step(policy(obs))
        r = env.ref.at(env.t)
        pos = env.bc.getBasePositionAndOrientation(env.robot)[0]
        js = env.bc.getJointStates(env.robot, env.ids)
        log["c_robot"].append(env._foot_contacts())
        log["c_ref"].append(r["contact"])
        log["q"].append([s[0] for s in js])
        log["q_ref"].append(r["q"])
        log["x"].append(pos[0] - x0)
        log["vx_ref"].append(r["vx"])
        if term:
            fell = True
        if term or trunc:
            break
    log = {k: np.array(v) for k, v in log.items()}
    log["fell"] = fell
    log["time"] = len(log["x"]) * CONTROL_DT
    return log


def summarize(name, log, ref_stats):
    c, cr = log["c_robot"], log["c_ref"]
    dt = CONTROL_DT
    st = gait_stats(c, dt)
    speed = log["x"][-1] / log["time"] if log["time"] > 0 else 0.0
    ref_speed = float(np.mean(log["vx_ref"]))
    return dict(
        name=name,
        survived="yes" if not log["fell"] else "no",
        time=f"{log['time']:.1f} s",
        speed=f"{speed:.2f} m/s",
        speed_ratio=f"{speed / ref_speed:.2f}" if ref_speed > 1e-3 else "n/a",
        joint_err=f"{math.degrees(np.mean(np.abs(log['q'] - log['q_ref']))):.1f} deg",
        contact=f"{100 * np.mean(c == cr):.0f} %",
        freq=f"{st['freq']:.2f} Hz",
        duty=" ".join(f"{d:.2f}" for d in st["duty"]),
        phase=" ".join(f"{x:.2f}" for x in st["phase"]),
        _contacts=c, _dt=dt,
    )


def print_table(rows):
    keys = [("survived", "survived"), ("time", "time"), ("speed", "speed"),
            ("speed_ratio", "speed ratio (vs horse ref)"), ("joint_err", "joint error"),
            ("contact", "contact match"), ("freq", "stride freq"),
            ("duty", "duty factor FR FL RR RL"), ("phase", "phase vs FR  FR FL RR RL")]
    w = max(len(k[1]) for k in keys) + 2
    cw = max(26, max(len(r["name"]) for r in rows) + 2)
    print("\n" + "metric".ljust(w) + "".join(r["name"].ljust(cw) for r in rows))
    print("-" * (w + cw * len(rows)))
    for k, label in keys:
        print(label.ljust(w) + "".join(str(r.get(k, "")).ljust(cw) for r in rows))
    print()


def save_diagram(rows, path, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.6 * len(rows) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        c = r["_contacts"]
        dt = r["_dt"]
        t = np.arange(len(c)) * dt
        for k in range(4):
            on = c[:, k] > 0.5
            # draw contiguous stance blocks
            start = None
            for i, v in enumerate(np.append(on, False)):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    ax.broken_barh([(t[start], (i - start) * dt)], (3 - k - 0.4, 0.8), color="0.25")
                    start = None
        ax.set_yticks(range(4))
        ax.set_yticklabels(LEGS[::-1])
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(r["name"], fontsize=10, loc="left")
    axes[-1].set_xlabel("time (s)  —  dark = foot on the ground")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved gait diagram -> {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--norm", help="vecnormalize.pkl (default: next to the model)")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--plot", default="gait_diagram.png")
    a = ap.parse_args()

    env = A1ImitationEnv(a.csv, gui=False, randomize=False, seed=0)

    # Horse reference row (contacts from the video, no physics).
    ref_c = env.ref.contact
    ref_dt = env.ref.dt
    st = gait_stats(ref_c, ref_dt)
    ref_row = dict(
        name="horse (reference)", survived="-", time=f"{env.ref.duration:.1f} s",
        speed=f"{np.mean(env.ref.vx):.2f} m/s", speed_ratio="1.00", joint_err="-",
        contact="100 %", freq=f"{st['freq']:.2f} Hz",
        duty=" ".join(f"{d:.2f}" for d in st["duty"]),
        phase=" ".join(f"{x:.2f}" for x in st["phase"]),
        _contacts=ref_c, _dt=ref_dt,
    )

    rows = [ref_row]
    base = rollout(env, lambda obs: np.zeros(12))
    rows.append(summarize("blind replay", base, st))

    if not a.baseline_only:
        if not a.model:
            ap.error("give a model .zip or use --baseline-only")
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        norm_path = a.norm or os.path.join(os.path.dirname(a.model), "vecnormalize.pkl")
        dummy = A1ImitationEnv(a.csv, gui=False, randomize=False, seed=1)
        vn = VecNormalize.load(norm_path, DummyVecEnv([lambda: dummy]))
        vn.training = False
        model = PPO.load(a.model, device="cpu")
        policy = lambda obs: model.predict(vn.normalize_obs(obs), deterministic=True)[0]
        rl = rollout(env, policy)
        rows.append(summarize("RL policy", rl, st))

    print_table(rows)
    print("How to read it: closer to the horse column = closer copy.")
    print("Trot check: phase offsets FL and RR near 0.5, RL near 0.0 (diagonal legs together).")
    save_diagram(rows, a.plot, CONTROL_DT)
    env.close()


if __name__ == "__main__":
    main()
