# Horse Gait Imitation on a Quadruped Robot

I wanted to see if a robot could copy how a horse moves just from a normal video.
I tracked a trotting horse with DeepLabCut, mapped the leg motion onto a Unitree A1
in PyBullet, and then trained an RL policy so the robot can do it with gravity on
and keep its balance.

Right now it works for trot. Gallop is next.

## Pipeline

video -> DeepLabCut keypoints -> retarget to A1 -> physics replay -> RL policy -> evaluation

- DeepLabCut gives me shoulder/hip, knee and paw for each leg.
- For retargeting I use the paw position relative to the hip, scale it to the A1 leg
  length and solve IK. Body height and tilt also come from the video.
- If you just replay the joint angles with gravity on, the robot falls or shuffles.
- The RL policy (PPO) adds small corrections on top of the horse motion so it can
  stay balanced. If it falls, the episode ends.
- eval_gait.py compares the horse and the robot (foot timing, stride frequency,
  speed, joint error).

## Files

- scripts/run_dlc.py - run DeepLabCut on a video
- scripts/retarget_dlc_to_a1.py - keypoints to A1 joint angles (CSV), --play to view it
- scripts/physics_replay_a1.py - replay with gravity, --assist adds a helper force
- scripts/rl_track_a1.py - train / play / check the RL policy
- scripts/eval_gait.py - compare horse vs robot, saves a gait diagram

## Setup

I keep DeepLabCut and the sim in separate conda envs.

```
conda create -n DEEPLABCUT python=3.10 -y
conda activate DEEPLABCUT
pip install "deeplabcut[gui]"

conda create -n a1rl python=3.10 -y
conda activate a1rl
pip install -r requirements.txt
```

## Run

```
# keypoints (DEEPLABCUT env)
python scripts/run_dlc.py <dlc_project>/config.yaml data/Trot_10s.mp4

# everything else (a1rl env)
python scripts/retarget_dlc_to_a1.py data/Trot_10sDLC_<...>.h5 --fps 30 --out data/trot_a1.csv --play
python scripts/physics_replay_a1.py data/trot_a1.csv --assist 0
python scripts/rl_track_a1.py train data/trot_a1.csv --steps 20000000 --envs 8
python scripts/rl_track_a1.py play data/trot_a1.csv runs/trot_a1/final_model.zip
python scripts/eval_gait.py data/trot_a1.csv runs/trot_a1/final_model.zip
```

## Results

(adding soon)

## Notes

- My first try used a CNN to predict joint angles from skeleton images. The output
  barely moved and some angles went to the wrong joints, so I dropped it and compute
  everything from the keypoints now.
- Horse and robot knees bend differently, so copying joint angles directly looks wrong.
  Using paw positions fixed that.
- The far side legs are partly hidden in a side view video, so they are tracked worse.
  That makes one pair of legs push less. Better labels should help.
- The A1 is much smaller than a horse, so it won't move exactly the same. I'm going for
  the same gait pattern and timing.
- Simulation only for now.

## References

- DeepMimic: https://arxiv.org/abs/1804.02717
- Learning Agile Robotic Locomotion Skills by Imitating Animals: https://arxiv.org/abs/2004.00784
- SLoMo: https://arxiv.org/abs/2304.14389

Videos are not included in this repo.
