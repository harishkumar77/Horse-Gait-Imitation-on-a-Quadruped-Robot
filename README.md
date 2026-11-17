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

- run_dlc.py - run DeepLabCut on a video
- retarget_dlc_to_a1.py - keypoints to A1 joint angles (CSV), --play to view it
- physics_replay_a1.py - replay with gravity, --assist adds a helper force
- rl_track_a1.py - train / play / check the RL policy
- eval_gait.py - compare horse vs robot, saves a gait diagram
- trot_a1.csv - retargeted trot from my video
- Trot_10sDLC_...h5 - DeepLabCut keypoints for that video

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
python run_dlc.py <dlc_project>/config.yaml Trot_10s.mp4

# everything else (a1rl env)
python retarget_dlc_to_a1.py Trot_10sDLC_<...>.h5 --fps 30 --out trot_a1.csv --play
python physics_replay_a1.py trot_a1.csv --assist 0
python rl_track_a1.py train trot_a1.csv --steps 20000000 --envs 8
python rl_track_a1.py play trot_a1.csv runs/trot_a1/final_model.zip
python eval_gait.py trot_a1.csv runs/trot_a1/final_model.zip
```
Videos are not included in this repo.
