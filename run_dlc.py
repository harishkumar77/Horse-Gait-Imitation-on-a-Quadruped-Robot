"""
Run DeepLabCut on one or more videos and save keypoints (.h5 + .csv).

Usage:
  python scripts/run_dlc.py /path/to/dlc_project/config.yaml data/Trot_10s.mp4 [more videos...]
  python scripts/run_dlc.py /path/to/dlc_project/config.yaml data/Trot_10s.mp4 --labeled

Expected bodyparts in the DLC project (used by retarget_dlc_to_a1.py):
  front_left_thai, front_left_knee, front_left_paw,
  front_right_thai, front_right_knee, front_right_paw,
  back_left_thai,  back_left_knee,  back_left_paw,
  back_right_thai, back_right_knee, back_right_paw
"""

import argparse
import os

import deeplabcut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="DeepLabCut project config.yaml")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--labeled", action="store_true", help="also make a labeled video")
    a = ap.parse_args()

    videos = [os.path.abspath(v) for v in a.videos]
    deeplabcut.analyze_videos(a.config, videos, videotype="mp4", save_as_csv=True)
    if a.labeled:
        deeplabcut.create_labeled_video(a.config, videos, videotype="mp4")


if __name__ == "__main__":
    main()
