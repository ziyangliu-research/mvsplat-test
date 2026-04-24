# 调整渲染相机位置旋转角度的脚本
import argparse
import json
import math
from pathlib import Path

import numpy as np


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float64,
    )


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_pose", type=Path, required=True)
    p.add_argument("--out_pose", type=Path, required=True)

    # camera-local translation: +x forward, +y right, +z down
    p.add_argument("--tx_cam", type=float, default=0.0)
    p.add_argument("--ty_cam", type=float, default=0.0)
    p.add_argument("--tz_cam", type=float, default=0.0)

    # camera-local rotation in degrees
    p.add_argument("--yaw_deg", type=float, default=0.0, help="+right, -left")
    p.add_argument("--pitch_deg", type=float, default=0.0, help="+up, -down")
    p.add_argument("--roll_deg", type=float, default=0.0, help="image roll")

    args = p.parse_args()

    Twc = np.array(json.loads(args.in_pose.read_text()), dtype=np.float64)
    if Twc.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose, got {Twc.shape}")

    R = Twc[:3, :3]
    t = Twc[:3, 3]

    # Local camera translation: world displacement = Rwc @ local displacement.
    dt_cam = np.array([args.tx_cam, args.ty_cam, args.tz_cam], dtype=np.float64)
    t_new = t + R @ dt_cam

    # Local camera rotation.
    # Order: yaw -> pitch -> roll in camera-local frame.
    R_delta = rot_z(args.yaw_deg) @ rot_y(args.pitch_deg) @ rot_x(args.roll_deg)
    R_new = R @ R_delta

    Twc_new = Twc.copy()
    Twc_new[:3, :3] = R_new
    Twc_new[:3, 3] = t_new

    args.out_pose.parent.mkdir(parents=True, exist_ok=True)
    args.out_pose.write_text(json.dumps(Twc_new.tolist(), indent=2), encoding="utf-8")
    print(f"Saved adjusted pose to {args.out_pose}")
    print(f"camera-local translation = {dt_cam.tolist()}")
    print(f"yaw={args.yaw_deg}, pitch={args.pitch_deg}, roll={args.roll_deg}")


if __name__ == "__main__":
    main()