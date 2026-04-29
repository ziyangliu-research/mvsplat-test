#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def pose_to_Twc(row):
    tx, ty, tz, qx, qy, qz, qw = row
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def load_poses(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=np.float64)


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_tartanair_stereo_extrinsic.py <P000_dir>")
        sys.exit(1)

    seq = Path(sys.argv[1])
    left_path = seq / "pose_lcam_front.txt"
    right_path = seq / "pose_rcam_front.txt"

    left = load_poses(left_path)
    right = load_poses(right_path)

    assert len(left) == len(right), (len(left), len(right))
    assert left.shape[1] == 7
    assert right.shape[1] == 7

    Ts = []
    for l, r in zip(left, right):
        T_w_l = pose_to_Twc(l)
        T_w_r = pose_to_Twc(r)
        T_l_r = np.linalg.inv(T_w_l) @ T_w_r
        Ts.append(T_l_r)

    Ts = np.stack(Ts, axis=0)

    # Translation statistics
    trans = Ts[:, :3, 3]
    print("num frames:", len(Ts))
    print("translation mean:", trans.mean(axis=0))
    print("translation std :", trans.std(axis=0))
    print("baseline norm mean:", np.linalg.norm(trans, axis=1).mean())
    print("baseline norm std :", np.linalg.norm(trans, axis=1).std())

    # Use the first frame as calibration if all frames are stable.
    T = Ts[0]

    print("\nT_c1_c2 from first frame:")
    print(T)

    print("\nOpenCV YAML format:")
    flat = T.reshape(-1)
    print("Stereo.T_c1_c2: !!opencv-matrix")
    print("  rows: 4")
    print("  cols: 4")
    print("  dt: f")
    print("  data: [")

    for i in range(4):
        vals = flat[i * 4:(i + 1) * 4]
        line = ", ".join(f"{v:.12f}" for v in vals)
        if i < 3:
            print(f"    {line},")
        else:
            print(f"    {line}")
    print("  ]")


if __name__ == "__main__":
    main()