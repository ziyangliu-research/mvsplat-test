from pathlib import Path
import torch
import math

def quat_to_rotmat(qx, qy, qz, qw):
    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float64)
    q = q / q.norm()
    x, y, z, w = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.tensor([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=torch.float64)

def build_Twc(vals):
    tx, ty, tz, qx, qy, qz, qw = vals
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
    T[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
    return T

# 你当前确认正确的轴变换
T_tartanCam_from_cvCam = torch.eye(4, dtype=torch.float64)
T_tartanCam_from_cvCam[:3, :3] = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=torch.float64)

def load_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            T = build_Twc(vals[:7]) @ T_tartanCam_from_cvCam
            poses.append(T)
    return poses

scene_root = Path("/workspace/mvsplat/datasets/tartanair_v2/House/Data_easy/House/Data_easy/P000")
left_poses = load_poses(scene_root / "pose_lcam_front.txt")
right_poses = load_poses(scene_root / "pose_rcam_front.txt")

Ts = []
for Tl, Tr in zip(left_poses, right_poses):
    Ts.append(torch.linalg.inv(Tl) @ Tr)

# translation stats
t = torch.stack([T[:3, 3] for T in Ts], dim=0)
print("translation mean:", t.mean(0))
print("translation std :", t.std(0))
print("translation min :", t.min(0).values)
print("translation max :", t.max(0).values)
print("norm mean/std   :", t.norm(dim=1).mean().item(), t.norm(dim=1).std().item())

# rotation stats relative to first frame
R0 = Ts[0][:3, :3]
angles = []
for T in Ts:
    R = R0.T @ T[:3, :3]
    trace = torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0)
    angle = math.degrees(math.acos(trace.item()))
    angles.append(angle)

angles = torch.tensor(angles, dtype=torch.float64)
print("rotation rel-to-first deg mean/std/max:",
      angles.mean().item(), angles.std().item(), angles.max().item())