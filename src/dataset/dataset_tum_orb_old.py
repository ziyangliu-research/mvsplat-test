'''
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import bisect

import torch
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetTUMORBCfg(DatasetCfgCommon):
    name: Literal["tum_orb"]
    root: Path
    trajectory_file: Path                  # ORB-SLAM3 输出的轨迹 txt
    association_file: Optional[Path] = None
    image_dirname: str = "image"
    depth_dirname: str = "depth"          # 目前先不强制使用
    pose_time_tolerance: float = 0.02     # 秒，图像时间戳与轨迹时间戳允许的最大差
    near: float = 0.1
    far: float = 10.0
    fx: float = 0.0                       # 原始像素内参
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    normalize_intrinsics: bool = True     # 按现有 MVSplat 约定，建议先 True
    test_len: int = -1                    # -1 表示使用全部
    frame_stride: int = 1                 # 可选降采样


class DatasetTUMORB(Dataset):
    def __init__(
        self,
        cfg: DatasetTUMORBCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.image_root = cfg.root / cfg.image_dirname
        self.depth_root = cfg.root / cfg.depth_dirname
        self.scene_name = cfg.root.name

        # 1) 读取图像列表（优先 association_file；否则直接扫 image/*.png）
        rgb_entries = self._load_rgb_entries()

        # 2) 读取 ORB-SLAM3 轨迹
        pose_times, pose_mats = self._load_orb_trajectory(cfg.trajectory_file)

        # 3) 时间戳对齐：每张图像匹配最近的轨迹位姿
        self.samples = []
        for ts, img_path, depth_path in rgb_entries:
            matched = self._match_pose(ts, pose_times, pose_mats, cfg.pose_time_tolerance)
            if matched is None:
                continue
            Twc = matched
            self.samples.append(
                {
                    "timestamp": ts,
                    "image_path": img_path,
                    "depth_path": depth_path,   # 暂时不强制使用
                    "extrinsics": Twc,          # 直接使用 4x4 Twc
                }
            )

        if cfg.frame_stride > 1:
            self.samples = self.samples[:: cfg.frame_stride]

        if cfg.test_len > 0 and stage == "test":
            self.samples = self.samples[: cfg.test_len]

        if len(self.samples) == 0:
            raise RuntimeError("No valid RGB-pose pairs were found for TUM dataset.")

        # 4) 预先整理整段序列的 extrinsics / intrinsics，后面 view_sampler 直接用
        self.all_extrinsics = torch.stack([x["extrinsics"] for x in self.samples], dim=0)
        self.all_intrinsics = self._build_intrinsics(len(self.samples))

    def _load_rgb_entries(self):
        """
        返回 list of:
            (timestamp: float, image_path: Path, depth_path: Path | None)
        """
        entries = []

        if self.cfg.association_file is not None and self.cfg.association_file.exists():
            with self.cfg.association_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # TUM 常见 association 格式:
                    # rgb_ts rgb_path depth_ts depth_path
                    if len(parts) >= 4:
                        rgb_ts = float(parts[0])
                        rgb_rel = parts[1]
                        depth_rel = parts[3]
                        img_path = self.cfg.root / rgb_rel
                        depth_path = self.cfg.root / depth_rel
                        entries.append((rgb_ts, img_path, depth_path))
                    # 兜底：如果只有 timestamp + path
                    elif len(parts) >= 2:
                        rgb_ts = float(parts[0])
                        img_path = self.cfg.root / parts[1]
                        entries.append((rgb_ts, img_path, None))
        else:
            # 直接遍历 image/*.png，文件名即时间戳
            for p in sorted(self.image_root.glob("*.png")):
                ts = float(p.stem)
                depth_path = self.depth_root / f"{p.stem}.png"
                if not depth_path.exists():
                    depth_path = None
                entries.append((ts, p, depth_path))

        entries.sort(key=lambda x: x[0])
        return entries

    def _load_orb_trajectory(self, traj_file: Path):
        """
        读取 ORB-SLAM3/TUM 风格轨迹:
            timestamp tx ty tz qx qy qz qw
        返回:
            pose_times: list[float]
            pose_mats:  list[Tensor(4,4)]  # Twc
        """
        pose_times = []
        pose_mats = []

        with traj_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts[:8])
                Twc = self._build_Twc_from_tum(tx, ty, tz, qx, qy, qz, qw)

                pose_times.append(ts)
                pose_mats.append(Twc)

        if len(pose_times) == 0:
            raise RuntimeError(f"No valid poses found in trajectory file: {traj_file}")

        return pose_times, pose_mats

    def _quat_to_rotmat(self, qx, qy, qz, qw):
        """
        输入顺序: qx qy qz qw
        输出: Rwc
        """

        # 归一化
        q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
        q = q / q.norm()
        qx, qy, qz, qw = q.tolist()

        x, y, z, w = qx, qy, qz, qw

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.tensor(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
                [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
            ],
            dtype=torch.float32,
        )
        return R

    def _build_Twc_from_tum(self, tx, ty, tz, qx, qy, qz, qw):
        """
        ORB-SLAM3 TUM 风格输出通常可解释为:
            twc + q(Rwc)
        这里直接组装成 4x4 的 Twc
        """
        Twc = torch.eye(4, dtype=torch.float32)
        Twc[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)
        Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
        return Twc

    def _match_pose(self, ts, pose_times, pose_mats, tol):
        """
        最近邻匹配图像时间戳和轨迹时间戳
        """
        idx = bisect.bisect_left(pose_times, ts)

        candidates = []
        if idx < len(pose_times):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)

        best_idx = None
        best_dt = None
        for j in candidates:
            dt = abs(pose_times[j] - ts)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = j

        if best_idx is None or best_dt is None or best_dt > tol:
            return None

        return pose_mats[best_idx]

    def _build_intrinsics(self, n_views: int):
        """
        按当前项目的习惯，先采用归一化 K（与 RE10K loader 注释一致）
        """
        fx, fy, cx, cy = self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy

        # 从第一张图读分辨率
        with Image.open(self.samples[0]["image_path"]) as im:
            w, h = im.size

        if self.cfg.normalize_intrinsics:
            fx = fx / w
            fy = fy / h
            cx = cx / w
            cy = cy / h

        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        K = K.unsqueeze(0).repeat(n_views, 1, 1)
        return K

    def _load_image(self, p: Path) -> Tensor:
        with Image.open(p) as im:
            im = im.convert("RGB")
            return self.to_tensor(im)

    def _get_bound(self, value: float, n_views: int):
        return torch.full((n_views,), float(value), dtype=torch.float32)

    def __len__(self):
        # 对单个序列，最简单地把长度设为可用帧数
        return len(self.samples)

    def __getitem__(self, idx):
        # 对一整段序列做 view sampling
        context_indices, target_indices = self.view_sampler.sample(
            self.scene_name,
            self.all_extrinsics,
            self.all_intrinsics,
        )

        # if self.stage == "test":
        #     target_indices = target_indices[len(target_indices)//2 : len(target_indices)//2 + 1]

        # print("context idx =", context_indices)
        # print("target idx  =", target_indices)
        # print("context ext =", self.all_extrinsics[context_indices].shape)
        # print("K =", self.all_intrinsics[context_indices])


        context_images = torch.stack(
            [self._load_image(self.samples[i]["image_path"]) for i in context_indices.tolist()],
            dim=0,
        )
        target_images = torch.stack(
            [self._load_image(self.samples[i]["image_path"]) for i in target_indices.tolist()],
            dim=0,
        )

        example = {
            "context": {
                "extrinsics": self.all_extrinsics[context_indices],
                "intrinsics": self.all_intrinsics[context_indices],
                "image": context_images,
                "near": self._get_bound(self.cfg.near, len(context_indices)),
                "far": self._get_bound(self.cfg.far, len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": self.all_extrinsics[target_indices],
                "intrinsics": self.all_intrinsics[target_indices],
                "image": target_images,
                "near": self._get_bound(self.cfg.near, len(target_indices)),
                "far": self._get_bound(self.cfg.far, len(target_indices)),
                "index": target_indices,
            },
            "scene": self.scene_name,
        }
        print("===============================")
        return example
'''