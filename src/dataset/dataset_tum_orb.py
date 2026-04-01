from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import bisect

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetTUMORBCfg(DatasetCfgCommon):
    name: Literal["tum_orb"]

    # ---------- 数据路径 ----------
    root: Path
    trajectory_file: Path
    association_file: Optional[Path] = None
    image_dirname: str = "image"
    depth_dirname: str = "depth"

    # ---------- 图像-轨迹时间戳匹配 ----------
    pose_time_tolerance: float = 0.02

    # ---------- 原始像素内参 ----------
    fx: float = 542.822841
    fy: float = 542.576870
    cx: float = 315.593520
    cy: float = 237.756098

    # 输出 normalized K
    normalize_intrinsics: bool = True

    # ---------- 采样 ----------
    frame_stride: int = 1

    # ---------- train/val/test 切分 ----------
    train_split_ratio: float = 0.7
    val_split_ratio: float = 0.15
    # test 比例默认用剩余部分，不单独配也行；这里留着更清晰
    test_split_ratio: float = 0.15

    # ---------- debug / evaluation ----------
    test_len: int = -1

    # ---------- 视锥范围 ----------
    near: float = 0.1
    far: float = 8.0


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

        # 1) 读取 RGB 文件列表（优先 associations；否则直接遍历 image/*.png）
        rgb_entries = self._load_rgb_entries()

        # 2) 读取轨迹（TUM / ORB / GT 格式）
        pose_times, pose_mats = self._load_trajectory(cfg.trajectory_file)

        # 3) 时间戳对齐：每张图像匹配最近的 pose
        all_samples = []
        for ts, img_path, depth_path in rgb_entries:
            Twc = self._match_pose(ts, pose_times, pose_mats, cfg.pose_time_tolerance)
            if Twc is None:
                continue
            all_samples.append(
                {
                    "timestamp": ts,
                    "image_path": img_path,
                    "depth_path": depth_path,
                    "extrinsics": Twc,  # 4x4 Twc
                }
            )

        if cfg.frame_stride > 1:
            all_samples = all_samples[:: cfg.frame_stride]

        if len(all_samples) == 0:
            raise RuntimeError("No valid RGB-pose pairs found for TUM/ORB dataset.")

        self.all_samples = all_samples
        self.eval_len = None

        sampler_name = getattr(self.view_sampler.cfg, "name", None)

        # ------------------------------------------------------------------
        # 训练/验证：按 contiguous split 切分序列
        # 测试：
        #   - 如果是 evaluation sampler，保留完整 self.all_samples，
        #     因为你的 JSON 索引是基于完整序列定义的
        #   - 否则走普通 split
        # ------------------------------------------------------------------
        if stage == "test" and sampler_name == "evaluation":
            self.samples = self.all_samples
            if cfg.test_len > 0:
                self.eval_len = cfg.test_len
            else:
                # evaluation sampler 通常有 index_path；如果没有，就退回全长
                self.eval_len = getattr(self.view_sampler, "index", None)
                if self.eval_len is not None:
                    self.eval_len = len(self.view_sampler.index)
                else:
                    self.eval_len = len(self.samples)
        else:
            self.samples = self._split_samples(self.all_samples, stage)

            if stage == "test" and cfg.test_len > 0:
                self.samples = self.samples[: cfg.test_len]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples left after stage split. stage={stage}, root={cfg.root}"
            )

        print(
            f">>> DatasetTUMORB init: stage={stage}, sampler={sampler_name}, "
            f"all={len(self.all_samples)}, used={len(self.samples)}, scene={self.scene_name}"
        )

    # -------------------------------------------------------------------------
    # split
    # -------------------------------------------------------------------------
    def _split_samples(self, samples, stage: Stage):
        n = len(samples)

        train_r = float(self.cfg.train_split_ratio)
        val_r = float(self.cfg.val_split_ratio)
        test_r = float(self.cfg.test_split_ratio)

        if train_r < 0 or val_r < 0 or test_r < 0:
            raise ValueError("Split ratios must be non-negative.")

        total = train_r + val_r + test_r
        if total <= 0:
            raise ValueError("At least one split ratio must be positive.")

        # 归一化，避免用户写成 70/15/15 或 0.7/0.15/0.15 都能工作
        if total > 1.0 + 1e-6:
            train_r /= total
            val_r /= total
            test_r /= total

        train_end = int(round(n * train_r))
        val_end = int(round(n * (train_r + val_r)))

        # 边界保护
        train_end = max(1, min(train_end, n))
        val_end = max(train_end + 1, min(val_end, n)) if n >= 2 else n

        if stage == "train":
            subset = samples[:train_end]
        elif stage == "val":
            subset = samples[train_end:val_end]
            if len(subset) == 0:
                # 回退：至少给 1 个样本，避免 val loader 直接空
                subset = samples[max(0, train_end - 1):train_end]
        elif stage == "test":
            subset = samples[val_end:]
            if len(subset) == 0:
                subset = samples[max(0, n - 1):]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return subset

    # -------------------------------------------------------------------------
    # 数据读取
    # -------------------------------------------------------------------------
    def _load_rgb_entries(self):
        """
        返回:
            list[(timestamp, image_path, depth_path_or_None)]
        """
        entries = []

        # 如果存在 associations.txt，优先使用
        if self.cfg.association_file is not None and self.cfg.association_file.exists():
            with self.cfg.association_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()

                    # 常见 TUM association:
                    # rgb_ts rgb_path depth_ts depth_path
                    if len(parts) >= 4:
                        rgb_ts = float(parts[0])
                        img_path = self.cfg.root / parts[1]
                        depth_path = self.cfg.root / parts[3]
                        entries.append((rgb_ts, img_path, depth_path))
                    elif len(parts) >= 2:
                        rgb_ts = float(parts[0])
                        img_path = self.cfg.root / parts[1]
                        entries.append((rgb_ts, img_path, None))
        else:
            # 直接遍历 image/*.png，文件名 stem 作为时间戳
            for p in sorted(self.image_root.glob("*.png")):
                ts = float(p.stem)
                depth_path = self.depth_root / f"{p.stem}.png"
                if not depth_path.exists():
                    depth_path = None
                entries.append((ts, p, depth_path))

        entries.sort(key=lambda x: x[0])
        return entries

    def _load_trajectory(self, traj_file: Path):
        """
        读取 TUM / ORB / GT 常见格式:
            timestamp tx ty tz qx qy qz qw

        返回:
            pose_times: list[float]
            pose_mats: list[Tensor(4, 4)]  # Twc
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

    def _match_pose(self, ts, pose_times, pose_mats, tol):
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

    # -------------------------------------------------------------------------
    # 位姿 / 内参
    # -------------------------------------------------------------------------
    def _quat_to_rotmat(self, qx, qy, qz, qw):
        """
        输入顺序: qx qy qz qw
        输出: Rwc
        """
        q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
        q = q / q.norm()
        x, y, z, w = q.tolist()

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.tensor(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=torch.float32,
        )
        return R

    def _build_Twc_from_tum(self, tx, ty, tz, qx, qy, qz, qw):
        """
        TUM / ORB / GT 常见语义:
            twc + q(Rwc)
        组装成 4x4 Twc
        """
        Twc = torch.eye(4, dtype=torch.float32)
        Twc[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)
        Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
        return Twc

    def _base_pixel_K(self):
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = self.cfg.fx
        K[1, 1] = self.cfg.fy
        K[0, 2] = self.cfg.cx
        K[1, 2] = self.cfg.cy
        return K

    # -------------------------------------------------------------------------
    # 图像几何处理
    # -------------------------------------------------------------------------
    def _process_image_and_K(self, image_path: Path):
        """
        原图 -> 保持比例 resize -> center crop
             -> 同步更新像素 K -> 可选 normalized K
        """
        with Image.open(image_path) as im:
            im = im.convert("RGB")

            orig_w, orig_h = im.size
            K = self._base_pixel_K()

            target_h, target_w = self.cfg.image_shape

            # 保持比例 resize，使短边覆盖目标大小
            scale = max(target_w / orig_w, target_h / orig_h)
            resized_w = int(round(orig_w * scale))
            resized_h = int(round(orig_h * scale))

            im = im.resize((resized_w, resized_h), Image.BILINEAR)

            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] *= scale
            K[1, 2] *= scale

            # center crop
            left = int(round((resized_w - target_w) / 2.0))
            top = int(round((resized_h - target_h) / 2.0))
            right = left + target_w
            bottom = top + target_h
            im = im.crop((left, top, right, bottom))

            K[0, 2] -= left
            K[1, 2] -= top

            if self.cfg.normalize_intrinsics:
                K[0, 0] /= target_w
                K[1, 1] /= target_h
                K[0, 2] /= target_w
                K[1, 2] /= target_h

            image_tensor = self.to_tensor(im)

        return image_tensor, K

    # -------------------------------------------------------------------------
    # 辅助
    # -------------------------------------------------------------------------
    def _get_bound(self, value: float, n_views: int):
        return torch.full((n_views,), float(value), dtype=torch.float32)

    def __len__(self):
        if self.eval_len is not None:
            return self.eval_len
        return len(self.samples)

    # -------------------------------------------------------------------------
    # 主接口
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        支持两种模式：
        1) bounded / arbitrary 等普通 sampler
        2) evaluation sampler：通过 JSON 精确控制 context / target
        """
        sampler_name = getattr(self.view_sampler.cfg, "name", None)

        # 当前 self.samples:
        # - train/val/bounded-test: 已是 stage split 后的子序列
        # - test/evaluation: 保留完整序列
        all_extrinsics = torch.stack([x["extrinsics"] for x in self.samples], dim=0)
        dummy_intrinsics = torch.stack(
            [self._base_pixel_K() for _ in range(len(self.samples))], dim=0
        )

        if sampler_name == "evaluation":
            scene_key = f"{self.scene_name}_{idx:04d}"
        else:
            scene_key = self.scene_name

        context_indices, target_indices = self.view_sampler.sample(
            scene_key,
            all_extrinsics,
            dummy_intrinsics,
        )

        context_images = []
        context_intrinsics = []
        for i in context_indices.tolist():
            img, K = self._process_image_and_K(self.samples[i]["image_path"])
            context_images.append(img)
            context_intrinsics.append(K)

        target_images = []
        target_intrinsics = []
        for i in target_indices.tolist():
            img, K = self._process_image_and_K(self.samples[i]["image_path"])
            target_images.append(img)
            target_intrinsics.append(K)

        context_images = torch.stack(context_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        context_intrinsics = torch.stack(context_intrinsics, dim=0)
        target_intrinsics = torch.stack(target_intrinsics, dim=0)

        example = {
            "context": {
                "extrinsics": all_extrinsics[context_indices],
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": self._get_bound(self.cfg.near, len(context_indices)),
                "far": self._get_bound(self.cfg.far, len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": all_extrinsics[target_indices],
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self._get_bound(self.cfg.near, len(target_indices)),
                "far": self._get_bound(self.cfg.far, len(target_indices)),
                "index": target_indices,
            },
            "scene": scene_key,
        }
        return example