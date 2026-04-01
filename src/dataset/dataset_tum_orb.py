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

    # 最终是否输出 normalized K
    normalize_intrinsics: bool = True

    # ---------- 测试长度 / 采样 ----------
    test_len: int = 50
    frame_stride: int = 1

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
        self.samples = []
        for ts, img_path, depth_path in rgb_entries:
            Twc = self._match_pose(ts, pose_times, pose_mats, cfg.pose_time_tolerance)
            if Twc is None:
                continue

            self.samples.append(
                {
                    "timestamp": ts,
                    "image_path": img_path,
                    "depth_path": depth_path,
                    "extrinsics": Twc,   # 直接保存 4x4 Twc
                }
            )

        if cfg.frame_stride > 1:
            self.samples = self.samples[:: cfg.frame_stride]

        
        # evaluation 模式下：
        # - 保留完整 self.samples（因为 JSON 里的 context/target 索引依赖完整帧池）
        # - 只限制 dataset 的长度，不要裁剪 self.samples
        self.eval_len = None
        sampler_name = getattr(self.view_sampler.cfg, "name", None)

        if stage == "test" and sampler_name == "evaluation":
            if cfg.test_len > 0:
                self.eval_len = cfg.test_len
            else:
                self.eval_len = len(self.view_sampler.index)
        else:
            if cfg.test_len > 0 and stage == "test":
                self.samples = self.samples[: cfg.test_len]

                if cfg.test_len > 0 and stage == "test":
                    self.samples = self.samples[: cfg.test_len]

        if len(self.samples) == 0:
            raise RuntimeError("No valid RGB-pose pairs found for TUM dataset.")


    # -------------------------------------------------------------------------
    # 数据读取
    # -------------------------------------------------------------------------
    def _load_rgb_entries(self):
        """
        返回:
            list[(timestamp, image_path, depth_path_or_None)]
        """
        entries = []

        # 如果你以后想用 TUM 的 associations.txt，可以保留这个入口。
        # 当前如果只是 RGB + pose -> MVSplat，association_file 不是必须的。
        if self.cfg.association_file is not None and self.cfg.association_file.exists():
            with self.cfg.association_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()

                    # TUM 常见 association:
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
            # 直接遍历 image/*.png，文件名即时间戳
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
            pose_mats: list[Tensor(4,4)]   # Twc
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

    # -------------------------------------------------------------------------
    # 位姿 / 内参
    # -------------------------------------------------------------------------
    def _quat_to_rotmat(self, qx, qy, qz, qw):
        """
        输入顺序: qx qy qz qw
        输出: Rwc
        """
        q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
        q = q / q.norm()  # 防止非单位四元数导致非法旋转矩阵

        x, y, z, w = q.tolist()

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
        TUM / ORB / GT 常见语义:
            twc + q(Rwc)
        组装成 4x4 Twc
        """
        Twc = torch.eye(4, dtype=torch.float32)
        Twc[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)
        Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
        return Twc

    def _base_pixel_K(self):
        """
        原始像素 K（尚未做 resize/crop）
        """
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = self.cfg.fx
        K[1, 1] = self.cfg.fy
        K[0, 2] = self.cfg.cx
        K[1, 2] = self.cfg.cy
        return K

    # -------------------------------------------------------------------------
    # 图像几何处理（严谨版）
    # -------------------------------------------------------------------------
    def _process_image_and_K(self, image_path: Path):
        """
        严谨做法：
            原图 -> (可选) undistort -> 保持比例 resize -> center crop
            -> 同步更新像素 K -> 最终再 normalized K

        返回:
            image_tensor: [3,H,W]
            K_final:      [3,3]   # 最终 normalized 或 pixel K
        """
        with Image.open(image_path) as im:
            im = im.convert("RGB")

            # -----------------------------------------------------------------
            # 如果以后要处理 TUM 畸变，这里就是入口：
            # 1) 用 ORB-SLAM3 / TUM 的 k1,k2,p1,p2,k3 先做 undistort
            # 2) 同时把原始像素 K 更新成去畸变后的新 K
            # 当前按你的要求：畸变先不处理，只保留这个注释位置。
            # -----------------------------------------------------------------

            orig_w, orig_h = im.size
            K = self._base_pixel_K()

            target_h, target_w = self.cfg.image_shape

            # 1) 保持比例 resize，使得短边至少覆盖 target
            scale = max(target_w / orig_w, target_h / orig_h)
            resized_w = int(round(orig_w * scale))
            resized_h = int(round(orig_h * scale))

            im = im.resize((resized_w, resized_h), Image.BILINEAR)

            # 像素 K 随 resize 缩放
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] *= scale
            K[1, 2] *= scale

            # 2) center crop 到最终尺寸
            left = int(round((resized_w - target_w) / 2.0))
            top = int(round((resized_h - target_h) / 2.0))
            right = left + target_w
            bottom = top + target_h

            im = im.crop((left, top, right, bottom))

            # principal point 随 crop 平移
            K[0, 2] -= left
            K[1, 2] -= top

            # 3) 如果需要，最终转成 normalized K
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

        # 先准备整段序列的 pose / K（给 sampler 用）
        all_extrinsics = torch.stack([x["extrinsics"] for x in self.samples], dim=0)
        dummy_intrinsics = torch.stack(
            [self._base_pixel_K() for _ in range(len(self.samples))], dim=0
        )

        # ------------------------------------------------------------
        # 关键点：
        # evaluation sampler 是按 scene 字符串查 JSON 的。
        # 所以如果想让每个 idx 对应一条 JSON 记录，就必须给每个样本不同的 scene key。
        # ------------------------------------------------------------
        sampler_name = getattr(self.view_sampler.cfg, "name", None)
        if sampler_name == "evaluation":
            scene_key = f"{self.scene_name}_{idx:04d}"
        else:
            scene_key = self.scene_name

        context_indices, target_indices = self.view_sampler.sample(
            scene_key,
            all_extrinsics,
            dummy_intrinsics,
        )

        # 读取 context 图像并同步构造最终 K
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
        # print("context idx =", context_indices)
        # print("target idx  =", target_indices)
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
            "scene": scene_key,   # 注意：这里也返回 scene_key
        }
        return example

    '''
    def __getitem__(self, idx):
        """
        用整个序列的 pose 做 view sampling，
        但在 test 阶段强制只保留 1 个 target，避免一整段 target 过慢。
        """
        # 先拿整段序列的 pose / K（K 先用虚拟占位，仅供 sampler 使用）
        # sampler 主要依赖视图数量 / 序列布局，真正的 K 会在下方按具体帧构造
        dummy_intrinsics = torch.stack(
            [self._base_pixel_K() for _ in range(len(self.samples))], dim=0
        )

        context_indices, target_indices = self.view_sampler.sample(
            self.scene_name,
            torch.stack([x["extrinsics"] for x in self.samples], dim=0),
            dummy_intrinsics,
        )

        # 测试阶段只保留 1 个 target
        # if self.stage == "test":
        #     if len(target_indices) > 1:
        #         # 取中间的一个 target，比 target_indices[:1] 更稳一点
        #         mid = len(target_indices) // 2
        #         target_indices = target_indices[mid : mid + 1]

        # 读取 context 图像并同步构造最终 K
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

        all_extrinsics = torch.stack([x["extrinsics"] for x in self.samples], dim=0)
        print("context idx =", context_indices)
        print("target idx  =", target_indices)
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
            "scene": self.scene_name,
        }
        return example
        '''