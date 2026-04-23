from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class SequenceCfg:
    scene: str
    root: Path
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class DatasetTartanAirCfg(DatasetCfgCommon):
    name: Literal["tartanair"]

    sequences: list[SequenceCfg] = field(default_factory=list)
    root: Optional[Path] = None
    auto_discover_sequences: bool = True
    scene_glob: str = "P*"

    # TartanAir stereo folders / pose files.
    left_camera_dirname: str = "image_lcam_front"
    right_camera_dirname: str = "image_rcam_front"
    left_pose_filename: str = "pose_lcam_front.txt"
    right_pose_filename: str = "pose_rcam_front.txt"

    # MVSplat stereo setting.
    stereo_as_context: bool = True
    target_camera: Literal["left", "right", "both"] = "left"

    # Pose parsing.
    # Default assumes each line is: tx ty tz qx qy qz qw
    pose_format: Literal[
        "tx_ty_tz_qx_qy_qz_qw",
        "tx_ty_tz_qw_qx_qy_qz",
    ] = "tx_ty_tz_qx_qy_qz_qw"
    pose_matrix_type: Literal["Twc", "Tcw"] = "Twc"

    # Intrinsics must be provided explicitly.
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    normalize_intrinsics: bool = True
    frame_stride: int = 1
    frame_start: int = 0
    max_frames: int = -1

    train_split_ratio: float = 0.7
    val_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    test_len: int = -1

    near: float = 0.1
    far: float = 50.0

    # Length mismatch handling.
    strict_length_check: bool = False


class DatasetTartanAir(Dataset):
    def __init__(self, cfg: DatasetTartanAirCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if not self.cfg.stereo_as_context:
            raise NotImplementedError(
                "DatasetTartanAir currently only supports stereo_as_context=True."
            )

        seq_cfgs = self._build_sequence_cfgs()
        self.scenes = []
        self.eval_items = None
        sampler_name = getattr(self.view_sampler.cfg, "name", None)

        for seq_cfg in seq_cfgs:
            samples = self._build_stereo_samples_for_sequence(seq_cfg)
            if len(samples) == 0:
                continue

            if sampler_name != "evaluation":
                samples = self._split_samples(samples, stage)
            if len(samples) == 0:
                continue

            self.scenes.append(
                {
                    "scene_name": seq_cfg.scene,
                    "samples": samples,
                    "fx": seq_cfg.fx,
                    "fy": seq_cfg.fy,
                    "cx": seq_cfg.cx,
                    "cy": seq_cfg.cy,
                }
            )

        if len(self.scenes) == 0:
            raise RuntimeError(f"No valid TartanAir scenes found for stage={stage}.")

        if sampler_name == "evaluation":
            self.eval_items = self._build_eval_items()
            if self.cfg.test_len > 0 and stage == "test":
                self.eval_items = self.eval_items[: self.cfg.test_len]
            if len(self.eval_items) == 0:
                raise RuntimeError(
                    f"Evaluation sampler active but no matching scene keys found for stage={stage}."
                )

    def _build_sequence_cfgs(self) -> list[SequenceCfg]:
        if len(self.cfg.sequences) > 0:
            seq_cfgs = self.cfg.sequences
        else:
            if self.cfg.root is None:
                raise RuntimeError(
                    "tartanair config must provide either `sequences` or `root`."
                )
            root = Path(self.cfg.root)
            if not root.exists():
                raise RuntimeError(f"TartanAir root does not exist: {root}")

            discovered = []
            if self.cfg.auto_discover_sequences:
                discovered = [
                    p for p in sorted(root.glob(self.cfg.scene_glob)) if p.is_dir()
                ]

            if len(discovered) == 0:
                discovered = [root]

            seq_cfgs = [
                SequenceCfg(
                    scene=p.name,
                    root=p,
                    fx=self.cfg.fx,
                    fy=self.cfg.fy,
                    cx=self.cfg.cx,
                    cy=self.cfg.cy,
                )
                for p in discovered
            ]

        for seq_cfg in seq_cfgs:
            self._validate_intrinsics(seq_cfg)
        return seq_cfgs

    def _validate_intrinsics(self, seq_cfg: SequenceCfg) -> None:
        vals = [seq_cfg.fx, seq_cfg.fy, seq_cfg.cx, seq_cfg.cy]
        if not all(v > 0 for v in vals):
            raise RuntimeError(
                f"Invalid intrinsics for scene={seq_cfg.scene}. "
                f"Please set fx/fy/cx/cy explicitly in tartanair config. "
                f"Got fx={seq_cfg.fx}, fy={seq_cfg.fy}, cx={seq_cfg.cx}, cy={seq_cfg.cy}."
            )

    def _build_eval_items(self):
        if not hasattr(self.view_sampler, "index") or self.view_sampler.index is None:
            raise RuntimeError(
                "Evaluation sampler does not expose `index`. "
                "Please check your evaluation view sampler implementation."
            )

        raw_index = self.view_sampler.index
        if isinstance(raw_index, dict):
            all_keys = list(raw_index.keys())
        elif isinstance(raw_index, list):
            all_keys = list(raw_index)
        else:
            raise RuntimeError(f"Unsupported evaluation index type: {type(raw_index)}")

        eval_items = []

        def sort_key(scene_key: str):
            suffix = scene_key.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return (0, int(suffix))
            return (1, scene_key)

        for scene_idx, scene in enumerate(self.scenes):
            scene_name = scene["scene_name"]
            matched_keys = [
                k for k in all_keys if k == scene_name or k.startswith(f"{scene_name}_")
            ]
            matched_keys = sorted(matched_keys, key=sort_key)
            for local_eval_idx, scene_key in enumerate(matched_keys):
                eval_items.append(
                    {
                        "scene_idx": scene_idx,
                        "scene_name": scene_name,
                        "scene_key": scene_key,
                        "local_eval_idx": local_eval_idx,
                    }
                )
        return eval_items
    
    def _build_stereo_samples_for_sequence(self, seq_cfg: SequenceCfg):
        left_root = seq_cfg.root / self.cfg.left_camera_dirname
        right_root = seq_cfg.root / self.cfg.right_camera_dirname
        left_pose_path = seq_cfg.root / self.cfg.left_pose_filename
        right_pose_path = seq_cfg.root / self.cfg.right_pose_filename

        left_entries = self._load_image_entries_from_dir(left_root)
        right_entries = self._load_image_entries_from_dir(right_root)
        left_poses = self._load_pose_file(left_pose_path)
        right_poses = self._load_pose_file(right_pose_path)

        n_left_imgs = len(left_entries)
        n_right_imgs = len(right_entries)
        n_left_pose = len(left_poses)
        n_right_pose = len(right_poses)
        n = min(n_left_imgs, n_right_imgs, n_left_pose, n_right_pose)

        if n == 0:
            return []

        if self.cfg.strict_length_check:
            if not (
                n_left_imgs == n_right_imgs == n_left_pose == n_right_pose
            ):
                raise RuntimeError(
                    f"Length mismatch in scene={seq_cfg.scene}: "
                    f"left_imgs={n_left_imgs}, right_imgs={n_right_imgs}, "
                    f"left_pose={n_left_pose}, right_pose={n_right_pose}"
                )

        left_entries = left_entries[:n]
        right_entries = right_entries[:n]
        left_poses = left_poses[:n]
        right_poses = right_poses[:n]

        if self.cfg.frame_start > 0:
            left_entries = left_entries[self.cfg.frame_start :]
            right_entries = right_entries[self.cfg.frame_start :]
            left_poses = left_poses[self.cfg.frame_start :]
            right_poses = right_poses[self.cfg.frame_start :]

        if self.cfg.max_frames > 0:
            left_entries = left_entries[: self.cfg.max_frames]
            right_entries = right_entries[: self.cfg.max_frames]
            left_poses = left_poses[: self.cfg.max_frames]
            right_poses = right_poses[: self.cfg.max_frames]

        K_left = self._build_K(seq_cfg.fx, seq_cfg.fy, seq_cfg.cx, seq_cfg.cy)
        K_right = self._build_K(seq_cfg.fx, seq_cfg.fy, seq_cfg.cx, seq_cfg.cy)

        samples = []
        for i, ((_, left_path), (_, right_path), Twc_left, Twc_right) in enumerate(
            zip(left_entries, right_entries, left_poses, right_poses)
        ):
            samples.append(
                {
                    "timestamp": float(i),
                    "frame_index": i,
                    "left_image_path": left_path,
                    "right_image_path": right_path,
                    "left_extrinsics": Twc_left,
                    "right_extrinsics": Twc_right,
                    "left_K": K_left.clone(),
                    "right_K": K_right.clone(),
                    "scene_name": seq_cfg.scene,
                }
            )

        if self.cfg.frame_stride > 1:
            samples = samples[:: self.cfg.frame_stride]
        return samples
    
    def _build_stereo_samples_for_sequence1(self, seq_cfg: SequenceCfg):
        left_root = seq_cfg.root / self.cfg.left_camera_dirname
        right_root = seq_cfg.root / self.cfg.right_camera_dirname
        left_pose_path = seq_cfg.root / self.cfg.left_pose_filename
        right_pose_path = seq_cfg.root / self.cfg.right_pose_filename

        left_entries = self._load_image_entries_from_dir(left_root)
        right_entries = self._load_image_entries_from_dir(right_root)
        left_poses = self._load_pose_file(left_pose_path)
        right_poses = self._load_pose_file(right_pose_path)

        n_left_imgs = len(left_entries)
        n_right_imgs = len(right_entries)
        n_left_pose = len(left_poses)
        n_right_pose = len(right_poses)
        n = min(n_left_imgs, n_right_imgs, n_left_pose, n_right_pose)

        if n == 0:
            return []

        if self.cfg.strict_length_check:
            if not (n_left_imgs == n_right_imgs == n_left_pose == n_right_pose):
                raise RuntimeError(
                    f"Length mismatch in scene={seq_cfg.scene}: "
                    f"left_imgs={n_left_imgs}, right_imgs={n_right_imgs}, "
                    f"left_pose={n_left_pose}, right_pose={n_right_pose}"
                )

        left_entries = left_entries[:n]
        right_entries = right_entries[:n]
        left_poses = left_poses[:n]
        right_poses = right_poses[:n]

        if self.cfg.frame_start > 0:
            left_entries = left_entries[self.cfg.frame_start :]
            right_entries = right_entries[self.cfg.frame_start :]
            left_poses = left_poses[self.cfg.frame_start :]
            right_poses = right_poses[self.cfg.frame_start :]

        if self.cfg.max_frames > 0:
            left_entries = left_entries[: self.cfg.max_frames]
            right_entries = right_entries[: self.cfg.max_frames]
            left_poses = left_poses[: self.cfg.max_frames]
            right_poses = right_poses[: self.cfg.max_frames]

        if len(left_poses) == 0 or len(right_poses) == 0:
            return []

        # 固定 stereo 外参：右目相对于左目的变换
        # 先用第一帧做 debug 版。后面如果你想更稳，可以改成多帧平均。
        T_lr = torch.linalg.inv(left_poses[0]) @ right_poses[0]

        K_left = self._build_K(seq_cfg.fx, seq_cfg.fy, seq_cfg.cx, seq_cfg.cy)
        K_right = self._build_K(seq_cfg.fx, seq_cfg.fy, seq_cfg.cx, seq_cfg.cy)

        samples = []
        for i, ((_, left_path), (_, right_path), Twc_left) in enumerate(
            zip(left_entries, right_entries, left_poses)
        ):
            # 不再直接使用 right_poses[i]
            Twc_right = Twc_left @ T_lr

            samples.append(
                {
                    "timestamp": float(i),
                    "frame_index": i,
                    "left_image_path": left_path,
                    "right_image_path": right_path,
                    "left_extrinsics": Twc_left,
                    "right_extrinsics": Twc_right,
                    "left_K": K_left.clone(),
                    "right_K": K_right.clone(),
                    "scene_name": seq_cfg.scene,
                }
            )

        if self.cfg.frame_stride > 1:
            samples = samples[:: self.cfg.frame_stride]
        return samples

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
        if total > 1.0 + 1e-6:
            train_r /= total
            val_r /= total
            test_r /= total

        train_end = int(round(n * train_r))
        val_end = int(round(n * (train_r + val_r)))
        train_end = max(1, min(train_end, n))
        val_end = max(train_end + 1, min(val_end, n)) if n >= 2 else n

        if stage == "train":
            subset = samples[:train_end]
        elif stage == "val":
            subset = samples[train_end:val_end]
            if len(subset) == 0:
                subset = samples[max(0, train_end - 1):train_end]
        elif stage == "test":
            subset = samples[val_end:]
            if len(subset) == 0:
                subset = samples[max(0, n - 1):]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return subset

    def _load_image_entries_from_dir(self, image_root: Path):
        if not image_root.exists():
            raise RuntimeError(f"Image directory not found: {image_root}")

        image_paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(image_root.glob(pattern))

        image_paths = sorted(image_paths, key=self._path_sort_key)
        return [(float(i), p) for i, p in enumerate(image_paths)]

    def _load_pose_file(self, pose_path: Path):
        if not pose_path.exists():
            raise RuntimeError(f"Pose file not found: {pose_path}")

        poses = []
        with pose_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 7:
                    continue
                values = list(map(float, parts[:7]))
                poses.append(self._build_pose_from_values(values))

        if len(poses) == 0:
            raise RuntimeError(f"No valid poses found in file: {pose_path}")
        return poses

    def _build_pose_from_values(self, values: list[float]) -> torch.Tensor:
        if self.cfg.pose_format == "tx_ty_tz_qx_qy_qz_qw":
            tx, ty, tz, qx, qy, qz, qw = values
        elif self.cfg.pose_format == "tx_ty_tz_qw_qx_qy_qz":
            tx, ty, tz, qw, qx, qy, qz = values
        else:
            raise ValueError(f"Unsupported pose_format: {self.cfg.pose_format}")

        Twc = self._build_Twc_from_pose(tx, ty, tz, qx, qy, qz, qw)
        if self.cfg.pose_matrix_type == "Twc":
            return Twc
        if self.cfg.pose_matrix_type == "Tcw":
            return torch.linalg.inv(Twc)
        raise ValueError(f"Unsupported pose_matrix_type: {self.cfg.pose_matrix_type}")

    def _path_sort_key(self, path: Path):
        stem = path.stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    def _quat_to_rotmat(self, qx, qy, qz, qw):
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

    def _build_Twc_from_pose(self, tx, ty, tz, qx, qy, qz, qw):
        # 1) TartanAir pose itself: treat as Twc in NED/world
        Twc_pose = torch.eye(4, dtype=torch.float32)
        Twc_pose[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)
        Twc_pose[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)

        # 2) Camera-axis conversion:
        # CV camera:     x right, y down, z forward
        # Tartan/AirSim: x forward, y right, z down
        T_tartanCam_from_cvCam = torch.eye(4, dtype=torch.float32)
        T_tartanCam_from_cvCam[:3, :3] = torch.tensor(
            [
                [0.0, 0.0, 1.0],  # cv z (forward) -> tartan x
                [1.0, 0.0, 0.0],  # cv x (right)   -> tartan y
                [0.0, 1.0, 0.0],  # cv y (down)    -> tartan z
            ],
            dtype=torch.float32,
        )

        Twc = Twc_pose @ T_tartanCam_from_cvCam
        return Twc

    def _build_K(self, fx: float, fy: float, cx: float, cy: float):
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def _base_pixel_K_from_sample(self, sample, camera: Optional[str] = None):
        if camera == "left":
            return sample["left_K"].clone()
        if camera == "right":
            return sample["right_K"].clone()
        raise ValueError(f"camera must be left/right in tartanair stereo mode, got {camera}")

    def _process_image_and_K(self, image_path: Path, K: torch.Tensor):
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            K = K.clone()
            target_h, target_w = self.cfg.image_shape
            scale = max(target_w / orig_w, target_h / orig_h)
            resized_w = int(round(orig_w * scale))
            resized_h = int(round(orig_h * scale))
            im = im.resize((resized_w, resized_h), Image.BILINEAR)
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] *= scale
            K[1, 2] *= scale
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

    def _get_bound(self, value: float, n_views: int):
        return torch.full((n_views,), float(value), dtype=torch.float32)

    def __len__(self):
        if self.eval_items is not None:
            return len(self.eval_items)
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.eval_items is not None:
            eval_item = self.eval_items[idx]
            scene = self.scenes[eval_item["scene_idx"]]
            scene_key = eval_item["scene_key"]
        else:
            scene = self.scenes[idx]
            scene_key = scene["scene_name"]

        scene_name = scene["scene_name"]
        samples = scene["samples"]

        all_extrinsics = torch.stack([x["left_extrinsics"] for x in samples], dim=0)
        dummy_intrinsics = torch.stack([x["left_K"] for x in samples], dim=0)
        context_indices, target_indices = self.view_sampler.sample(
            scene_key, all_extrinsics, dummy_intrinsics
        )

        context_images = []
        context_intrinsics = []
        context_extrinsics = []
        for i in context_indices.tolist():
            sample = samples[i]
            img_l, K_l = self._process_image_and_K(
                sample["left_image_path"],
                self._base_pixel_K_from_sample(sample, "left"),
            )
            img_r, K_r = self._process_image_and_K(
                sample["right_image_path"],
                self._base_pixel_K_from_sample(sample, "right"),
            )
            context_images.extend([img_l, img_r])
            context_intrinsics.extend([K_l, K_r])
            context_extrinsics.extend(
                [sample["left_extrinsics"], sample["right_extrinsics"]]
            )

        context_images = torch.stack(context_images, dim=0)
        context_intrinsics = torch.stack(context_intrinsics, dim=0)
        context_extrinsics = torch.stack(context_extrinsics, dim=0)

        target_images = []
        target_intrinsics = []
        target_extrinsics = []
        target_index_expanded = []
        target_camera_id = []

        for i in target_indices.tolist():
            sample = samples[i]

            if self.cfg.target_camera in ("left", "both"):
                img_l, K_l = self._process_image_and_K(
                    sample["left_image_path"],
                    self._base_pixel_K_from_sample(sample, "left"),
                )
                target_images.append(img_l)
                target_intrinsics.append(K_l)
                target_extrinsics.append(sample["left_extrinsics"])
                target_index_expanded.append(i)
                target_camera_id.append(0)

            if self.cfg.target_camera in ("right", "both"):
                img_r, K_r = self._process_image_and_K(
                    sample["right_image_path"],
                    self._base_pixel_K_from_sample(sample, "right"),
                )
                target_images.append(img_r)
                target_intrinsics.append(K_r)
                target_extrinsics.append(sample["right_extrinsics"])
                target_index_expanded.append(i)
                target_camera_id.append(1)

        target_images = torch.stack(target_images, dim=0)
        target_intrinsics = torch.stack(target_intrinsics, dim=0)
        target_extrinsics = torch.stack(target_extrinsics, dim=0)
        target_index_expanded = torch.tensor(target_index_expanded, dtype=torch.long)
        target_camera_id = torch.tensor(target_camera_id, dtype=torch.long)

        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": self._get_bound(self.cfg.near, context_images.shape[0]),
                "far": self._get_bound(self.cfg.far, context_images.shape[0]),
                "index": context_indices,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self._get_bound(self.cfg.near, target_images.shape[0]),
                "far": self._get_bound(self.cfg.far, target_images.shape[0]),
                "index": target_index_expanded,
                "camera_id": target_camera_id,
            },
            "scene": scene_key,
            "scene_name": scene_name,
        }
        return example
