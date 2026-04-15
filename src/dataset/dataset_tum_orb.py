from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import bisect
import csv

import torch
import torchvision.transforms as tf
import yaml
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class SequenceCfg:
    scene: str
    root: Path
    trajectory_file: Path
    association_file: Optional[Path] = None
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class DatasetTUMORBCfg(DatasetCfgCommon):
    name: Literal["tum_orb"]

    sequences: list[SequenceCfg] = field(default_factory=list)

    root: Optional[Path] = None
    trajectory_file: Optional[Path] = None
    association_file: Optional[Path] = None

    image_dirname: str = "image"
    depth_dirname: str = "depth"

    # -------- EuRoC / stereo-first 支持 --------
    stereo_as_context: bool = False
    left_camera_dirname: str = "cam0/data"
    right_camera_dirname: str = "cam1/data"
    left_sensor_yaml: str = "cam0/sensor.yaml"
    right_sensor_yaml: str = "cam1/sensor.yaml"
    target_camera: Literal["left", "right", "both"] = "left"

    pose_time_tolerance: float = 0.02

    fx: float = 542.822841
    fy: float = 542.576870
    cx: float = 315.593520
    cy: float = 237.756098

    normalize_intrinsics: bool = True
    frame_stride: int = 1

    train_split_ratio: float = 0.7
    val_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    test_len: int = -1

    near: float = 0.1
    far: float = 8.0


class DatasetTUMORB(Dataset):
    def __init__(self, cfg: DatasetTUMORBCfg, stage: Stage, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        seq_cfgs = self._build_sequence_cfgs()
        self.scenes = []
        self.eval_items = None
        sampler_name = getattr(self.view_sampler.cfg, "name", None)

        for seq_cfg in seq_cfgs:
            samples = self._build_samples_for_sequence(seq_cfg)
            if len(samples) == 0:
                continue

            # evaluation sampler: train/val/test 都直接依赖 JSON，不做 split
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
            raise RuntimeError(f"No valid scenes found for stage={stage}.")

        if sampler_name == "evaluation":
            self.eval_items = self._build_eval_items()
            if self.cfg.test_len > 0 and stage == "test":
                self.eval_items = self.eval_items[: self.cfg.test_len]
            if len(self.eval_items) == 0:
                raise RuntimeError(
                    f"Evaluation sampler active but no matching scene keys found for stage={stage}."
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
            matched_keys = [k for k in all_keys if k == scene_name or k.startswith(f"{scene_name}_")]
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

    def _build_sequence_cfgs(self):
        if len(self.cfg.sequences) > 0:
            return self.cfg.sequences
        if self.cfg.root is None or self.cfg.trajectory_file is None:
            raise RuntimeError(
                "tum_orb config must provide either `sequences` or (`root`, `trajectory_file`)."
            )
        scene_name = self.cfg.root.name
        return [
            SequenceCfg(
                scene=scene_name,
                root=self.cfg.root,
                trajectory_file=self.cfg.trajectory_file,
                association_file=self.cfg.association_file,
                fx=self.cfg.fx,
                fy=self.cfg.fy,
                cx=self.cfg.cx,
                cy=self.cfg.cy,
            )
        ]

    def _build_samples_for_sequence(self, seq_cfg: SequenceCfg):
        if self.cfg.stereo_as_context:
            return self._build_stereo_samples_for_sequence(seq_cfg)
        image_root = seq_cfg.root / self.cfg.image_dirname
        depth_root = seq_cfg.root / self.cfg.depth_dirname
        rgb_entries = self._load_rgb_entries(seq_cfg, image_root, depth_root)
        pose_times, pose_mats = self._load_trajectory(seq_cfg.trajectory_file)

        samples = []
        for ts, img_path, depth_path in rgb_entries:
            Twc = self._match_pose(ts, pose_times, pose_mats, self.cfg.pose_time_tolerance)
            if Twc is None:
                continue
            samples.append(
                {
                    "timestamp": ts,
                    "image_path": img_path,
                    "depth_path": depth_path,
                    "extrinsics": Twc,
                    "scene_name": seq_cfg.scene,
                    "fx": seq_cfg.fx,
                    "fy": seq_cfg.fy,
                    "cx": seq_cfg.cx,
                    "cy": seq_cfg.cy,
                }
            )

        if self.cfg.frame_stride > 1:
            samples = samples[:: self.cfg.frame_stride]
        return samples

    def _build_stereo_samples_for_sequence(self, seq_cfg: SequenceCfg):
        left_root = seq_cfg.root / self.cfg.left_camera_dirname
        right_root = seq_cfg.root / self.cfg.right_camera_dirname
        left_entries = self._load_image_entries_from_dir(left_root)
        right_entries = self._load_image_entries_from_dir(right_root)

        pose_times, Twb_list = self._load_trajectory(seq_cfg.trajectory_file)
        Tbc_left, K_left = self._load_euroc_sensor_yaml(seq_cfg.root / self.cfg.left_sensor_yaml)
        Tbc_right, K_right = self._load_euroc_sensor_yaml(seq_cfg.root / self.cfg.right_sensor_yaml)

        right_times = [x[0] for x in right_entries]
        right_paths = [x[1] for x in right_entries]

        samples = []
        for ts, left_path in left_entries:
            right_idx = self._match_timestamp_index(ts, right_times, self.cfg.pose_time_tolerance)
            if right_idx is None:
                continue
            pose_idx = self._match_timestamp_index(ts, pose_times, self.cfg.pose_time_tolerance)
            if pose_idx is None:
                continue

            Twb = Twb_list[pose_idx]
            Twc_left = Twb @ Tbc_left
            Twc_right = Twb @ Tbc_right

            samples.append(
                {
                    "timestamp": ts,
                    "left_image_path": left_path,
                    "right_image_path": right_paths[right_idx],
                    "left_extrinsics": Twc_left,
                    "right_extrinsics": Twc_right,
                    "body_extrinsics": Twb,
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

    def _load_rgb_entries(self, seq_cfg: SequenceCfg, image_root: Path, depth_root: Path):
        entries = []
        if seq_cfg.association_file is not None and seq_cfg.association_file.exists():
            with seq_cfg.association_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        rgb_ts = float(parts[0])
                        img_path = seq_cfg.root / parts[1]
                        depth_path = seq_cfg.root / parts[3]
                        entries.append((rgb_ts, img_path, depth_path))
                    elif len(parts) >= 2:
                        rgb_ts = float(parts[0])
                        img_path = seq_cfg.root / parts[1]
                        entries.append((rgb_ts, img_path, None))
        else:
            for p in sorted(image_root.glob("*.png")):
                ts = float(p.stem)
                depth_path = depth_root / f"{p.stem}.png"
                if not depth_path.exists():
                    depth_path = None
                entries.append((ts, p, depth_path))
        entries.sort(key=lambda x: x[0])
        return entries

    def _load_image_entries_from_dir(self, image_root: Path):
        entries = []
        for p in sorted(image_root.glob("*.png")):
            ts = float(p.stem) * 1e-9  # EuRoC ns -> s
            entries.append((ts, p))
        entries.sort(key=lambda x: x[0])
        return entries

    def _load_trajectory(self, traj_file: Path):
        if traj_file.suffix.lower() == ".csv":
            return self._load_trajectory_euroc_csv(traj_file)
        return self._load_trajectory_tum_txt(traj_file)

    def _load_trajectory_tum_txt(self, traj_file: Path):
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

    def _load_trajectory_euroc_csv(self, traj_file: Path):
        pose_times = []
        pose_mats = []
        with traj_file.open("r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0:
                    continue
                if row[0].startswith("#"):
                    continue
                if len(row) < 8:
                    continue
                try:
                    ts_ns = int(row[0])
                    px = float(row[1])
                    py = float(row[2])
                    pz = float(row[3])
                    qw = float(row[4])
                    qx = float(row[5])
                    qy = float(row[6])
                    qz = float(row[7])
                except ValueError:
                    continue
                Twb = self._build_Twc_from_tum(px, py, pz, qx, qy, qz, qw)
                pose_times.append(ts_ns * 1e-9)
                pose_mats.append(Twb)
        if len(pose_times) == 0:
            raise RuntimeError(f"No valid EuRoC poses found in trajectory file: {traj_file}")
        return pose_times, pose_mats

    def _load_euroc_sensor_yaml(self, yaml_path: Path):
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)
        T_BS = data["T_BS"]
        if isinstance(T_BS, dict) and "data" in T_BS:
            vals = T_BS["data"]
            rows = int(T_BS.get("rows", 4))
            cols = int(T_BS.get("cols", 4))
            Tbc = torch.tensor(vals, dtype=torch.float32).view(rows, cols)
        else:
            Tbc = torch.tensor(T_BS, dtype=torch.float32)
        intr = data["intrinsics"]
        fx, fy, cx, cy = map(float, intr[:4])
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return Tbc, K

    def _match_timestamp_index(self, ts, all_times, tol):
        idx = bisect.bisect_left(all_times, ts)
        candidates = []
        if idx < len(all_times):
            candidates.append(idx)
        if idx - 1 >= 0:
            candidates.append(idx - 1)
        best_idx = None
        best_dt = None
        for j in candidates:
            dt = abs(all_times[j] - ts)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = j
        if best_idx is None or best_dt is None or best_dt > tol:
            return None
        return best_idx

    def _match_pose(self, ts, pose_times, pose_mats, tol):
        idx = self._match_timestamp_index(ts, pose_times, tol)
        if idx is None:
            return None
        return pose_mats[idx]

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

    def _build_Twc_from_tum(self, tx, ty, tz, qx, qy, qz, qw):
        Twc = torch.eye(4, dtype=torch.float32)
        Twc[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)
        Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
        return Twc

    def _base_pixel_K_from_sample(self, sample, camera: Optional[str] = None):
        if self.cfg.stereo_as_context:
            if camera == "left":
                return sample["left_K"].clone()
            if camera == "right":
                return sample["right_K"].clone()
            raise ValueError(f"camera must be left/right in stereo mode, got {camera}")
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = sample["fx"]
        K[1, 1] = sample["fy"]
        K[0, 2] = sample["cx"]
        K[1, 2] = sample["cy"]
        return K

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

        if self.cfg.stereo_as_context:
            all_extrinsics = torch.stack([x["left_extrinsics"] for x in samples], dim=0)
            dummy_intrinsics = torch.stack([x["left_K"] for x in samples], dim=0)
            context_indices, target_indices = self.view_sampler.sample(
                scene_key, all_extrinsics, dummy_intrinsics
            )

            # ---------- context ----------
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

            # ---------- target ----------
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
                    target_camera_id.append(0)   # left

                if self.cfg.target_camera in ("right", "both"):
                    img_r, K_r = self._process_image_and_K(
                        sample["right_image_path"],
                        self._base_pixel_K_from_sample(sample, "right"),
                    )
                    target_images.append(img_r)
                    target_intrinsics.append(K_r)
                    target_extrinsics.append(sample["right_extrinsics"])
                    target_index_expanded.append(i)
                    target_camera_id.append(1)   # right

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

        all_extrinsics = torch.stack([x["extrinsics"] for x in samples], dim=0)
        dummy_intrinsics = torch.stack([self._base_pixel_K_from_sample(x) for x in samples], dim=0)
        context_indices, target_indices = self.view_sampler.sample(scene_key, all_extrinsics, dummy_intrinsics)

        context_images = []
        context_intrinsics = []
        for i in context_indices.tolist():
            img, K = self._process_image_and_K(samples[i]["image_path"], self._base_pixel_K_from_sample(samples[i]))
            context_images.append(img)
            context_intrinsics.append(K)

        target_images = []
        target_intrinsics = []
        for i in target_indices.tolist():
            img, K = self._process_image_and_K(samples[i]["image_path"], self._base_pixel_K_from_sample(samples[i]))
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
            "scene_name": scene_name,
        }
        return example
