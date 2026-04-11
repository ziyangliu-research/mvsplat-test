import argparse
import bisect
import csv
import json
from math import floor
from pathlib import Path


def load_image_entries(image_root: Path):
    entries = []
    for p in sorted(image_root.glob("*.png")):
        ts = float(p.stem) * 1e-9  # EuRoC ns -> s
        entries.append((ts, p))
    return entries


def load_euroc_pose_times(traj_file: Path):
    pose_times = []
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
            except ValueError:
                continue
            pose_times.append(ts_ns * 1e-9)
    if len(pose_times) == 0:
        raise RuntimeError(f"No valid EuRoC poses found in trajectory file: {traj_file}")
    return pose_times


def match_timestamp_index(ts, all_times, tol):
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


def build_valid_time_steps(root: Path, trajectory_file: Path, left_camera_dirname: str, right_camera_dirname: str, pose_time_tolerance: float, frame_stride: int):
    left_entries = load_image_entries(root / left_camera_dirname)
    right_entries = load_image_entries(root / right_camera_dirname)
    right_times = [x[0] for x in right_entries]
    pose_times = load_euroc_pose_times(trajectory_file)

    valid = []
    for ts, _ in left_entries:
        right_idx = match_timestamp_index(ts, right_times, pose_time_tolerance)
        if right_idx is None:
            continue
        pose_idx = match_timestamp_index(ts, pose_times, pose_time_tolerance)
        if pose_idx is None:
            continue
        valid.append(ts)

    if frame_stride > 1:
        valid = valid[::frame_stride]
    return valid


def evenly_pick_targets(start: int, end: int, num_targets: int):
    if end < start:
        return []
    if num_targets == 1:
        return [(start + end) // 2]
    step = (end - start) / (num_targets - 1)
    out = []
    for i in range(num_targets):
        x = int(round(start + i * step))
        if x not in out:
            out.append(x)
    return out


def build_split_json(scene_name: str, start_idx: int, end_idx: int, offsets: list[int], items_per_offset: int):
    margin = max(offsets)
    valid_start = start_idx
    valid_end = end_idx - margin
    targets = evenly_pick_targets(valid_start, valid_end, items_per_offset)

    out = {}
    counter = 0
    for t in targets:
        for off in offsets:
            tg = t + off
            if tg > end_idx:
                continue
            key = f"{scene_name}_{counter:04d}"
            out[key] = {
                "context": [t],
                "target": [tg],
                "meta": {
                    "context_time_index": t,
                    "target_time_index": tg,
                    "target_offset": off,
                },
            }
            counter += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trajectory_file", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--left_camera_dirname", type=str, default="cam0/data")
    parser.add_argument("--right_camera_dirname", type=str, default="cam1/data")
    parser.add_argument("--pose_time_tolerance", type=float, default=0.01)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--train_items_per_offset", type=int, default=200)
    parser.add_argument("--val_items_per_offset", type=int, default=20)
    parser.add_argument("--test_items_per_offset", type=int, default=20)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    valid_steps = build_valid_time_steps(
        root=args.root,
        trajectory_file=args.trajectory_file,
        left_camera_dirname=args.left_camera_dirname,
        right_camera_dirname=args.right_camera_dirname,
        pose_time_tolerance=args.pose_time_tolerance,
        frame_stride=args.frame_stride,
    )
    n = len(valid_steps)
    if n == 0:
        raise RuntimeError("No valid stereo+pose matched time steps found.")

    total = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total
    val_ratio = args.val_ratio / total

    train_end = int(floor(n * train_ratio)) - 1
    val_end = int(floor(n * (train_ratio + val_ratio))) - 1
    train_end = max(0, min(train_end, n - 1))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_json = build_split_json(args.scene_name, 0, train_end, args.offsets, args.train_items_per_offset)
    val_json = build_split_json(args.scene_name, train_end + 1, val_end, args.offsets, args.val_items_per_offset)
    test_json = build_split_json(args.scene_name, val_end + 1, n - 1, args.offsets, args.test_items_per_offset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.scene_name}_stereo_train.json"
    val_path = args.out_dir / f"{args.scene_name}_stereo_val.json"
    test_path = args.out_dir / f"{args.scene_name}_stereo_test.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_json, f, indent=2, ensure_ascii=False)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_json, f, indent=2, ensure_ascii=False)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2, ensure_ascii=False)

    print(f"Valid matched time steps: {n}")
    print(f"Saved train json: {train_path} ({len(train_json)} items)")
    print(f"Saved val json  : {val_path} ({len(val_json)} items)")
    print(f"Saved test json : {test_path} ({len(test_json)} items)")
    print(f"offsets={args.offsets}, train_end={train_end}, val_end={val_end}")


if __name__ == "__main__":
    main()
