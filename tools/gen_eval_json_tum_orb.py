'''
    扫描 associations.txt 或 image/*.png
    自动得到有效 frame 数
    生成一个 15 组的小诊断 JSON
    用于：

    small gap
    medium gap
    large gap
    的 checkpoint selection
'''
import argparse
import json
from pathlib import Path


def load_frame_entries(root: Path, association_file: Path | None, image_dirname: str):
    entries = []

    if association_file is not None and association_file.exists():
        with association_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rgb_ts = float(parts[0])
                    entries.append((rgb_ts, parts[1]))
    else:
        image_root = root / image_dirname
        for p in sorted(image_root.glob("*.png")):
            ts = float(p.stem)
            entries.append((ts, str(p.relative_to(root))))

    entries.sort(key=lambda x: x[0])
    return entries


def pick_targets(n_frames: int, max_gap: int, num_targets: int):
    """
    在 [max_gap, n_frames-max_gap-1] 范围里均匀挑 target index
    """
    left = max_gap
    right = n_frames - max_gap - 1

    if right <= left:
        raise RuntimeError(
            f"Not enough frames for max_gap={max_gap}. n_frames={n_frames}"
        )

    if num_targets == 1:
        return [(left + right) // 2]

    step = (right - left) / (num_targets - 1)
    targets = [int(round(left + i * step)) for i in range(num_targets)]
    # 去重并保持顺序
    dedup = []
    for t in targets:
        if t not in dedup:
            dedup.append(t)
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--association_file", type=Path, default=None)
    parser.add_argument("--image_dirname", type=str, default="image")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--gaps", type=int, nargs="+", default=[5, 15, 30])
    parser.add_argument("--num_targets", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = load_frame_entries(args.root, args.association_file, args.image_dirname)
    if args.frame_stride > 1:
        entries = entries[:: args.frame_stride]

    n_frames = len(entries)
    if n_frames == 0:
        raise RuntimeError("No frames found.")

    max_gap = max(args.gaps)
    targets = pick_targets(n_frames, max_gap=max_gap, num_targets=args.num_targets)

    eval_dict = {}
    counter = 0

    for t in targets:
        for g in args.gaps:
            c0 = t - g
            c1 = t + g
            if c0 < 0 or c1 >= n_frames:
                continue

            key = f"{args.scene_name}_{counter:04d}"
            eval_dict[key] = {
                "context": [c0, c1],
                "target": [t],
            }
            counter += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(eval_dict)} eval items to {args.output}")


if __name__ == "__main__":
    main()
