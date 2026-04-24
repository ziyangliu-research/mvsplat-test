import argparse
import json
from pathlib import Path


def sorted_pngs(folder: Path):
    return sorted(folder.glob('*.png'))


def count_pose_lines(path: Path) -> int:
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def build_valid_indices(scene_root: Path, left_dir: str, right_dir: str, left_pose: str, right_pose: str):
    left_imgs = sorted_pngs(scene_root / left_dir)
    right_imgs = sorted_pngs(scene_root / right_dir)
    left_pose_n = count_pose_lines(scene_root / left_pose)
    right_pose_n = count_pose_lines(scene_root / right_pose)
    n = min(len(left_imgs), len(right_imgs), left_pose_n, right_pose_n)
    if n <= 1:
        raise RuntimeError(
            f"Not enough aligned frames. left_imgs={len(left_imgs)}, right_imgs={len(right_imgs)}, "
            f"left_pose={left_pose_n}, right_pose={right_pose_n}"
        )
    return list(range(n))


def build_pair_entries(indices, scene_name: str, offset: int = 1, pair_step: int = 2):
    entries = []
    for ctx in range(0, len(indices) - offset, pair_step):
        tgt = ctx + offset
        entries.append({
            "scene_key": f"{scene_name}_{len(entries):04d}",
            "context": [indices[ctx]],
            "target": [indices[tgt]],
            "meta": {
                "context_index": indices[ctx],
                "target_index": indices[tgt],
                "target_offset": offset,
            },
        })
    return entries


def slice_contiguous(entries, train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    n = len(entries)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = entries[:train_end]
    val = entries[train_end:val_end]
    test = entries[val_end:]
    return train, val, test


def slice_interval(entries, holdout_every: int, val_mod: int, test_mod: int):
    train, val, test = [], [], []
    for i, e in enumerate(entries):
        r = i % holdout_every
        if r == val_mod:
            val.append(e)
        elif r == test_mod:
            test.append(e)
        else:
            train.append(e)
    return train, val, test


def finalize(entries, stage: str):
    out = {}
    for e in entries:
        meta = dict(e["meta"])
        meta["stage"] = stage
        out[e["scene_key"]] = {
            "context": e["context"],
            "target": e["target"],
            "meta": meta,
        }
    return out


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description='Generate simple sequential TartanAir pair splits, e.g. 0->1, 2->3.')
    p.add_argument('--scene_root', type=Path, required=True)
    p.add_argument('--scene_name', type=str, required=True)
    p.add_argument('--split_mode', choices=['contiguous', 'interval'], default='contiguous')
    p.add_argument('--left_dir', type=str, default='image_lcam_front')
    p.add_argument('--right_dir', type=str, default='image_rcam_front')
    p.add_argument('--left_pose', type=str, default='pose_lcam_front.txt')
    p.add_argument('--right_pose', type=str, default='pose_rcam_front.txt')
    p.add_argument('--offset', type=int, default=1, help='Target = context + offset. Default 1.')
    p.add_argument('--pair_step', type=int, default=2, help='Context step in raw frame order. Default 2 gives 0->1,2->3,...')
    p.add_argument('--train_ratio', type=float, default=1)
    p.add_argument('--val_ratio', type=float, default=0)
    p.add_argument('--test_ratio', type=float, default=0)
    p.add_argument('--holdout_every', type=int, default=10)
    p.add_argument('--val_mod', type=int, default=8)
    p.add_argument('--test_mod', type=int, default=9)
    p.add_argument('--train_max_entries', type=int, default=-1)
    p.add_argument('--val_max_entries', type=int, default=-1)
    p.add_argument('--test_max_entries', type=int, default=-1)
    p.add_argument('--out_dir', type=Path, required=True)
    args = p.parse_args()

    indices = build_valid_indices(args.scene_root, args.left_dir, args.right_dir, args.left_pose, args.right_pose)
    entries = build_pair_entries(indices, args.scene_name, offset=args.offset, pair_step=args.pair_step)
    if len(entries) == 0:
        raise RuntimeError('No pair entries were generated.')

    if args.split_mode == 'contiguous':
        train, val, test = slice_contiguous(entries, args.train_ratio, args.val_ratio, args.test_ratio)
    else:
        train, val, test = slice_interval(entries, args.holdout_every, args.val_mod, args.test_mod)

    if args.train_max_entries > 0:
        train = train[:args.train_max_entries]
    if args.val_max_entries > 0:
        val = val[:args.val_max_entries]
    if args.test_max_entries > 0:
        test = test[:args.test_max_entries]

    train_json = finalize(train, 'train')
    val_json = finalize(val, 'val')
    test_json = finalize(test, 'test')

    write_json(args.out_dir / f'{args.scene_name}_train.json', train_json)
    write_json(args.out_dir / f'{args.scene_name}_val.json', val_json)
    write_json(args.out_dir / f'{args.scene_name}_test.json', test_json)

    print(f'raw valid frames: {len(indices)}')
    print(f'generated pair entries: {len(entries)}')
    print(f'train/val/test: {len(train_json)} / {len(val_json)} / {len(test_json)}')
    if train:
        print('first train:', train[0])
    if val:
        print('first val:', val[0])
    if test:
        print('first test:', test[0])


if __name__ == '__main__':
    main()
