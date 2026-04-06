from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# 帧读取
# =========================

def load_frame_entries(root: Path, association_file: Optional[Path], image_dirname: str) -> List[Tuple[float, str]]:
    entries = []
    if association_file is not None and association_file.exists():
        with association_file.open("r", encoding="utf-8") as f:
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
            entries.append((float(p.stem), str(p.relative_to(root))))
    entries.sort(key=lambda x: x[0])
    return entries


# =========================
# Pair family 定义
# =========================

def default_pair_families() -> List[Dict]:
    """
    v1 先固定四类：
    - near:      左右都靠近 probe
    - medium:    左右都中等距离
    - far:       左右都较远
    - asym_lr:   左近右远，测试不对称互补
    所有 packet 都保持两视图。
    """
    return [
        {"name": "near", "left_inner": 2, "left_outer": 7, "right_inner": 2, "right_outer": 7},
        {"name": "medium", "left_inner": 4, "left_outer": 10, "right_inner": 4, "right_outer": 10},
        {"name": "far", "left_inner": 6, "left_outer": 16, "right_inner": 6, "right_outer": 16},
        {"name": "asym_lr", "left_inner": 2, "left_outer": 7, "right_inner": 4, "right_outer": 12},
    ]


def max_required_margin(families: List[Dict]) -> int:
    return max(max(f["left_outer"], f["right_outer"]) for f in families)


# =========================
# target 选择
# =========================

def pick_targets(n_frames: int, margin: int, num_targets: int) -> List[int]:
    left = margin
    right = n_frames - margin - 1
    if right <= left:
        raise RuntimeError(f"Not enough frames for margin={margin}. n_frames={n_frames}")

    if num_targets == 1:
        return [(left + right) // 2]

    step = (right - left) / (num_targets - 1)
    targets = [int(round(left + i * step)) for i in range(num_targets)]
    dedup = []
    for t in targets:
        if t not in dedup:
            dedup.append(t)
    return dedup


# =========================
# Packet / Pair 构造
# =========================

def build_packet_contexts(target_idx: int, family: Dict) -> Tuple[List[int], List[int]]:
    packet_a = [target_idx - family["left_outer"], target_idx - family["left_inner"]]
    packet_b = [target_idx + family["right_inner"], target_idx + family["right_outer"]]
    return packet_a, packet_b


def build_jsons(scene_name: str, n_frames: int, targets: List[int], families: List[Dict]) -> Tuple[Dict, Dict]:
    packet_eval_dict = {}
    pair_benchmark_dict = {}
    packet_key_map = {}
    packet_counter = 0
    pair_counter = 0

    def register_packet(context: List[int], target: List[int]) -> str:
        nonlocal packet_counter
        key_tuple = (tuple(context), tuple(target))
        if key_tuple not in packet_key_map:
            packet_key = f"{scene_name}_pkt_{packet_counter:04d}"
            packet_key_map[key_tuple] = packet_key
            packet_eval_dict[packet_key] = {
                "context": list(context),
                "target": list(target),
            }
            packet_counter += 1
        return packet_key_map[key_tuple]

    for t in targets:
        for family in families:
            packet_a_ctx, packet_b_ctx = build_packet_contexts(t, family)

            if min(packet_a_ctx) < 0 or max(packet_b_ctx) >= n_frames:
                continue
            if not (packet_a_ctx[0] < packet_a_ctx[1] < t < packet_b_ctx[0] < packet_b_ctx[1]):
                continue

            packet_a_key = register_packet(packet_a_ctx, [t])
            packet_b_key = register_packet(packet_b_ctx, [t])

            pair_key = f"{scene_name}_pair_{pair_counter:04d}"
            pair_benchmark_dict[pair_key] = {
                "scene": scene_name,
                "packet_a_key": packet_a_key,
                "packet_b_key": packet_b_key,
                "target": [t],
                "packet_a_context": packet_a_ctx,
                "packet_b_context": packet_b_ctx,
                "family": family["name"],
                "left_inner_gap": family["left_inner"],
                "left_outer_gap": family["left_outer"],
                "right_inner_gap": family["right_inner"],
                "right_outer_gap": family["right_outer"],
                "pair_span": packet_b_ctx[-1] - packet_a_ctx[0],
            }
            pair_counter += 1

    return packet_eval_dict, pair_benchmark_dict


# =========================
# 主函数
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--association_file", type=Path, default=None)
    parser.add_argument("--image_dirname", type=str, default="image")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--num_targets", type=int, default=5)
    parser.add_argument("--packet_output", type=Path, required=True, help="输出给 evaluation sampler 使用的 packet-level JSON")
    parser.add_argument("--pair_output", type=Path, required=True, help="输出给离线 fusion benchmark 使用的 pair benchmark JSON")
    args = parser.parse_args()

    entries = load_frame_entries(args.root, args.association_file, args.image_dirname)
    if args.frame_stride > 1:
        entries = entries[:: args.frame_stride]
    n_frames = len(entries)
    if n_frames == 0:
        raise RuntimeError("No frames found.")

    families = default_pair_families()
    margin = max_required_margin(families)
    targets = pick_targets(n_frames=n_frames, margin=margin, num_targets=args.num_targets)

    packet_eval_dict, pair_benchmark_dict = build_jsons(
        scene_name=args.scene_name,
        n_frames=n_frames,
        targets=targets,
        families=families,
    )

    args.packet_output.parent.mkdir(parents=True, exist_ok=True)
    args.pair_output.parent.mkdir(parents=True, exist_ok=True)

    with args.packet_output.open("w", encoding="utf-8") as f:
        json.dump(packet_eval_dict, f, indent=2, ensure_ascii=False)
    with args.pair_output.open("w", encoding="utf-8") as f:
        json.dump(pair_benchmark_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved packet-eval JSON to: {args.packet_output}")
    print(f"  num packet items = {len(packet_eval_dict)}")
    print(f"Saved pair-benchmark JSON to: {args.pair_output}")
    print(f"  num pair items   = {len(pair_benchmark_dict)}")
    print(f"Chosen targets     = {targets}")
    print(f"Pair families      = {[f['name'] for f in families]}")


if __name__ == "__main__":
    main()
