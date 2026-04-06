from __future__ import annotations

import argparse
from pathlib import Path

import torch

from packet_fusion_utils import load_packet, merge_with_policy, packet_basic_stats, save_packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet_a", type=str, required=True)
    parser.add_argument("--packet_b", type=str, required=True)
    parser.add_argument("--policy", type=str, default="raw", choices=["raw", "opacity", "opacity_topk", "voxel", "opacity_topk_voxel"])
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    packet_a = load_packet(args.root / args.packet_a)
    packet_b = load_packet(args.root / args.packet_b)
    _, _, merged = merge_with_policy(
        packet_a,
        packet_b,
        policy=args.policy,
        opacity_thresh=args.opacity_thresh,
        topk=args.topk,
        voxel_size=args.voxel_size,
    )
    save_packet(merged, args.out)

    stats = packet_basic_stats(merged)
    print(f"Saved merged packet to: {args.out}")
    print(f"Policy        : {args.policy}")
    print(f"Num gaussians : {stats['num_gaussians']}")
    print(f"Opacity mean  : {stats['opacity_mean']:.6f}")
    print(f"Center        : {stats['center']}")
    print(f"BBox min      : {stats['bbox_min']}")
    print(f"BBox max      : {stats['bbox_max']}")


if __name__ == "__main__":
    main()
