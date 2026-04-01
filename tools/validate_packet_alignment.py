from pathlib import Path
import argparse
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def load_packet(path):
    packet = torch.load(path, map_location="cpu")
    required = ["means", "opacities"]
    for k in required:
        if k not in packet:
            raise KeyError(f"{path} missing key: {k}")
    return packet


def filter_packet(packet, opacity_thresh=0.05, topk=50000):
    means = packet["means"]          # [N, 3]
    opacities = packet["opacities"]  # [N]

    mask = opacities > opacity_thresh
    if mask.sum() == 0:
        return means[:0], opacities[:0]

    means = means[mask]
    opacities = opacities[mask]

    if topk is not None and means.shape[0] > topk:
        idx = torch.topk(opacities, k=topk).indices
        means = means[idx]
        opacities = opacities[idx]

    return means, opacities


def bbox_stats(points):
    if points.shape[0] == 0:
        return None
    center = points.mean(dim=0)
    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values
    extent = bbox_max - bbox_min
    return {
        "center": center,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "extent": extent,
        "num_points": points.shape[0],
    }


def compute_bbox_overlap(points0, points1):
    s0 = bbox_stats(points0)
    s1 = bbox_stats(points1)
    if s0 is None or s1 is None:
        return None

    overlap_min = torch.maximum(s0["bbox_min"], s1["bbox_min"])
    overlap_max = torch.minimum(s0["bbox_max"], s1["bbox_max"])
    overlap_extent = (overlap_max - overlap_min).clamp(min=0)
    overlap_volume = overlap_extent.prod().item()

    return {
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "overlap_extent": overlap_extent,
        "overlap_volume": overlap_volume,
    }


def crop_to_overlap(points, overlap_min, overlap_max):
    mask = ((points >= overlap_min) & (points <= overlap_max)).all(dim=1)
    return points[mask]


def random_subsample(points, max_points=2000):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0])[:max_points]
    return points[idx]


def bidirectional_nn_distance(points0, points1, max_points=2000):
    """
    用 overlap 区域里的点做一个简单的双向 1-NN 距离。
    这不是最终论文指标，但非常适合当前诊断“是否基本对齐”。
    """
    if points0.shape[0] == 0 or points1.shape[0] == 0:
        return None

    p0 = random_subsample(points0, max_points=max_points)
    p1 = random_subsample(points1, max_points=max_points)

    # [n0, n1]
    dist = torch.cdist(p0, p1)

    d0 = dist.min(dim=1).values  # p0 -> p1
    d1 = dist.min(dim=0).values  # p1 -> p0

    return {
        "p0_to_p1_mean": d0.mean().item(),
        "p0_to_p1_median": d0.median().item(),
        "p1_to_p0_mean": d1.mean().item(),
        "p1_to_p0_median": d1.median().item(),
        "sym_mean": (d0.mean() + d1.mean()).item() / 2.0,
        "sym_median": (d0.median() + d1.median()).item() / 2.0,
        "n0_eval": p0.shape[0],
        "n1_eval": p1.shape[0],
    }


def make_color(n, rgb):
    return torch.tensor(rgb, dtype=torch.uint8).unsqueeze(0).repeat(n, 1)


def export_points_ply(points, colors, out_path):
    points = points.cpu().numpy().astype(np.float32)
    colors = colors.cpu().numpy().astype(np.uint8)

    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(str(out_path))
    print(f"Saved ply to {out_path}")


def print_stats(name, stats):
    if stats is None:
        print(f"{name}: empty")
        return
    print(f"{name} center: {stats['center']}")
    print(f"{name} bbox min: {stats['bbox_min']}")
    print(f"{name} bbox max: {stats['bbox_max']}")
    print(f"{name} bbox extent: {stats['extent']}")
    print(f"{name} num points: {stats['num_points']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Directory containing saved packet .pt files")
    parser.add_argument("--packet0", type=str, required=True,
                        help="First packet filename, e.g. rgbd_bonn_static_0000.pt")
    parser.add_argument("--packet1", type=str, required=True,
                        help="Second packet filename, e.g. rgbd_bonn_static_0001.pt")
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50000)
    parser.add_argument("--nn_eval_points", type=int, default=2000)
    parser.add_argument("--export_ply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    packet0 = load_packet(root / args.packet0)
    packet1 = load_packet(root / args.packet1)

    means0, op0 = filter_packet(packet0, opacity_thresh=args.opacity_thresh, topk=args.topk)
    means1, op1 = filter_packet(packet1, opacity_thresh=args.opacity_thresh, topk=args.topk)

    stats0 = bbox_stats(means0)
    stats1 = bbox_stats(means1)
    print_stats("packet0", stats0)
    print_stats("packet1", stats1)

    if stats0 is None or stats1 is None:
        print("One packet is empty after filtering.")
        return

    center_dist = torch.norm(stats0["center"] - stats1["center"]).item()
    print(f"center distance: {center_dist}")

    overlap = compute_bbox_overlap(means0, means1)
    print(f"bbox overlap min: {overlap['overlap_min']}")
    print(f"bbox overlap max: {overlap['overlap_max']}")
    print(f"bbox overlap extent: {overlap['overlap_extent']}")
    print(f"bbox overlap volume approx: {overlap['overlap_volume']}")

    overlap0 = crop_to_overlap(means0, overlap["overlap_min"], overlap["overlap_max"])
    overlap1 = crop_to_overlap(means1, overlap["overlap_min"], overlap["overlap_max"])

    print(f"overlap packet0 points: {overlap0.shape[0]}")
    print(f"overlap packet1 points: {overlap1.shape[0]}")

    nn_stats = bidirectional_nn_distance(
        overlap0, overlap1, max_points=args.nn_eval_points
    )
    if nn_stats is None:
        print("NN stats unavailable (empty overlap points).")
    else:
        print("=== overlap bidirectional 1-NN distance ===")
        print(f"p0 -> p1 mean   : {nn_stats['p0_to_p1_mean']:.6f}")
        print(f"p0 -> p1 median : {nn_stats['p0_to_p1_median']:.6f}")
        print(f"p1 -> p0 mean   : {nn_stats['p1_to_p0_mean']:.6f}")
        print(f"p1 -> p0 median : {nn_stats['p1_to_p0_median']:.6f}")
        print(f"sym mean        : {nn_stats['sym_mean']:.6f}")
        print(f"sym median      : {nn_stats['sym_median']:.6f}")
        print(f"eval points     : {nn_stats['n0_eval']} / {nn_stats['n1_eval']}")

    if args.export_ply:
        colors0 = make_color(means0.shape[0], [255, 0, 0])
        colors1 = make_color(means1.shape[0], [0, 255, 0])

        export_points_ply(means0, colors0, root / f"{Path(args.packet0).stem}_red.ply")
        export_points_ply(means1, colors1, root / f"{Path(args.packet1).stem}_green.ply")

        means_all = torch.cat([means0, means1], dim=0)
        colors_all = torch.cat([colors0, colors1], dim=0)
        export_points_ply(
            means_all,
            colors_all,
            root / f"merged_{Path(args.packet0).stem}_{Path(args.packet1).stem}_red_green.ply"
        )

        # 只导出 overlap 区域，便于你专门看“重叠部分是不是双层”
        overlap_colors0 = make_color(overlap0.shape[0], [255, 0, 0])
        overlap_colors1 = make_color(overlap1.shape[0], [0, 255, 0])
        overlap_means = torch.cat([overlap0, overlap1], dim=0)
        overlap_colors = torch.cat([overlap_colors0, overlap_colors1], dim=0)
        export_points_ply(
            overlap_means,
            overlap_colors,
            root / f"overlap_{Path(args.packet0).stem}_{Path(args.packet1).stem}_red_green.ply"
        )


if __name__ == "__main__":
    main()
