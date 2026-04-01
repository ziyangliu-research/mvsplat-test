from pathlib import Path
import torch
import numpy as np
from plyfile import PlyData, PlyElement


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
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=True)
    ply.write(str(out_path))
    print(f"Saved ply to {out_path}")


def load_packet(path):
    return torch.load(path, map_location="cpu")


def filter_packet(packet, opacity_thresh=0.05, topk=50000):
    means = packet["means"]          # [N, 3]
    opacities = packet["opacities"]  # [N]

    mask = opacities > opacity_thresh
    if mask.sum() == 0:
        print(f"[Warning] no points survive opacity_thresh={opacity_thresh}")
        return means[:0], opacities[:0]

    means = means[mask]
    opacities = opacities[mask]

    if topk is not None and means.shape[0] > topk:
        idx = torch.topk(opacities, k=topk).indices
        means = means[idx]
        opacities = opacities[idx]

    return means, opacities


def make_color(n, rgb):
    return torch.tensor(rgb, dtype=torch.uint8).unsqueeze(0).repeat(n, 1)


def print_packet_stats(name, means):
    if means.shape[0] == 0:
        print(f"{name}: empty")
        return

    center = means.mean(dim=0)
    bbox_min = means.min(dim=0).values
    bbox_max = means.max(dim=0).values
    extent = bbox_max - bbox_min

    print(f"{name} center: {center}")
    print(f"{name} bbox min: {bbox_min}")
    print(f"{name} bbox max: {bbox_max}")
    print(f"{name} bbox extent: {extent}")
    print(f"{name} num points: {means.shape[0]}")


if __name__ == "__main__":
    # 这里改成你的实际输出目录
    root = Path("outputs/test/tum_orb/gaussian_packets")

    packet0 = load_packet(root / "rgbd_bonn_static_0000.pt")
    packet1 = load_packet(root / "rgbd_bonn_static_0001.pt")

    # 你可以调这两个超参数
    opacity_thresh = 0.05
    topk = 50000

    means0, op0 = filter_packet(packet0, opacity_thresh=opacity_thresh, topk=topk)
    means1, op1 = filter_packet(packet1, opacity_thresh=opacity_thresh, topk=topk)

    print_packet_stats("packet0", means0)
    print_packet_stats("packet1", means1)

    if means0.shape[0] > 0 and means1.shape[0] > 0:
        center_dist = torch.norm(means0.mean(dim=0) - means1.mean(dim=0)).item()
        print(f"center distance: {center_dist}")

        bbox0_min = means0.min(dim=0).values
        bbox0_max = means0.max(dim=0).values
        bbox1_min = means1.min(dim=0).values
        bbox1_max = means1.max(dim=0).values

        overlap_min = torch.maximum(bbox0_min, bbox1_min)
        overlap_max = torch.minimum(bbox0_max, bbox1_max)
        overlap_extent = (overlap_max - overlap_min).clamp(min=0)

        print(f"bbox overlap min: {overlap_min}")
        print(f"bbox overlap max: {overlap_max}")
        print(f"bbox overlap extent: {overlap_extent}")
        print(f"bbox overlap volume approx: {overlap_extent.prod().item()}")

    # 红色：packet0；绿色：packet1
    colors0 = make_color(means0.shape[0], [255, 0, 0])
    colors1 = make_color(means1.shape[0], [0, 255, 0])

    # 分别导出（可选，但推荐）
    export_points_ply(means0, colors0, root / "rgbd_bonn_static_0000_red.ply")
    export_points_ply(means1, colors1, root / "rgbd_bonn_static_0001_green.ply")

    # 合并导出（最关键）
    means_all = torch.cat([means0, means1], dim=0)
    colors_all = torch.cat([colors0, colors1], dim=0)
    export_points_ply(means_all, colors_all, root / "merged_0000_0001_red_green.ply")