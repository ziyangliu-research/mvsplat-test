from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch

# 保证可以从仓库根目录导入 render_cuda。
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.decoder.cuda_splatting import render_cuda  # noqa: E402


# =========================
# 基础 I/O
# =========================

def load_packet(path: Path, map_location: str = "cpu") -> Dict:
    packet = torch.load(path, map_location=map_location)
    required = [
        "scene",
        "context_index",
        "target_index",
        "target_extrinsics",
        "target_intrinsics",
        "target_near",
        "target_far",
        "target_image",
        "image_shape",
        "background_color",
        "means",
        "covariances",
        "harmonics",
        "opacities",
    ]
    missing = [k for k in required if k not in packet]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return packet


def save_packet(packet: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packet, path)


# =========================
# Packet / Probe 辅助
# =========================

def select_target_view(x: torch.Tensor, idx: int, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} is not a tensor")

    if name in ["target_extrinsics", "target_intrinsics"]:
        if x.ndim == 3:
            return x[idx]
        if x.ndim == 2:
            return x
        raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    if name in ["target_near", "target_far", "target_index"]:
        if x.ndim == 1:
            return x[idx]
        if x.ndim == 0:
            return x
        raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    if name == "target_image":
        if x.ndim == 4:
            return x[idx]
        if x.ndim == 3:
            return x
        raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    raise KeyError(f"Unsupported field name: {name}")


def build_probe_from_packet(packet: Dict, probe_idx: int, device: str = "cuda") -> Dict:
    extr = select_target_view(packet["target_extrinsics"], probe_idx, "target_extrinsics").unsqueeze(0).to(device)
    intr = select_target_view(packet["target_intrinsics"], probe_idx, "target_intrinsics").unsqueeze(0).to(device)
    near = select_target_view(packet["target_near"], probe_idx, "target_near").reshape(1).to(device)
    far = select_target_view(packet["target_far"], probe_idx, "target_far").reshape(1).to(device)
    gt = select_target_view(packet["target_image"], probe_idx, "target_image").to(device)
    bg = packet["background_color"]
    if bg.ndim == 1:
        bg = bg.unsqueeze(0)
    elif bg.ndim != 2 or bg.shape[0] != 1:
        raise ValueError(f"Unexpected background_color shape: {tuple(bg.shape)}")
    bg = bg.to(device)

    return {
        "extrinsics": extr,
        "intrinsics": intr,
        "near": near,
        "far": far,
        "background_color": bg,
        "image_shape": tuple(packet["image_shape"]),
        "gt": gt,
        "target_index": int(select_target_view(packet["target_index"], probe_idx, "target_index").item()),
        "scene": packet["scene"],
    }


def check_probe_consistency(packet_a: Dict, packet_b: Dict, probe_idx: int = 0, atol: float = 1e-6) -> Dict:
    idx_a = select_target_view(packet_a["target_index"], probe_idx, "target_index")
    idx_b = select_target_view(packet_b["target_index"], probe_idx, "target_index")
    extr_a = select_target_view(packet_a["target_extrinsics"], probe_idx, "target_extrinsics")
    extr_b = select_target_view(packet_b["target_extrinsics"], probe_idx, "target_extrinsics")
    intr_a = select_target_view(packet_a["target_intrinsics"], probe_idx, "target_intrinsics")
    intr_b = select_target_view(packet_b["target_intrinsics"], probe_idx, "target_intrinsics")
    near_a = select_target_view(packet_a["target_near"], probe_idx, "target_near")
    near_b = select_target_view(packet_b["target_near"], probe_idx, "target_near")
    far_a = select_target_view(packet_a["target_far"], probe_idx, "target_far")
    far_b = select_target_view(packet_b["target_far"], probe_idx, "target_far")

    extr_diff = torch.norm(extr_a - extr_b).item()
    intr_diff = torch.norm(intr_a - intr_b).item()
    near_diff = torch.abs(near_a - near_b).item()
    far_diff = torch.abs(far_a - far_b).item()
    same_shape = tuple(packet_a["image_shape"]) == tuple(packet_b["image_shape"])
    same = (
        idx_a.item() == idx_b.item()
        and extr_diff < atol
        and intr_diff < atol
        and near_diff < atol
        and far_diff < atol
        and same_shape
    )
    return {
        "same": bool(same),
        "target_index_a": int(idx_a.item()),
        "target_index_b": int(idx_b.item()),
        "extr_diff": extr_diff,
        "intr_diff": intr_diff,
        "near_diff": near_diff,
        "far_diff": far_diff,
        "same_image_shape": bool(same_shape),
    }


def packet_basic_stats(packet: Dict) -> Dict:
    means = packet["means"]
    opacities = packet["opacities"]
    return {
        "num_gaussians": int(means.shape[0]),
        "opacity_min": float(opacities.min().item()),
        "opacity_max": float(opacities.max().item()),
        "opacity_mean": float(opacities.mean().item()),
        "center": means.mean(dim=0).cpu(),
        "bbox_min": means.min(dim=0).values.cpu(),
        "bbox_max": means.max(dim=0).values.cpu(),
    }


# =========================
# 融合前处理 / 融合策略
# =========================

def _copy_packet_with_fields(packet: Dict, means: torch.Tensor, covariances: torch.Tensor, harmonics: torch.Tensor, opacities: torch.Tensor, scene_name: Optional[str] = None) -> Dict:
    out = dict(packet)
    out["means"] = means
    out["covariances"] = covariances
    out["harmonics"] = harmonics
    out["opacities"] = opacities
    if scene_name is not None:
        out["scene"] = scene_name
    return out


def filter_packet(packet: Dict, opacity_thresh: Optional[float] = None, topk: Optional[int] = None) -> Dict:
    means = packet["means"]
    covs = packet["covariances"]
    shs = packet["harmonics"]
    opacities = packet["opacities"]

    keep = torch.ones_like(opacities, dtype=torch.bool)
    if opacity_thresh is not None:
        keep = keep & (opacities > opacity_thresh)
    if keep.sum().item() == 0:
        raise ValueError(
            f"No gaussians survive filtering: opacity_thresh={opacity_thresh}, topk={topk}"
        )

    means = means[keep]
    covs = covs[keep]
    shs = shs[keep]
    opacities = opacities[keep]

    if topk is not None and means.shape[0] > topk:
        idx = torch.topk(opacities, k=topk).indices
        means = means[idx]
        covs = covs[idx]
        shs = shs[idx]
        opacities = opacities[idx]

    return _copy_packet_with_fields(packet, means, covs, shs, opacities)


def merge_raw(packet_a: Dict, packet_b: Dict, merged_scene_name: Optional[str] = None) -> Dict:
    scene_name = merged_scene_name or f"{packet_a['scene']}__MERGED__{packet_b['scene']}"
    return _copy_packet_with_fields(
        packet_a,
        torch.cat([packet_a["means"], packet_b["means"]], dim=0),
        torch.cat([packet_a["covariances"], packet_b["covariances"]], dim=0),
        torch.cat([packet_a["harmonics"], packet_b["harmonics"]], dim=0),
        torch.cat([packet_a["opacities"], packet_b["opacities"]], dim=0),
        scene_name,
    )


def _voxel_keep_max_opacity(packet: Dict, voxel_size: float) -> Dict:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    means = packet["means"]
    covs = packet["covariances"]
    shs = packet["harmonics"]
    opacities = packet["opacities"]

    coords = torch.floor(means / voxel_size).to(torch.int64)
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)
    num_groups = int(inverse.max().item()) + 1

    group_max = torch.full((num_groups,), -float("inf"), dtype=opacities.dtype, device=opacities.device)
    group_max.scatter_reduce_(0, inverse, opacities, reduce="amax", include_self=True)

    candidate_mask = opacities >= (group_max[inverse] - 1e-12)
    candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    candidate_groups = inverse[candidate_idx]

    order = torch.argsort(candidate_groups)
    candidate_idx = candidate_idx[order]
    candidate_groups = candidate_groups[order]

    first_mask = torch.ones_like(candidate_groups, dtype=torch.bool)
    if candidate_groups.numel() > 1:
        first_mask[1:] = candidate_groups[1:] != candidate_groups[:-1]
    keep_idx = candidate_idx[first_mask]

    return _copy_packet_with_fields(
        packet,
        means[keep_idx],
        covs[keep_idx],
        shs[keep_idx],
        opacities[keep_idx],
    )


def merge_with_policy(
    packet_a: Dict,
    packet_b: Dict,
    policy: str,
    opacity_thresh: Optional[float] = None,
    topk: Optional[int] = None,
    voxel_size: Optional[float] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    返回 (single_a_for_eval, single_b_for_eval, merged_packet)。
    这样每个策略都可以和“同策略下的单包”公平比较。
    """
    policy = policy.lower()

    if policy == "raw":
        a_eval = packet_a
        b_eval = packet_b
        merged = merge_raw(a_eval, b_eval)
        return a_eval, b_eval, merged

    if policy == "opacity":
        a_eval = filter_packet(packet_a, opacity_thresh=opacity_thresh, topk=None)
        b_eval = filter_packet(packet_b, opacity_thresh=opacity_thresh, topk=None)
        merged = merge_raw(a_eval, b_eval)
        return a_eval, b_eval, merged

    if policy == "opacity_topk":
        a_eval = filter_packet(packet_a, opacity_thresh=opacity_thresh, topk=topk)
        b_eval = filter_packet(packet_b, opacity_thresh=opacity_thresh, topk=topk)
        merged = merge_raw(a_eval, b_eval)
        return a_eval, b_eval, merged

    if policy == "voxel":
        a_eval = packet_a
        b_eval = packet_b
        merged = _voxel_keep_max_opacity(merge_raw(a_eval, b_eval), voxel_size=voxel_size)
        return a_eval, b_eval, merged

    if policy == "opacity_topk_voxel":
        a_eval = filter_packet(packet_a, opacity_thresh=opacity_thresh, topk=topk)
        b_eval = filter_packet(packet_b, opacity_thresh=opacity_thresh, topk=topk)
        merged = _voxel_keep_max_opacity(merge_raw(a_eval, b_eval), voxel_size=voxel_size)
        return a_eval, b_eval, merged

    raise ValueError(f"Unknown policy: {policy}")


# =========================
# 渲染 / 指标
# =========================

def render_packet_with_probe(packet: Dict, probe: Dict, device: str = "cuda") -> torch.Tensor:
    means = packet["means"].to(device)
    covs = packet["covariances"].to(device)
    shs = packet["harmonics"].to(device)
    opacities = packet["opacities"].to(device)

    image = render_cuda(
        extrinsics=probe["extrinsics"],
        intrinsics=probe["intrinsics"],
        near=probe["near"],
        far=probe["far"],
        image_shape=probe["image_shape"],
        background_color=probe["background_color"],
        gaussian_means=means.unsqueeze(0),
        gaussian_covariances=covs.unsqueeze(0),
        gaussian_sh_coefficients=shs.unsqueeze(0),
        gaussian_opacities=opacities.unsqueeze(0),
    )
    return image[0].detach().cpu()


def mse_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> Tuple[float, float]:
    pred = pred.float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    mse = torch.mean((pred - gt) ** 2).item()
    psnr = -10.0 * math.log10(max(mse, eps))
    return mse, psnr


def tensor_to_imshow(img: torch.Tensor):
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def save_four_panel(gt: torch.Tensor, img_a: torch.Tensor, img_b: torch.Tensor, img_m: torch.Tensor, save_path: Path, title: str = "") -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mse_a, psnr_a = mse_psnr(img_a, gt)
    mse_b, psnr_b = mse_psnr(img_b, gt)
    mse_m, psnr_m = mse_psnr(img_m, gt)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    images = [gt, img_a, img_b, img_m]
    titles = [
        "GT",
        f"Packet A\nPSNR={psnr_a:.2f}, MSE={mse_a:.6f}",
        f"Packet B\nPSNR={psnr_b:.2f}, MSE={mse_b:.6f}",
        f"Merged\nPSNR={psnr_m:.2f}, MSE={mse_m:.6f}",
    ]
    for ax, image, tt in zip(axes, images, titles):
        ax.imshow(tensor_to_imshow(image))
        ax.set_title(tt)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_triplet(packet_a: Dict, packet_b: Dict, merged_packet: Dict, probe: Dict, device: str = "cuda") -> Dict:
    img_a = render_packet_with_probe(packet_a, probe, device=device)
    img_b = render_packet_with_probe(packet_b, probe, device=device)
    img_m = render_packet_with_probe(merged_packet, probe, device=device)
    gt = probe["gt"].detach().cpu()

    mse_a, psnr_a = mse_psnr(img_a, gt)
    mse_b, psnr_b = mse_psnr(img_b, gt)
    mse_m, psnr_m = mse_psnr(img_m, gt)

    best_single_psnr = max(psnr_a, psnr_b)
    return {
        "gt": gt,
        "img_a": img_a,
        "img_b": img_b,
        "img_merged": img_m,
        "A_mse": mse_a,
        "A_psnr": psnr_a,
        "B_mse": mse_b,
        "B_psnr": psnr_b,
        "Merged_mse": mse_m,
        "Merged_psnr": psnr_m,
        "best_single_psnr": best_single_psnr,
        "delta_best_psnr": psnr_m - best_single_psnr,
        "delta_avg_psnr": psnr_m - 0.5 * (psnr_a + psnr_b),
        "merged_beats_best": bool(psnr_m > best_single_psnr),
    }
