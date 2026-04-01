import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

from src.model.decoder.cuda_splatting import render_cuda


# =========================
# Utilities
# =========================

def load_packet(path: Path, map_location="cpu"):
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
    for k in required:
        if k not in packet:
            raise KeyError(f"{path} missing key: {k}")
    return packet


def select_target_view(x, idx: int, name: str):
    """
    Robustly select one target view from a packet field.
    Handles both current saved format and a hypothetically squeezed format.

    Expected typical shapes:
      - target_extrinsics: [V, 4, 4]
      - target_intrinsics: [V, 3, 3]
      - target_near/far:   [V]
      - target_image:      [V, 3, H, W]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} is not a tensor")

    if name in ["target_extrinsics", "target_intrinsics"]:
        if x.ndim == 3:
            # [V, 4, 4] or [V, 3, 3]
            return x[idx]
        elif x.ndim == 2:
            # already squeezed: [4,4] or [3,3]
            return x
        else:
            raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    if name in ["target_near", "target_far", "target_index"]:
        if x.ndim == 1:
            return x[idx]
        elif x.ndim == 0:
            return x
        else:
            raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    if name == "target_image":
        if x.ndim == 4:
            # [V, 3, H, W]
            return x[idx]
        elif x.ndim == 3:
            # already squeezed: [3, H, W]
            return x
        else:
            raise ValueError(f"Unexpected shape for {name}: {tuple(x.shape)}")

    raise KeyError(f"Unsupported field name: {name}")


def packet_basic_stats(packet):
    means = packet["means"]
    opacities = packet["opacities"]

    stats = {
        "num_gaussians": int(means.shape[0]),
        "opacity_min": float(opacities.min().item()),
        "opacity_max": float(opacities.max().item()),
        "opacity_mean": float(opacities.mean().item()),
        "bbox_min": means.min(dim=0).values,
        "bbox_max": means.max(dim=0).values,
        "bbox_extent": means.max(dim=0).values - means.min(dim=0).values,
        "center": means.mean(dim=0),
    }
    return stats


def print_packet_summary(name: str, packet):
    stats = packet_basic_stats(packet)
    print(f"\n===== {name} =====")
    print(f"scene            : {packet['scene']}")
    print(f"context_index    : {packet['context_index'].tolist()}")
    print(f"target_index     : {packet['target_index'].tolist() if packet['target_index'].ndim > 0 else packet['target_index'].item()}")
    print(f"means shape      : {tuple(packet['means'].shape)}")
    print(f"covariances shape: {tuple(packet['covariances'].shape)}")
    print(f"harmonics shape  : {tuple(packet['harmonics'].shape)}")
    print(f"opacities shape  : {tuple(packet['opacities'].shape)}")
    print(f"image_shape      : {tuple(packet['image_shape'])}")
    print(f"num gaussians    : {stats['num_gaussians']}")
    print(
        f"opacity min/mean/max: "
        f"{stats['opacity_min']:.6f} / {stats['opacity_mean']:.6f} / {stats['opacity_max']:.6f}"
    )
    print(f"center           : {stats['center']}")
    print(f"bbox min         : {stats['bbox_min']}")
    print(f"bbox max         : {stats['bbox_max']}")
    print(f"bbox extent      : {stats['bbox_extent']}")


def filter_packet(packet, opacity_thresh=None, topk=None):
    """
    Return a shallow-copied packet with filtered gaussians.
    Only filters means/covariances/harmonics/opacities.
    """
    means = packet["means"]
    covs = packet["covariances"]
    shs = packet["harmonics"]
    opacities = packet["opacities"]

    keep = torch.ones_like(opacities, dtype=torch.bool)

    if opacity_thresh is not None:
        keep = keep & (opacities > opacity_thresh)

    if keep.sum() == 0:
        raise ValueError(
            f"No gaussians survive filtering: opacity_thresh={opacity_thresh}"
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

    out = dict(packet)
    out["means"] = means
    out["covariances"] = covs
    out["harmonics"] = shs
    out["opacities"] = opacities
    return out


def build_probe_from_packet(packet, probe_idx: int, device: str):
    """
    Build a single probe camera from packet[target_*][probe_idx].
    Output shapes match render_cuda expectations:
      extrinsics: [1, 4, 4]
      intrinsics: [1, 3, 3]
      near/far:   [1]
      bg:         [1, 3]
    """
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

    image_shape = tuple(packet["image_shape"])

    return {
        "extrinsics": extr,
        "intrinsics": intr,
        "near": near,
        "far": far,
        "background_color": bg,
        "image_shape": image_shape,
        "gt": gt,
        "target_index": select_target_view(packet["target_index"], probe_idx, "target_index").item(),
        "scene": packet["scene"],
    }


def compare_probe_meta(packet_a, packet_b, probe_idx=0, atol=1e-6):
    """
    Compare whether A and B store the same probe target camera / index.
    For your current setup (both target=32), this should ideally be the same.
    """
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

    img_shape_same = tuple(packet_a["image_shape"]) == tuple(packet_b["image_shape"])

    print("\n===== Probe consistency check (A vs B) =====")
    print(f"target index A / B : {idx_a.item()} / {idx_b.item()}")
    print(f"same target index  : {bool(idx_a.item() == idx_b.item())}")

    extr_diff = torch.norm(extr_a - extr_b).item()
    intr_diff = torch.norm(intr_a - intr_b).item()
    near_diff = torch.abs(near_a - near_b).item()
    far_diff = torch.abs(far_a - far_b).item()

    print(f"||extr_A - extr_B|| : {extr_diff:.8f}")
    print(f"||intr_A - intr_B|| : {intr_diff:.8f}")
    print(f"|near_A - near_B|   : {near_diff:.8f}")
    print(f"|far_A - far_B|     : {far_diff:.8f}")
    print(f"same image_shape    : {img_shape_same}")

    same = (
        idx_a.item() == idx_b.item()
        and extr_diff < atol
        and intr_diff < atol
        and near_diff < atol
        and far_diff < atol
        and img_shape_same
    )
    print(f"probe fully consistent (atol={atol}) : {same}")
    return same


def merge_packets(packet_a, packet_b):
    merged = dict(packet_a)  # meta is copied from A, but render will use explicit probe
    merged["means"] = torch.cat([packet_a["means"], packet_b["means"]], dim=0)
    merged["covariances"] = torch.cat([packet_a["covariances"], packet_b["covariances"]], dim=0)
    merged["harmonics"] = torch.cat([packet_a["harmonics"], packet_b["harmonics"]], dim=0)
    merged["opacities"] = torch.cat([packet_a["opacities"], packet_b["opacities"]], dim=0)
    merged["scene"] = f"{packet_a['scene']}__MERGED__{packet_b['scene']}"
    return merged


def render_packet_with_probe(packet, probe, device="cuda"):
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


def mse_psnr(pred, gt, eps=1e-12):
    """
    pred, gt: [3, H, W], range assumed roughly [0,1]
    """
    pred = pred.float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    mse = torch.mean((pred - gt) ** 2).item()
    psnr = -10.0 * torch.log10(torch.tensor(max(mse, eps))).item()
    return mse, psnr


def tensor_to_imshow(img):
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def visualize(gt, img_a, img_b, img_merged, save_path, title_suffix=""):
    mse_a, psnr_a = mse_psnr(img_a, gt)
    mse_b, psnr_b = mse_psnr(img_b, gt)
    mse_m, psnr_m = mse_psnr(img_merged, gt)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    imgs = [gt, img_a, img_b, img_merged]
    titles = [
        "GT @ probe",
        f"Packet A @ probe\nPSNR={psnr_a:.2f}  MSE={mse_a:.6f}",
        f"Packet B @ probe\nPSNR={psnr_b:.2f}  MSE={mse_b:.6f}",
        f"Merged (A+B) @ probe\nPSNR={psnr_m:.2f}  MSE={mse_m:.6f}",
    ]

    for ax, im, tt in zip(axes, imgs, titles):
        ax.imshow(tensor_to_imshow(im))
        ax.set_title(tt)
        ax.axis("off")

    plt.suptitle(title_suffix)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved visualization] {save_path}")

    return {
        "A": {"mse": mse_a, "psnr": psnr_a},
        "B": {"mse": mse_b, "psnr": psnr_b},
        "Merged": {"mse": mse_m, "psnr": psnr_m},
    }


def maybe_save_merged_packet(merged_packet, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_packet, save_path)
    print(f"[Saved merged packet] {save_path}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="outputs/test/tum_orb/gaussian_packets",
        help="Directory containing packet .pt files",
    )
    parser.add_argument(
        "--packet_a",
        type=str,
        default="rgbd_bonn_static_0000.pt",
        help="Filename of packet A",
    )
    parser.add_argument(
        "--packet_b",
        type=str,
        default="rgbd_bonn_static_0001.pt",
        help="Filename of packet B",
    )
    parser.add_argument(
        "--probe_packet",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Which packet provides the probe camera + GT",
    )
    parser.add_argument(
        "--probe_idx",
        type=int,
        default=0,
        help="Which target view inside packet to use as probe",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--opacity_thresh",
        type=float,
        default=None,
        help="Optional opacity threshold before rendering",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Optional top-k gaussians by opacity after thresholding",
    )
    parser.add_argument(
        "--save_merged_pt",
        action="store_true",
        help="Also save merged packet as .pt",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="merged_render_comparison.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    root = Path(args.root)
    path_a = root / args.packet_a
    path_b = root / args.packet_b

    print("Loading packets...")
    packet_a = load_packet(path_a, map_location="cpu")
    packet_b = load_packet(path_b, map_location="cpu")

    print_packet_summary("Packet A (raw)", packet_a)
    print_packet_summary("Packet B (raw)", packet_b)

    compare_probe_meta(packet_a, packet_b, probe_idx=args.probe_idx)

    if args.opacity_thresh is not None or args.topk is not None:
        print("\nApplying optional filtering before rendering...")
        packet_a = filter_packet(
            packet_a, opacity_thresh=args.opacity_thresh, topk=args.topk
        )
        packet_b = filter_packet(
            packet_b, opacity_thresh=args.opacity_thresh, topk=args.topk
        )
        print_packet_summary("Packet A (filtered)", packet_a)
        print_packet_summary("Packet B (filtered)", packet_b)

    probe_source = packet_a if args.probe_packet == "a" else packet_b
    probe = build_probe_from_packet(probe_source, probe_idx=args.probe_idx, device=args.device)

    print("\n===== Probe used for all renders =====")
    print(f"probe packet scene : {probe['scene']}")
    print(f"probe target index : {probe['target_index']}")
    print(f"probe image_shape  : {probe['image_shape']}")
    print(f"probe near / far   : {probe['near'].item():.6f} / {probe['far'].item():.6f}")
    print(f"device             : {args.device}")

    print("\nRendering A @ probe ...")
    img_a = render_packet_with_probe(packet_a, probe, device=args.device)

    print("Rendering B @ probe ...")
    img_b = render_packet_with_probe(packet_b, probe, device=args.device)

    print("Merging packets ...")
    merged_packet = merge_packets(packet_a, packet_b)
    print_packet_summary("Merged packet", merged_packet)

    print("Rendering Merged(A+B) @ probe ...")
    img_merged = render_packet_with_probe(merged_packet, probe, device=args.device)

    gt = probe["gt"].detach().cpu()

    title_suffix = (
        f"Probe target index = {probe['target_index']} | "
        f"probe packet = {args.probe_packet.upper()} | "
        f"opacity_thresh = {args.opacity_thresh} | topk = {args.topk}"
    )
    out_png = root / args.out_png
    metrics = visualize(gt, img_a, img_b, img_merged, out_png, title_suffix=title_suffix)

    print("\n===== Image metrics against GT @ probe =====")
    print(f"A      : PSNR={metrics['A']['psnr']:.4f}, MSE={metrics['A']['mse']:.8f}")
    print(f"B      : PSNR={metrics['B']['psnr']:.4f}, MSE={metrics['B']['mse']:.8f}")
    print(f"Merged : PSNR={metrics['Merged']['psnr']:.4f}, MSE={metrics['Merged']['mse']:.8f}")

    if args.save_merged_pt:
        merged_name = f"merged__{Path(args.packet_a).stem}__{Path(args.packet_b).stem}.pt"
        maybe_save_merged_packet(merged_packet, root / merged_name)

    print("\nDone.")


if __name__ == "__main__":
    main()