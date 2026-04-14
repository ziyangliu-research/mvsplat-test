import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from jaxtyping import install_import_hook

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
    from src.misc.image_io import save_image
    from src.model.decoder.cuda_splatting import render_cuda
    from src.model.types import Gaussians


REQUIRED_PACKET_FIELDS = {
    "scene",
    "means",
    "covariances",
    "harmonics",
    "opacities",
    "target_extrinsics",
    "target_intrinsics",
    "target_near",
    "target_far",
    "target_image",
    "image_shape",
    "background_color",
    "context_index",
    "target_index",
}


@dataclass
class LoadedPacket:
    path: Path
    data: dict[str, Any]
    context_sort_index: int
    scene_key: str
    base_scene: str


@dataclass
class RenderJob:
    packet: LoadedPacket
    target_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline prefix fusion baseline with fixed-target evaluation, metrics, and curve plots."
    )
    parser.add_argument("--packet_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_packets", type=int, required=True)
    parser.add_argument(
        "--render_targets",
        choices=["last_only", "all_prefix_targets", "fixed_packet"],
        default="fixed_packet",
    )
    parser.add_argument(
        "--fixed_packet_index",
        type=int,
        default=-1,
        help="0-based index after packet sorting. Negative values follow Python indexing. Used when render_targets=fixed_packet.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Also compute LPIPS. This is slower than PSNR/SSIM.",
    )
    return parser.parse_args()


def canonicalize_scene_name(scene: str) -> str:
    parts = scene.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scene


def resolve_device(device_str: str) -> torch.device:
    try:
        device = torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device '{device_str}': {exc}") from exc

    if device.type != "cuda":
        raise RuntimeError(
            "This script uses diff_gaussian_rasterization via render_cuda(), which is CUDA-only. "
            f"Requested device='{device_str}'. Please use a CUDA device such as 'cuda:0'."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested a CUDA device, but torch.cuda.is_available() is False."
        )

    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device '{device_str}', but only {torch.cuda.device_count()} CUDA "
            "device(s) are available."
        )

    return device


def as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def packet_id(packet: LoadedPacket) -> str:
    return packet.path.name


def save_png_tensor(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape [C, H, W], got {tuple(image.shape)}"
        )
    save_image(image.detach().cpu().float(), path)


def validate_packet(packet: dict[str, Any], path: Path) -> None:
    missing = sorted(REQUIRED_PACKET_FIELDS - set(packet.keys()))
    if missing:
        raise KeyError(f"Packet '{path}' is missing required fields: {missing}")

    if not isinstance(packet["scene"], str):
        raise TypeError(f"Packet '{path}' field 'scene' must be a string.")

    for field in ("means", "covariances", "harmonics", "opacities"):
        if not isinstance(packet[field], torch.Tensor):
            raise TypeError(f"Packet '{path}' field '{field}' must be a torch.Tensor.")

    means = packet["means"]
    covariances = packet["covariances"]
    harmonics = packet["harmonics"]
    opacities = packet["opacities"]

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Packet '{path}' has invalid means shape {tuple(means.shape)}.")
    if covariances.ndim != 3 or covariances.shape[-2:] != (3, 3):
        raise ValueError(
            f"Packet '{path}' has invalid covariances shape {tuple(covariances.shape)}."
        )
    if harmonics.ndim != 3 or harmonics.shape[1] != 3:
        raise ValueError(
            f"Packet '{path}' has invalid harmonics shape {tuple(harmonics.shape)}."
        )
    if opacities.ndim != 1:
        raise ValueError(
            f"Packet '{path}' has invalid opacities shape {tuple(opacities.shape)}."
        )

    num_gaussians = means.shape[0]
    if covariances.shape[0] != num_gaussians:
        raise ValueError(f"Packet '{path}' gaussian field lengths do not match.")
    if harmonics.shape[0] != num_gaussians:
        raise ValueError(f"Packet '{path}' gaussian field lengths do not match.")
    if opacities.shape[0] != num_gaussians:
        raise ValueError(f"Packet '{path}' gaussian field lengths do not match.")

    context_index = packet["context_index"]
    target_index = packet["target_index"]
    if not isinstance(context_index, torch.Tensor) or context_index.numel() == 0:
        raise ValueError(
            f"Packet '{path}' has invalid context_index; expected a non-empty tensor."
        )
    if not isinstance(target_index, torch.Tensor) or target_index.numel() == 0:
        raise ValueError(
            f"Packet '{path}' has invalid target_index; expected a non-empty tensor."
        )

    target_extrinsics = packet["target_extrinsics"]
    target_intrinsics = packet["target_intrinsics"]
    target_near = packet["target_near"]
    target_far = packet["target_far"]
    target_image = packet["target_image"]

    if target_extrinsics.ndim != 3 or target_extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            f"Packet '{path}' has invalid target_extrinsics shape {tuple(target_extrinsics.shape)}."
        )
    if target_intrinsics.ndim != 3 or target_intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Packet '{path}' has invalid target_intrinsics shape {tuple(target_intrinsics.shape)}."
        )
    if target_image.ndim != 4 or target_image.shape[1] != 3:
        raise ValueError(
            f"Packet '{path}' has invalid target_image shape {tuple(target_image.shape)}."
        )
    if target_near.ndim != 1 or target_far.ndim != 1:
        raise ValueError(
            f"Packet '{path}' target_near/target_far must be 1D tensors; got "
            f"{tuple(target_near.shape)} and {tuple(target_far.shape)}."
        )

    num_target_views = target_extrinsics.shape[0]
    if target_intrinsics.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_image.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_near.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_far.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")
    if target_index.shape[0] != num_target_views:
        raise ValueError(f"Packet '{path}' target view counts do not match.")

    image_shape = packet["image_shape"]
    if len(image_shape) != 2:
        raise ValueError(
            f"Packet '{path}' has invalid image_shape={image_shape}; expected (H, W)."
        )

    background_color = packet["background_color"]
    if not isinstance(background_color, torch.Tensor):
        raise TypeError(
            f"Packet '{path}' field 'background_color' must be a torch.Tensor."
        )
    if background_color.numel() != 3:
        raise ValueError(
            f"Packet '{path}' has invalid background_color shape {tuple(background_color.shape)}."
        )


def load_packets(packet_dir: Path) -> list[LoadedPacket]:
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet_dir does not exist: {packet_dir}")
    if not packet_dir.is_dir():
        raise NotADirectoryError(f"packet_dir is not a directory: {packet_dir}")

    packet_paths = sorted(packet_dir.glob("*.pt"))
    if not packet_paths:
        raise FileNotFoundError(f"No .pt packets found under: {packet_dir}")

    loaded_packets: list[LoadedPacket] = []
    for path in packet_paths:
        packet = torch.load(path, map_location="cpu")
        if not isinstance(packet, dict):
            raise TypeError(f"Packet '{path}' must contain a dict, got {type(packet)}.")
        validate_packet(packet, path)
        scene_key = packet["scene"]
        loaded_packets.append(
            LoadedPacket(
                path=path,
                data=packet,
                context_sort_index=int(packet["context_index"][0].item()),
                scene_key=scene_key,
                base_scene=canonicalize_scene_name(scene_key),
            )
        )

    base_scene_names = sorted({packet.base_scene for packet in loaded_packets})
    if len(base_scene_names) != 1:
        raise RuntimeError(
            "The current baseline only supports fusing packets from one base sequence at a time. "
            f"Found multiple base sequences in packet_dir: {base_scene_names}"
        )

    loaded_packets.sort(key=lambda item: item.context_sort_index)
    return loaded_packets


def validate_packet_collection(packets: list[LoadedPacket]) -> tuple[int, int]:
    image_shapes = {
        tuple(int(x) for x in packet.data["image_shape"]) for packet in packets
    }
    if len(image_shapes) != 1:
        raise ValueError(
            "All packets must share the same image_shape, got "
            f"{sorted(image_shapes)}."
        )

    harmonic_dims = {int(packet.data["harmonics"].shape[-1]) for packet in packets}
    if len(harmonic_dims) != 1:
        raise ValueError(
            "All packets must share the same SH dimensionality, got "
            f"{sorted(harmonic_dims)}."
        )

    return next(iter(image_shapes))


def concat_fused_gaussians(packets: list[LoadedPacket], device: torch.device) -> Gaussians:
    means = torch.cat([packet.data["means"] for packet in packets], dim=0).to(device)
    covariances = torch.cat([packet.data["covariances"] for packet in packets], dim=0).to(device)
    harmonics = torch.cat([packet.data["harmonics"] for packet in packets], dim=0).to(device)
    opacities = torch.cat([packet.data["opacities"] for packet in packets], dim=0).to(device)

    return Gaussians(
        means=means.unsqueeze(0).contiguous(),
        covariances=covariances.unsqueeze(0).contiguous(),
        harmonics=harmonics.unsqueeze(0).contiguous(),
        opacities=opacities.unsqueeze(0).contiguous(),
    )


def resolve_fixed_packet(packets: list[LoadedPacket], fixed_packet_index: int) -> LoadedPacket:
    num_packets = len(packets)
    resolved_index = fixed_packet_index
    if resolved_index < 0:
        resolved_index = num_packets + resolved_index
    if resolved_index < 0 or resolved_index >= num_packets:
        raise IndexError(
            f"fixed_packet_index={fixed_packet_index} is out of range for {num_packets} packets."
        )
    return packets[resolved_index]


def build_render_jobs(prefix_packets: list[LoadedPacket], render_targets: str, fixed_packet: LoadedPacket | None) -> list[RenderJob]:
    if render_targets == "last_only":
        packet = prefix_packets[-1]
        return [RenderJob(packet=packet, target_idx=i) for i in range(packet.data["target_image"].shape[0])]

    if render_targets == "all_prefix_targets":
        jobs: list[RenderJob] = []
        for packet in prefix_packets:
            jobs.extend(RenderJob(packet=packet, target_idx=i) for i in range(packet.data["target_image"].shape[0]))
        return jobs

    if render_targets == "fixed_packet":
        if fixed_packet is None:
            raise RuntimeError("fixed_packet must be provided when render_targets='fixed_packet'.")
        return [RenderJob(packet=fixed_packet, target_idx=i) for i in range(fixed_packet.data["target_image"].shape[0])]

    raise ValueError(f"Unsupported render_targets mode: {render_targets}")


@torch.no_grad()
def render_fused_views(fused: Gaussians, jobs: list[RenderJob], image_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    extrinsics = torch.stack([job.packet.data["target_extrinsics"][job.target_idx] for job in jobs], dim=0).to(device=device, dtype=torch.float32)
    intrinsics = torch.stack([job.packet.data["target_intrinsics"][job.target_idx] for job in jobs], dim=0).to(device=device, dtype=torch.float32)
    near = torch.stack([job.packet.data["target_near"][job.target_idx] for job in jobs], dim=0).to(device=device, dtype=torch.float32)
    far = torch.stack([job.packet.data["target_far"][job.target_idx] for job in jobs], dim=0).to(device=device, dtype=torch.float32)
    background = torch.stack([job.packet.data["background_color"].reshape(-1) for job in jobs], dim=0).to(device=device, dtype=torch.float32)

    num_views = len(jobs)
    return render_cuda(
        extrinsics=extrinsics.contiguous(),
        intrinsics=intrinsics.contiguous(),
        near=near.contiguous(),
        far=far.contiguous(),
        image_shape=image_shape,
        background_color=background.contiguous(),
        gaussian_means=fused.means.expand(num_views, -1, -1).contiguous(),
        gaussian_covariances=fused.covariances.expand(num_views, -1, -1, -1).contiguous(),
        gaussian_sh_coefficients=fused.harmonics.expand(num_views, -1, -1, -1).contiguous(),
        gaussian_opacities=fused.opacities.expand(num_views, -1).contiguous(),
    )


@torch.no_grad()
def compute_metrics_for_batch(rendered: torch.Tensor, gt_images: torch.Tensor, compute_lpips_flag: bool) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {
        "psnr": compute_psnr(gt_images, rendered),
        "ssim": compute_ssim(gt_images, rendered),
    }
    if compute_lpips_flag:
        metrics["lpips"] = compute_lpips(gt_images, rendered)
    return metrics


def metric_mean_as_float(metric_values: torch.Tensor) -> float:
    return float(metric_values.detach().cpu().mean().item())


def plot_curve(x_values: list[int], y_values: list[float], title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Number of fused packets")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.max_packets <= 0:
        raise ValueError(f"--max_packets must be > 0, got {args.max_packets}")

    device = resolve_device(args.device)
    packets = load_packets(args.packet_dir)
    image_shape = validate_packet_collection(packets)

    max_packets = min(args.max_packets, len(packets))
    if max_packets < args.max_packets:
        print(f"Requested max_packets={args.max_packets}, but only found {len(packets)} packet(s). Processing {max_packets} prefix step(s).")

    fixed_packet = None
    if args.render_targets == "fixed_packet":
        fixed_packet = resolve_fixed_packet(packets, args.fixed_packet_index)
        print(f"Using fixed target packet: sorted_index={packets.index(fixed_packet)}, file={fixed_packet.path.name}, scene_key={fixed_packet.scene_key}")

    images_dir = args.output_dir / "images"
    gt_dir = args.output_dir / "gt"
    plots_dir = args.output_dir / "plots"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_steps: list[dict[str, Any]] = []
    curve_packet_counts: list[int] = []
    curve_num_gaussians: list[float] = []
    curve_psnr: list[float] = []
    curve_ssim: list[float] = []
    curve_lpips: list[float] = []

    for k in range(1, max_packets + 1):
        prefix_packets = packets[:k]
        fused = concat_fused_gaussians(prefix_packets, device)
        jobs = build_render_jobs(prefix_packets, args.render_targets, fixed_packet)
        if not jobs:
            raise RuntimeError(f"No render jobs were created for prefix step k={k}.")

        rendered = render_fused_views(fused=fused, jobs=jobs, image_shape=image_shape, device=device)
        gt_images = torch.stack([job.packet.data["target_image"][job.target_idx] for job in jobs], dim=0).to(device=device, dtype=torch.float32)
        metrics = compute_metrics_for_batch(rendered=rendered, gt_images=gt_images, compute_lpips_flag=args.compute_lpips)

        rendered_filenames: list[str] = []
        gt_filenames: list[str] = []
        per_view_metrics: list[dict[str, Any]] = []

        for render_idx, (job, rendered_image, gt_image) in enumerate(zip(jobs, rendered, gt_images)):
            target_index = int(job.packet.data["target_index"][job.target_idx].item())
            packet_stem = job.packet.path.stem
            filename = f"k_{k:03d}_{packet_stem}_target_{target_index:06d}_view_{render_idx:03d}.png"

            save_png_tensor(rendered_image, images_dir / filename)
            save_png_tensor(gt_image, gt_dir / filename)

            rendered_filenames.append(filename)
            gt_filenames.append(filename)

            one_view = {
                "packet_file": job.packet.path.name,
                "scene_key": job.packet.scene_key,
                "target_index": target_index,
                "psnr": float(metrics["psnr"][render_idx].detach().cpu().item()),
                "ssim": float(metrics["ssim"][render_idx].detach().cpu().item()),
            }
            if args.compute_lpips:
                one_view["lpips"] = float(metrics["lpips"][render_idx].detach().cpu().item())
            per_view_metrics.append(one_view)

        mean_psnr = metric_mean_as_float(metrics["psnr"])
        mean_ssim = metric_mean_as_float(metrics["ssim"])
        mean_lpips = metric_mean_as_float(metrics["lpips"]) if args.compute_lpips else None

        curve_packet_counts.append(k)
        curve_num_gaussians.append(float(fused.means.shape[1]))
        curve_psnr.append(mean_psnr)
        curve_ssim.append(mean_ssim)
        if mean_lpips is not None:
            curve_lpips.append(mean_lpips)

        step_summary: dict[str, Any] = {
            "num_packets": k,
            "num_gaussians": int(fused.means.shape[1]),
            "base_scene": prefix_packets[0].base_scene,
            "render_mode": args.render_targets,
            "packet_filenames": [packet_id(packet) for packet in prefix_packets],
            "context_indices": [as_list(packet.data["context_index"]) for packet in prefix_packets],
            "target_indices": [as_list(packet.data["target_index"]) for packet in prefix_packets],
            "rendered_images": rendered_filenames,
            "gt_images": gt_filenames,
            "metrics_mean": {"psnr": mean_psnr, "ssim": mean_ssim},
            "metrics_per_view": per_view_metrics,
        }
        if mean_lpips is not None:
            step_summary["metrics_mean"]["lpips"] = mean_lpips

        summary_steps.append(step_summary)

        log_message = f"[{k}/{max_packets}] fused {k} packet(s) -> {int(fused.means.shape[1])} gaussians, rendered {len(rendered_filenames)} target view(s), PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}"
        if mean_lpips is not None:
            log_message += f", LPIPS={mean_lpips:.4f}"
        print(log_message)

    plot_curve(curve_packet_counts, curve_num_gaussians, "Number of Gaussians vs Number of Fused Packets", "Number of Gaussians", plots_dir / "num_gaussians_vs_packets.png")
    plot_curve(curve_packet_counts, curve_psnr, "PSNR vs Number of Fused Packets", "PSNR", plots_dir / "psnr_vs_packets.png")
    plot_curve(curve_packet_counts, curve_ssim, "SSIM vs Number of Fused Packets", "SSIM", plots_dir / "ssim_vs_packets.png")
    if args.compute_lpips:
        plot_curve(curve_packet_counts, curve_lpips, "LPIPS vs Number of Fused Packets", "LPIPS", plots_dir / "lpips_vs_packets.png")

    summary: dict[str, Any] = {
        "packet_dir": str(args.packet_dir),
        "output_dir": str(args.output_dir),
        "render_targets": args.render_targets,
        "fixed_packet_index": args.fixed_packet_index if args.render_targets == "fixed_packet" else None,
        "compute_lpips": bool(args.compute_lpips),
        "device": str(device),
        "available_packets": len(packets),
        "processed_packets": max_packets,
        "base_scene": packets[0].base_scene,
        "curves": {"num_packets": curve_packet_counts, "num_gaussians": curve_num_gaussians, "psnr": curve_psnr, "ssim": curve_ssim},
        "steps": summary_steps,
    }
    if args.compute_lpips:
        summary["curves"]["lpips"] = curve_lpips
    if fixed_packet is not None:
        summary["fixed_target_packet"] = {
            "packet_file": fixed_packet.path.name,
            "scene_key": fixed_packet.scene_key,
            "context_index": as_list(fixed_packet.data["context_index"]),
            "target_index": as_list(fixed_packet.data["target_index"]),
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
