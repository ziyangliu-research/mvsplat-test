from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from packet_fusion_utils import (
    build_probe_from_packet,
    check_probe_consistency,
    evaluate_triplet,
    load_packet,
    merge_with_policy,
    packet_basic_stats,
    save_four_panel,
    save_packet,
)


def aggregate_results(df: pd.DataFrame) -> Dict:
    summary = {}
    for policy, g in df.groupby("policy"):
        summary[policy] = {
            "num_pairs": int(len(g)),
            "A_psnr_mean": float(g["A_psnr"].mean()),
            "B_psnr_mean": float(g["B_psnr"].mean()),
            "Merged_psnr_mean": float(g["Merged_psnr"].mean()),
            "delta_best_psnr_mean": float(g["delta_best_psnr"].mean()),
            "delta_avg_psnr_mean": float(g["delta_avg_psnr"].mean()),
            "win_rate": float(g["merged_beats_best"].mean()),
        }

    family_summary = {}
    for (policy, family), g in df.groupby(["policy", "family"]):
        family_summary.setdefault(policy, {})[family] = {
            "num_pairs": int(len(g)),
            "Merged_psnr_mean": float(g["Merged_psnr"].mean()),
            "delta_best_psnr_mean": float(g["delta_best_psnr"].mean()),
            "win_rate": float(g["merged_beats_best"].mean()),
        }

    return {
        "policy_summary": summary,
        "family_summary": family_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet_root", type=Path, required=True, help="保存 packet .pt 的目录")
    parser.add_argument("--pair_json", type=Path, required=True, help="pair_fusion_v1 的 JSON")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["raw", "opacity", "opacity_topk", "voxel", "opacity_topk_voxel"],
        help="可选: raw opacity opacity_topk voxel opacity_topk_voxel",
    )
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50000)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--probe_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_merged_pt", action="store_true")
    parser.add_argument("--strict_probe", action="store_true", help="若 A/B 的 probe 不一致则直接报错")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "visualizations"
    merged_dir = args.out_dir / "merged_packets"

    with args.pair_json.open("r", encoding="utf-8") as f:
        pair_dict = json.load(f)

    records: List[Dict] = []

    for pair_key, item in pair_dict.items():
        packet_a_path = args.packet_root / f"{item['packet_a_key']}.pt"
        packet_b_path = args.packet_root / f"{item['packet_b_key']}.pt"
        if not packet_a_path.exists():
            raise FileNotFoundError(f"Missing packet A: {packet_a_path}")
        if not packet_b_path.exists():
            raise FileNotFoundError(f"Missing packet B: {packet_b_path}")

        packet_a_raw = load_packet(packet_a_path)
        packet_b_raw = load_packet(packet_b_path)

        probe_meta = check_probe_consistency(packet_a_raw, packet_b_raw, probe_idx=args.probe_idx)
        if args.strict_probe and not probe_meta["same"]:
            raise RuntimeError(f"Probe mismatch in {pair_key}: {probe_meta}")

        probe = build_probe_from_packet(packet_a_raw, probe_idx=args.probe_idx, device=args.device)
        stats_a_raw = packet_basic_stats(packet_a_raw)
        stats_b_raw = packet_basic_stats(packet_b_raw)

        for policy in args.policies:
            single_a, single_b, merged = merge_with_policy(
                packet_a_raw,
                packet_b_raw,
                policy=policy,
                opacity_thresh=args.opacity_thresh,
                topk=args.topk,
                voxel_size=args.voxel_size,
            )

            result = evaluate_triplet(single_a, single_b, merged, probe, device=args.device)
            stats_a = packet_basic_stats(single_a)
            stats_b = packet_basic_stats(single_b)
            stats_m = packet_basic_stats(merged)

            record = {
                "pair_key": pair_key,
                "scene": item["scene"],
                "family": item["family"],
                "target_index": int(item["target"][0]),
                "packet_a_key": item["packet_a_key"],
                "packet_b_key": item["packet_b_key"],
                "packet_a_context": item["packet_a_context"],
                "packet_b_context": item["packet_b_context"],
                "left_inner_gap": int(item["left_inner_gap"]),
                "left_outer_gap": int(item["left_outer_gap"]),
                "right_inner_gap": int(item["right_inner_gap"]),
                "right_outer_gap": int(item["right_outer_gap"]),
                "pair_span": int(item["pair_span"]),
                "policy": policy,
                "opacity_thresh": args.opacity_thresh,
                "topk": args.topk,
                "voxel_size": args.voxel_size,
                "probe_same": probe_meta["same"],
                "probe_extr_diff": probe_meta["extr_diff"],
                "probe_intr_diff": probe_meta["intr_diff"],
                "A_num_raw": stats_a_raw["num_gaussians"],
                "B_num_raw": stats_b_raw["num_gaussians"],
                "A_num_eval": stats_a["num_gaussians"],
                "B_num_eval": stats_b["num_gaussians"],
                "Merged_num": stats_m["num_gaussians"],
                "A_psnr": result["A_psnr"],
                "B_psnr": result["B_psnr"],
                "Merged_psnr": result["Merged_psnr"],
                "A_mse": result["A_mse"],
                "B_mse": result["B_mse"],
                "Merged_mse": result["Merged_mse"],
                "best_single_psnr": result["best_single_psnr"],
                "delta_best_psnr": result["delta_best_psnr"],
                "delta_avg_psnr": result["delta_avg_psnr"],
                "merged_beats_best": int(result["merged_beats_best"]),
            }
            records.append(record)

            if args.save_vis:
                title = (
                    f"pair={pair_key} | family={item['family']} | policy={policy} | "
                    f"target={item['target'][0]}"
                )
                save_four_panel(
                    result["gt"],
                    result["img_a"],
                    result["img_b"],
                    result["img_merged"],
                    vis_dir / policy / f"{pair_key}.png",
                    title=title,
                )

            if args.save_merged_pt:
                save_packet(merged, merged_dir / policy / f"{pair_key}.pt")

    df = pd.DataFrame(records)
    csv_path = args.out_dir / "pair_fusion_results.csv"
    json_path = args.out_dir / "pair_fusion_results.json"
    summary_path = args.out_dir / "pair_fusion_summary.json"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary = aggregate_results(df)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved CSV    : {csv_path}")
    print(f"Saved JSON   : {json_path}")
    print(f"Saved summary: {summary_path}")
    print("\n=== Policy summary ===")
    for policy, s in summary["policy_summary"].items():
        print(
            f"[{policy}] num_pairs={s['num_pairs']} | "
            f"Merged_PSNR={s['Merged_psnr_mean']:.4f} | "
            f"delta_best={s['delta_best_psnr_mean']:.4f} | "
            f"win_rate={s['win_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
