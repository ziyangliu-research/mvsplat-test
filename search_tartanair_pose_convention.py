#!/usr/bin/env python3
"""
Brute-force search for the camera-frame convention used by TartanAir poses by
running the repository's existing test pipeline on a small same-view benchmark.

What it does
------------
1. Generates a small evaluation JSON with entries like context=[i], target=[i].
2. Temporarily patches src/dataset/dataset_tartanair.py so that
   _build_Twc_from_pose() uses one candidate camera-axis transform A and one
   inverse flag.
3. Runs `python -m src.main ... mode=test` for each candidate.
4. Collects scores_all_avg.json, saves rendered images, and writes a ranked
   summary.
5. Restores the original dataset_tartanair.py at the end, even if a candidate
   fails.

Important
---------
- Set pose_matrix_type=Twc in your dataset config while using this script.
- This script edits dataset_tartanair.py on disk temporarily. It restores the
  original file afterward.
- Candidates are signed permutation matrices with det=+1 (24 candidates), with
  and without pose inversion (48 candidates total).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=Path, default=Path("/workspace/mvsplat"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--experiment", type=str, default="tartanair_p000_ft")
    p.add_argument("--scene_name", type=str, default="P000")
    p.add_argument("--test_json", type=Path, default=None,
                   help="Path to write the generated same-view evaluation JSON. Defaults under assets/tartanair/.")
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="Explicit raw/sample indices to test, e.g. 0 10 20 30 40")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=10,
                   help="Stride when auto-generating indices")
    p.add_argument("--count", type=int, default=5,
                   help="Number of indices when auto-generating")
    p.add_argument("--output_root", type=Path, default=Path("outputs/axis_search"))
    p.add_argument("--image_shape_h", type=int, default=None,
                   help="Optional override: dataset.image_shape[0]")
    p.add_argument("--image_shape_w", type=int, default=None,
                   help="Optional override: dataset.image_shape[1]")
    p.add_argument("--target_camera", type=str, default="left", choices=["left", "right", "both"])
    p.add_argument("--max_candidates", type=int, default=-1,
                   help="Limit number of candidates for a quick smoke run. -1 means all 48.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--extra_override", type=str, nargs="*", default=[],
                   help="Extra Hydra overrides appended as-is.")
    return p.parse_args()


def generate_right_handed_signed_permutation_matrices() -> list[list[list[float]]]:
    mats: list[list[list[float]]] = []
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product([-1.0, 1.0], repeat=3))
    for perm in perms:
        for sign in signs:
            M = [[0.0, 0.0, 0.0] for _ in range(3)]
            for r, c in enumerate(perm):
                M[r][c] = sign[r]
            det = round(
                M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
            )
            if det == 1:
                mats.append(M)
    # stable order
    return mats


def matrix_to_name(M: list[list[float]]) -> str:
    axis_names = []
    basis = ["x", "y", "z"]
    for row in M:
        c = max(range(3), key=lambda j: abs(row[j]))
        s = "p" if row[c] > 0 else "n"
        axis_names.append(f"{s}{basis[c]}")
    return "_".join(axis_names)


def build_same_view_json(scene_name: str, indices: Iterable[int]) -> dict:
    out = {}
    for k, i in enumerate(indices):
        out[f"{scene_name}_{k:04d}"] = {
            "context": [int(i)],
            "target": [int(i)],
            "meta": {
                "index": int(i),
                "stage": "test",
                "kind": "same_view_debug",
            },
        }
    return out


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def patch_dataset_file(dataset_path: Path, matrix: list[list[float]], use_inverse: bool) -> str:
    original = dataset_path.read_text(encoding="utf-8")

    replacement = f'''def _build_Twc_from_pose(self, tx, ty, tz, qx, qy, qz, qw):\n        Twc_pose = torch.eye(4, dtype=torch.float32)\n        Twc_pose[:3, :3] = self._quat_to_rotmat(qx, qy, qz, qw)\n        Twc_pose[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)\n\n        if {str(use_inverse)}:\n            Twc_pose = torch.linalg.inv(Twc_pose)\n\n        A = torch.eye(4, dtype=torch.float32)\n        A[:3, :3] = torch.tensor({matrix!r}, dtype=torch.float32)\n        Twc = Twc_pose @ A\n        return Twc\n\n    def _build_K(self, fx: float, fy: float, cx: float, cy: float):'''

    pattern = re.compile(
        r"def _build_Twc_from_pose\(self, tx, ty, tz, qx, qy, qz, qw\):.*?\n\s*def _build_K\(self, fx: float, fy: float, cx: float, cy: float\):",
        re.DOTALL,
    )
    patched, count = pattern.subn(replacement, original)
    if count != 1:
        raise RuntimeError(f"Failed to patch _build_Twc_from_pose in {dataset_path}")
    dataset_path.write_text(patched, encoding="utf-8")
    return original


def run_candidate(
    args: argparse.Namespace,
    candidate_name: str,
    json_path: Path,
) -> tuple[int, str]:
    run_name = f"axis_search__{candidate_name}"
    cmd = [
        args.python,
        "-m",
        "src.main",
        f"+experiment={args.experiment}",
        "mode=test",
        f"checkpointing.load={args.checkpoint}",
        "checkpointing.resume=false",
        "wandb.mode=disabled",
        f"wandb.name={run_name}",
        f"test.output_path={args.output_root}",
        "test.compute_scores=true",
        "test.save_image=true",
        "test.save_video=false",
        f"dataset.view_sampler.test_index_path={json_path}",
        f"dataset.target_camera={args.target_camera}",
        "dataset.pose_matrix_type=Twc",
    ]
    if args.image_shape_h is not None and args.image_shape_w is not None:
        cmd.append(f"dataset.image_shape=[{args.image_shape_h},{args.image_shape_w}]")
    cmd.extend(args.extra_override)

    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=args.repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def read_scores(run_dir: Path) -> dict | None:
    score_path = run_dir / "scores_all_avg.json"
    if not score_path.exists():
        return None
    return json.loads(score_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_path = repo_root / "src" / "dataset" / "dataset_tartanair.py"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_tartanair.py not found: {dataset_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.indices is None or len(args.indices) == 0:
        indices = [args.start + i * args.stride for i in range(args.count)]
    else:
        indices = list(args.indices)

    test_json = args.test_json
    if test_json is None:
        test_json = repo_root / "assets" / "tartanair" / f"{args.scene_name}_same_view_test.json"
    else:
        test_json = test_json if test_json.is_absolute() else (repo_root / test_json)

    write_json(test_json, build_same_view_json(args.scene_name, indices))
    print(f"[INFO] Wrote same-view test JSON to: {test_json}")
    print(f"[INFO] Indices: {indices}")

    matrices = generate_right_handed_signed_permutation_matrices()
    candidates: list[tuple[str, list[list[float]], bool]] = []
    for M in matrices:
        base = matrix_to_name(M)
        candidates.append((f"Twc__{base}", M, False))
        candidates.append((f"InvTwc__{base}", M, True))

    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    original_text = dataset_path.read_text(encoding="utf-8")
    results = []
    logs_dir = args.output_root / "axis_search_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, (name, M, use_inverse) in enumerate(candidates, start=1):
            print(f"\n[INFO] ({idx}/{len(candidates)}) Testing candidate: {name}")
            patch_dataset_file(dataset_path, M, use_inverse)
            code, out = run_candidate(args, name, test_json)
            log_path = logs_dir / f"{name}.log"
            log_path.write_text(out, encoding="utf-8")

            run_dir = args.output_root / f"axis_search__{name}"
            scores = read_scores(run_dir)
            result = {
                "candidate": name,
                "use_inverse": use_inverse,
                "matrix": M,
                "returncode": code,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "scores": scores,
            }
            if scores is not None:
                psnr = scores.get("psnr")
                ssim = scores.get("ssim")
                lpips = scores.get("lpips")
                print(f"[INFO] scores: psnr={psnr}, ssim={ssim}, lpips={lpips}")
            else:
                print(f"[WARN] No scores_all_avg.json found for {name}; check log: {log_path}")
            results.append(result)
    finally:
        dataset_path.write_text(original_text, encoding="utf-8")
        print(f"\n[INFO] Restored original dataset file: {dataset_path}")

    def sort_key(item: dict):
        scores = item.get("scores") or {}
        psnr = scores.get("psnr", float("-inf"))
        ssim = scores.get("ssim", float("-inf"))
        return (psnr, ssim)

    results_sorted = sorted(results, key=sort_key, reverse=True)
    summary_path = args.output_root / "axis_search_summary.json"
    summary_path.write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")

    print("\n===== TOP CANDIDATES =====")
    for item in results_sorted[:10]:
        s = item.get("scores") or {}
        print(
            f"{item['candidate']}: "
            f"psnr={s.get('psnr')}, ssim={s.get('ssim')}, lpips={s.get('lpips')}, "
            f"dir={item['run_dir']}"
        )
    print(f"\n[INFO] Saved ranked summary to: {summary_path}")
    print(f"[INFO] Rendered images are under: {args.output_root} / axis_search__<candidate>/...")


if __name__ == "__main__":
    main()
