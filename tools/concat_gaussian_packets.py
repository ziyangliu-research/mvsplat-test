from pathlib import Path
import torch

def load_packet(path):
    return torch.load(path, map_location="cpu")

def concat_packets(packet_paths, out_path):
    packets = [load_packet(p) for p in packet_paths]

    merged = {
        "scenes": [p["scene"] for p in packets],
        "means": torch.cat([p["means"] for p in packets], dim=0),
        "covariances": torch.cat([p["covariances"] for p in packets], dim=0),
        "harmonics": torch.cat([p["harmonics"] for p in packets], dim=0),
        "opacities": torch.cat([p["opacities"] for p in packets], dim=0),
    }

    torch.save(merged, out_path)
    print(f"Saved merged packet to: {out_path}")
    print("Num gaussians =", merged["means"].shape[0])

if __name__ == "__main__":
    root = Path("outputs/test/tum_orb/gaussian_packets")

    packet_paths = [
        root / "rgbd_bonn_static_0000.pt",
        root / "rgbd_bonn_static_0001.pt",
    ]

    out_path = root / "merged_0000_0001.pt"
    concat_packets(packet_paths, out_path)