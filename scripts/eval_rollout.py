import argparse
import torch
from pathlib import Path

from wm.data.dataset_npz import EpisodeNPZDataset
from wm.data.collate import collate_fixed_length
from wm.models.world_model import WorldModelVPF
from wm.utils.rollout import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    npz_files = sorted(args.dataset_root.glob("*.npz"))
    dataset = EpisodeNPZDataset(npz_files, config={"sequence_length": 64})
    batch = collate_fixed_length([dataset[0]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: v.to(device) for k, v in batch.items()}

    j_dim = batch["q"].shape[-1]
    action_dim = batch["action"].shape[-1]
    model = WorldModelVPF(j_dim=j_dim, action_dim=action_dim).to(device)
    model.eval()

    preds = rollout(model, batch, horizon=args.horizon)

    # ground truth
    q_gt = batch["q"][:, 1:args.horizon + 1]

    err = torch.mean((preds["q_hat"] - q_gt) ** 2).sqrt()
    print(f"Rollout RMSE (q): {err.item():.6f}")


if __name__ == "__main__":
    main()
