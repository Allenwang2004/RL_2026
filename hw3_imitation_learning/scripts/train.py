"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

If the data contains DAgger episodes, the validation set is drawn from the
original teleop episodes only; all DAgger samples go to training.

Trains one model per seed (default: 5 seeds) and saves them as
checkpoints/<task>/<run-name>/seed{k}.pt. Pick the best one afterwards with
scripts/eval_folder.py.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ...
        --action-keys ...
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, Subset

# TODO: Choose your own hyperparameters!
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1


def episode_is_dagger(zarr_path: Path) -> np.ndarray:
    """Per-episode DAgger flag of a processed zarr (written by compute_actions.py)."""
    z = zarr_lib.open_group(str(zarr_path), mode="r")
    if "episode_is_dagger" in z["meta"]:
        return np.asarray(z["meta"]["episode_is_dagger"][:], dtype=bool)
    n_eps = int(z["meta"]["episode_ends"].shape[0])
    n_dagger = int(z.attrs.get("num_dagger_episodes", 0))
    if n_dagger in (0, n_eps):
        return np.full(n_eps, n_dagger > 0)
    raise SystemExit(
        f"ERROR: {zarr_path} mixes teleop and DAgger episodes but has no "
        "meta/episode_is_dagger. Re-run scripts/compute_actions.py to regenerate it."
    )


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        loss = model.compute_loss(states, action_chunks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the evaluation step for one batch here.
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # TODO: You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="+",
        default=None,
        help="Additional processed .zarr stores to merge with --zarr (e.g. DAgger data).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Hidden layer width of the policy MLP (default: 128).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of hidden layers of the policy MLP (default: 2).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds; one model is trained per seed (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output folder name under checkpoints/<task>/ "
        "(default: best_model_<action_space>_<policy>[_dagger<N>ep]_<timestamp>).",
    )
    args = parser.parse_args()

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # select gpu 2
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    ep_is_dagger = np.concatenate([episode_is_dagger(zp) for zp in zarr_paths])
    assert ep_is_dagger.size == ep_ends.size
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    # Validation uses teleop samples only; DAgger samples always go to training.
    sample_ep = np.searchsorted(ep_ends, dataset.indices, side="right")
    teleop_idx = np.flatnonzero(~ep_is_dagger[sample_ep])
    dagger_idx = np.flatnonzero(ep_is_dagger[sample_ep])
    print(
        f"  episodes: {int((~ep_is_dagger).sum())} teleop, {int(ep_is_dagger.sum())} dagger "
        f"({teleop_idx.size} / {dagger_idx.size} samples)"
    )

    # ── output folder ─────────────────────────────────────────────────
    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    run_name = f"best_model_{action_space}_{args.policy}"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        run_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep"
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"{run_name}_{ts}"
    # Default: checkpoints/<task>/<run_name>/seed{k}.pt
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube") / run_name
    else:
        ckpt_dir = Path("./checkpoints/single_cube") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_vals: dict[int, float] = {}
    for seed in args.seeds:
        print(f"\n═══ Seed {seed} ({len(best_vals) + 1}/{len(args.seeds)}) ═══")
        torch.manual_seed(seed)

        # ── train / val split (val from teleop only) ──────────────────
        # Same permutation as random_split when there is no DAgger data.
        n_val = max(1, int(teleop_idx.size * VAL_SPLIT))
        n_train = teleop_idx.size - n_val
        perm = torch.randperm(
            teleop_idx.size, generator=torch.Generator().manual_seed(seed)
        ).numpy()
        train_ds = Subset(
            dataset, np.concatenate([teleop_idx[perm[:n_train]], dagger_idx]).tolist()
        )
        val_ds = Subset(dataset, teleop_idx[perm[n_train:]].tolist())
        print(f"Split: {len(train_ds)} train, {len(val_ds)} val (teleop only)")

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # ── model ─────────────────────────────────────────────────────
        model = build_policy(
            args.policy,
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            chunk_size=args.chunk_size,
            d_model=args.d_model,
            depth=args.depth,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # TODO: implement an optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # ── training loop ─────────────────────────────────────────────
        best_val = float("inf")
        save_path = ckpt_dir / f"seed{seed}.pt"

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)
            scheduler.step()

            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "seed": seed,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "normalizer": {
                            "state_mean": normalizer.state_mean,
                            "state_std": normalizer.state_std,
                            "action_mean": normalizer.action_mean,
                            "action_std": normalizer.action_std,
                        },
                        "chunk_size": args.chunk_size,
                        # eval_utils.load_checkpoint rebuilds the model from these
                        "d_model": args.d_model,
                        "depth": args.depth,
                        "policy_type": args.policy,
                        "state_keys": args.state_keys,
                        "action_keys": args.action_keys,
                        "state_dim": int(states.shape[1]),
                        "action_dim": int(actions.shape[1]),
                        "val_loss": val_loss,
                    },
                    save_path,
                )
                tag = " ✓ saved"

            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
            )

        best_vals[seed] = best_val
        print(f"Seed {seed}: best val loss {best_val:.6f} -> {save_path}")

    print(f"\nCheckpoints in {ckpt_dir}:")
    for seed, v in best_vals.items():
        print(f"  seed{seed}.pt  best val {v:.6f}")
    print(f"Pick the best one with: python scripts/eval_folder.py --exercise <N> --folder {ckpt_dir}")


if __name__ == "__main__":
    main()
