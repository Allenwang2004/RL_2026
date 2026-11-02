"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size) # which calls the BasePolicy constructor to set state_dim, action_dim, and chunk_size
        self.d_model = d_model
        self.depth = depth

        # depth hidden layers of width d_model: Linear -> ReLU -> ... -> Linear
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, d_model), nn.ReLU()]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat_actions = self.mlp(state)
        return flat_actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Sample actions from the policy given a state."""
        return self.forward(state)


# State layout MultiTaskPolicy expects; pass exactly these as --state-keys.
MULTITASK_STATE_KEYS = [
    "state_ee_xyz",
    "state_gripper",
    "original_pos_cube_red[:3]",
    "original_pos_cube_green[:3]",
    "original_pos_cube_blue[:3]",
    "state_goal",
    "goal_pos",
]
_EE, _GRIPPER, _CUBES, _GOAL, _BIN = slice(0, 3), slice(3, 4), slice(4, 13), slice(13, 16), slice(16, 19)
MULTITASK_STATE_DIM = 19


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    The target cube is selected inside the model with the one-hot goal
    (target = sum_i goal_i * cube_i), so the MLP never sees cube colours:
    "red target at y=0.35" and "green target at y=0.35" become the same
    input and every demo trains all three colours.

    MLP input: ee xyz, gripper, target xyz, bin xyz, target - ee, bin - ee.
    The state arrives normalized per column, so the dataset normalizer is
    kept as buffers (saved in the state_dict) to recover raw positions.
    """

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        if state_dim != MULTITASK_STATE_DIM:
            raise ValueError(
                f"MultiTaskPolicy expects state_dim={MULTITASK_STATE_DIM} "
                f"(--state-keys {' '.join(MULTITASK_STATE_KEYS)}), got {state_dim}"
            )
        self.d_model = d_model
        self.depth = depth
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

        layers: list[nn.Module] = []
        in_dim = 16
        for _ in range(depth):
            layers += [nn.Linear(in_dim, d_model), nn.ReLU()]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def set_state_normalizer(self, state_mean, state_std) -> None:
        """Copy the dataset state normalizer in; call once before training."""
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(state_std, dtype=torch.float32))

    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Map the normalized state to colour-invariant MLP inputs."""
        raw = state * self.state_std + self.state_mean
        cubes = raw[:, _CUBES].view(-1, 3, 3)  # (B, colour, xyz)
        goal = raw[:, _GOAL]
        goal = goal / goal.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target = (goal.unsqueeze(-1) * cubes).sum(dim=1)

        # One shared scale for all cubes, so the selected target is
        # normalized the same way whichever colour it is.
        cube_mean = self.state_mean[_CUBES].view(3, 3).mean(dim=0)
        cube_std = self.state_std[_CUBES].view(3, 3).mean(dim=0)
        ee_std = self.state_std[_EE]
        return torch.cat(
            [
                state[:, _EE],
                state[:, _GRIPPER],
                (target - cube_mean) / cube_std,
                state[:, _BIN],
                (target - raw[:, _EE]) / ee_std,
                (raw[:, _BIN] - raw[:, _EE]) / ee_std,
            ],
            dim=1,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat_actions = self.mlp(self.features(state))
        return flat_actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Sample actions from the policy given a state."""
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 128,
    depth: int = 2,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
