"""SpatialGridPolicy: the RAM-vector + GRU policy of ``src.deep_policy`` with the ResNet-18 screen trunk replaced by a small
conv net over the semantic grid of ``src.spatial_grid`` (~0.7M parameters instead of ~11.6M).

It is a ``SpatialVisionPolicy`` subclass: the rollout (act / probs / forward_recurrent), the PPO paths (evaluate_actions,
evaluate_actions_sequence), action masking, the battle A-bias and the aux head are inherited unchanged. Only the "screen"
argument changes meaning: a ``(GRID_CHANNELS, 15, 15)`` uint8 grid (``(B, C, 15, 15)`` / ``(B, T, C, 15, 15)`` batched)
instead of a ``(3, 144, 160)`` image. Checkpoints carry ``grid_schema`` so a grid checkpoint can never load as a pixel one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.deep_policy import SpatialVisionPolicy, TORCH_AVAILABLE, CheckpointIncompatibleError

GRID_SCHEMA = "grid-v1-7ch-15x15"

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from src.entity_classifier import ENTITY_CLASSES
    from src.spatial_grid import (
        CH_COUNTER_FRONT, CH_ENTITY, CH_ENTRY, CH_FACING, CH_SIGHT, CH_TERRAIN, CH_VISITED,
        GRID_CHANNELS, GRID_SIZE, TERRAIN_CLASSES,
    )

    FACING_CLASSES = 5   # none, down, up, left, right
    # one-hot terrain + one-hot entity + sight + 4 entry-direction bits + counter front + one-hot facing + visited
    GRID_CONV_CHANNELS = TERRAIN_CLASSES + ENTITY_CLASSES + 1 + 4 + 1 + FACING_CLASSES + 1

    class GridTrunk(nn.Module):
        """Categorical planes -> one-hot, then conv(3x3) -> conv(3x3, stride 2) -> 4x4 average pool -> 1024 features."""

        OUT_DIM = 64 * 4 * 4

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(GRID_CONV_CHANNELS, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 15x15 -> 8x8
            self.pool = nn.AdaptiveAvgPool2d((4, 4))

        @staticmethod
        def expand(x: torch.Tensor) -> torch.Tensor:
            """(B, GRID_CHANNELS, 15, 15) integer/float planes -> (B, GRID_CONV_CHANNELS, 15, 15) float."""
            terrain = F.one_hot(x[:, CH_TERRAIN].long().clamp(0, TERRAIN_CLASSES - 1), TERRAIN_CLASSES).float()
            entity = F.one_hot(x[:, CH_ENTITY].long().clamp(0, ENTITY_CLASSES - 1), ENTITY_CLASSES).float()
            entry = x[:, CH_ENTRY].long()
            entry_bits = torch.stack([(entry >> i) & 1 for i in range(4)], dim=-1).float()
            facing = F.one_hot(x[:, CH_FACING].long().clamp(0, FACING_CLASSES - 1), FACING_CLASSES).float()
            binary = torch.stack([x[:, CH_SIGHT], x[:, CH_COUNTER_FRONT], x[:, CH_VISITED]], dim=-1).float()
            planes = torch.cat([terrain, entity, binary[..., :1], entry_bits, binary[..., 1:2], facing, binary[..., 2:]],
                               dim=-1)
            return planes.permute(0, 3, 1, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.conv1(self.expand(x)))
            h = F.relu(self.conv2(h))
            return torch.flatten(self.pool(h), 1)

    class SpatialGridPolicy(SpatialVisionPolicy):
        """Grid + RAM-vector policy; see the module docstring. Defaults are the small configuration."""

        VISION_OUT_DIM = GridTrunk.OUT_DIM

        def __init__(self, obs_dim: int = 80, num_actions: int = 6, ram_hidden_dim: int = 128, fusion_dim: int = 256,
                     use_gru: bool = False, gru_hidden_dim: int = 256, aux_heads: bool = False):
            super().__init__(obs_dim=obs_dim, num_actions=num_actions, ram_hidden_dim=ram_hidden_dim,
                             fusion_dim=fusion_dim, use_gru=use_gru, gru_hidden_dim=gru_hidden_dim,
                             aux_heads=aux_heads)

        def _build_vision_trunk(self) -> nn.Module:
            return GridTrunk()

        @staticmethod
        def _screen_tensor(screen_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
            """Numpy grid (GRID_CHANNELS, 15, 15) -> (1, GRID_CHANNELS, 15, 15) float."""
            if screen_np.shape != (GRID_CHANNELS, GRID_SIZE, GRID_SIZE):
                raise ValueError(f"grid must be {(GRID_CHANNELS, GRID_SIZE, GRID_SIZE)}, got {screen_np.shape}")
            return torch.from_numpy(np.ascontiguousarray(screen_np)).unsqueeze(0).float().to(device)

        def save(self, path: str | Path) -> None:
            super().save(path)
            blob = torch.load(str(path), map_location="cpu")
            blob["model_type"] = "SpatialGridPolicy"
            blob["grid_schema"] = GRID_SCHEMA
            torch.save(blob, str(path))

        @classmethod
        def load(cls, path: str | Path, device: str = "cpu", **kwargs) -> "SpatialGridPolicy":
            p = Path(path)
            if not p.exists() and (Path("models") / p.name).exists():
                p = Path("models") / p.name
            schema = torch.load(str(p), map_location="cpu").get("grid_schema")
            if schema != GRID_SCHEMA:
                raise CheckpointIncompatibleError(
                    f"{p}: grid schema {schema!r} != {GRID_SCHEMA!r}; not a SpatialGridPolicy checkpoint for this grid layout.")
            return super().load(p, device=device, **kwargs)

else:

    class SpatialGridPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SpatialGridPolicy. Install torch.")
