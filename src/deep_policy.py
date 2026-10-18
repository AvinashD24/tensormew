"""Spatial Vision + RAM Embedding Deep Policy (~11.6M parameters).

Combines a ResNet-18 style convolutional visual backbone over the raw Game Boy
144x160 RGB screen buffer with a multi-layer RAM embedding trunk to produce
spatially-aware, quest-aware action distributions and value estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions.categorical import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Basic residual block with two 3x3 convs and a skip connection."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out, inplace=True)

    class ResNet18VisionTrunk(nn.Module):
        """Convolutional vision trunk matching ResNet-18 dimensions (~11.17M params)."""

        def __init__(self, in_channels: int = 3):
            super().__init__()
            # Initial 7x7 conv: 144x160 -> 72x80
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 72x80 -> 36x40

            # 4 residual stages
            self.layer1 = nn.Sequential(
                ConvBlock(64, 64, stride=1),
                ConvBlock(64, 64, stride=1),
            )
            self.layer2 = nn.Sequential(
                ConvBlock(64, 128, stride=2),  # 36x40 -> 18x20
                ConvBlock(128, 128, stride=1),
            )
            self.layer3 = nn.Sequential(
                ConvBlock(128, 256, stride=2),  # 18x20 -> 9x10
                ConvBlock(256, 256, stride=1),
            )
            self.layer4 = nn.Sequential(
                ConvBlock(256, 512, stride=2),  # 9x10 -> 5x5
                ConvBlock(512, 512, stride=1),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)  # (B, 512)

    class SpatialVisionPolicy(nn.Module):
        """Combined ResNet-18 Vision + RAM State policy (~11.64M parameters)."""

        def __init__(
            self,
            obs_dim: int = 17,
            num_actions: int = 5,
            ram_hidden_dim: int = 256,
            fusion_dim: int = 512,
        ):
            super().__init__()
            self.obs_dim = obs_dim
            self.num_actions = num_actions

            # 1. Visual Backbone (~11.17M params)
            self.vision_trunk = ResNet18VisionTrunk(in_channels=3)

            # 2. RAM State Trunk (~70k params)
            self.ram_trunk = nn.Sequential(
                nn.Linear(obs_dim, ram_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ram_hidden_dim, ram_hidden_dim),
                nn.ReLU(inplace=True),
            )

            # 3. Multimodal Fusion Layer (~393k params)
            self.fusion = nn.Sequential(
                nn.Linear(512 + ram_hidden_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True),
            )

            # 4. Heads (~3k params)
            self.actor = nn.Linear(fusion_dim, num_actions)
            self.critic = nn.Linear(fusion_dim, 1)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            # Small initial gain for actor head to keep initial policy uniform
            nn.init.orthogonal_(self.actor.weight, gain=0.01)

        @property
        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def forward(
            self, screen: torch.Tensor, ram_obs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning (action_logits, value)."""
            vis_feat = self.vision_trunk(screen)
            ram_feat = self.ram_trunk(ram_obs)
            fused = self.fusion(torch.cat([vis_feat, ram_feat], dim=-1))
            logits = self.actor(fused)
            value = self.critic(fused).squeeze(-1)
            return logits, value

        @torch.no_grad()
        def act(
            self,
            screen_np: np.ndarray,
            ram_np: np.ndarray,
            deterministic: bool = False,
            device: str = "cpu",
        ) -> Tuple[int, float, float]:
            """Take numpy screen (144, 160, 3) and ram (17,) and return (action, logp, value)."""
            self.eval()
            # Format screen: (H, W, C) -> (1, C, H, W) normalized to [0, 1]
            if screen_np.ndim == 3 and screen_np.shape[2] >= 3:
                rgb = screen_np[:, :, :3]
            else:
                rgb = np.repeat(screen_np[:, :, None], 3, axis=2)
            screen_t = (
                torch.from_numpy(rgb.transpose(2, 0, 1))
                .unsqueeze(0)
                .float()
                .to(device)
                .div_(255.0)
            )
            ram_t = torch.from_numpy(ram_np).unsqueeze(0).float().to(device)

            logits, value = self.forward(screen_t, ram_t)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item()), float(value.item())

        def evaluate_actions(
            self, screen: torch.Tensor, ram_obs: torch.Tensor, actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for PPO loss updates."""
            logits, values = self.forward(screen, ram_obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy, values

        def save(self, path: str | Path) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "obs_dim": self.obs_dim,
                    "num_actions": self.num_actions,
                    "param_count": self.param_count,
                    "model_type": "SpatialVisionPolicy",
                },
                str(path),
            )

        @classmethod
        def load(cls, path: str | Path, device: str = "cpu") -> "SpatialVisionPolicy":
            checkpoint = torch.load(str(path), map_location=device)
            model = cls(
                obs_dim=checkpoint.get("obs_dim", 17),
                num_actions=checkpoint.get("num_actions", 5),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            model.eval()
            return model

else:

    class SpatialVisionPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SpatialVisionPolicy. Install torch.")
