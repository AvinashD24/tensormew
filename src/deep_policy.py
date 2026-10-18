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
            obs_dim: int = 21,
            num_actions: int = 5,
            ram_hidden_dim: int = 256,
            fusion_dim: int = 512,
            use_gru: bool = False,
            gru_hidden_dim: int = 512,
        ):
            super().__init__()
            self.obs_dim = obs_dim
            self.num_actions = num_actions
            self.use_gru = use_gru
            self.gru_hidden_dim = gru_hidden_dim

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

            # 4. Recurrent Temporal Memory Layer (~1.57M params)
            if use_gru:
                self.gru = nn.GRU(fusion_dim, gru_hidden_dim, batch_first=True)
                head_dim = gru_hidden_dim
            else:
                self.gru = None
                head_dim = fusion_dim

            # 5. Heads (~3k params)
            self.actor = nn.Linear(head_dim, num_actions)
            self.critic = nn.Linear(head_dim, 1)

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
            self,
            screen: torch.Tensor,
            ram_obs: torch.Tensor,
            hx: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning (action_logits, value)."""
            vis_feat = self.vision_trunk(screen)
            ram_feat = self.ram_trunk(ram_obs)
            fused = self.fusion(torch.cat([vis_feat, ram_feat], dim=-1))

            if self.use_gru and self.gru is not None:
                if fused.dim() == 2:
                    gru_out, _ = self.gru(fused.unsqueeze(1), hx)
                    out = gru_out.squeeze(1)
                else:
                    out, _ = self.gru(fused, hx)
            else:
                out = fused

            logits = self.actor(out)
            value = self.critic(out).squeeze(-1)
            return logits, value

        def forward_recurrent(
            self,
            screen: torch.Tensor,
            ram_obs: torch.Tensor,
            hx: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """Recurrent forward pass returning (action_logits, value, new_hx)."""
            vis_feat = self.vision_trunk(screen)
            ram_feat = self.ram_trunk(ram_obs)
            fused = self.fusion(torch.cat([vis_feat, ram_feat], dim=-1))

            if self.use_gru and self.gru is not None:
                if fused.dim() == 2:
                    gru_out, new_hx = self.gru(fused.unsqueeze(1), hx)
                    out = gru_out.squeeze(1)
                else:
                    out, new_hx = self.gru(fused, hx)
            else:
                out = fused
                new_hx = None

            logits = self.actor(out)
            value = self.critic(out).squeeze(-1)
            return logits, value, new_hx


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

        @torch.no_grad()
        def probs(
            self,
            screen_np: np.ndarray,
            ram_np: np.ndarray,
            device: str = "cpu",
        ) -> np.ndarray:
            """Compute action probability distribution from raw screen and RAM vector."""
            self.eval()
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
            logits, _ = self.forward(screen_t, ram_t)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return p


        @torch.no_grad()
        def probs_recurrent(
            self,
            screen_np: np.ndarray,
            ram_np: np.ndarray,
            hx: torch.Tensor | None = None,
            device: str = "cpu",
        ) -> Tuple[np.ndarray, torch.Tensor | None]:
            """Compute action probabilities and next hidden state for recurrent models."""
            self.eval()
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
            logits, _, next_hx = self.forward_recurrent(screen_t, ram_t, hx)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return p, next_hx

        def evaluate_actions(
            self, screen: torch.Tensor, ram_obs: torch.Tensor, actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for PPO loss updates."""
            if screen.dim() == 5:
                # Sequential evaluation (B, T, C, H, W)
                return self.evaluate_actions_sequence(screen, ram_obs, actions)
            logits, values = self.forward(screen, ram_obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy, values

        def evaluate_actions_sequence(
            self,
            screens: torch.Tensor,
            ram_obs: torch.Tensor,
            actions: torch.Tensor,
            hx: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate sequential trajectory chunk for recurrent PPO loss updates.

            Args:
                screens: (B, T, 3, H, W)
                ram_obs: (B, T, obs_dim)
                actions: (B, T)
                hx: (1, B, gru_hidden_dim) initial hidden state for each chunk
            """
            B, T = screens.shape[:2]
            screens_flat = screens.reshape(B * T, *screens.shape[2:])
            ram_flat = ram_obs.reshape(B * T, -1)
            vis_feat = self.vision_trunk(screens_flat)
            ram_feat = self.ram_trunk(ram_flat)
            fused = self.fusion(torch.cat([vis_feat, ram_feat], dim=-1)).reshape(B, T, -1)

            if self.use_gru and self.gru is not None:
                gru_out, _ = self.gru(fused, hx)
                out = gru_out.reshape(B * T, -1)
            else:
                out = fused.reshape(B * T, -1)

            logits = self.actor(out)
            values = self.critic(out).squeeze(-1).reshape(B, T)
            dist = Categorical(logits=logits.reshape(B, T, self.num_actions))
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
                    "use_gru": self.use_gru,
                    "gru_hidden_dim": self.gru_hidden_dim,
                    "model_type": "SpatialVisionPolicy",
                },
                str(path),
            )

        @classmethod
        def load(
            cls,
            path: str | Path,
            device: str = "cpu",
            override_use_gru: bool | None = None,
            override_obs_dim: int | None = None,
        ) -> "SpatialVisionPolicy":
            p = Path(path)
            if not p.exists():
                fallback = Path("models") / p.name
                if fallback.exists():
                    p = fallback
            checkpoint = torch.load(str(p), map_location=device)
            saved_use_gru = checkpoint.get("use_gru", False)
            use_gru = saved_use_gru if override_use_gru is None else override_use_gru
            gru_hidden_dim = checkpoint.get("gru_hidden_dim", 512)
            checkpoint_obs_dim = checkpoint.get("obs_dim", 17)
            target_obs_dim = checkpoint_obs_dim if override_obs_dim is None else override_obs_dim

            model = cls(
                obs_dim=target_obs_dim,
                num_actions=checkpoint.get("num_actions", 5),
                use_gru=use_gru,
                gru_hidden_dim=gru_hidden_dim,
            )

            state_dict = checkpoint["state_dict"]
            if target_obs_dim > checkpoint_obs_dim and "ram_trunk.0.weight" in state_dict:
                old_w = state_dict["ram_trunk.0.weight"]
                new_w = torch.zeros(old_w.shape[0], target_obs_dim, device=old_w.device, dtype=old_w.dtype)
                new_w[:, :old_w.shape[1]] = old_w
                torch.nn.init.orthogonal_(new_w[:, old_w.shape[1]:], gain=0.1)
                state_dict["ram_trunk.0.weight"] = new_w

            strict = (use_gru == saved_use_gru)
            model.load_state_dict(state_dict, strict=strict)
            model.to(device)
            model.eval()
            return model

else:

    class SpatialVisionPolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SpatialVisionPolicy. Install torch.")
