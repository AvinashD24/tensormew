"""Auxiliary prediction tasks for the recurrent policy (``long_v4``): targets, loss, no emulator cost.

Literature: Jaderberg et al. 2016, "Reinforcement Learning with Unsupervised Auxiliary Tasks" (UNREAL, arXiv
1611.05397): training the shared representation to also predict pseudo-rewards / reward / value improved Atari to
880 % of expert human performance and gave a mean 10x learning speed-up on Labyrinth; Lample & Chaplot 2016/2017,
"Playing FPS Games with Deep RL" (arXiv 1609.05521): predicting game features (enemy / item present) from the shared
convolutional features "dramatically improves training speed and performance".  Those results are for A3C / DQN-style
agents on other games.  There is NO published evidence for this exact setup (PPO + GRU, Pokemon Crystal RAM features),
and the loss weight (0.05) and horizon (8) are guesses; the heads are config-gated for that reason.

Three tasks, all with targets that are already in the stored RAM observation vector ``rams[b, t]`` (so nothing extra
is stored or computed in the workers).  For step ``t`` and horizon ``k`` the target is read at step ``t + k`` of the
same rollout sequence; it is masked when ``t + k`` is past the end of the rollout or an episode ended in between:

* ``map``: the map the agent is on at ``t + k`` (group and number, two classifications; story-block features);
* ``hp``: change of the lead Pokemon's HP fraction between ``t`` and ``t + k`` (regression, scaled x5 so a 20 % drop is
  1.0), from the battle-feature block (``party0_hp``);
* ``battle``: whether a battle is running at some step in ``(t, t + k]`` given none at ``t`` (binary; steps already in
  a battle are masked), from the in-battle flag.

The heads sit on the recurrent features that also feed the actor and critic, so the auxiliary gradients flow into the
shared trunk (GRU, fusion, RAM and vision trunks).  They never touch the action / value outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.battle_obs import BATTLE_FEATURE_NAMES
from src.rl_env import BASE_OBS_DIM
from src.story_path import MILESTONE_COUNT, STORY_FEATURE_DIM

AUX_MAP_GROUPS = 32     # map group classes (Crystal groups are 1..26)
AUX_MAP_NUMBERS = 64    # map number classes (clipped)
AUX_OUT_DIM = AUX_MAP_GROUPS + AUX_MAP_NUMBERS + 2   # + lead-HP-change regression + battle-start logit
HP_TARGET_SCALE = 5.0

IDX_MAP_GROUP = BASE_OBS_DIM + MILESTONE_COUNT       # g / 32 in the story block
IDX_MAP_NUMBER = IDX_MAP_GROUP + 1                   # n / 32
IDX_LEAD_HP = BASE_OBS_DIM + STORY_FEATURE_DIM + BATTLE_FEATURE_NAMES.index("party0_hp")
IDX_BATTLE = 7                                       # base block: tiles(4), x, y, badges, battle (pinned in tests)

AUX_KEYS = ("map", "hp", "battle")


def split_outputs(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the head output ``(..., AUX_OUT_DIM)`` into logits / regression / battle logit."""
    g = AUX_MAP_GROUPS
    n = AUX_MAP_NUMBERS
    return {"group": raw[..., :g], "number": raw[..., g:g + n], "hp": raw[..., g + n], "battle": raw[..., g + n + 1]}


def valid_mask(dones: torch.Tensor | None, T: int, horizon: int, batch: int, device) -> torch.Tensor:
    """``(B, T)`` bool: step ``t`` has a target ``horizon`` steps ahead inside this sequence and no episode boundary
    in ``[t, t + horizon - 1]`` (``dones[b, t]`` = the episode ended after step ``t``)."""
    mask = torch.zeros(batch, T, dtype=torch.bool, device=device)
    if horizon >= T:
        return mask
    ok = torch.ones(batch, T - horizon, dtype=torch.bool, device=device)
    if dones is not None:
        d = dones.to(device).float()
        window_has_done = d.unfold(1, horizon, 1)[:, : T - horizon].amax(dim=-1) > 0   # (B, T - horizon)
        ok = ~window_has_done
    mask[:, : T - horizon] = ok
    return mask


def make_targets(rams: torch.Tensor, dones: torch.Tensor | None, horizon: int) -> dict[str, torch.Tensor]:
    """Targets and masks for a ``(B, T, D)`` RAM observation sequence."""
    B, T = rams.shape[:2]
    h = int(horizon)
    valid = valid_mask(dones, T, h, B, rams.device)
    ahead = torch.zeros_like(rams)
    if h < T:
        ahead[:, : T - h] = rams[:, h:]
    group = torch.round(ahead[..., IDX_MAP_GROUP] * 32.0).long().clamp_(0, AUX_MAP_GROUPS - 1)
    number = torch.round(ahead[..., IDX_MAP_NUMBER] * 32.0).long().clamp_(0, AUX_MAP_NUMBERS - 1)
    hp_delta = (ahead[..., IDX_LEAD_HP] - rams[..., IDX_LEAD_HP]) * HP_TARGET_SCALE
    in_battle_now = rams[..., IDX_BATTLE] > 0.5
    battle_soon = torch.zeros(B, T, dtype=torch.bool, device=rams.device)
    if h < T:
        flags = (rams[..., IDX_BATTLE] > 0.5).float()
        # any battle flag in (t, t + h]: the unfolded window of the next h steps
        nxt = flags[:, 1:].unfold(1, h, 1).amax(dim=-1) > 0   # (B, T - h): window starts at t + 1
        battle_soon[:, : T - h] = nxt[:, : T - h]
    return {
        "group": group, "number": number, "hp": hp_delta, "battle": battle_soon.float(),
        "valid": valid, "battle_valid": valid & ~in_battle_now,
    }


def aux_loss(out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean losses over the valid steps; returns ``(total, {"map", "hp", "battle"})`` (zero loss when nothing is valid)."""
    zero = out["hp"].sum() * 0.0
    valid, bvalid = tgt["valid"], tgt["battle_valid"]
    parts: dict[str, torch.Tensor] = {}
    if valid.any():
        ce_g = F.cross_entropy(out["group"][valid], tgt["group"][valid])
        ce_n = F.cross_entropy(out["number"][valid], tgt["number"][valid])
        parts["map"] = ce_g + ce_n
        parts["hp"] = F.smooth_l1_loss(out["hp"][valid], tgt["hp"][valid])
    else:
        parts["map"] = parts["hp"] = zero
    if bvalid.any():
        parts["battle"] = F.binary_cross_entropy_with_logits(out["battle"][bvalid], tgt["battle"][bvalid])
    else:
        parts["battle"] = zero
    total = parts["map"] + parts["hp"] + parts["battle"]
    return total, {k: float(v.detach()) for k, v in parts.items()}
