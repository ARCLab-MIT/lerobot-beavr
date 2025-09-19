#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.dact.configuration_dact_a import DACTConfigA
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, populate_queues


class DACTPolicyA(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = DACTConfigA
    name = "dact_a"

    def __init__(
        self,
        config: DACTConfigA,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = DACT(config)

        self.temporal_attn_pool_1d = TemporalAttentionPool1D(config)
        if config.image_features:
            self.temporal_attn_pool_2d = TemporalAttentionPool2D(
                config,
                backbone=self.model.backbone,
                backbone_out_channels=self.model.backbone_out_channels,
            )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

        self._obs_steps_seen = 0
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        # ------------------------------------------------------------------ #
        # 1. Pre-process the incoming observation (normalise + image stacking)
        # ------------------------------------------------------------------ #
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        # Normalize the batch inputs
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self.config.image_features], dim=-4
            )

        # ------------------------------------------------------------------ #
        # 2. Initialise / update rolling queues of past observations
        # ------------------------------------------------------------------ #
        # Belt and suspenders approach to ensure that the action is not added to the queues
        # Populate queues ensures that the queues are initialised and updated with the latest observations
        # For the first timestep it copies the first observation several times
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        self._obs_steps_seen = min(self._obs_steps_seen + 1, self.config.n_obs_steps) # Increment the number of observations seen

        # ------------------------------------------------------------------ #
        # 3. Branch A – temporal ensembling (no action queue needed)
        # Stacks observations inside
        # ------------------------------------------------------------------ #
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk()
            action = self.temporal_ensembler.update(actions)
            return action

        # ------------------------------------------------------------------ #
        # 4. Branch B – uses up actions in the queue then calls the model
        # ------------------------------------------------------------------ #
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk()[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _stack_obs_from_queues(self, queues: dict[str, deque]) -> dict[str, Tensor]:
        """Materialize the last T=n_obs_steps as a time axis (B, T, ...)."""
        assert queues is not None, "Queues must be initialised before calling this method"
        stacked = {}
        for k, dq in queues.items():
            if k == ACTION:
                continue
            # dq is a deque of length n_obs_steps with per-step tensors shaped like the incoming batch
            stacked[k] = torch.stack(list(dq), dim=1)
        return stacked

    def _masked_mean(self, x: Tensor, mask: Tensor, dim: int = 1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
        """Compute the mean of x along the time axis, ignoring the masked values."""
        mask = mask.to(dtype=x.dtype, device=x.device)
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        weighted_x = x * mask
        return weighted_x.sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).clamp(min=eps)

    def _time_mask(self, b: int) -> Tensor:
        """(B, T) mask with ones for valid timesteps, zeros for duplicated/padded."""
        t = self.config.n_obs_steps
        k = min(self._obs_steps_seen, t)
        device = get_device_from_parameters(self)
        mask = torch.zeros(b, t, dtype=torch.bool, device=device)
        mask[:, :k] = 1
        return mask

    # def _temporal_pool(self, hist: dict[str, Tensor], t_mask: Tensor) -> dict[str, Tensor]:
    #     """Pool the history of observations to a single vector.
    #     This is a simple average pooling over the time axis.
    #     Args:
    #         hist: Dictionary of tensors with a time axis.
    #     Returns:
    #         Dictionary of tensors without a time axis.
    #     """
    #     pooled = {}
    #     for k, x in hist.items():
    #         if k in {OBS_STATE, OBS_ENV_STATE}:
    #             assert x.ndim >= 3 and x.size(1) == self.config.n_obs_steps
    #             pooled[k] = self._masked_mean(x, t_mask, dim=1, keepdim=False)
    #         elif k in {OBS_IMAGES}:
    #             assert x.ndim >= 5 and x.size(1) == self.config.n_obs_steps
    #             pooled[k] = self._masked_mean(x, t_mask, dim=1, keepdim=False)
    #         else:
    #             pooled[k] = x
    #     return pooled

    def _temporal_pool(self, hist: dict[str, Tensor], t_mask: Tensor) -> dict[str, Tensor]:
        """Pool the history of observations to a single vector.
        This is a temporal attention pooling over the time axis. It uses a learnable query to attend to the history of observations.
        Args:
            hist: Dictionary of tensors with a time axis.
            t_mask: Time mask to ignore padded values.
        Returns:
            Dictionary of tensors without a time axis.
        """
        pooled = {}
        for k, x in hist.items():
            if k in {OBS_STATE, OBS_ENV_STATE}:
                pooled[k] = self.temporal_attn_pool_1d(x, t_mask)
            elif k in {OBS_IMAGES}:
                # Loop through n_cams and pool each camera independently
                pooled[k] = []
                for i in range(x.shape[2]): # Reminder: x.shape = (B, T, n_cams, C, H, W)
                    # Reminder: hist[OBS_STATE].shape = (B, T, D)
                    current_robot_state = hist[OBS_STATE][:, -1] # Use the last robot state for each camera
                    pooled[k].append(self.temporal_attn_pool_2d(x=x[:, :, i], context=current_robot_state, mask=t_mask))
                pooled[k] = torch.stack(pooled[k], dim=1)  # (B, n_cams, C, H, W)
            else:
                pooled[k] = x
        return pooled

    @torch.no_grad()
    def predict_action_chunk(self) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        hist = self._stack_obs_from_queues(self._queues)
        b = next(iter(hist.values())).shape[0]
        t_mask = self._time_mask(b)
        hist = self._temporal_pool(hist, t_mask)

        # Convert images to list format
        if self.config.image_features and OBS_IMAGES in hist:
            hist = dict(hist)  # shallow copy
            # hist[OBS_IMAGES] already contains the pooled images (B, n_cams, C, H, W)
            # Convert to list format: (B, n_cams, C, H, W) -> list of (B, C, H, W)
            hist[OBS_IMAGES] = [hist[OBS_IMAGES][:, i] for i in range(hist[OBS_IMAGES].shape[1])]

        actions = self.model(hist)[0]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # 1) Normalize inputs first so pooling works on normalized values
        batch = self.normalize_inputs(batch)

        # 2) Build time-stacked hist like inference expects
        hist: dict[str, Tensor] = {}

        # State: (B, T, D) or (B, D)
        if OBS_STATE in batch:
            x = batch[OBS_STATE]
            if x.ndim == 2:  # no time axis
                x = x.unsqueeze(1)  # (B, 1, D)
            hist[OBS_STATE] = x

        # Env state (optional): (B, T, E) or (B, E)
        if OBS_ENV_STATE in batch:
            x = batch[OBS_ENV_STATE]
            if x.ndim == 2:
                x = x.unsqueeze(1)  # (B, 1, E)
            hist[OBS_ENV_STATE] = x

        # Images: stack per camera into (B, T, n_cams, C, H, W)
        if self.config.image_features:
            cams = []
            for cam_key in self.config.image_features:
                img = batch[cam_key]
                if img.ndim == 4:   # (B, C, H, W) -> (B, 1, C, H, W)
                    img = img.unsqueeze(1)
                cams.append(img)     # each (B, T, C, H, W)
            hist[OBS_IMAGES] = torch.stack(cams, dim=2)  # (B, T, n_cams, C, H, W)

        # 3) Derive a valid time mask (B, T) from any available *_is_pad (prefer state)
        valid_mask = None
        mask_key_candidates = []
        if OBS_STATE in hist:
            mask_key_candidates.append(f"{OBS_STATE}_is_pad")
        if self.config.image_features:
            for key in self.config.image_features:
                mask_key_candidates.append(f"{key}_is_pad")
        if OBS_ENV_STATE in hist:
            mask_key_candidates.append(f"{OBS_ENV_STATE}_is_pad")

        for mk in mask_key_candidates:
            if mk in batch:
                valid_mask = ~batch[mk]  # invert pad -> valid
                break
        if valid_mask is None:
            # No pad mask present; fall back to all valid
            b, t = next(iter(hist.values())).shape[:2]
            valid_mask = torch.ones(b, t, dtype=torch.bool, device=next(iter(hist.values())).device)

        # 4) Temporal pool using the same logic as inference
        pooled = self._temporal_pool(hist, valid_mask)

        # 5) Repack images to the model’s expected list format and update batch
        if OBS_STATE in pooled:
            batch[OBS_STATE] = pooled[OBS_STATE]  # (B, D)
        if OBS_ENV_STATE in pooled:
            batch[OBS_ENV_STATE] = pooled[OBS_ENV_STATE]  # (B, E)
        if OBS_IMAGES in pooled:
            # pooled[OBS_IMAGES]: (B, n_cams, C, H, W) -> list of (B, C, H, W)
            batch[OBS_IMAGES] = [pooled[OBS_IMAGES][:, i] for i in range(pooled[OBS_IMAGES].shape[1])]

        # 6) Normalize targets and run model
        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # 7) Loss (masking unchanged)
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch[f"{ACTION}_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

class TemporalAttentionPool1D(nn.Module):
    """
    Pools a temporal sequence (B, T, D(feature dimension)) -> (B, D (model dimension)) with a learnable query.
    'mask': (B, T) with True=valid will be converted for the attention mask.
    """
    def __init__(self, config: DACTConfigA):
        super().__init__()
        self.query =  nn.Parameter(torch.randn(1, 1, config.dim_model)) # (1, 1, D)
        self.attn = nn.MultiheadAttention(num_heads=config.n_heads, embed_dim=config.dim_model, batch_first=True)
        # Time positions correspond to observation history length, not chunked action length
        self.time_pos_embed = nn.Embedding(config.n_obs_steps, config.dim_model) # (T, D)
        # Support projecting either robot state or env state into model dim
        self._robot_dim = (
            config.robot_state_feature.shape[0] if config.robot_state_feature is not None else None
        )
        self._env_dim = (
            config.env_state_feature.shape[0] if config.env_state_feature is not None else None
        )
        if self._robot_dim is not None:
            self.robot_state_input_proj = nn.Linear(self._robot_dim, config.dim_model)
            self.robot_state_output_proj = nn.Linear(config.dim_model, self._robot_dim)
        else:
            self.robot_state_input_proj = None
            self.robot_state_output_proj = None
        if self._env_dim is not None:
            self.env_state_input_proj = nn.Linear(self._env_dim, config.dim_model)
            self.env_state_output_proj = nn.Linear(config.dim_model, self._env_dim)
        else:
            self.env_state_input_proj = None
            self.env_state_output_proj = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x: (B, T, D_in), mask: (B, T) True=valid
        # Project input to model dimension depending on feature type
        d_in = x.size(-1)
        if self._robot_dim is not None and d_in == self._robot_dim and self.robot_state_input_proj is not None:
            x = self.robot_state_input_proj(x)
        elif self._env_dim is not None and d_in == self._env_dim and self.env_state_input_proj is not None:
            x = self.env_state_input_proj(x)
        elif d_in == self.attn.embed_dim:
            # Already in model dimension
            pass
        else:
            raise ValueError(
                f"TemporalAttentionPool1D received input with dim {d_in}, which does not match robot ({self._robot_dim}) "
                f"or env ({self._env_dim}) dims, and is not equal to model dim ({self.attn.embed_dim})."
            )
        b, t, d = x.shape

        # Fetch the first T positional vectors and broadcast across the batch
        device = get_device_from_parameters(self)
        t_ids = torch.arange(t, device=device)
        t_pos_embed = self.time_pos_embed(t_ids).unsqueeze(0).expand(b, t, d)

        x = x + t_pos_embed

        kpm = None if mask is None else ~mask
        # Note: Expanded to batch size
        # The query encodes a trainable notion of "how should I pool a sequence into one vector?"
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=kpm)
        out = out.squeeze(1) # (B, D_model)

        # Project back to original dimension
        if self._robot_dim is not None and d_in == self._robot_dim and self.robot_state_output_proj is not None:
            out = self.robot_state_output_proj(out)
        elif self._env_dim is not None and d_in == self._env_dim and self.env_state_output_proj is not None:
            out = self.env_state_output_proj(out)
        # If already in model dimension, keep as is

        return out # (B, D_original)

class TemporalAttentionPool2D(nn.Module):
    """
    Pools a stack of feature maps (B, T, C, H, W) -> (B, C, H, W) with frame-level attention.
    Weights are derived from global pooled descriptors (B, T, C).
    """
    def __init__(self, config: DACTConfigA, backbone: nn.Module, backbone_out_channels: int):
        super().__init__()
        # Use a MLP to create a query informed by the robot state
        self.query = nn.Sequential(
            nn.Linear(config.robot_state_feature.shape[0], config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )
        self.attn = nn.MultiheadAttention(num_heads=config.n_heads, embed_dim=config.dim_model, batch_first=True)
        # Time positions correspond to observation history length
        self.time_pos_embed = nn.Embedding(config.n_obs_steps, config.dim_model) # (T, D)
        # Use the shared CNN backbone to extract per-frame features
        self.backbone = backbone
        self.backbone_out_channels = backbone_out_channels
        # Project frame descriptors from backbone channels to model dimension for attention
        self.frame_feat_proj = nn.Linear(self.backbone_out_channels, config.dim_model)

    def forward(self, x: Tensor, context: Tensor, mask: Tensor | None = None) -> Tensor:
        # x: (B, T, C, H, W), mask: (B, T) True=valid
        b, t, c, h, w = x.shape

        # 1) Extract per-frame CNN features using the shared backbone
        x_bt = x.reshape(b * t, c, h, w)                       # (B*T, C, H, W)
        feat = self.backbone(x_bt)["feature_map"]              # (B*T, C_b, H_b, W_b)
        h_b, w_b = feat.shape[2], feat.shape[3]                # Extract spatial dimensions

        # 2) Frame-level descriptors for attention weights
        g = feat.mean(dim=(2, 3))                              # (B*T, C_b)
        g = self.frame_feat_proj(g)                            # (B*T, D)
        g = g.view(b, t, -1)                                   # (B, T, D)

        # 3) Temporal attention to get per-frame weights
        device = get_device_from_parameters(self)
        t_ids = torch.arange(t, device=device)
        g = g + self.time_pos_embed(t_ids).unsqueeze(0).expand(b, t, g.size(-1))
        kpm = None if mask is None else ~mask
        q = self.query(context).unsqueeze(1) # (B, 1, D)
        _, attn = self.attn(q, g, g, key_padding_mask=kpm)
        # attn: (B, 1, T) softmax over time t
        w = attn.squeeze(1).view(b, t, 1, 1, 1) # (B, T, 1, 1, 1)

        # 4) Pool in feature space
        feat = feat.view(b, t, self.backbone_out_channels, h_b, w_b)
        out = (feat * w).sum(dim=1) # (B, C_b, H_b, W_b)
        return out # (B, C_b, H_b, W_b)

class DACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: DACTConfigA):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            # Expose backbone output channels for consumers
            self.backbone_out_channels = backbone_model.fc.in_features

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self.backbone_out_channels, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]) # (B, D)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch[f"{ACTION}_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for cam_features in batch[OBS_IMAGES]:
                # Assume we already have the backbone features
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: DACTConfigA, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: DACTConfigA):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: DACTConfigA):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: DACTConfigA):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
