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

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.dact.configuration_dact_a import DACTConfigA
from lerobot.policies.dact.mamba_policy import Mamba2, FrozenDinov2, CrossCameraAttention, CrossModalAttention
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy

HISTORY_TOKEN = "history_token"


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

        self.model = DACT(config)

        # History encoder
        self.history_encoder = HistoryEncoder(
            config=config,
        )
        # Wire shared modules so HistoryEncoder can reuse them
        if self.config.image_features:
            self.history_encoder.backbone = self.backbone
            self.history_encoder.model = self.model
            self.history_encoder.encoder_cam_feat_pos_embed = self.model.encoder_cam_feat_pos_embed
            self.history_encoder.encoder_img_feat_input_proj = self.model.encoder_img_feat_input_proj

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

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
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.history_cache = None
        # Debug: track reset calls
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Convert images to list format for model consumption
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Compute/update history token h_t
        x_t = self.history_encoder.fuse_obs_to_history_vec(batch)
        if self.history_cache is None:
            self.history_cache = self.history_encoder.init_cache(x_t.shape[0])
        h_t, self.history_cache = self.history_encoder.step(x_t, self.history_cache)
        batch[HISTORY_TOKEN] = h_t

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Compute/update history token h_t
        x_t = self.history_encoder.fuse_obs_to_history_vec(batch)
        if self.history_cache is None:
            self.history_cache = self.history_encoder.init_cache(x_t.shape[0])

        h_t, self.history_cache = self.history_encoder.step(x_t, self.history_cache)
        batch[HISTORY_TOKEN] = h_t

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # Compute L1 loss only over non-padded actions
        # Mask shape: (B, chunk_size, 1) where True = valid (not padded)
        action_mask = ~batch["action_is_pad"].unsqueeze(-1)
        l1_per_element = F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
        # Normalize by number of valid elements, not total elements
        l1_loss = l1_per_element.sum() / action_mask.sum().clamp(min=1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


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
            # Projection for history token into encoder hidden dimension
            self.vae_encoder_history_input_proj = nn.Linear(config.dim_model, config.dim_model)
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + 1 + config.chunk_size  # +1 for history token
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
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        # +1 history token
        n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        # Projection for history token (B, D) -> (B, D)
        self.encoder_history_input_proj = nn.Linear(config.dim_model, config.dim_model)
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

        # print(f"observation.state size: {batch['observation.state'].shape}")
        # print(f"observation.images size: {batch['observation.images'][0].shape}")
        # print(f"action size: {batch['action'].shape}")

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

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
            history_embed = self.vae_encoder_history_input_proj(batch[HISTORY_TOKEN])
            hist_embed = history_embed.unsqueeze(1)  # (B, 1, D)

            # print(f"robot_state_embed.shape: {robot_state_embed.shape}")
            # print(f"hist_embed.shape: {hist_embed.shape}")
            # print(f"action_embed.shape: {action_embed.shape}")
            # print(f"cls_embed.shape: {cls_embed.shape}")

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, hist_embed, action_embed]  # (B, S+3, D)
            else:
                vae_encoder_input = [cls_embed, hist_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, (2 if self.config.robot_state_feature else 1) + 1),  # +1 for history
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
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        # History token
        hist_token = self.encoder_history_input_proj(batch[HISTORY_TOKEN])
        encoder_in_tokens.append(hist_token)

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
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

class HistoryEncoder(nn.Module):
    """Tiny recurrent history encoder based on a single Mamba2 block.

    Exposes step() with cached states for online inference and forward() for
    sequence processing in training.
    """

    def __init__(self, config: DACTConfigA):
        super().__init__()
        # One Mamba2 mixer operating at dim_model
        self.mamba = Mamba2(
            d_model=config.dim_model,
            d_state=config.history_d_state,
            d_conv=config.history_d_conv,
            expand=config.history_expand,
            headdim=config.history_headdim,
            chunk_size=config.chunk_size,
            use_mem_eff_path=config.history_use_mem_eff_path,
        )
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(config.dinov2_dim, 512, 3, padding=1), #[B, 512, 45, 34]
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), #[B, 256, 45, 34]
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.Flatten(1),
            nn.Linear(128*23*18, config.dim_model),  # (B, D)
            nn.LayerNorm(config.dim_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10)
        )

        self.config = config
        self.num_cameras = config.num_cameras
        self.hist_backbone = FrozenDinov2(config)
        self.in_dim = config.dim_model * self.num_cameras
        self.cross_camera_attn = self.cross_cam_attn = CrossCameraAttention(config)
        self.cross_modal_attn = CrossModalAttention(config)

        self.encoder_history_input_proj = nn.Linear(self.in_dim, config.dim_model)

    def init_cache(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return self.mamba.allocate_inference_cache(batch_size=batch_size, max_seqlen=1, dtype=None)

    def step(
        self,
        x_t: Tensor, # (B, D)
        cache: tuple[Tensor, Tensor],
        # valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run one recurrent step.

        Args:
            x_t: (B, D)
            cache: tuple(conv_state, ssm_state)
            valid_mask: optional (B,) bool; if False for an item, cache is preserved
        Returns:
            h_t: (B, D) and updated cache
        """
        # conv_state is a (B, D_conv) tensor
        # ssm_state is a (B, nheads, headdim, d_state) tensor
        # These tensors are used to store the state of the Mamba2 block
        conv_state, ssm_state = cache
        # Treat prior cache as constants
        conv_state = conv_state.detach()
        ssm_state = ssm_state.detach()
        h_t, new_conv, new_ssm = self.mamba.step(x_t.unsqueeze(1), conv_state, ssm_state)
        # if valid_mask is not None:
        #     # Preserve states for invalid items
        #     keep = ~valid_mask

        #     if keep.any():
        #         # Update the conv_state and ssm_state for the valid items
        #         new_conv = torch.where(
        #             keep.view(-1, 1, 1), conv_state, new_conv
        #         )
        #         new_ssm = torch.where(
        #             keep.view(-1, 1, 1, 1), ssm_state, new_ssm
        #         )
        # Return the updated conv_state and ssm_state
        return h_t, (new_conv.detach(), new_ssm.detach())

    def forward(self, x_seq: Tensor) -> Tensor:
        """Process a full sequence.

        Args:
            x_seq: (B, D)
        Returns:
            h_seq: (B, D)
        """
        return self.mamba(x_seq)

    def fuse_obs_to_history_vec(self, batch: dict[str, Tensor]) -> Tensor:
        """Fuse per-step observation into a rich vector x_t for the HistoryEncoder.
        """
        # 1
        features = []
        for img in batch[OBS_IMAGES]:
            raw_img_features = self.hist_backbone(img) # (B, D, H_patch, W_patch)
            img_features = self.spatial_adapter(raw_img_features)
            features.append(img_features)

        cam_features = torch.cat(features, dim=1)

        # If there are multiple cameras, we need to pool them together
        if self.num_cameras > 1:
            cam_features = self.cross_camera_attn(
                cam_features.unsqueeze(1),
                cam_features.unsqueeze(1),
                cam_features.unsqueeze(1),
            ).squeeze(1)

        # 2
        if self.config.robot_state_feature:
            low_dim_features = batch[OBS_STATE].unsqueeze(1)  # (B, 1, D)
            cam_features_proj = self.encoder_history_input_proj(cam_features) # (B, D)
            fused_features = self.cross_modal_attn(
                query=cam_features_proj.unsqueeze(1),
                key=low_dim_features,
                value=low_dim_features,
            ).squeeze(1)

        x_t = fused_features # (B, D)

        return x_t
