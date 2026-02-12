from collections import deque
from itertools import chain
from typing import Any

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

try:
    import dinov2  # noqa: F401

    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
)
from lerobot.policies.mact.configuration_mact import MACTConfig
from lerobot.policies.mact.cross_attention import CrossCameraAttention
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

HISTORY_TOKEN = "history_token"


class MACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = MACTConfig
    name = "mact"

    def __init__(
        self,
        config: MACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = MACT(config)

        self.history_encoder = HistoryEncoder(config)

        self._queues = None

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
        self._queues = {
            OBS_STATE: deque[Any](maxlen=self.config.n_obs_steps),
            ACTION: deque[Any](maxlen=self.config.n_action_steps)
            if self.config.temporal_ensemble_coeff is None
            else None,
        }
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque[Any](maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque[Any](maxlen=self.config.n_obs_steps)
        self._mamba_cache = None
        self._history_tokens = deque[Tensor](maxlen=self.config.n_history_tokens)
        self._inference_step_counter = 0  # Track steps for stride-based history updates

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # Keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict[str, Tensor](batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Mirrors the logic in forward() but for single timestep inference.
        """
        self.eval()

        # Prepare batch - shallow copy to avoid modifying input
        model_batch = dict[str, Tensor](batch)

        # Stack images along camera dimension: (B, N_cam, C, H, W)
        if self.config.image_features:
            model_batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Process through history encoder
        # Initialize cache on first call
        if self._mamba_cache is None:
            self._mamba_cache = self.history_encoder.init_cache(
                batch_size=model_batch[OBS_IMAGES].shape[0],
                dtype=model_batch[OBS_IMAGES].dtype,
            )
            self._strided_obs_idx = 0  # Track how many strided observations we've processed

        # Only update history on stride boundaries (matches training)
        if self._inference_step_counter % self.config.observation_stride == 0:
            # Get spatial tokens (no pooling yet - step() handles that after Mamba)
            x_t = self.history_encoder.fuse_one_timestep(
                model_batch[OBS_IMAGES],
                timestep_idx=self._strided_obs_idx,
            )  # (B, N, D)
            # Process through Mamba (one step per patch, then pool)
            h_t, self._mamba_cache = self.history_encoder.step(
                x_t, self._mamba_cache, timestep_idx=self._strided_obs_idx
            )

            self._history_tokens.append(h_t.detach())  # h_t is (B, k, D)
            self._strided_obs_idx += 1

        self._inference_step_counter += 1
        history_tokens = list(self._history_tokens)
        # Each entry is (B, k, D), concatenate to (B, n_frames*k, D)
        model_batch[HISTORY_TOKEN] = torch.cat(history_tokens, dim=1)

        # Get action predictions from model
        actions = self.model(model_batch)[0]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""

        # Prepare batch for model input
        batch = dict[str, Tensor](batch)  # shallow copy

        # Stack images for model input if needed
        if self.config.image_features:
            # Stack images along camera dimension for model input: (B, L, N_cam, C, H, W)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Prepare batch for history encoder

        # Input: images (B, L, n_cameras, C, H, W), states (B, L, D_state)
        # Output: (B, L*k, D) where k = n_spatial_tokens
        h_seq = self.history_encoder.forward(batch)

        # Extract the last n_history_tokens * n_spatial_tokens for decoder conditioning
        tokens_per_frame = self.config.n_spatial_tokens
        n_tokens = min(self.config.n_history_tokens * tokens_per_frame, h_seq.shape[1])
        batch[HISTORY_TOKEN] = h_seq[:, -n_tokens:, :]  # (B, n_tokens, D)

        # Create model batch with only the most recent timestep
        # History encoder uses full sequences, but main model uses only last timestep
        model_batch = {HISTORY_TOKEN: batch[HISTORY_TOKEN]}

        # Extract last timestep from sequences for main model
        if OBS_IMAGES in batch:
            # batch[OBS_IMAGES] is (B, L, N_cam, C, H, W) -> take last timestep (B, N_cam, C, H, W)
            model_batch[OBS_IMAGES] = batch[OBS_IMAGES][:, -1, ...]

        if OBS_STATE in batch:
            # batch[OBS_STATE] is (B, L, D_state) -> take last timestep (B, D_state)
            model_batch[OBS_STATE] = batch[OBS_STATE][:, -1, ...]

        if OBS_ENV_STATE in batch:
            # batch[OBS_ENV_STATE] is (B, L, D_env) -> take last timestep (B, D_env)
            model_batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE][:, -1, ...]

        if ACTION in batch:
            # batch[ACTION] is (B, chunk_size, action_dim) - no sequence dimension for actions
            # Actions are the target predictions, not part of history sequence
            model_batch[ACTION] = batch[ACTION]
            # Also copy the action padding mask
            if "action_is_pad" in batch:
                model_batch["action_is_pad"] = batch["action_is_pad"]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)

        # Compute loss using the last timestep's action
        l1_loss = (
            F.l1_loss(model_batch[ACTION], actions_hat, reduction="none")
            * ~model_batch["action_is_pad"].unsqueeze(-1)
        ).mean()

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


class MACT(nn.Module):
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

    def __init__(self, config: MACTConfig):
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
            # dimension. Includes: cls + (robot_state) + chunk_size + n_history_tokens * n_spatial_tokens
            num_input_token_encoder = (
                1 + config.chunk_size + config.n_history_tokens * config.n_spatial_tokens
            )
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
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
        # [history_tokens, latent, (robot_state), (env_state), (image_feature_map_pixels)].
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
        # Note: history tokens get their own separate positional embeddings
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Positional embeddings for history tokens (used as encoder inputs)
        n_hist_pos_embed = config.n_history_tokens * config.n_spatial_tokens
        self.history_pos_embed = nn.Embedding(n_hist_pos_embed, config.dim_model)

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

        # Get batch size - batch[OBS_IMAGES] is now (B, N_cam, C, H, W), not a list
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES].shape[0]
        elif OBS_STATE in batch:
            batch_size = batch[OBS_STATE].shape[0]
        elif OBS_ENV_STATE in batch:
            batch_size = batch[OBS_ENV_STATE].shape[0]
        else:
            raise ValueError("Batch must contain at least one observation")

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                # Ensure robot_state_embed is (B, 1, D)
                if robot_state_embed.dim() == 3:
                    robot_state_embed = robot_state_embed[:, -1:, :]  # (B, 1, D)
                else:
                    robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)

            # Ensure ACTION has chunk_size dimension: (B, chunk_size, action_dim)
            action_input = batch[ACTION]
            if action_input.dim() == 2:
                # ACTION is (B, action_dim) - missing chunk dimension
                # This likely means the dataloader provides actions for a single timestep
                # Reshape to (B, 1, action_dim) to match expected VAE encoder input
                action_input = action_input.unsqueeze(1)  # (B, 1, action_dim)
            elif action_input.dim() == 3:
                # ACTION is already (B, chunk_size, action_dim) - correct format
                pass
            else:
                raise ValueError(f"Unexpected ACTION shape: {action_input.shape}")

            history_embed = batch[HISTORY_TOKEN]  # (B, n_tokens, D)

            action_embed = self.vae_encoder_action_input_proj(action_input)  # (B, chunk_size, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [
                    cls_embed,
                    robot_state_embed,
                    action_embed,
                    history_embed,
                ]  # (B, 1+1+chunk_size+n_tokens, D)
            else:
                vae_encoder_input = [cls_embed, action_embed, history_embed]  # (B, 1+chunk_size+n_tokens, D)
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, 1+chunk_size+n_tokens, D)

            # Prepare key padding mask for the transformer encoder.
            # Tokens: [cls, (robot_state), actions, history_tokens]
            # cls, robot_state, and history tokens are never padded (False)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            history_is_pad = torch.full(
                (batch_size, history_embed.shape[1]),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"], history_is_pad], axis=1
            )  # (bs, 1 or 2 + chunk_size + n_history_tokens)

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
        # Start with history tokens
        history_cond = batch[HISTORY_TOKEN]  # (B, n_tokens, D)
        n_tokens = history_cond.shape[1]
        history_cond_seq = history_cond.transpose(0, 1)  # (n_tokens, B, D)

        encoder_in_tokens = []
        encoder_in_pos_embed = []

        # Add history tokens and their positional embeddings
        for i in range(n_tokens):
            encoder_in_tokens.append(history_cond_seq[i])
            encoder_in_pos_embed.append(self.history_pos_embed.weight[-n_tokens + i].unsqueeze(0))

        # Add latent token
        encoder_in_tokens.append(self.encoder_latent_input_proj(latent_sample))
        encoder_in_pos_embed.extend(list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)))

        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # batch[OBS_IMAGES] has shape (B, N_cam, C, H, W)
            # Iterate over each camera
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            n_cameras = batch[OBS_IMAGES].shape[1]
            for cam_idx in range(n_cameras):
                cam_img = batch[OBS_IMAGES][:, cam_idx]  # (B, C, H, W)
                cam_features = self.backbone(cam_img)["feature_map"]
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
        # Standard decoder cross-attention with encoder outputs (which now include history tokens)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            encoder_pos_embed=encoder_in_pos_embed,
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


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


class ImageEncoder(nn.Module):
    """Base class for image encoders that extract features from images."""

    def forward(self, images: Tensor) -> Tensor:
        """Encode images to feature tokens.

        Args:
            images: (B, C, H, W) batch of images
        Returns:
            (B, num_tokens, D) feature tokens
        """
        raise NotImplementedError


class ResNetImageEncoder(ImageEncoder):
    """ResNet-based image encoder with spatial tokens."""

    def __init__(self, config: MACTConfig):
        super().__init__()
        self.config = config
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[
                False,
                False,
                config.replace_final_stride_with_dilation,
            ],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Spatial adapter: preserve spatial features as 49 tokens (7x7 grid)
        backbone_out_channels = backbone_model.fc.in_features
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                backbone_out_channels,
                config.spatial_adapter_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.spatial_adapter_hidden_dim,
                config.dim_model,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )
        # Per-token processing after flattening spatial dims
        self.spatial_token_proj = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.ReLU(inplace=True),
            nn.Dropout(config.spatial_adapter_dropout),
        )

        if config.freeze_history_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, n_tokens, D) - all spatial tokens from 7x7 feature map."""
        if self.config.freeze_history_backbone:
            with torch.no_grad():
                features = self.backbone(images)["feature_map"]  # (B, C, H, W)
        else:
            features = self.backbone(images)["feature_map"]  # (B, C, H, W)

        # Apply conv layers
        features = self.spatial_conv(features)  # (B, D, H, W)

        # Flatten spatial dims to sequence: (B, D, H, W) -> (B, H*W, D)
        features = einops.rearrange(features, "b d h w -> b (h w) d")

        # Apply per-token projection
        tokens = self.spatial_token_proj(features)  # (B, H*W, D)

        return tokens


class DinoV2ImageEncoder(ImageEncoder):
    """DINOv2-based image encoder that preserves spatial patch tokens."""

    def __init__(self, config: MACTConfig):
        super().__init__()
        self.config = config
        if not DINOV2_AVAILABLE:
            raise ImportError(
                "DINOv2 is not available. Install with: pip install dinov2 or use torch.hub.load"
            )

        # Load DINOv2 model from torch hub
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", config.vision_backbone)

        # Get embedding dimension based on model variant
        dinov2_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        self.embed_dim = dinov2_dims.get(config.vision_backbone, 768)

        # Project to model dimension if needed
        if self.embed_dim != config.dim_model:
            self.proj = nn.Linear(self.embed_dim, config.dim_model)
        else:
            self.proj = nn.Identity()

        # Freeze backbone if requested
        if config.freeze_history_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, num_patches, D) - all spatial patch tokens."""
        # DINOv2 expects float images in [0, 1]
        if images.dtype == torch.uint8:
            images = images.float().div_(255)

        # Get tokens from DINOv2: (B, num_patches+1, embed_dim)
        if self.config.freeze_history_backbone:
            with torch.no_grad():
                tokens = self.dinov2.forward_features(images)
        else:
            tokens = self.dinov2.forward_features(images)

        # Extract patch tokens (excluding CLS token at position 0)
        patch_tokens = tokens["x_norm_patchtokens"] if isinstance(tokens, dict) else tokens[:, 1:]

        # Project to model dimension
        projected = self.proj(patch_tokens)  # (B, num_patches, D)

        return projected


def _create_mamba_block(config: MACTConfig, layer_idx: int) -> Block:
    """Create a Mamba2 block using the official Block wrapper.

    Args:
        config: MACT configuration
        layer_idx: Index of this layer in the stack
    Returns:
        Official Block wrapping a Mamba2 mixer with optional MLP
    """
    d = config.dim_model

    # Mamba2 SSM head dimension (NOT the same as attention headdim).
    # With expand=2: d_inner = 2*512 = 1024, headdim=128 → nheads=8.
    mamba_headdim = 128
    mamba_d_state = 512

    mixer_cls = lambda dim: Mamba2(  # noqa: E731
        d_model=dim,
        d_state=mamba_d_state,
        headdim=mamba_headdim,
        ngroups=1,
        dt_max=0.02,
        layer_idx=layer_idx,
    )

    if config.history_use_mlp:
        mlp_cls = lambda dim: nn.Sequential(  # noqa: E731
            nn.Linear(dim, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, dim),
        )
    else:
        mlp_cls = nn.Identity

    return Block(
        dim=d,
        mixer_cls=mixer_cls,
        mlp_cls=mlp_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
    )


class MultiQueryPooling(nn.Module):
    """Multi-query attention pooling for spatial tokens.

    Uses k learnable queries to preserve spatial structure instead of
    compressing to a single token. This helps prevent mode collapse by
    retaining more fine-grained spatial information.
    """

    def __init__(self, config: MACTConfig):
        super().__init__()
        self.n_queries = config.n_spatial_tokens
        self.queries = nn.Parameter(
            torch.randn(1, self.n_queries, config.dim_model) * (1 / np.sqrt(config.dim_model))
        )
        self.attn = nn.MultiheadAttention(
            config.dim_model,
            num_heads=config.n_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x: Tensor) -> Tensor:
        """Summarize spatial tokens using multi-query attention.

        Args:
            x: (B, N_tokens, D) spatial tokens
        Returns:
            (B, n_queries, D) summarized representations preserving spatial structure
        """
        queries = self.queries.expand(x.shape[0], -1, -1)  # (B, k, D)
        summary, _ = self.attn(queries, x, x)  # (B, k, D)
        return self.norm(queries + summary)  # (B, k, D)


class SpatialTransformer(nn.Module):
    """Per-frame spatial self-attention over patches.

    Processes spatial relationships within each frame independently,
    preserving temporal structure. Used in spatial-then-temporal architecture.
    """

    def __init__(self, config: MACTConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_spatial_attn_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial self-attention to each frame independently.

        Args:
            x: (B, L, N, D) where L=frames, N=patches per frame
        Returns:
            (B, L, N, D) with spatial attention applied per-frame
        """
        B, L, N, D = x.shape  # noqa: N806
        # Reshape to process all frames as a single batch
        x = x.view(B * L, N, D)  # (B*L, N, D)
        x = self.encoder(x)  # Self-attention over patches
        return x.view(B, L, N, D)  # (B, L, N, D)


class PositionalEmbedding3D(nn.Module):
    """3D positional embeddings combining spatial (patch position) + temporal (frame index).

    Spatial embeddings are learned per-patch position within a frame.
    Temporal embeddings are learned per-frame position within the sequence.
    Both are added to the input tokens via broadcasting.
    """

    def __init__(self, n_patches: int, n_frames: int, dim_model: int):
        super().__init__()
        # Learnable spatial embeddings: one per patch position
        self.spatial = nn.Parameter(torch.randn(1, 1, n_patches, dim_model) * 0.02)
        # Learnable temporal embeddings: one per frame position
        self.temporal = nn.Parameter(torch.randn(1, n_frames, 1, dim_model) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add 3D positional embeddings to spatiotemporal tokens.

        Args:
            x: (B, L, N, D) spatiotemporal tokens where L=frames, N=patches
        Returns:
            (B, L, N, D) with positional embeddings added
        """
        B, L, N, D = x.shape  # noqa: N806
        # Slice to actual sequence/patch lengths (handles variable sizes)
        spatial = self.spatial[:, :, :N, :]  # (1, 1, N, D)
        temporal = self.temporal[:, :L, :, :]  # (1, L, 1, D)
        return x + spatial + temporal  # Broadcasting adds both

    def get_embeddings_for_timestep(self, n_patches: int, timestep_idx: int) -> tuple[Tensor, Tensor]:
        """Get spatial and temporal embeddings for a single timestep (inference).

        Args:
            n_patches: Number of patches for this frame
            timestep_idx: Current timestep index (for temporal embedding)
        Returns:
            spatial: (1, N, D) spatial embeddings
            temporal: (1, 1, D) temporal embedding for this timestep
        """
        spatial = self.spatial[:, 0, :n_patches, :]  # (1, N, D)
        # Use modular indexing for long-horizon inference
        temporal_idx = timestep_idx % self.temporal.shape[1]
        temporal = self.temporal[:, temporal_idx : temporal_idx + 1, 0, :]  # (1, 1, D)
        return spatial, temporal


class HistoryEncoder(nn.Module):
    """Recurrent history encoder based on stacked Mamba2 blocks.

    Exposes step() with cached states for online inference and forward() for
    sequence processing in training.
    """

    def __init__(self, config: MACTConfig):
        super().__init__()
        self.config = config

        # Stack of Mamba2 blocks controlled by n_mamba2_layers config parameter
        self.blocks = nn.ModuleList(
            [_create_mamba_block(config, layer_idx=i) for i in range(config.n_mamba2_layers)]
        )

        # Image encoder (ResNet or DINOv2)
        if self.config.image_features:
            if config.vision_backbone.startswith("dinov2"):
                self.image_encoder = DinoV2ImageEncoder(config)
            else:
                self.image_encoder = ResNetImageEncoder(config)

        self.cross_camera_attn = self.cross_cam_attn = CrossCameraAttention(config)

        self.encoder_history_input_proj = nn.Linear(config.dim_model, config.dim_model)

        # Spatial-then-temporal architecture:
        # Step 1: Spatial transformer (self-attention over patches per frame)
        self.spatial_transformer = SpatialTransformer(config)
        # Step 2: Multi-query pooling (compress patches to n_spatial_tokens per frame)
        self.spatial_pool = MultiQueryPooling(config)

        # Compute number of spatial patches based on vision backbone
        # ResNet with 224x224 input produces 7x7=49 tokens
        # DINOv2 with 224x224 input produces 16x16=256 tokens (patch_size=14)
        if config.vision_backbone.startswith("dinov2"):
            # DINOv2 patch size is 14, so 224/14 = 16 patches per side
            img_h, img_w = config.history_image_size or (224, 224)
            n_patches_per_img = (img_h // 14) * (img_w // 14)
        else:
            # ResNet produces 7x7 feature map regardless of input size (due to adaptive pooling)
            n_patches_per_img = 49

        # Total patches per frame = n_cameras * n_patches_per_image
        n_cameras = len(config.image_features) if config.image_features else 1
        self.n_patches_per_frame = n_cameras * n_patches_per_img

        # 3D positional embeddings: spatial (per patch) + temporal (per frame)
        self.pos_embed_3d = PositionalEmbedding3D(
            n_patches=self.n_patches_per_frame,
            n_frames=config.n_obs_steps,
            dim_model=config.dim_model,
        )

    def _downsample_images(
        self,
        images: Tensor,
        height: int,
        width: int,
    ) -> tuple[Tensor, int, int]:
        """Downsample images for history encoder efficiency if configured.

        Args:
            images: Input images tensor. Can be:
                - (B, L, N_cam, C, H, W) for training batch
                - (B, N_cam, C, H, W) for single timestep inference
            height: Current height of the images
            width: Current width of the images

        Returns:
            Tuple of:
                - Downsampled images tensor (same shape as input)
                - New height
                - New width
        """
        if self.config.history_image_size is None:
            return images, height, width

        target_h, target_w = self.config.history_image_size
        if (height, width) == (target_h, target_w):
            return images, height, width

        # Store original shape to restore it after downsampling
        original_shape = images.shape
        num_dims = len(original_shape)

        # Reshape to (N, C, H, W) for interpolation where N = batch_size * [seq_len] * num_cameras
        if num_dims == 6:
            # Training path: (B, L, N_cam, C, H, W)
            batch_size, seq_len, num_cameras, channels = original_shape[:4]
            images = images.reshape(batch_size * seq_len * num_cameras, channels, height, width)
        elif num_dims == 5:
            # Inference path: (B, N_cam, C, H, W)
            batch_size, num_cameras, channels = original_shape[:3]
            images = images.reshape(batch_size * num_cameras, channels, height, width)
        else:
            raise ValueError(f"Unexpected image tensor shape: {original_shape}")

        # Downsample
        images = F.interpolate(
            images.float() if images.dtype == torch.uint8 else images,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to original structure with new spatial dimensions
        if num_dims == 6:
            images = images.reshape(batch_size, seq_len, num_cameras, channels, target_h, target_w)
        elif num_dims == 5:
            images = images.reshape(batch_size, num_cameras, channels, target_h, target_w)

        return images, target_h, target_w

    @torch.no_grad()
    def init_cache(self, batch_size: int, dtype: torch.dtype) -> list[tuple[Tensor, Tensor]]:
        # Return a list of caches - one for each Mamba2 block
        return [
            block.allocate_inference_cache(batch_size=batch_size, max_seqlen=1, dtype=dtype)
            for block in self.blocks
        ]

    @torch.no_grad()
    def step(
        self,
        x_t: Tensor,  # (B, N, D) - spatial tokens for this frame
        cache: list[tuple[Tensor, Tensor]],
        timestep_idx: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run one frame through spatial-then-temporal pipeline (inference).

        Flow: spatial attention → pool → Mamba steps for k pooled tokens.
        Matches training: pool BEFORE Mamba, so Mamba sees k tokens per frame.

        Args:
            x_t: (B, N, D) - spatial tokens for this frame (N patches)
            cache: list of (conv_state, ssm_state) tuples - one per block
            timestep_idx: Current timestep index for temporal positional embedding
        Returns:
            h_t: (B, k, D) - pooled & temporally processed tokens for this frame
            updated_cache: Updated cache list
        """
        batch_size, n_patches, dim_model = x_t.shape

        # Add 3D positional embeddings for this timestep
        spatial_embed, temporal_embed = self.pos_embed_3d.get_embeddings_for_timestep(n_patches, timestep_idx)
        x_t = x_t + spatial_embed + temporal_embed  # (B, N, D)

        # Step 1: Spatial self-attention over patches (treat as single frame)
        x_t = x_t.unsqueeze(1)  # (B, 1, N, D) - add frame dim for SpatialTransformer
        x_t = self.spatial_transformer(x_t)  # (B, 1, N, D)
        x_t = x_t.squeeze(1)  # (B, N, D)

        # Step 2: Pool patches → k summary tokens
        pooled = self.spatial_pool(x_t)  # (B, k, D)

        # Step 3: Feed k pooled tokens through Mamba sequentially
        outputs = []
        for token_idx in range(pooled.shape[1]):
            hidden = pooled[:, token_idx, :]  # (B, D)
            residual = None

            for block_idx, block in enumerate(self.blocks):
                conv_state, ssm_state = cache[block_idx]

                if residual is None:
                    residual = hidden

                # Pre-norm (official Block uses block.norm for first norm)
                hidden_norm = block.norm(residual.to(dtype=block.norm.weight.dtype))

                # Mamba2 mixer step
                y_t, new_conv, new_ssm = block.mixer.step(hidden_norm.unsqueeze(1), conv_state, ssm_state)
                y_t = y_t.squeeze(1)  # (B, D)

                # Residual connection
                hidden = y_t + residual

                # Optional MLP (official Block uses block.norm2 for MLP norm)
                if block.mlp is not None:
                    hidden = block.mlp(block.norm2(hidden)) + hidden

                cache[block_idx] = (new_conv, new_ssm)
                residual = hidden

            outputs.append(hidden)

        # Stack and project: (B, k, D)
        h_t = torch.stack(outputs, dim=1)
        h_t = self.encoder_history_input_proj(h_t)  # (B, k, D)

        return h_t, cache

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Process a full observation sequence (training).

        Flow: encode → spatial attention → pool → Mamba → project

        Args:
            batch: Full batch of observations and actions
        Returns:
            h_seq: (B, L*k, D) where k = n_spatial_tokens
        """
        # Get spatial tokens per frame
        x = self.fuse_observations(batch)  # (B, L, N, D)
        batch_size, seq_len, n_patches, dim_model = x.shape

        # Add 3D positional embeddings (spatial + temporal)
        x = self.pos_embed_3d(x)  # (B, L, N, D)

        # Step 1: Spatial self-attention per frame
        x = self.spatial_transformer(x)  # (B, L, N, D)

        # Step 2: Pool each frame's patches → k summary tokens
        h_frames = []
        for t in range(seq_len):
            h_t = self.spatial_pool(x[:, t, :, :])  # (B, k, D)
            h_frames.append(h_t)
        x = torch.stack(h_frames, dim=1)  # (B, L, k, D)

        # Step 3: Flatten for Mamba temporal processing
        x = einops.rearrange(x, "b l k d -> b (l k) d")  # (B, L*k, D)

        # Step 4: Mamba processes temporal relationships
        # Official Block.forward returns (hidden_states, residual) tuples.
        # hidden_states is the mixer output, residual is the accumulated skip connection.
        # After the final block, combine them to get the full output.
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual=residual)
        x = x + residual  # Combine final mixer output with skip connection

        # Project to final representation
        h_seq = self.encoder_history_input_proj(x)  # (B, L*k, D)

        return h_seq

    def fuse_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract and process spatial tokens per frame for spatiotemporal Mamba.

        Returns spatial tokens WITHOUT pooling - pooling happens AFTER Mamba.
        Flow: images -> encode -> cross-camera attn -> return spatial tokens

        Args:
            batch: Full batch of observations with OBS_IMAGES as stacked tensor (B, L, N_cam, C, H, W)
        Returns:
            x: (B, L, N, D) spatial tokens per frame ready for Mamba
        """
        # OBS_IMAGES is already stacked: (B, L, N_cam, C, H, W)
        img_stack = batch[OBS_IMAGES]
        batch_size, seq_len, num_cameras, channels, height, width = img_stack.shape

        # Downsample images for history encoder efficiency if configured
        img_stack, height, width = self._downsample_images(img_stack, height, width)

        # Process images in chunks to reduce peak memory
        # Image encoder has no temporal dependency, so we can chunk freely
        max_images_per_chunk = self.config.max_images_per_chunk
        images_per_timestep = num_cameras * batch_size
        timesteps_per_chunk = max(1, max_images_per_chunk // images_per_timestep)

        img_tokens_list = []
        for t_start in range(0, seq_len, timesteps_per_chunk):
            t_end = min(t_start + timesteps_per_chunk, seq_len)

            # Extract chunk: (B, chunk_len, N_cam, C, H, W)
            img_chunk = img_stack[:, t_start:t_end]
            chunk_len = t_end - t_start

            # Flatten to (N_cam*B*chunk_len, C, H, W)
            img_batch = img_chunk.reshape(num_cameras * batch_size * chunk_len, channels, height, width)

            # Convert uint8 to float if needed
            if img_batch.dtype == torch.uint8:
                img_batch = img_batch.float().div_(255)

            # Convert to channels_last for cuDNN optimization
            img_batch = img_batch.contiguous(memory_format=torch.channels_last)

            # Encoder forward pass for this chunk (no temporal dependency)
            chunk_tokens = self.image_encoder(img_batch)  # (N_cam*B*chunk_len, num_tokens, D)
            img_tokens_list.append(chunk_tokens)

        # Concatenate all chunks - now we have tokens for all timesteps
        img_tokens = torch.cat(img_tokens_list, dim=0)  # (N_cam*B*L, num_tokens, D)

        num_tokens_per_img = img_tokens.shape[1]
        dim_model = img_tokens.shape[-1]

        # Reshape to (N_cam, B, L, num_tokens, D)
        img_tokens = img_tokens.view(num_cameras, batch_size, seq_len, num_tokens_per_img, dim_model)

        # Combine camera and token dimensions for cross-camera attention
        # Reshape to (B, L, N_cam*num_tokens, D)
        cam_tokens = einops.rearrange(img_tokens, "n_cam b l n_tok d -> b l (n_cam n_tok) d")

        # Reshape for attention: (B*L, N_cam*num_tokens, D)
        num_tokens_per_frame = num_cameras * num_tokens_per_img
        x = cam_tokens.reshape(batch_size * seq_len, num_tokens_per_frame, dim_model)

        # Cross-camera attention operates on all tokens from all cameras
        x = self.cross_camera_attn(x, x, x)  # (B*L, N_cam*num_tokens, D)

        # Reshape to (B, L, N, D) - NO POOLING, return spatial tokens
        x = x.reshape(batch_size, seq_len, num_tokens_per_frame, dim_model)

        return x

    def fuse_one_timestep(self, obs_images: Tensor, timestep_idx: int = 0) -> Tensor:
        """Process a single observation timestep for streaming inference.

        Returns spatial tokens (NOT pooled) for step() to process per-patch.
        Flow: images -> encode -> cross-camera attn -> return spatial tokens

        Args:
            obs_images: (B, N_cam, C, H, W) images from all cameras at one timestep
            timestep_idx: Current timestep index (unused here, used in step())
        Returns:
            x_t: (B, N, D) spatial tokens ready for step()
        """
        batch_size, num_cameras, channels, height, width = obs_images.shape

        # Downsample images for history encoder efficiency if configured
        obs_images, height, width = self._downsample_images(obs_images, height, width)

        # Flatten cameras and batch for backbone processing
        img_batch = obs_images.reshape(num_cameras * batch_size, channels, height, width)

        # Convert uint8 to float if needed
        if img_batch.dtype == torch.uint8:
            img_batch = img_batch.float().div_(255)

        # Convert to channels_last for cuDNN optimization (helps ResNet)
        img_batch = img_batch.contiguous(memory_format=torch.channels_last)

        # Encoder forward pass for all cameras
        # Returns (N_cam*B, num_tokens, D) where num_tokens=49 for ResNet, 256 for DINOv2
        img_tokens = self.image_encoder(img_batch)  # (N_cam*B, num_tokens, D)

        num_tokens_per_img = img_tokens.shape[1]
        dim_model = img_tokens.shape[-1]

        # Reshape to (B, N_cam, num_tokens, D)
        img_tokens = img_tokens.view(batch_size, num_cameras, num_tokens_per_img, dim_model)

        # Flatten camera and token dimensions: (B, N_cam*num_tokens, D)
        img_tokens_flat = img_tokens.reshape(batch_size, num_cameras * num_tokens_per_img, dim_model)

        # Cross-camera attention operates on all tokens from all cameras
        x_t = self.cross_camera_attn(
            img_tokens_flat, img_tokens_flat, img_tokens_flat
        )  # (B, N_cam*num_tokens, D)

        # Return spatial tokens (NO POOLING - step() will handle that after Mamba)
        return x_t
