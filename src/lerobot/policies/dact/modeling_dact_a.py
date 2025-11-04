from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn


from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
from lerobot.policies.dact.configuration_dact_a import DACTConfigA
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from lerobot.policies.dact.mamba.mtil import Mamba2, CrossCameraAttention, CrossModalAttention
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues

HISTORY_TOKEN = "history_token"

class DACTPolicyA(PreTrainedPolicy):

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

        # History encoder (only if history conditioning is enabled)
        if config.use_history_conditioning:
            self.history_encoder = HistoryEncoder(
                config=config,
            )
        else:
            self.history_encoder = None

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # Separate main model backbone from other parameters for different learning rates
        # History backbone is frozen, so exclude it entirely
        params = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and
                       not (n.startswith("history_encoder") and not self.config.use_history_conditioning) and
                       p.requires_grad
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

        # Only include history encoder backbone if history conditioning is enabled and backbone is not frozen
        if self.config.use_history_conditioning and not self.config.freeze_history_backbone:
            params.append({
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("history_encoder.hist_backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            })

        return params

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        
        # Reset Mamba streaming cache for history encoder (only if using history conditioning)
        # Cache will be initialized on first use in predict_action_chunk
        if self.config.use_history_conditioning:
            self._mamba_cache = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The model generates `chunk_size` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        batch = self.normalize_inputs(batch)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # Stack images along camera dimension: list of (B, C, H, W) -> (B, n_cameras, C, H, W)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=1)

        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()


    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Stack n latest observations from the queue (similar to diffusion policy)
        batch_stacked = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        # Determine batch size and device for cache initialization
        batch_size = batch_stacked[OBS_IMAGES].shape[0] if OBS_IMAGES in batch_stacked else batch_stacked[OBS_STATE].shape[0]
        device = batch_stacked[OBS_IMAGES].device if OBS_IMAGES in batch_stacked else batch_stacked[OBS_STATE].device
        dtype = next(self.parameters()).dtype  # Match model dtype

        # Process history conditioning if enabled
        if self.config.use_history_conditioning:
            # Initialize Mamba cache if not present (first call after reset)
            if self._mamba_cache is None:
                self._mamba_cache = self.history_encoder.init_cache(batch_size, dtype)
                # Move cache to correct device
                self._mamba_cache = tuple(c.to(device) for c in self._mamba_cache)

            # Streaming history processing: only process the CURRENT (most recent) observation
            # The Mamba cache already contains history from previous steps
            if self.config.image_features:
                obs_images_current = batch_stacked[OBS_IMAGES][:, -1, :, :, :, :]  # (B, n_cameras, C, H, W)
            else:
                obs_images_current = None

            obs_state_current = batch_stacked[OBS_STATE][:, -1, :] if OBS_STATE in batch_stacked else None

            # Fuse current observation into feature vector x_t
            x_t = self.history_encoder.fuse_one_timestep(obs_images_current, obs_state_current)  # (B, D)

            # Run one Mamba recurrent step with cached states
            h_t, self._mamba_cache = self.history_encoder.step(x_t, self._mamba_cache)

            # Detach to prevent gradient accumulation across inference steps
            h_t = h_t.detach()

            # Use the hidden state as the history token
            batch = {HISTORY_TOKEN: h_t}
        else:
            # No history conditioning - behave exactly like ACT
            batch = {}

        # Extract current frame per camera for ACT model
        if self.config.image_features:
            if self.config.use_history_conditioning:
                # obs_images_current was already extracted above
                n_cameras = obs_images_current.shape[1]
                imgs_last = [obs_images_current[:, i].contiguous(memory_format=torch.channels_last)
                            for i in range(n_cameras)]
            else:
                # Extract current frame when no history conditioning
                obs_images_current = batch_stacked[OBS_IMAGES][:, -1, :, :, :, :]  # (B, n_cameras, C, H, W)
                n_cameras = obs_images_current.shape[1]
                imgs_last = [obs_images_current[:, i].contiguous(memory_format=torch.channels_last)
                            for i in range(n_cameras)]
            batch[OBS_IMAGES] = imgs_last

        # Extract current state if present
        if OBS_STATE in batch_stacked:
            if self.config.use_history_conditioning:
                # obs_state_current was already extracted above
                batch[OBS_STATE] = obs_state_current.contiguous()
            else:
                # Extract current state when no history conditioning
                obs_state_current = batch_stacked[OBS_STATE][:, -1, :]
                batch[OBS_STATE] = obs_state_current.contiguous()

        actions = self.model(batch)[0]

        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation.
        
        Expects batch to contain observation sequences fetched by factory.py via delta_timestamps.
        Images: (B, L, C, H, W), States: (B, L, D_state), Actions: (B, chunk_size, D_action)
        """
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # Stack images: list of (B, L, C, H, W) -> (B, L, n_cameras, C, H, W)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # Process observation sequence through history encoder if conditioning is enabled
        if self.config.use_history_conditioning:
            # Input: images (B, L, n_cameras, C, H, W), states (B, L, D_state)
            # Output: h_seq (B, L, D)
            h_seq = self.history_encoder.forward(batch)  # (B, L, D)

            # Take the last timestep's history token as the current history state
            h_last = h_seq[:, -1, :]  # (B, D)
            batch[HISTORY_TOKEN] = h_last

        # Extract last frame per camera for ACT model (only current observation goes to ACT)
        if self.config.image_features:
            # batch[OBS_IMAGES] is (B, L, n_cameras, C, H, W), extract last timestep
            imgs_last_stacked = batch[OBS_IMAGES][:, -1, :, :, :, :]  # (B, n_cameras, C, H, W)
            n_cameras = imgs_last_stacked.shape[1]
            imgs_last = [imgs_last_stacked[:, i].contiguous(memory_format=torch.channels_last) 
                        for i in range(n_cameras)]
            batch[OBS_IMAGES] = imgs_last
        
        # Extract last state if present
        if OBS_STATE in batch and batch[OBS_STATE].dim() > 2:
            batch[OBS_STATE] = batch[OBS_STATE][:, -1, :].contiguous()

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
            loss = l1_loss * self.config.l1_weight + mean_kld * self.config.kl_weight
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
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size  # cls + actions (history removed)
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # # Backbone for image feature extraction.
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
            # self.backbone = self.backbone.to(memory_format=torch.channels_last)

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = DACTDecoder(config)

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
            # Project visual backbone feature maps (C_backbone -> dim_model)
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            # self.encoder_img_feat_input_proj = self.encoder_img_feat_input_proj.to(memory_format=torch.channels_last)

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
                (batch_size, (2 if self.config.robot_state_feature else 1)),
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
            for img in batch[OBS_IMAGES]:
                img = img.contiguous(memory_format=torch.channels_last)
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
        # Pass history conditioning to decoder if enabled
        history_cond = batch.get(HISTORY_TOKEN) if self.config.use_history_conditioning else None
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            history_cond=history_cond,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class DACTDecoderLayer(nn.Module):
    """ACT decoder layer with optional cross-attention to history conditioning."""

    def __init__(self, config: DACTConfigA):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Optional: Additional cross-attention for history conditioning
        if config.use_history_conditioning:
            self.history_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        else:
            self.history_attn = None

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        if config.use_history_conditioning:
            self.norm4 = nn.LayerNorm(config.dim_model)  # for history cross-attention
        else:
            self.norm4 = None
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        if config.use_history_conditioning:
            self.dropout4 = nn.Dropout(config.dropout)  # for history cross-attention
        else:
            self.dropout4 = None

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
        self.use_history_conditioning = config.use_history_conditioning

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        history_cond: Tensor | None = None,  # Optional: (1, B, D) history conditioning
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder.
            history_cond: Optional (1, B, C) history conditioning vector.
            decoder_pos_embed: (DS, 1, C) positional embedding for decoder queries.
            encoder_pos_embed: (ES, 1, C) positional embedding for encoder keys.
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        # 1. Self-attention (same as ACT)
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # 2. Cross-attention to encoder output (same as ACT)
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
        )[0]
        x = skip + self.dropout2(x)

        # 3. Optional: Cross-attention to history conditioning
        if self.use_history_conditioning and history_cond is not None:
            if self.pre_norm:
                skip = x
                x = self.norm3(x)
            else:
                x = self.norm2(x)
                skip = x
            x = self.history_attn(
                query=x,  # No positional encoding for history
                key=history_cond,
                value=history_cond,
            )[0]
            x = skip + self.dropout3(x)

            # 4. Feed-forward (with history conditioning - norm3 used)
            if self.pre_norm:
                skip = x
                x = self.norm4(x)
            else:
                x = self.norm3(x)
                skip = x
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = skip + self.dropout4(x)
            if not self.pre_norm:
                x = self.norm4(x)
        else:
            # 4. Feed-forward (without history conditioning - same as ACT)
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


class DACTDecoder(nn.Module):
    """ACT decoder with history conditioning support."""

    def __init__(self, config: DACTConfigA):
        super().__init__()
        self.layers = nn.ModuleList([DACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        history_cond: Tensor | None = None,  # Optional: (1, B, D) or (B, D)
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # Ensure history_cond has shape (1, B, D) for cross-attention if provided
        if history_cond is not None and history_cond.dim() == 2:
            history_cond = history_cond.unsqueeze(0)  # (B, D) -> (1, B, D)

        for layer in self.layers:
            x = layer(
                x, encoder_out, history_cond,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class HistoryEncoder(nn.Module):
    """Recurrent history encoder based on a single Mamba2 block.

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
            nn.Conv2d(config.image_backbone_out_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(256, config.dim_model),  # (B, D)
            nn.LayerNorm(config.dim_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10)
        )

        self.config = config
        # Backbone for image feature extraction.
        if self.config.image_features:
            history_backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.hist_backbone = IntermediateLayerGetter(history_backbone_model, return_layers={"layer4": "feature_map"})
            # Expose backbone output channels for consumers
            self.hist_backbone_out_channels = history_backbone_model.fc.in_features
            # self.hist_backbone = self.hist_backbone.to(memory_format=torch.channels_last)

            # Freeze the history backbone parameters
            if config.freeze_history_backbone:
                for param in self.hist_backbone.parameters():
                    param.requires_grad = False

        self.cross_camera_attn = self.cross_cam_attn = CrossCameraAttention(config)
        self.cross_modal_attn = CrossModalAttention(config)

        self.encoder_history_input_proj = nn.Linear(config.dim_model, config.dim_model)

        # Predictor head for auxiliary next-state supervision (OBS_STATE[t+1])
        if self.config.robot_state_feature:
            self.next_state_predictor = nn.Sequential(
                nn.Linear(self.config.dim_model, self.config.dim_model),
                nn.GELU(),
                nn.Linear(self.config.dim_model, self.config.robot_state_feature.shape[0]),
            )
        else:
            self.next_state_predictor = None

    @torch.no_grad()
    def init_cache(self, batch_size: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self.mamba.allocate_inference_cache(batch_size=batch_size, max_seqlen=1, dtype=dtype)

    @torch.no_grad()
    def step(
        self,
        x_t: Tensor, # (B, D)
        cache: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run one recurrent step (inference only).

        Args:
            x_t: (B, D)
            cache: tuple(conv_state, ssm_state)
        Returns:
            h_t: (B, D) and updated cache
        """
        conv_state, ssm_state = cache
        h_t, new_conv, new_ssm = self.mamba.step(x_t.unsqueeze(1), conv_state, ssm_state)
        return h_t.squeeze(1), (new_conv, new_ssm)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Process a full sequence using fused Mamba2 forward (training).

        Args:
            batch: Full batch of observations and actions
        Returns:
            h_seq: (B, L, D) processed history features
        """
        # Fuse observations into history vectors
        x_seq = self.fuse_observations(batch)  # (B, L, D)
        # Process through Mamba
        h_seq = self.mamba(x_seq)  # (B, L, D)
        return h_seq


    def fuse_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Function to fuse observations into history vectors.

        Processes observation sequences of length n_obs_steps.
        Processes all cameras and timesteps in a single batched backbone forward pass.

        Args:
            batch: Full batch of observations with OBS_IMAGES as stacked tensor (B, L, N_cam, C, H, W)
        Returns:
            x: (B, L, D) processed history features
        """
        # OBS_IMAGES is already stacked: (B, L, N_cam, C, H, W)
        img_stack = batch[OBS_IMAGES]
        B, L, N_cam, C, H, W = img_stack.shape

        # Flatten to (N_cam*B*L, C, H, W) for batch processing
        img_batch = img_stack.reshape(N_cam * B * L, C, H, W)

        # Convert uint8 to float if needed
        if img_batch.dtype == torch.uint8:
            img_batch = img_batch.float().div_(255)

        # Convert to channels_last for cuDNN optimization
        img_batch = img_batch.contiguous(memory_format=torch.channels_last)

        # Single backbone forward pass for all cameras and timesteps
        raw = self.hist_backbone(img_batch)
        raw_img_features = raw["feature_map"]
        img_features = self.spatial_adapter(raw_img_features)  # (N_cam*B*L, D)

        D = img_features.shape[-1]

        # Reshape to (N_cam, B, L, D)
        img_features = img_features.view(N_cam, B, L, D)

        # Stack cameras then cross-camera attention
        cam_tokens = torch.stack([img_features[i] for i in range(N_cam)], dim=2)  # (B, L, N_cam, D)
        x = cam_tokens.reshape(B*L, N_cam, D)  # (B*L, N_cam, D)
        x = self.cross_camera_attn(x, x, x)  # (B*L, N_cam, D)
        x = x.mean(dim=1).reshape(B, L, D)  # (B, L, D)
        cam_features_proj = self.encoder_history_input_proj(x)  # (B, L, D)

        # Cross-modal fusion with robot state if available
        if self.config.robot_state_feature and OBS_STATE in batch:
            k = v = self.cross_modal_attn.proj_lowdim(batch[OBS_STATE])  # (B, L, D)
            q = cam_features_proj  # (B, L, D)
            x_out = self.cross_modal_attn(q, k, v)  # (B, L, D)
        else:
            x_out = cam_features_proj

        return x_out

    def fuse_one_timestep(self, obs_images: Tensor, obs_state: Tensor = None) -> Tensor:
        """Process a single observation timestep for streaming inference.

        Args:
            obs_images: (B, N_cam, C, H, W) images from all cameras at one timestep
            obs_state: (B, D_state) robot state at one timestep (optional)
        Returns:
            x_t: (B, D) fused observation feature for this timestep
        """
        B, N_cam, C, H, W = obs_images.shape

        # Flatten cameras and batch for backbone processing
        img_batch = obs_images.reshape(N_cam * B, C, H, W)

        # Convert uint8 to float if needed
        if img_batch.dtype == torch.uint8:
            img_batch = img_batch.float().div_(255)

        # Convert to channels_last for cuDNN optimization
        img_batch = img_batch.contiguous(memory_format=torch.channels_last)

        # Backbone forward pass for all cameras
        raw = self.hist_backbone(img_batch)
        raw_img_features = raw["feature_map"]
        img_features = self.spatial_adapter(raw_img_features)  # (N_cam*B, D)

        D = img_features.shape[-1]

        # Reshape to (B, N_cam, D)
        img_features = img_features.view(B, N_cam, D)

        # Cross-camera attention
        cam_features_fused = self.cross_camera_attn(img_features, img_features, img_features)  # (B, N_cam, D)
        cam_features_fused = cam_features_fused.mean(dim=1)  # (B, D)
        cam_features_proj = self.encoder_history_input_proj(cam_features_fused)  # (B, D)

        # Cross-modal fusion with robot state if available
        if self.config.robot_state_feature and obs_state is not None:
            k = v = self.cross_modal_attn.proj_lowdim(obs_state.unsqueeze(1))  # (B, 1, D)
            q = cam_features_proj.unsqueeze(1)  # (B, 1, D)
            x_out = self.cross_modal_attn(q, k, v)  # (B, 1, D)
            x_out = x_out.squeeze(1)  # (B, D)
        else:
            x_out = cam_features_proj

        return x_out
