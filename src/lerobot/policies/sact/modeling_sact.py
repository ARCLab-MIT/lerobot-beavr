from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sact.configuration_sact import SACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class SACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = SACTConfig
    name = "sact"

    def __init__(
        self,
        config: SACTConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = SACT(config)

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

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Get original image size for scaling
        H_img, W_img = batch[OBS_IMAGES][0].shape[-2:]

        logits = self.model(batch)[0]
        B, S, Hp, Wp = logits.shape  # Hp, Wp are number of patches

        # Compute scale factors to convert from patch indices to image coords
        # Each patch covers patch_height x patch_width pixels in the original image
        scale_h = self.model.patch_height  # height per patch in pixels
        scale_w = self.model.patch_width   # width per patch in pixels

        idx = logits.view(B, S, -1).argmax(dim=-1)
        patch_xs = idx % Wp  # patch x index
        patch_ys = idx // Wp  # patch y index

        # Convert patch indices to pixel coordinates (center of each patch)
        xs_scaled = (patch_xs.float() + 0.5) * scale_w
        ys_scaled = (patch_ys.float() + 0.5) * scale_h

        # Apply refinement network for pixel-accurate predictions
        if self.config.use_refinement and self.config.image_features:
            # Get high-resolution backbone features
            backbone_features = self.model.backbone(batch[OBS_IMAGES][0])["feature_map"]  # (B, C, H_feat, W_feat)

            # Extract patches around selected patch centers for refinement
            refinement_patches = self._extract_refinement_patches(
                backbone_features, patch_ys, patch_xs, batch_size=B, seq_len=S
            )  # (B*S, C, patch_size, patch_size)

            # Predict offsets using refinement network
            offsets = self.model.refinement_net(refinement_patches)  # (B*S, 2)

            # Reshape offsets back to (B, S, 2)
            offsets = offsets.view(B, S, 2)

            # Add offsets to patch center coordinates
            xs_scaled = xs_scaled + offsets[:, :, 0]
            ys_scaled = ys_scaled + offsets[:, :, 1]

        actions = torch.stack([xs_scaled, ys_scaled], dim=-1)

        return actions

    def _extract_refinement_patches(
        self,
        backbone_features: Tensor,
        patch_ys: Tensor,
        patch_xs: Tensor,
        batch_size: int,
        seq_len: int
    ) -> Tensor:
        """
        Extract patches around selected patch centers for refinement.

        Args:
            backbone_features: (B, C, H_feat, W_feat) - high-resolution features
            patch_ys: (B, S) - selected patch y indices
            patch_xs: (B, S) - selected patch x indices
            batch_size: batch size B
            seq_len: sequence length S

        Returns:
            refinement_patches: (B*S, C, patch_size, patch_size)
        """
        B, C, H_feat, W_feat = backbone_features.shape
        patch_size = self.config.refinement_patch_size

        # Convert patch indices to feature map coordinates
        # Assuming backbone features have the same spatial resolution as input image
        feat_ys = patch_ys * self.model.patch_height + self.model.patch_height // 2
        feat_xs = patch_xs * self.model.patch_width + self.model.patch_width // 2

        # Flatten for batched extraction
        feat_ys_flat = feat_ys.view(-1)  # (B*S,)
        feat_xs_flat = feat_xs.view(-1)  # (B*S,)

        # Extract patches using unfold and gather
        # First, unfold the feature map to get all possible patches
        patches = backbone_features.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        # Shape: (B, C, H_feat-patch_size+1, W_feat-patch_size+1, patch_size, patch_size)

        # Gather the patches at the selected locations
        refinement_patches = []
        for b in range(batch_size):
            for s in range(seq_len):
                y_idx = feat_ys_flat[b * seq_len + s]
                x_idx = feat_xs_flat[b * seq_len + s]

                # Clamp indices to valid range
                y_idx = torch.clamp(y_idx, 0, H_feat - patch_size)
                x_idx = torch.clamp(x_idx, 0, W_feat - patch_size)

                patch = patches[b, :, y_idx, x_idx, :, :]  # (C, patch_size, patch_size)
                refinement_patches.append(patch)

        refinement_patches = torch.stack(refinement_patches, dim=0)  # (B*S, C, patch_size, patch_size)
        return refinement_patches

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        logits, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        B, S, Hf, Wf = logits.shape

        # Get original image size for coordinate conversion
        if self.config.image_features:
            H_img, W_img = batch[OBS_IMAGES][0].shape[-2:]
        else:
            H_img, W_img = None, None

        obs = batch[OBS_STATE]                 # (B,3) or (B,S,3)
        if obs.dim() == 2:                     # (B,3) → repeat across time
            obs = obs.unsqueeze(1).expand(B, S, -1)

        active = obs.argmax(dim=-1)            # (B,S) in {0,1,2}
        actions = batch[ACTION]                # (B,S,6) = [a0x,a0y,a1x,a1y,a2x,a2y]

        xs_all = actions[..., 0::2]            # (B,S,3)
        ys_all = actions[..., 1::2]            # (B,S,3)

        idx = active.unsqueeze(-1)             # (B,S,1)
        x = torch.gather(xs_all, -1, idx).squeeze(-1)    # (B,S), in [0,1) if normalized
        y = torch.gather(ys_all, -1, idx).squeeze(-1)    # (B,S)

        # Ground truth actions are in pixel coordinates [0, H_img) x [0, W_img)
        # Convert to patch indices [0, n_patches_h) x [0, n_patches_w)
        x = x.clamp(0, W_img - 1e-8)
        y = y.clamp(0, H_img - 1e-8)
        xs = (x / self.model.patch_width).long().clamp(0, self.model.n_patches_w - 1)  # (B,S)
        ys = (y / self.model.patch_height).long().clamp(0, self.model.n_patches_h - 1)  # (B,S)

        # For pixel-accurate training, compute predicted pixel coordinates and use regression loss
        if self.config.use_refinement and self.config.image_features:
            # Get predicted patch indices
            pred_idx = logits.view(B, S, -1).argmax(dim=-1)
            pred_patch_xs = pred_idx % self.model.n_patches_w  # (B, S)
            pred_patch_ys = pred_idx // self.model.n_patches_w  # (B, S)

            # Get backbone features for refinement patch extraction
            backbone_features = self.model.backbone(batch[OBS_IMAGES][0])["feature_map"]  # (B, C, H_feat, W_feat)

            # Extract refinement patches for predicted patches
            refinement_patches = self._extract_refinement_patches(
                backbone_features, pred_patch_ys, pred_patch_xs, batch_size=B, seq_len=S
            )  # (B*S, C, patch_size, patch_size)

            # Predict offsets using refinement network
            pred_offsets = self.model.refinement_net(refinement_patches)  # (B*S, 2)
            pred_offsets = pred_offsets.view(B, S, 2)  # (B, S, 2)

            # Compute predicted pixel coordinates: patch_center + offsets
            pred_patch_centers_x = (pred_patch_xs.float() + 0.5) * self.model.patch_width
            pred_patch_centers_y = (pred_patch_ys.float() + 0.5) * self.model.patch_height
            pred_coords = torch.stack([pred_patch_centers_x + pred_offsets[:, :, 0],
                                     pred_patch_centers_y + pred_offsets[:, :, 1]], dim=-1)  # (B, S, 2)

            # Ground truth pixel coordinates
            gt_coords = torch.stack([x, y], dim=-1)  # (B, S, 2)

            # Pixel-accurate loss: L1 loss between predicted and ground truth pixel coordinates
            pixel_loss = F.l1_loss(pred_coords, gt_coords)
            loss_dict = {"pixel_loss": pixel_loss.item()}

            # For backwards compatibility, also compute CE loss but don't use it
            target_idx  = (ys * self.model.n_patches_w + xs).view(-1)  # (B*S,)
            logits_flat = logits.view(B * S, self.model.n_patches)
            ce_loss = F.cross_entropy(logits_flat, target_idx)
            loss_dict["ce_loss"] = ce_loss.item()
        else:
            # Fallback to original patch-based loss when refinement is disabled
            target_idx  = (ys * self.model.n_patches_w + xs).view(-1)  # (B*S,)
            logits_flat = logits.view(B * S, self.model.n_patches)
            pixel_loss = F.cross_entropy(logits_flat, target_idx)
            loss_dict = {"ce_loss": pixel_loss.item()}


        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = pixel_loss * self.config.ce_weight + mean_kld * self.config.kl_weight
        else:
            loss = pixel_loss

        return loss, loss_dict

class SACT(nn.Module):
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

    def __init__(self, config: SACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        # Calculate patch dimensions for efficient tokenization
        if self.config.image_features:
            # Assume single camera for now - get image shape from first image feature
            first_img_key = next(iter(self.config.image_features.keys()))
            img_shape = self.config.image_features[first_img_key].shape  # [C, H, W]
            self.img_height, self.img_width = img_shape[1], img_shape[2]
            self.patch_height, self.patch_width = self.config.patch_size
            self.n_patches_h = self.img_height // self.patch_height
            self.n_patches_w = self.img_width // self.patch_width
            self.n_patches = self.n_patches_h * self.n_patches_w

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
            self.backbone = ConvBackbone(config)
            if self.config.use_attention_pooling:
                self.attention_pool = AttentionPooling(config.hidden_channels, self.config.patch_size, config.n_heads)
            if self.config.use_refinement:
                self.refinement_net = RefinementNetwork(config)

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
            self.encoder_img_feat_input_proj = nn.Conv2d(config.hidden_channels, config.dim_model, kernel_size=1)
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
        self.action_head = PointerHead(config)

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
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
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

        pixel_hw = None
        pixel_len = None

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        pixel_tokens = None
        pixel_pos_tokens = None

        if self.config.image_features:
            # For now, only 1 camera supported
            assert len(batch[OBS_IMAGES]) == 1, "Only 1 camera is supported for now"
            backbone_features = self.backbone(batch[OBS_IMAGES][0])["feature_map"]                  # (B, C, H, W)

            # Create patch tokens using pooling for efficient attention
            if self.config.use_attention_pooling:
                # Use attention pooling to aggregate pixels within each patch
                patch_features = self.attention_pool(backbone_features)  # (B, C, n_patches_h, n_patches_w)
            else:
                # Use standard average pooling
                patch_features = F.adaptive_avg_pool2d(backbone_features,
                                                      (self.n_patches_h, self.n_patches_w))  # (B, C, n_patches_h, n_patches_w)

            # Project to model dim for pointer memory
            patch_tokens = self.encoder_img_feat_input_proj(patch_features)  # (B, D, n_patches_h, n_patches_w)

            pixel_hw = (self.n_patches_h, self.n_patches_w)
            pixel_len = self.n_patches

            # 2D pos enc we'll add to patch memory (helps pointer localization)
            patch_pos_embed = self.encoder_cam_feat_pos_embed(patch_tokens)  # (B, D, n_patches_h, n_patches_w)

            # Rearrange to seq form for einsum: (N_patches, B, D)
            pixel_tokens = einops.rearrange(patch_tokens, "b d h w -> (h w) b d")
            pixel_pos_tokens = einops.rearrange(patch_pos_embed, "b d h w -> (h w) b d")

            # Extend immediately instead of accumulating and concatenating
            # Convert to list to extend properly
            encoder_in_tokens.extend(list(pixel_tokens))
            encoder_in_pos_embed.extend(list(pixel_pos_tokens))

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

        assert pixel_len is not None and pixel_hw is not None, "Pointer head requires pixel_hw and pixel_len to be set"

        pixel_tokens_with_pos = pixel_tokens + pixel_pos_tokens
        logits = self.action_head(decoder_out=decoder_out, pixel_tokens=pixel_tokens_with_pos, hw=pixel_hw)

        return logits, (mu, log_sigma_x2)

class PointerHead(nn.Module):
    def __init__(self, config: SACTConfig):
        super().__init__()
        self.scale = config.dim_model ** -0.5

    def forward(self, decoder_out: Tensor, pixel_tokens: Tensor, hw: tuple[int, int]) -> Tensor:
        # Decoder output: (S, B, D)
        # Pixel tokens: (N, B, D)
        logits = torch.einsum("sbd,nbd->bsn", decoder_out, pixel_tokens) * self.scale
        H, W = hw
        return logits.view(logits.size(0), logits.size(1), H, W)  # (B, S, H, W)

class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating pixels within each patch into tokens."""

    def __init__(self, channels: int, patch_size: tuple[int, int], num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.patch_h, self.patch_w = patch_size
        self.patch_pixels = self.patch_h * self.patch_w
        self.num_heads = num_heads

        # Learnable query for pooling each patch
        self.query = nn.Parameter(torch.randn(1, 1, channels))

        # Projections for attention
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, backbone_features: Tensor) -> Tensor:
        """
        Args:
            backbone_features: (B, C, H_img, W_img) - full resolution features

        Returns:
            (B, C, H_patches, W_patches) - attention-pooled patch features
        """
        B, C, H_img, W_img = backbone_features.shape

        # Extract patches manually (unfold operation)
        patches = backbone_features.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        # Shape: (B, C, H_patches, W_patches, patch_h, patch_w)

        B, C, Hp, Wp, ph, pw = patches.shape

        # Reshape to treat each patch as a sequence
        patch_seq = patches.permute(0, 2, 3, 4, 5, 1).reshape(B * Hp * Wp, ph * pw, C)
        # Shape: (B*Hp*Wp, patch_pixels, C)

        # Create queries for each patch
        query = self.query.expand(B * Hp * Wp, 1, C)  # (B*Hp*Wp, 1, C)

        # Apply attention within each patch
        attn_out, _ = self.attn(query, patch_seq, patch_seq)

        # Take the attended output and project
        pooled = self.out_proj(attn_out.squeeze(1))  # (B*Hp*Wp, C)

        # Reshape back to spatial patch layout
        pooled = pooled.view(B, Hp, Wp, C).permute(0, 3, 1, 2)  # (B, C, Hp, Wp)

        return pooled


class ConvBackbone(nn.Module):
    def __init__(self, config: SACTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        f = self.net(x)                              # (B, H, Hf, Wf)

        return {"feature_map": f}


class RefinementNetwork(nn.Module):
    """Refinement network for pixel-accurate predictions within selected patches."""

    def __init__(self, config: SACTConfig):
        super().__init__()
        self.patch_size = config.refinement_patch_size
        self.hidden_channels = config.refinement_hidden_channels

        # CNN to process the extracted patch
        self.net = nn.Sequential(
            nn.Conv2d(config.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 2),  # Predict (dx, dy) offsets
        )

    def forward(self, patch_features: Tensor) -> Tensor:
        """
        Args:
            patch_features: (B*S, C, patch_size, patch_size) - features for selected patches

        Returns:
            offsets: (B*S, 2) - predicted (dx, dy) offsets within the patch
        """
        offsets = self.net(patch_features)  # (B*S, 2)
        # Scale offsets to be within [-patch_size/2, patch_size/2]
        offsets = torch.tanh(offsets) * (self.patch_size / 2)
        return offsets
