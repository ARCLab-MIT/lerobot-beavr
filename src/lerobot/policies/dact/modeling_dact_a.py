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
)
from lerobot.policies.dact.configuration_dact_a import DACTConfigA
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from lerobot.policies.dact.mamba.mtil import Mamba2, CrossCameraAttention, CrossModalAttention
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy

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

        # History encoder
        self.history_encoder = HistoryEncoder(
            config=config,
        )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # Separate main model backbone from other parameters for different learning rates
        # History backbone is frozen, so exclude it entirely
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
        # Separate caches to avoid type collisions:
        self.mamba_cache = None       # tuple(conv_state, ssm_state) for inference .step()
        self.hist_vec_cache = None    # Tensor (B, D) for TBPTT in training

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

    def _ensure_hist_cache(self, B: int, device: torch.device, dtype: torch.dtype):
        if getattr(self, "hist_vec_cache", None) is None \
        or self.hist_vec_cache.shape[0] != B \
        or self.hist_vec_cache.device != device \
        or self.hist_vec_cache.dtype != dtype:
            self.hist_vec_cache = torch.zeros(B, self.config.dim_model, device=device, dtype=dtype)

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

        # Check if cache batch size matches current batch size, reinitialize if needed
        current_batch_size = x_t.shape[0]
        if self.mamba_cache is None:
            self.mamba_cache = self.history_encoder.init_cache(current_batch_size, x_t.dtype)
        else:
            # Check if cache batch size matches current batch size
            cache_batch_size = self.mamba_cache[0].shape[0] if self.mamba_cache[0] is not None else 0
            if cache_batch_size != current_batch_size:
                self.mamba_cache = self.history_encoder.init_cache(current_batch_size, x_t.dtype)

        # Ensure cache is on the same device as input
        if hasattr(self.mamba_cache[0], 'device'):
            target_device = x_t.device
            if self.mamba_cache[0].device != target_device:
                self.mamba_cache = tuple(cache.to(target_device) if hasattr(cache, 'to') else cache
                                         for cache in self.mamba_cache)
            h_t, self.mamba_cache = self.history_encoder.step(x_t, self.mamba_cache)

        batch[HISTORY_TOKEN] = h_t

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        import time
        import os
        timing_dict = {}
        
        # Time the entire forward pass
        total_forward_start = time.perf_counter()
        
        # Time batch preprocessing
        preprocessing_start = time.perf_counter()
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        preprocessing_time = (time.perf_counter() - preprocessing_start) * 1000
        timing_dict['preprocessing_ms'] = preprocessing_time

        # ----- TBPTT stateful history -----
        # 1) make sure we have a cache shaped to (B, D)
        cache_start = time.perf_counter()
        device = next(self.parameters()).device
        if self.config.image_features:
            B = batch[OBS_IMAGES][0].shape[0]
            dtype = batch[OBS_IMAGES][0].dtype
        else:
            B = batch[OBS_STATE].shape[0]
            dtype = batch[OBS_STATE].dtype

        self._ensure_hist_cache(B, device, dtype)  # ensures self.hist_vec_cache

        # 2) DETACH previous hidden (truncation)
        h_prev = self.hist_vec_cache.detach()
        cache_time = (time.perf_counter() - cache_start) * 1000
        timing_dict['cache_ops_ms'] = cache_time

        # 3) Run encoder over the window with prefix-priming
        history_start = time.perf_counter()
        # Optional precise profiling
        if torch.cuda.is_available() and os.getenv("LEROBOT_TIMING_SYNC") == "1":
            torch.cuda.synchronize()
        h_seq = self.history_encoder.forward(batch, h_prev=h_prev)  # (B, L, D)
        if torch.cuda.is_available() and os.getenv("LEROBOT_TIMING_SYNC") == "1":
            torch.cuda.synchronize()
        history_time = (time.perf_counter() - history_start) * 1000
        timing_dict['history_ms'] = history_time

        # 4) Take the last valid history vector for this window
        postprocess_start = time.perf_counter()
        # In-place, zero-alloc update of history cache and select() views
        self._ensure_hist_cache(B, device, dtype)
        # Keep gradients for TBPTT - only detach the cache tensor itself, not the source
        h_last = h_seq.select(1, -1)  # (B, D) view
        self.hist_vec_cache.copy_(h_last.detach())  # Copy detached version to cache

        # 5) If any streams ended in this window, reset their cache
        #    Prefer the explicit ended mask if present; otherwise derive from valid_mask.
        if "ended_mask" in batch:
            ended = batch["ended_mask"]                    # (B,)
        elif "valid_mask" in batch:
            ended = ~batch["valid_mask"][:, -1]            # (B,)
        else:
            ended = None

        if ended is not None and ended.any():
            # zero them in-place on the cache tensor
            self.hist_vec_cache[ended] = 0

        # 6) Update cache for next window  
        # Use the original h_last (with gradients) for TBPTT, cache is just for storage
        # ----------------------------------

        # The rest stays the same, except use the TBPTT history token:
        batch[HISTORY_TOKEN] = h_last

        # Only last frame per cam/low-dim goes to ACT for this window (as you already do)
        if self.config.image_features:
            # Use view-based selection and set channels_last only after selecting last frame
            imgs_last = [imgs.select(1, -1).contiguous(memory_format=torch.channels_last) for imgs in batch[OBS_IMAGES]]
            batch[OBS_IMAGES] = imgs_last
        if OBS_STATE in batch and batch[OBS_STATE].dim() > 2:
            batch[OBS_STATE] = batch[OBS_STATE].select(1, -1)
        if ACTION in batch and batch[ACTION].dim() > 3:
            batch[ACTION] = batch[ACTION].select(1, -1)
        if "action_is_pad" in batch and batch["action_is_pad"].dim() > 2:
            batch["action_is_pad"] = batch["action_is_pad"].select(1, -1)
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        timing_dict['postprocess_ms'] = postprocess_time

        # Time the main model forward pass
        model_start = time.perf_counter()
        if torch.cuda.is_available() and os.getenv("LEROBOT_TIMING_SYNC") == "1":
            torch.cuda.synchronize()
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        if torch.cuda.is_available() and os.getenv("LEROBOT_TIMING_SYNC") == "1":
            torch.cuda.synchronize()
        model_time = (time.perf_counter() - model_start) * 1000
        timing_dict['model_forward_ms'] = model_time

        # Time loss computation
        loss_start = time.perf_counter()
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

        loss_time = (time.perf_counter() - loss_start) * 1000
        timing_dict['loss_computation_ms'] = loss_time
        
        # Calculate total forward time and add all timing info
        total_forward_time = (time.perf_counter() - total_forward_start) * 1000
        timing_dict['total_forward_ms'] = total_forward_time
        
        # Add all timing info to loss_dict
        loss_dict.update(timing_dict)

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
            self.backbone = self.backbone.to(memory_format=torch.channels_last)

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
            # Project visual backbone feature maps (C_backbone -> dim_model)
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_img_feat_input_proj = self.encoder_img_feat_input_proj.to(memory_format=torch.channels_last)

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
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )
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
        self.num_cameras = config.num_cameras
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
            self.hist_backbone = self.hist_backbone.to(memory_format=torch.channels_last)
            
            # Freeze the history backbone parameters
            for param in self.hist_backbone.parameters():
                param.requires_grad = False

        self.cross_camera_attn = self.cross_cam_attn = CrossCameraAttention(config)
        self.cross_modal_attn = CrossModalAttention(config)

        self.encoder_history_input_proj = nn.Linear(config.dim_model, config.dim_model)

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

    def forward(self, batch_seq: dict[str, Tensor], h_prev: Tensor | None = None) -> Tensor:
        """Process a full sequence using fused Mamba2 forward (training).

        Args:
            x_seq: (B, L, D) sequence of history vectors
        Returns:
            h_seq: (B, L, D) processed history features
        """
        # Use fused forward path for efficiency during training
        x_seq = self.fuse_obs_sequence_to_history(batch_seq) # (B, L, D)

        if h_prev is not None:
            if h_prev.dtype != x_seq.dtype:
                h_prev = h_prev.to(x_seq.dtype)
            if h_prev.device != x_seq.device:
                h_prev = h_prev.to(x_seq.device)
        
            x_seq = torch.cat([h_prev.unsqueeze(1), x_seq], dim=1)
            h_seq = self.mamba(x_seq) # (B, L+1, D)
            h_seq = h_seq[:, 1:, :] # (B, L, D)
        
        else:
            h_seq = self.mamba(x_seq) # (B, L, D)            
        
        return h_seq

    def fuse_obs_sequence_to_history(self, batch: dict[str, Tensor]) -> Tensor:
        """Fuse a sequence of observations into history vectors (B, L, D).
        
        Processes all cameras and timesteps in a single batched backbone forward pass (N_cam*B*L).
        Uses channels_last memory format for cuDNN optimization.
        
        Args:
            batch: dict with OBS_IMAGES [(B, L, C, H, W), ...] and optionally OBS_STATE (B, L, D_state)
        Returns:
            x_seq: (B, L, D) sequence of fused history vectors
        """
        # Extract dimensions
        first_img_key = list(batch[OBS_IMAGES].keys())[0] if isinstance(batch[OBS_IMAGES], dict) else 0
        first_img = batch[OBS_IMAGES][first_img_key] if isinstance(batch[OBS_IMAGES], dict) else batch[OBS_IMAGES][0]
        B, L = first_img.shape[:2]
        
        # Batch all cameras together for single backbone forward pass
        # Stack cameras: list of (B, L, C, H, W) → (N_cam, B, L, C, H, W)
        img_stack = torch.stack(batch[OBS_IMAGES], dim=0)  # (N_cam, B, L, C, H, W)
        N_cam = img_stack.shape[0]
        C, H, W = img_stack.shape[3:]
        
        # Flatten to (N_cam*B*L, C, H, W) and convert to channels_last for cuDNN fast path
        img_batch = img_stack.reshape(N_cam * B * L, C, H, W)
        # If frames are uint8 (0–255), convert before resizing
        if img_batch.dtype == torch.uint8:
            img_batch = img_batch.float().div_(255)

        # (Works fine under AMP)
        img_batch = torch.nn.functional.interpolate(
            img_batch, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
            ).contiguous(memory_format=torch.channels_last)
        
        # Single backbone forward pass for all cameras
        raw = self.hist_backbone(img_batch)  # dict with key "feature_map"
        raw_img_features = raw["feature_map"]
        img_features = self.spatial_adapter(raw_img_features)  # (N_cam*B*L, D)
        
        # Reshape to (N_cam, B, L, D) then split per camera
        D = img_features.shape[-1]
        img_features = img_features.view(N_cam, B, L, D)
        features = [img_features[i] for i in range(N_cam)]  # list of (B, L, D)
        
        # Stack cameras then cross-camera MHA
        per_cam = features               # list of (B, L, D)
        cam_tokens = torch.stack(per_cam, dim=2)  # (B, L, C, D)
        B, L, C, D = cam_tokens.shape
        x = cam_tokens.reshape(B*L, C, D)                 # (BL, C, D) cameras as sequence
        x = self.cross_camera_attn(x, x, x)               # residual + norm; (BL, C, D)
        x = x.mean(dim=1).reshape(B, L, D)                # pool cameras
        cam_features_proj = self.encoder_history_input_proj(x)  # (B, L, D)
        
        if self.config.robot_state_feature and OBS_STATE in batch:
            k = v = self.cross_modal_attn.proj_lowdim(batch[OBS_STATE])  # (B, L, D)
            q = cam_features_proj                                        # (B, L, D)
            x_seq = self.cross_modal_attn(q, k, v)                       # (B, L, D)
        else:
            x_seq = cam_features_proj
        
        return x_seq

    def fuse_obs_to_history_vec(self, batch: dict[str, Tensor]) -> Tensor:
        """Fuse per-step observation into a rich vector x_t for the HistoryEncoder.
        """
        # ---- per-step visual features for each camera -> (B, D) ----
        per_cam_feats: list[Tensor] = []
        for img in batch[OBS_IMAGES]:
            raw = self.hist_backbone(img)                 # {"feature_map": ...}
            fmap = raw["feature_map"]                     # (B, Cb, H', W')
            feat = self.spatial_adapter(fmap)             # (B, D)
            per_cam_feats.append(feat)

        # Cross-camera attention exactly like the sequence path but with L=1
        # Build (B, 1, C, D) then reshape to (B, C, D) and attend across C
        C = len(per_cam_feats)
        cam_tokens = torch.stack([f.unsqueeze(1) for f in per_cam_feats], dim=2)  # (B, 1, C, D)
        B = cam_tokens.shape[0]
        D = cam_tokens.shape[-1]
        x = cam_tokens.reshape(B, C, D)                     # (B, C, D)
        # MultiheadAttention usually expects (L, N, E) unless batch_first=True inside CrossCameraAttention.
        # CrossCameraAttention in your sequence path was called with (BL, C, D); to attend across cameras,
        # pass (C, B, D) so L=C (cameras), N=B.
        x = self.cross_camera_attn(x, x, x)                  # (B, C, D)
        x = x.mean(dim=1)                                    # (B, D)
        cam_features_proj = self.encoder_history_input_proj(x)  # (B, D)

        # ---- cross-modal fusion (project low-dim state first!) ----
        if self.config.robot_state_feature:
            low = self.cross_modal_attn.proj_lowdim(batch[OBS_STATE])  # (B, D)
            fused = self.cross_modal_attn(
                query=cam_features_proj.unsqueeze(1),                  # (B, 1, D)
                key=low.unsqueeze(1),                                  # (B, 1, D)
                value=low.unsqueeze(1),                                # (B, 1, D)
            ).squeeze(1)                                               # (B, D)
        else:
            fused = cam_features_proj

        return fused  # (B, D)
