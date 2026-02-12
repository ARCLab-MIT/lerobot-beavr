#!/usr/bin/env python
"""Diagnostic script to analyze decoder attention patterns in MACT.

This script checks whether the decoder is attending to history tokens or just current observations,
which helps diagnose mode collapse issues.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lerobot.policies.mact.modeling_mact import MACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps


def hook_attention_weights(model):
    """Add hooks to capture cross-attention weights from decoder layers."""
    attention_weights = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_output, attn_weights)
            # We need to modify multihead_attn to return weights
            attention_weights[f"layer_{layer_idx}"] = output[1].detach().cpu()

        return hook

    # Register hooks on decoder layers' multihead_attn
    hooks = []
    for idx, layer in enumerate(model.model.decoder.layers):
        # We need to temporarily modify the layer to return attention weights
        # Store original forward
        original_forward = layer.forward

        def new_forward(
            self,
            x,
            encoder_out,
            decoder_pos_embed=None,
            encoder_pos_embed=None,
            _original=original_forward,
            _idx=idx,
        ):
            """Modified forward that captures attention weights."""
            skip = x
            if self.pre_norm:
                x = self.norm1(x)
            q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
            x = self.self_attn(q, k, value=x)[0]
            x = skip + self.dropout1(x)
            if self.pre_norm:
                skip = x
                x = self.norm2(x)
            else:
                x = self.norm1(x)
                skip = x

            # MODIFIED: Capture cross-attention weights
            attn_out, attn_weights = self.multihead_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                value=encoder_out,
                average_attn_weights=True,  # Average over heads
            )
            attention_weights[f"layer_{_idx}"] = attn_weights.detach().cpu()

            x = skip + self.dropout2(attn_out)
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

        # Bind the new forward
        layer.forward = new_forward.__get__(layer, layer.__class__)

    return attention_weights


def analyze_attention_pattern(
    policy_path: str, dataset_repo_id: str, output_dir: str = "outputs/attention_analysis"
):
    """Analyze where the decoder attends during inference.

    Args:
        policy_path: Path to trained policy checkpoint
        dataset_repo_id: Dataset to sample from
        output_dir: Where to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    print(f"Loading policy from {policy_path}...")
    policy = MACTPolicy.from_pretrained(policy_path)
    policy.train()  # Keep in training mode to use full forward pass with VAE
    policy.cuda()

    # Load dataset
    print(f"Loading dataset {dataset_repo_id}...")
    # Load dataset with the same delta_timestamps as used during training
    # This ensures the batch has the correct temporal structure (B, L, ...)
    
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    
    dataset = LeRobotDataset(
        dataset_repo_id,
        delta_timestamps=delta_timestamps,
    )

    # Get a sample batch (use a later timestep to ensure history is populated)
    # Skip early frames to ensure we have full history context
    sample_idx = max(policy.config.n_history_tokens + 10, 50)
    print(f"Using sample at index {sample_idx} (to ensure history is populated)...")
    batch = dataset[sample_idx]
    
    # Debug: Check batch structure
    print(f"\nBatch keys: {batch.keys()}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # Add batch dimension and move to GPU
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Hook attention weights
    attention_weights = hook_attention_weights(policy)

    # Forward pass - use training mode to get full forward pass with VAE
    print("Running forward pass (training mode to capture all components)...")
    # No torch.no_grad() - we need gradients for VAE to work properly
    loss, loss_dict = policy.forward(batch)
    print(f"Forward pass complete. Loss: {loss.item():.4f}, Loss dict: {loss_dict}")

    # Analyze captured attention weights
    print("\n" + "=" * 70)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 70)

    # Count tokens in encoder sequence
    n_history_tokens = policy.config.n_history_tokens * policy.config.n_spatial_tokens
    n_1d_tokens = 1  # latent
    if policy.config.robot_state_feature:
        n_1d_tokens += 1
    if policy.config.env_state_feature:
        n_1d_tokens += 1

    # Image tokens (ResNet produces 7x7=49 per camera)
    n_cameras = len(policy.config.image_features) if policy.config.image_features else 0
    n_img_tokens = 49 * n_cameras

    total_tokens = n_1d_tokens + n_history_tokens + n_img_tokens

    print(f"\nEncoder token breakdown (total: {total_tokens}):")
    print(f"  - 1D tokens (latent, robot state, etc.): {n_1d_tokens}")
    print(
        f"  - History tokens: {n_history_tokens} ({policy.config.n_history_tokens} frames × {policy.config.n_spatial_tokens} tokens)"
    )
    print(f"  - Current image tokens: {n_img_tokens} ({n_cameras} cameras × 49 patches)")

    # Analyze each decoder layer
    for layer_name, weights in attention_weights.items():
        # weights shape: (batch, tgt_len, src_len) where src_len = encoder sequence length
        weights = weights[0]  # Remove batch dim: (chunk_size, encoder_len)

        # Average over decoder queries (chunk_size dimension)
        avg_weights = weights.mean(dim=0)  # (encoder_len,)

        # Split by token type
        attn_1d = avg_weights[:n_1d_tokens].sum().item()
        attn_history = avg_weights[n_1d_tokens : n_1d_tokens + n_history_tokens].sum().item()
        attn_img = avg_weights[n_1d_tokens + n_history_tokens :].sum().item()

        print(f"\n{layer_name.upper()}:")
        print(f"  Attention to 1D tokens:     {attn_1d * 100:5.1f}%")
        print(f"  Attention to history:       {attn_history * 100:5.1f}%")
        print(f"  Attention to current image: {attn_img * 100:5.1f}%")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Heatmap of all attention weights
        sns.heatmap(weights.numpy(), ax=axes[0], cmap="viridis", cbar_kws={"label": "Attention Weight"})
        axes[0].set_xlabel("Encoder Token Index")
        axes[0].set_ylabel("Decoder Query Index (action chunk)")
        axes[0].set_title(f"{layer_name}: Full Attention Matrix")

        # Add vertical lines to separate token types
        axes[0].axvline(x=n_1d_tokens, color="red", linestyle="--", linewidth=2, label="History start")
        axes[0].axvline(
            x=n_1d_tokens + n_history_tokens, color="orange", linestyle="--", linewidth=2, label="Image start"
        )

        # Plot 2: Bar chart of attention distribution
        categories = ["1D Tokens", f"History\n({n_history_tokens})", f"Current Obs\n({n_img_tokens})"]
        values = [attn_1d * 100, attn_history * 100, attn_img * 100]
        colors = ["gray", "blue", "green"]

        axes[1].bar(categories, values, color=colors)
        axes[1].set_ylabel("Attention Weight (%)")
        axes[1].set_title(f"{layer_name}: Attention Distribution")
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_dir / f"{layer_name}_attention.png", dpi=150)
        print(f"  Saved visualization: {output_dir / f'{layer_name}_attention.png'}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS:")

    # Get last layer attention
    last_layer_weights = attention_weights[f"layer_{len(attention_weights) - 1}"][0].mean(dim=0)
    attn_history_last = last_layer_weights[n_1d_tokens : n_1d_tokens + n_history_tokens].sum().item()
    attn_img_last = last_layer_weights[n_1d_tokens + n_history_tokens :].sum().item()

    if attn_history_last < 0.2:
        print("❌ PROBLEM: Decoder barely attends to history (<20%)")
        print("   → The model is ignoring shuffle information!")
        print("   → Solutions:")
        print("      1. Increase n_spatial_tokens to preserve more spatial info")
        print("      2. Try removing current images from encoder (force history use)")
        print("      3. Add stronger regularization on history token usage")
    elif attn_img_last > 0.7:
        print("⚠️  WARNING: Decoder heavily attends to current observation (>70%)")
        print("   → The model may be learning 'pick nearest cup' instead of 'pick correct cup'")
        print("   → Current observation may be overwhelming history signal")
    else:
        print("✓ OK: Decoder uses both history and current observation")
        print(f"  History: {attn_history_last * 100:.1f}%, Current: {attn_img_last * 100:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze MACT decoder attention patterns")
    parser.add_argument("--policy.path", type=str, required=True, help="Path to trained policy")
    parser.add_argument("--dataset.repo_id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/attention_analysis", help="Output directory"
    )

    args = parser.parse_args()

    analyze_attention_pattern(
        policy_path=args.__dict__["policy.path"],
        dataset_repo_id=args.__dict__["dataset.repo_id"],
        output_dir=args.output_dir,
    )
