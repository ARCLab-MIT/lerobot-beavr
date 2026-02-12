#!/usr/bin/env python
"""Visualize history embeddings using manual cup labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from lerobot.policies.mact.modeling_mact import MACTPolicy
from lerobot.utils.constants import OBS_IMAGES

# Manual ground truth: 0=left, 1=middle, 2=right
MANUAL_LABELS = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 0,
    5: 2,
    6: 0,
    7: 1,
    8: 0,
    9: 2,
    10: 1,
    11: 1,
    12: 0,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 0,
    18: 1,
    19: 1,
    20: 2,
    21: 2,
    22: 1,
    23: 2,
    24: 1,
    25: 2,
    26: 2,
    27: 0,
    28: 1,
    29: 2,
    30: 0,
}


def extract_embeddings(policy_path: str, dataset_repo_id: str):
    """Extract history embeddings from labeled episodes."""
    print(f"Loading policy from {policy_path}...")
    policy = MACTPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.cuda()

    print(f"Loading dataset {dataset_repo_id}...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id)

    embeddings = []
    labels = []

    print(f"Processing {len(MANUAL_LABELS)} labeled episodes...")

    for ep_idx in sorted(MANUAL_LABELS.keys()):
        ep = dataset.meta.episodes[ep_idx]
        start_idx = ep["dataset_from_index"]
        end_idx = ep["dataset_to_index"]

        # Use policy config for window and stride
        n_obs_steps = policy.config.n_obs_steps
        stride = policy.config.observation_stride
        total_window = (n_obs_steps - 1) * stride

        # Sample from NEAR THE END of episode (after shuffle, near grasp point)
        # Use last 20% of episode to ensure shuffle is complete
        # NOTE: We ensure the window fits within the episode
        sample_idx = start_idx + int((end_idx - start_idx) * 0.8)

        # Ensure we don't go out of bounds with our history window
        sample_idx = max(start_idx + total_window, sample_idx)

        # Calculate frame indices with striding (most recent first, going backwards)
        frame_indices = []
        for i in range(n_obs_steps):
            frame_idx = sample_idx - (n_obs_steps - 1 - i) * stride
            # Clamp to episode boundaries
            frame_idx = max(start_idx, min(end_idx - 1, frame_idx))
            frame_indices.append(frame_idx)

        # Extract strided frames
        frames = [dataset[idx] for idx in frame_indices]

        # Stack into batch (same as before)
        batch_seq = {}
        for k in frames[0]:
            if isinstance(frames[0][k], torch.Tensor):
                stacked = torch.stack([f[k] for f in frames], dim=0)  # (L, ...)
                if "image" in k:
                    batch_seq[k] = stacked.unsqueeze(0).cuda()  # (1, L, C, H, W)
                else:
                    batch_seq[k] = stacked.unsqueeze(0).cuda()  # (1, L, D)
            else:
                batch_seq[k] = frames[0][k]

        if policy.config.image_features:
            img_list = [batch_seq[key] for key in policy.config.image_features]
            batch_seq[OBS_IMAGES] = torch.stack(img_list, dim=2)  # (1, L, n_cameras, C, H, W)

        with torch.no_grad():
            h_seq = policy.history_encoder.forward(batch_seq)

        h_avg = h_seq.mean(dim=1).cpu().numpy()[0]
        embeddings.append(h_avg)
        labels.append(MANUAL_LABELS[ep_idx])

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"Sample idx (example): {sample_idx} (Ep range: {start_idx}-{end_idx})")
    print(f"Window: {n_obs_steps} frames, Stride: {stride}, Total window size: {total_window}")
    print(
        f"Distribution: Left={np.sum(labels == 0)}, Middle={np.sum(labels == 1)}, Right={np.sum(labels == 2)}"
    )

    return embeddings, labels


def visualize(embeddings, labels, output_dir="outputs/embedding_analysis"):
    """Visualize with PCA and t-SNE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # Variance analysis
    within_var = [embeddings[labels == c].var(axis=0).mean() for c in range(3)]
    between_var = embeddings.var(axis=0).mean()

    print("\nVariance:")
    print(f"  Within-cup avg: {np.mean(within_var):.6f}")
    print(f"  Between-cup: {between_var:.6f}")
    print(f"  Ratio (between/within): {between_var / np.mean(within_var):.2f}")

    if between_var / np.mean(within_var) < 1.1:
        print("  ⚠️  Low ratio - embeddings may not be discriminative!")

    # PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(embeddings)
    print(f"\nPCA variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(embeddings) // 4))
    x_tsne = tsne.fit_transform(embeddings)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["red", "green", "blue"]
    names = ["Left", "Middle", "Right"]

    for ax, x_proj, method in zip(axes, [x_pca, x_tsne], ["PCA", "t-SNE"], strict=True):
        for c in range(3):
            mask = labels == c
            ax.scatter(
                x_proj[mask, 0],
                x_proj[mask, 1],
                c=colors[c],
                label=names[c],
                alpha=0.7,
                s=80,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_title(f"History Embeddings - {method}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "embeddings.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {output_path}")

    # Silhouette scores
    sil_pca = silhouette_score(x_pca, labels)
    sil_tsne = silhouette_score(x_tsne, labels)

    print("\nSilhouette scores (>0.5 = good, >0.2 = weak):")
    print(f"  PCA:   {sil_pca:.3f}")
    print(f"  t-SNE: {sil_tsne:.3f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    best_score = max(sil_pca, sil_tsne)

    if best_score > 0.5:
        print("✅ GOOD: History embeddings CLUSTER by correct cup!")
        print("   → History encoder IS learning which cup is correct")
        print("   → Problem is in the DECODER or TRAINING DYNAMICS")
        print("\n   Next steps:")
        print("   1. Decoder may be ignoring history (despite 42% attention)")
        print("   2. Add auxiliary loss: predict cup from history")
        print("   3. Check if decoder learns shortcuts (always pick nearest)")
        print("   4. Visualize what decoder attention weights focus on")
    elif best_score > 0.2:
        print("⚠️  WEAK: Some clustering but not strong")
        print("   → History encoder captures SOME info but needs improvement")
        print("\n   Next steps:")
        print("   1. Increase n_spatial_tokens for finer spatial detail")
        print("   2. Add more n_spatial_attn_layers")
        print("   3. Verify ball reveal is in observation window")
    else:
        print("❌ BAD: NO clustering - history does NOT encode cup!")
        print("   → History encoder is FAILING to learn discriminative features")
        print("\n   Next steps:")
        print("   1. URGENT: Verify ball reveal is visible in observation window")
        print("   2. Test without Mamba (pass spatial tokens directly)")
        print("   3. Check if visual cue is too subtle")
        print("   4. Add colored markers in sim for debugging")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", type=str, required=True)
    parser.add_argument("--dataset.repo_id", type=str, required=True)
    args = parser.parse_args()

    embeddings, labels = extract_embeddings(args.__dict__["policy.path"], args.__dict__["dataset.repo_id"])
    visualize(embeddings, labels)
