"""
Test script for attention visualization with dummy data that mimics MACT policy output.

This script creates realistic dummy attention data and tests all visualization functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .attention_visualization import (
    plot_attention_heatmap,
    plot_episode_attention_heatmap,
    plot_detailed_encoder_history_attention,
    analyze_attention_patterns,
    create_attention_summary_report,
)

# ============================================================================
# TOKEN COUNT ANALYSIS FOR MACT POLICY
# ============================================================================
"""
Based on modeling_mact.py and configuration_mact.py:

DECODER HISTORY ATTENTION:
- Query tokens: chunk_size action positions (default: 100)
- Key tokens: n_history_tokens (default: 4)
- Shape: (batch, 100, 4)
- Visualization: 100 rows (action positions) × 4 columns (history tokens)

ENCODER-HISTORY CROSS-ATTENTION:
- Query tokens: encoder sequence length
  * 1 latent token
  * 1 robot state token (if robot_state_feature exists)
  * 0-1 env state token (if env_state_feature exists)
  * N_cameras × spatial_patches image tokens
    - For ResNet18 with 224×224 input: layer4 outputs 7×7 = 49 spatial patches
    - For 2 cameras: 2 × 49 = 98 image tokens
  * Total example (2 cameras + robot state): 1 + 1 + 98 = 100 tokens
- Key tokens: n_history_tokens (default: 4)
- Shape: (batch, 100, 4)
- Visualization: 100 rows (encoder tokens) × 4 columns (history tokens)

For typical Aloha setup:
- Decoder: 100 action positions × 4 history tokens
- Encoder: 100 encoder tokens × 4 history tokens
"""

def create_dummy_attention_data(
    batch_size=2,
    chunk_size=100,
    n_history_tokens=4,
    n_encoder_tokens=100,
    n_decoder_layers=1,
    timesteps=10
):
    """
    Create dummy attention data that mimics what MACT policy produces.
    
    Args:
        batch_size: Batch size
        chunk_size: Number of action chunk positions (default 100)
        n_history_tokens: Number of history tokens (default 4)
        n_encoder_tokens: Number of encoder tokens (latent + state + images)
        n_decoder_layers: Number of decoder layers
        timesteps: Number of timesteps to simulate in an episode
        
    Returns:
        List of attention weight dictionaries (one per timestep)
    """
    episode_attn_history = []
    
    for t in range(timesteps):
        attn_dict = {}
        
        # Decoder history attention for each layer
        for layer_idx in range(n_decoder_layers):
            # Shape: (batch, chunk_size, n_history_tokens)
            # Create pattern: earlier actions attend more to recent history
            decoder_attn = torch.zeros(batch_size, chunk_size, n_history_tokens)
            
            for i in range(chunk_size):
                # Earlier actions (i=0) attend more to most recent history (token 3)
                # Later actions spread attention more evenly
                recency_bias = 1.0 - (i / chunk_size) * 0.7
                probs = np.array([0.1, 0.15, 0.25, 0.5]) * recency_bias
                probs = probs / probs.sum()  # Normalize
                
                for b in range(batch_size):
                    # Add some noise for realism
                    noisy_probs = probs + np.random.randn(n_history_tokens) * 0.05
                    noisy_probs = np.maximum(noisy_probs, 0.01)
                    noisy_probs = noisy_probs / noisy_probs.sum()
                    decoder_attn[b, i, :] = torch.from_numpy(noisy_probs)
            
            attn_dict[f'layer_{layer_idx}_history_attn'] = decoder_attn
        
        # Decoder-encoder cross-attention for each layer
        for layer_idx in range(n_decoder_layers):
            # Shape: (batch, chunk_size, n_encoder_tokens)
            # Shows how each action position attends to encoder tokens
            encoder_cross_attn = torch.zeros(batch_size, chunk_size, n_encoder_tokens)
            
            for i in range(chunk_size):
                # Different actions attend to different encoder token types
                # Early actions focus on latent/state, later actions on images
                latent_focus = 1.0 - (i / chunk_size) * 0.6  # Decreases as we go through chunk
                image_focus = (i / chunk_size) * 0.8  # Increases as we go through chunk
                
                # Create attention pattern
                attn_pattern = torch.zeros(n_encoder_tokens)
                
                # Latent token (index 0) gets high attention from early actions
                attn_pattern[0] = 0.4 * latent_focus
                
                # State token (index 1) gets moderate attention
                attn_pattern[1] = 0.2
                
                # Image tokens (indices 2+) share the remaining attention
                remaining_attn = 1.0 - attn_pattern[0] - attn_pattern[1]
                if n_encoder_tokens > 2:
                    attn_pattern[2:] = remaining_attn / (n_encoder_tokens - 2)
                
                # Add some variation
                noise = torch.randn(n_encoder_tokens) * 0.05
                attn_pattern = torch.clamp(attn_pattern + noise, min=0.01)
                attn_pattern = attn_pattern / attn_pattern.sum()
                
                for b in range(batch_size):
                    encoder_cross_attn[b, i, :] = attn_pattern
            
            attn_dict[f'layer_{layer_idx}_encoder_attn'] = encoder_cross_attn
        
        # Encoder-history cross-attention
        # Shape: (batch, n_encoder_tokens, n_history_tokens)
        encoder_attn = torch.zeros(batch_size, n_encoder_tokens, n_history_tokens)
        
        for i in range(n_encoder_tokens):
            # Different encoder token types attend differently:
            # - Latent (token 0): attends to most recent
            # - State (token 1): attends evenly
            # - Image patches (tokens 2+): attend based on spatial position
            
            if i == 0:  # Latent token
                probs = np.array([0.05, 0.1, 0.25, 0.6])
            elif i == 1:  # State token
                probs = np.array([0.25, 0.25, 0.25, 0.25])
            else:  # Image tokens
                # Spatial pattern: some patches attend to recent, others to older history
                spatial_idx = i - 2
                phase = (spatial_idx / (n_encoder_tokens - 2)) * np.pi
                recent_weight = 0.3 + 0.5 * np.cos(phase)
                probs = np.array([
                    0.1 * (1 - recent_weight),
                    0.2 * (1 - recent_weight),
                    0.3 * (1 - recent_weight),
                    0.4 + 0.6 * recent_weight
                ])
            
            probs = probs / probs.sum()
            
            for b in range(batch_size):
                # Add noise
                noisy_probs = probs + np.random.randn(n_history_tokens) * 0.03
                noisy_probs = np.maximum(noisy_probs, 0.01)
                noisy_probs = noisy_probs / noisy_probs.sum()
                encoder_attn[b, i, :] = torch.from_numpy(noisy_probs)
        
        attn_dict['encoder_history_cross_attn'] = encoder_attn
        episode_attn_history.append(attn_dict)
    
    return episode_attn_history


def test_single_timestep_visualization():
    """Test visualization of attention at a single timestep."""
    print("=" * 70)
    print("TEST 1: Single Timestep Attention Visualization")
    print("=" * 70)
    
    # Create dummy data for one timestep
    batch_size = 4
    chunk_size = 100
    n_history = 4
    
    # Decoder history attention
    decoder_attn = torch.zeros(batch_size, chunk_size, n_history)
    for b in range(batch_size):
        for i in range(chunk_size):
            # Earlier actions attend to most recent history
            weight = 1.0 - (i / chunk_size) * 0.5
            probs = np.array([0.1, 0.2, 0.3, 0.4]) * weight
            probs = probs / probs.sum()
            decoder_attn[b, i, :] = torch.from_numpy(probs)
    
    print(f"\nDecoder History Attention Shape: {decoder_attn.shape}")
    print(f"  - Rows (Y-axis): {chunk_size} action chunk positions")
    print(f"  - Columns (X-axis): {n_history} history tokens")
    
    # Create visualization
    output_dir = Path("outputs/test_attention_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_attention_heatmap(
        decoder_attn,
        title="Single Timestep: Decoder → History Attention",
        save_path=output_dir / "test_single_decoder_history.png",
        figsize=(8, 12)
    )
    plt.close(fig)
    print(f"\n✓ Saved: {output_dir / 'test_single_decoder_history.png'}")


def test_episode_visualization():
    """Test episode-level attention visualization."""
    print("\n" + "=" * 70)
    print("TEST 2: Episode-Level Attention Visualization")
    print("=" * 70)

    # Create realistic episode data
    episode_data = create_dummy_attention_data(
        batch_size=2,
        chunk_size=100,
        n_history_tokens=4,
        n_encoder_tokens=100,
        n_decoder_layers=1,
        timesteps=15
    )

    print(f"\nEpisode statistics:")
    print(f"  - Timesteps: {len(episode_data)}")
    print(f"  - Decoder layers: 1")
    print(f"  - Attention types captured: {list(episode_data[0].keys())}")

    decoder_shape = episode_data[0]['layer_0_history_attn'].shape
    encoder_shape = episode_data[0]['encoder_history_cross_attn'].shape

    print(f"\nDecoder History Attention:")
    print(f"  - Shape per timestep: {decoder_shape}")
    print(f"  - Visualization: {decoder_shape[1]} rows × {decoder_shape[2]} columns")

    print(f"\nEncoder-History Cross-Attention:")
    print(f"  - Shape per timestep: {encoder_shape}")
    print(f"  - Encoder tokens breakdown (example with 2 cameras):")
    print(f"    * 1 latent token")
    print(f"    * 1 robot state token")
    print(f"    * 98 image tokens (2 cameras × 49 patches)")
    print(f"    = 100 total encoder tokens")
    print(f"  - Visualization: {encoder_shape[1]} rows × {encoder_shape[2]} columns")

    output_dir = Path("outputs/test_attention_viz")

    # Generate episode heatmap (simplified version)
    fig = plot_episode_attention_heatmap(
        episode_data,
        save_path=output_dir / "test_episode_attention.png",
        show_first_action_only=True,
        average_image_patches=True
    )
    plt.close(fig)
    print(f"\n✓ Saved: {output_dir / 'test_episode_attention.png'}")
    print(f"  - Decoder: Shows only first action (1 row) × 4 history tokens")
    print(f"  - Encoder: Shows latent + robot state + averaged images (3 rows) × 4 history tokens")

    # Generate detailed encoder visualization with labels (also simplified)
    encoder_labels = ["latent", "robot_state"] + [f"img_{i}" for i in range(98)]
    fig = plot_detailed_encoder_history_attention(
        episode_data,
        encoder_token_types=encoder_labels,
        save_path=output_dir / "test_encoder_detailed.png",
        average_image_patches=True
    )
    plt.close(fig)
    print(f"✓ Saved: {output_dir / 'test_encoder_detailed.png'}")
    print(f"  - Shows latent, robot_state, and avg_images (3 tokens total)")


def test_attention_analysis():
    """Test attention pattern analysis."""
    print("\n" + "=" * 70)
    print("TEST 3: Attention Pattern Analysis")
    print("=" * 70)
    
    episode_data = create_dummy_attention_data(
        batch_size=2,
        chunk_size=100,
        n_history_tokens=4,
        n_encoder_tokens=100,
        timesteps=10
    )
    
    # Analyze decoder attention
    decoder_attn = episode_data[0]['layer_0_history_attn']
    decoder_analysis = analyze_attention_patterns(
        decoder_attn.mean(dim=0),
        attn_type="decoder_history"
    )
    
    print("\nDecoder History Attention Analysis:")
    print(f"  - Average entropy: {decoder_analysis['avg_entropy']:.3f}")
    print(f"  - Attention sparsity: {decoder_analysis['attention_sparsity']:.3f}")
    print(f"  - Temporal focus (std): {decoder_analysis['temporal_focus']:.3f}")
    print(f"  - Mean max attention position: {decoder_analysis['mean_max_attention']:.1f}")
    
    # Analyze encoder attention
    encoder_attn = episode_data[0]['encoder_history_cross_attn']
    encoder_analysis = analyze_attention_patterns(
        encoder_attn.mean(dim=0),
        attn_type="encoder_history"
    )
    
    print("\nEncoder-History Cross-Attention Analysis:")
    print(f"  - Average entropy: {encoder_analysis['avg_entropy']:.3f}")
    print(f"  - Attention sparsity: {encoder_analysis['attention_sparsity']:.3f}")
    print(f"  - Temporal focus (std): {encoder_analysis['temporal_focus']:.3f}")
    
    # Generate summary report
    output_dir = Path("outputs/test_attention_viz")
    report = create_attention_summary_report(
        episode_data,
        save_path=output_dir / "test_attention_report.txt"
    )
    print(f"\n✓ Saved: {output_dir / 'test_attention_report.txt'}")
    print("\nReport preview:")
    print(report[:500] + "..." if len(report) > 500 else report)


def test_minimal_visualization():
    """Test with minimal, easy-to-read visualization."""
    print("\n" + "=" * 70)
    print("TEST 4: Minimal Easy-to-Read Visualization")
    print("=" * 70)

    # Create very simple pattern: small sizes for clarity
    batch_size = 1
    chunk_size = 10  # Much smaller for readability
    n_history = 4

    # Create clear pattern: each action attends primarily to one history token
    simple_attn = torch.zeros(batch_size, chunk_size, n_history)
    for i in range(chunk_size):
        # Rotate which history token gets most attention
        dominant_idx = i % n_history
        probs = torch.ones(n_history) * 0.1
        probs[dominant_idx] = 0.7
        simple_attn[0, i, :] = probs / probs.sum()

    print(f"\nSimplified visualization with:")
    print(f"  - {chunk_size} action positions (rows)")
    print(f"  - {n_history} history tokens (columns)")

    output_dir = Path("outputs/test_attention_viz")
    fig = plot_attention_heatmap(
        simple_attn,
        title="Simplified: 10 Actions × 4 History Tokens",
        save_path=output_dir / "test_simple_readable.png",
        figsize=(6, 8)
    )
    plt.close(fig)
    print(f"\n✓ Saved: {output_dir / 'test_simple_readable.png'}")


def test_simplified_mact_visualization():
    """Test the simplified MACT-style visualizations as requested by user."""
    print("\n" + "=" * 70)
    print("TEST 5: Simplified MACT Visualizations (User Requirements)")
    print("=" * 70)

    # Create full-size MACT data (100 action positions, 100 encoder tokens)
    episode_data = create_dummy_attention_data(
        batch_size=2,
        chunk_size=100,
        n_history_tokens=4,
        n_encoder_tokens=100,
        n_decoder_layers=1,
        timesteps=10
    )

    output_dir = Path("outputs/test_attention_viz")

    print(f"\nOriginal token counts:")
    print(f"  - Decoder: 100 action positions × 4 history tokens")
    print(f"  - Encoder: 100 encoder tokens × 4 history tokens")

    print(f"\nAfter simplification:")
    print(f"  - Decoder: 1 action position (first/next) × 4 history tokens")
    print(f"  - Encoder: 3 tokens (latent + robot_state + avg_images) × 4 history tokens")

    # Test simplified episode visualization
    fig = plot_episode_attention_heatmap(
        episode_data,
        save_path=output_dir / "test_mact_simplified.png",
        show_first_action_only=True,
        average_image_patches=True,
        figsize=(12, 5)  # Wider to accommodate side-by-side plots
    )
    plt.close(fig)
    print(f"\n✓ Saved simplified MACT visualization: {output_dir / 'test_mact_simplified.png'}")

    # Test single timestep with first action only
    decoder_attn = episode_data[5]['layer_0_history_attn']  # Middle timestep
    fig = plot_attention_heatmap(
        decoder_attn,
        title="Single Timestep: First Action Only × History Tokens",
        save_path=output_dir / "test_first_action_only.png",
        show_first_action_only=True,
        figsize=(8, 3)  # Very compact
    )
    plt.close(fig)
    print(f"✓ Saved first action visualization: {output_dir / 'test_first_action_only.png'}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MACT ATTENTION VISUALIZATION TEST SUITE")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_single_timestep_visualization()
    test_episode_visualization()
    test_attention_analysis()
    test_minimal_visualization()
    test_simplified_mact_visualization()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    print("\nVisualization files saved to: outputs/test_attention_viz/")
    print("\nKey findings:")
    print("  • Decoder attention: 100 action positions × 4 history tokens")
    print("  • Encoder attention: ~100 encoder tokens × 4 history tokens")
    print("    (1 latent + 1 state + N×49 image patches, where N = # cameras)")
    print("\nFor full-size plots, the Y-axis may show many tokens.")
    print("Consider using subsampling or aggregation for clearer visualizations.")


if __name__ == "__main__":
    main()

