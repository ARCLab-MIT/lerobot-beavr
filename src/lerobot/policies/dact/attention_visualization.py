"""
Attention visualization utilities for MACT models.

This module provides functions to visualize attention patterns in transformer decoders,
particularly focusing on cross-attention between action predictions and history tokens.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heatmap(attn_weights: torch.Tensor,
                          title: str = "Cross-Attention: Decoder Queries vs History Keys",
                          save_path: str | None = None,
                          figsize: tuple = (10, 8),
                          max_yticks: int = 20,
                          show_first_action_only: bool = False) -> plt.Figure:
    """
    Plot attention heatmap for decoder cross-attention with history tokens.

    Args:
        attn_weights: (B, chunk_size, n_history_tokens) attention weights
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        max_yticks: Maximum number of y-axis tick labels to show (for readability)
        show_first_action_only: If True, only show the first action position (next predicted action)

    Returns:
        matplotlib Figure object
    """
    # Average over batch dimension if present
    attn_avg = attn_weights.mean(dim=0).cpu().numpy() if attn_weights.dim() == 3 else attn_weights.cpu().numpy()

    # If showing only first action, select it
    if show_first_action_only and attn_avg.ndim == 2 and attn_avg.shape[0] > 1:
        attn_avg = attn_avg[0:1]  # Keep as 2D array with shape (1, n_history)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using matplotlib
    im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('History Token Position', fontsize=12)
    ax.set_ylabel('Action Chunk Position', fontsize=12)

    # X-axis: show all history tokens (usually just 4)
    ax.set_xticks(range(attn_avg.shape[1]))
    ax.set_xticklabels([f'H-{i}' for i in range(attn_avg.shape[1])])

    # Y-axis: intelligently subsample for readability
    n_rows = attn_avg.shape[0]
    if n_rows <= max_yticks:
        # Show all labels if few enough
        ytick_positions = list(range(n_rows))
        ytick_labels = [f'A-{i}' for i in range(n_rows)]
    else:
        # Subsample: show first, last, and evenly spaced intermediate ticks
        step = max(1, n_rows // (max_yticks - 2))
        ytick_positions = [0] + list(range(step, n_rows - 1, step)) + [n_rows - 1]
        ytick_labels = [f'A-{i}' for i in ytick_positions]

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attention_evolution(attn_history: list[dict[str, torch.Tensor]],
                           layer_idx: int = 0,
                           save_path: str | None = None) -> plt.Figure:
    """
    Plot how attention patterns evolve over time steps.

    Args:
        attn_history: List of attention weight dictionaries from each time step
        layer_idx: Which decoder layer to visualize
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, len(attn_history), figsize=(4*len(attn_history), 6))
    if len(attn_history) == 1:
        axes = [axes]

    for i, attn_dict in enumerate(attn_history):
        attn_weights = attn_dict[f'layer_{layer_idx}_history_attn']
        attn_avg = attn_weights.mean(dim=0).cpu().numpy()

        # Create heatmap using matplotlib
        axes[i].imshow(attn_avg, cmap='viridis', aspect='equal')
        axes[i].set_title(f'Time Step {i+1}')
        axes[i].set_xlabel('History Tokens')
        axes[i].set_ylabel('Action Positions')

        # Set tick labels
        axes[i].set_xticks(range(attn_avg.shape[1]))
        axes[i].set_yticks(range(attn_avg.shape[0]))
        axes[i].set_xticklabels([f'H-{j}' for j in range(attn_avg.shape[1])])
        axes[i].set_yticklabels([f'A-{j}' for j in range(attn_avg.shape[0])])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_single_timestep_attention(attn_dict: dict[str, torch.Tensor],
                                  save_path: str | None = None,
                                  figsize: tuple = None,
                                  dpi: int = 100,
                                  max_yticks: int = 20) -> plt.Figure:
    """
    Plot attention heatmap for a single timestep.

    Args:
        attn_dict: Single attention dictionary from one timestep
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches. If None, uses dynamic sizing based on number of plots
        dpi: DPI for saving the figure (affects final pixel dimensions)
        max_yticks: Maximum number of y-axis tick labels to show for readability

    Returns:
        matplotlib Figure object
    """
    if not attn_dict:
        return None

    # Filter attention data and combine decoder attentions
    decoder_history_attn = None
    decoder_encoder_attn = None
    encoder_history_attn = None

    for key, weights in attn_dict.items():
        if weights is None:
            continue

        if key == 'decoder_history_attn':
            decoder_history_attn = weights
        elif key == 'decoder_encoder_attn':
            decoder_encoder_attn = weights
        elif key == 'encoder_history_cross_attn':
            encoder_history_attn = weights

    # Create plot keys: one for decoder attention, one for post-hoc analysis
    plot_keys = []
    filtered_attn = {}

    # Combine decoder attention to both encoder and history tokens
    if decoder_encoder_attn is not None and decoder_history_attn is not None:
        # Concatenate along the key dimension to show attention to all tokens
        combined_decoder_attn = torch.cat([decoder_encoder_attn, decoder_history_attn], dim=-1)
        plot_keys.append('decoder_combined_attn')
        filtered_attn['decoder_combined_attn'] = combined_decoder_attn

    # Add encoder-history cross-attention (post-hoc analysis)
    if encoder_history_attn is not None:
        plot_keys.append('encoder_history_cross_attn')
        filtered_attn['encoder_history_cross_attn'] = encoder_history_attn

    if not plot_keys:
        return None

    # Determine subplot layout - we want 1 row, with columns for each attention type
    n_plots = len(plot_keys)
    # Set figsize dynamically: base width per plot, fixed height
    default_figsize = (4 * n_plots, 5) if figsize is None else figsize
    fig, axes = plt.subplots(1, n_plots, figsize=default_figsize)
    if n_plots == 1:
        axes = [axes]

    for i, layer_key in enumerate(plot_keys):
        avg_weights = None  # Initialize to ensure it's defined

        if layer_key == 'decoder_combined_attn':
            # Process combined decoder attention to both encoder and history tokens
            weights = filtered_attn[layer_key]
            avg_weights = weights.mean(dim=0).cpu().numpy() if weights.dim() == 3 else weights.cpu().numpy()

            # Show only first action attending to all tokens
            if avg_weights.shape[0] > 1:
                avg_weights = avg_weights[0:1]  # (1, total_seq_len)

            # Count non-image tokens to properly group encoder portion
            # Encoder tokens are ordered: [latent, robot_state?, env_state?, image_patches...]
            n_non_image_tokens = 1  # latent always present
            if attn_dict.get('robot_state_present', False):
                n_non_image_tokens += 1
            if attn_dict.get('env_state_present', False):
                n_non_image_tokens += 1

            # Split into encoder and history portions
            encoder_seq_len = n_non_image_tokens
            if attn_dict.get('n_image_features', 0) > 0:
                # Estimate encoder sequence length (this is approximate since we don't know exact number of image patches)
                # For now, assume most of the sequence is image patches
                encoder_seq_len = avg_weights.shape[1] - attn_dict.get('n_history_tokens', 4)  # Subtract history tokens

            encoder_weights = avg_weights[:, :encoder_seq_len]
            history_weights = avg_weights[:, encoder_seq_len:]

            # Process encoder portion (group image patches)
            should_process_images = (encoder_weights.shape[1] > n_non_image_tokens and
                                   not np.allclose(encoder_weights, 0, atol=1e-6))

            if should_process_images:
                non_image_tokens = encoder_weights[:, :n_non_image_tokens]  # (1, n_non_image)
                image_patches = encoder_weights[:, n_non_image_tokens:]  # (1, n_images)

                # Sum up all image patches
                image_max = image_patches.sum(axis=1, keepdims=True)  # (1, 1)
                processed_encoder = np.concatenate([non_image_tokens, image_max], axis=1)  # (1, n_non_image + 1)
            else:
                # When not processing images, truncate to non-image tokens only
                processed_encoder = encoder_weights[:, :n_non_image_tokens]  # (1, n_non_image)

            # Combine processed encoder with history
            avg_weights = np.concatenate([processed_encoder, history_weights], axis=1)  # (1, processed_len)

            # Swap axes: transpose so tokens are on y-axis, actions on x-axis
            avg_weights = avg_weights.T  # (n_tokens, 1)

        elif layer_key == 'encoder_history_cross_attn':
            weights = filtered_attn[layer_key]

            # Average over batch dimension if present - these are raw attention weights (not normalized)
            avg_weights = weights.mean(dim=0).cpu().numpy() if weights.dim() == 3 else weights.cpu().numpy()

            # Count non-image tokens to properly group them
            n_non_image_tokens = 1  # latent always present
            if attn_dict.get('robot_state_present', False):
                n_non_image_tokens += 1
            if attn_dict.get('env_state_present', False):
                n_non_image_tokens += 1

            # Average image patches if requested and there are enough tokens
            # Note: This averages multiple image patch attention weights, but still preserves actual values
            if avg_weights.shape[0] > n_non_image_tokens:  # More tokens than non-image tokens
                non_image_tokens = avg_weights[:n_non_image_tokens]  # (n_non_image, n_history)
                image_patches = avg_weights[n_non_image_tokens:]  # (n_images, n_history)
                image_avg = image_patches.mean(axis=0, keepdims=True)  # (1, n_history)
                avg_weights = np.concatenate([non_image_tokens, image_avg], axis=0)  # (n_non_image + 1, n_history)

        else:
            # Handle other attention types
            weights = filtered_attn[layer_key]

            # Average over batch dimension if present
            avg_weights = weights.mean(dim=0).cpu().numpy() if weights.dim() == 3 else weights.cpu().numpy()

        # Skip if we couldn't get weights for this layer
        if avg_weights is None:
            continue

        # Plot the attention heatmap for all types
        # Each plot will auto-scale to its own data range and have its own colorbar
        im = axes[i].imshow(avg_weights, cmap='viridis', aspect='auto', interpolation='nearest')
        # Add colorbar for each plot so each has its own scale legend
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label('Attention Weight', fontsize=12)

        # Set titles and labels based on layer_key
        if layer_key == 'decoder_combined_attn':
            axes[i].set_title('Decoder Final Layer\nCross-Attention', fontsize=13)
            axes[i].set_xlabel('Action Position', fontsize=11)
            axes[i].set_ylabel('Token Position', fontsize=11)
        elif layer_key == 'encoder_history_cross_attn':
            axes[i].set_title('Encoder → History\nCross-Attention (Analysis)', fontsize=13)
            axes[i].set_xlabel('History Token Position', fontsize=11)
            axes[i].set_ylabel('Encoder Token Position', fontsize=11)
        else:
            axes[i].set_title(f'{layer_key.replace("_", " ").title()}\n(Next Action)', fontsize=13)
            axes[i].set_xlabel('Action Position', fontsize=11)
            axes[i].set_ylabel('Token Position', fontsize=11)

        # Set ticks and labels
        n_rows, n_cols = avg_weights.shape

        # X-axis ticks and labels (after axis swap, this is Action Position)
        if layer_key == 'decoder_combined_attn':
            # X-axis: single action position
            axes[i].set_xticks([0])
            axes[i].set_xticklabels(['Next Action'], fontsize=9)
        elif layer_key == 'encoder_history_cross_attn':
            # X-axis: show all history tokens
            axes[i].set_xticks(range(n_cols))
            axes[i].set_xticklabels([f'H-{j}' for j in range(n_cols)])
        else:
            # Default x-axis
            axes[i].set_xticks(range(n_cols))
            axes[i].set_xticklabels([f'A-{j}' for j in range(n_cols)], fontsize=8)

        # Y-axis ticks and labels (after axis swap, this is Token Position for decoder)
        if layer_key == 'decoder_combined_attn':
            # Y-axis: combined tokens (encoder + history)
            combined_labels = []

            # Add encoder token labels
            combined_labels.append('Latent')
            if attn_dict.get('robot_state_present', False):
                combined_labels.append('Robot State')
            if attn_dict.get('env_state_present', False):
                combined_labels.append('Env State')
            if attn_dict.get('n_image_features', 0) > 0:
                combined_labels.append('Summed Images')

            # Add history token labels
            n_history = attn_dict.get('n_history_tokens', 4)
            for j in range(n_history):
                combined_labels.append(f'H-{j}')

            axes[i].set_yticks(range(len(combined_labels)))
            axes[i].set_yticklabels(combined_labels, fontsize=7)
        elif layer_key == 'encoder_history_cross_attn':
            # Y-axis: encoder tokens - use dynamic labeling based on features
            n_non_image_tokens = 1  # latent always present
            encoder_labels = ['Latent']
            if attn_dict.get('robot_state_present', False):
                n_non_image_tokens += 1
                encoder_labels.append('Robot State')
            if attn_dict.get('env_state_present', False):
                n_non_image_tokens += 1
                encoder_labels.append('Env State')
            if attn_dict.get('n_image_features', 0) > 0:
                encoder_labels.append('Avg Images')

            if len(encoder_labels) == n_rows:
                # We have labels for all tokens
                axes[i].set_yticks(range(n_rows))
                axes[i].set_yticklabels(encoder_labels, fontsize=9)
            else:
                # Fallback to default subsampling
                if n_rows <= max_yticks:
                    ytick_positions = list(range(n_rows))
                    ytick_labels = [f'E-{j}' for j in range(n_rows)]
                else:
                    step = max(1, n_rows // (max_yticks - 2))
                    ytick_positions = [0] + list(range(step, n_rows - 1, step)) + [n_rows - 1]
                    ytick_labels = [f'E-{j}' for j in ytick_positions]
                axes[i].set_yticks(ytick_positions)
                axes[i].set_yticklabels(ytick_labels, fontsize=8)
        else:
            # Default y-axis
            if n_rows <= max_yticks:
                ytick_positions = list(range(n_rows))
                ytick_labels = [f'T-{j}' for j in range(n_rows)]
            else:
                step = max(1, n_rows // (max_yticks - 2))
                ytick_positions = [0] + list(range(step, n_rows - 1, step)) + [n_rows - 1]
                ytick_labels = [f'T-{j}' for j in ytick_positions]
            axes[i].set_yticks(ytick_positions)
            axes[i].set_yticklabels(ytick_labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Ensure even pixel dimensions by adjusting figure size
        current_figsize = fig.get_size_inches()  # Get the actual figure size used
        fig_width_px = int(current_figsize[0] * dpi)
        fig_height_px = int(current_figsize[1] * dpi)

        # Make dimensions even
        if fig_width_px % 2 != 0:
            fig_width_px += 1
        if fig_height_px % 2 != 0:
            fig_height_px += 1

        # Set figure size to achieve even pixel dimensions
        fig.set_size_inches(fig_width_px / dpi, fig_height_px / dpi)

        plt.savefig(save_path, dpi=dpi, bbox_inches=None)  # Don't use tight cropping

    return fig


def plot_episode_attention_heatmap(episode_attn_history: list[dict[str, torch.Tensor]],
                                 save_path: str | None = None,
                                 max_yticks: int = 20,
                                 show_first_action_only: bool = False,
                                 average_image_patches: bool = False,
                                 figsize: tuple = None) -> plt.Figure:
    """
    Generate attention heatmap averaged over an entire episode.

    Args:
        episode_attn_history: List of attention weight dictionaries from episode
        save_path: Optional path to save the figure
        max_yticks: Maximum number of y-axis tick labels to show (for readability)
        show_first_action_only: If True, only show the first action position for decoder attention
        average_image_patches: If True, average all image patches into single token for encoder attention
        figsize: Figure size as (width, height). If None, uses default.

    Returns:
        matplotlib Figure object
    """
    if not episode_attn_history:
        return None

    # Aggregate attention weights across the episode
    episode_attn = {}
    feature_info = {}  # Store feature information from the first valid dict
    for attn_dict in episode_attn_history:
        if attn_dict is None:
            continue  # Skip None entries
        for key, weights in attn_dict.items():
            if weights is None and key not in ['robot_state_present', 'env_state_present', 'n_image_features']:
                continue  # Skip None weights but keep feature info
            if key not in episode_attn:
                episode_attn[key] = []
            if torch.is_tensor(weights):
                episode_attn[key].append(weights.cpu())
            else:
                # Store feature info (non-tensor values)
                episode_attn[key] = weights
                feature_info[key] = weights

    # Determine which keys to plot and their dimensions
    plot_keys = []
    for key in episode_attn:
        if key == 'decoder_history_attn' or key == 'encoder_history_cross_attn':
            plot_keys.append(key)

    if not plot_keys:
        return None

    # Create heatmaps for each attention type
    default_figsize = (8*len(plot_keys), 10) if figsize is None else figsize
    fig, axes = plt.subplots(1, len(plot_keys), figsize=default_figsize)
    if len(plot_keys) == 1:
        axes = [axes]

    for i, layer_key in enumerate(plot_keys):
        weight_list = episode_attn[layer_key]
        weights_tensor = torch.stack(weight_list, dim=0)  # (timesteps, ...)

        if layer_key == 'encoder_history_cross_attn':
            # Shape: (timesteps, batch, encoder_seq_len, n_history)
            avg_weights = weights_tensor.mean(dim=[0, 1]).numpy()  # (encoder_seq_len, n_history)

            # Count non-image tokens
            n_non_image_tokens = 1  # latent always present
            if feature_info.get('robot_state_present', False):
                n_non_image_tokens += 1
            if feature_info.get('env_state_present', False):
                n_non_image_tokens += 1

            # If averaging image patches, group tokens and average
            if average_image_patches and len(avg_weights) > n_non_image_tokens:  # More tokens than non-image tokens
                non_image_tokens = avg_weights[:n_non_image_tokens]  # (n_non_image, n_history)
                image_patches = avg_weights[n_non_image_tokens:]  # (n_images, n_history)
                image_avg = image_patches.mean(axis=0, keepdims=True)  # (1, n_history)
                avg_weights = np.concatenate([non_image_tokens, image_avg], axis=0)  # (n_non_image + 1, n_history)

            # Create heatmap using matplotlib
            # Auto-scale to data range for this plot
            im = axes[i].imshow(avg_weights, cmap='viridis', aspect='auto', interpolation='nearest')
            # Add colorbar for each plot so each has its own scale legend
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label('Attention Weight', fontsize=12)

            axes[i].set_title('Encoder → History\nCross-Attention', fontsize=13)
            axes[i].set_xlabel('History Token Position', fontsize=11)
            axes[i].set_ylabel('Encoder Token Position', fontsize=11)

            # X-axis: show all history tokens (usually just 4)
            n_history = avg_weights.shape[1]
            axes[i].set_xticks(range(n_history))
            axes[i].set_xticklabels([f'H-{j}' for j in range(n_history)])

            # Y-axis: with averaging, should be much fewer tokens
            n_encoder = avg_weights.shape[0]

            # Use dynamic labeling based on features
            encoder_labels = ['Latent']
            if feature_info.get('robot_state_present', False):
                encoder_labels.append('Robot State')
            if feature_info.get('env_state_present', False):
                encoder_labels.append('Env State')
            if average_image_patches and feature_info.get('n_image_features', 0) > 0:
                encoder_labels.append('Avg Images')

            if len(encoder_labels) == n_encoder:
                # We have labels for all tokens
                axes[i].set_yticks(range(n_encoder))
                axes[i].set_yticklabels(encoder_labels, fontsize=9)
            else:
                # Default subsampling
                if n_encoder <= max_yticks:
                    ytick_positions = list(range(n_encoder))
                    ytick_labels = [f'E-{j}' for j in range(n_encoder)]
                else:
                    step = max(1, n_encoder // (max_yticks - 2))
                    ytick_positions = [0] + list(range(step, n_encoder - 1, step)) + [n_encoder - 1]
                    ytick_labels = [f'E-{j}' for j in ytick_positions]
                axes[i].set_yticks(ytick_positions)
                axes[i].set_yticklabels(ytick_labels, fontsize=8)

        else:
            # Original decoder history attention: (timesteps, batch, chunk_size, n_history_tokens)
            avg_weights = weights_tensor.mean(dim=[0, 1]).numpy()  # (chunk_size, n_history_tokens)

            # If showing only first action, select it
            if show_first_action_only and avg_weights.shape[0] > 1:
                avg_weights = avg_weights[0:1]  # Keep as 2D array with shape (1, n_history)

            # Create heatmap using matplotlib
            im = axes[i].imshow(avg_weights, cmap='viridis', aspect='auto', interpolation='nearest')
            # Add colorbar for each plot so each has its own scale legend
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label('Attention Weight', fontsize=12)

            title_suffix = " (First Action Only)" if show_first_action_only else ""
            axes[i].set_title(f'{layer_key.replace("_", " ").title()}\nEpisode Average{title_suffix}', fontsize=13)
            axes[i].set_xlabel('History Token Position', fontsize=11)
            axes[i].set_ylabel('Action Position', fontsize=11)

            # X-axis: show all history tokens
            n_history = avg_weights.shape[1]
            axes[i].set_xticks(range(n_history))
            axes[i].set_xticklabels([f'H-{j}' for j in range(n_history)])

            # Y-axis: with first action only, should be just 1 token
            n_actions = avg_weights.shape[0]
            if show_first_action_only and n_actions == 1:
                axes[i].set_yticks([0])
                axes[i].set_yticklabels(['Next Action'], fontsize=9)
            else:
                # Default subsampling
                if n_actions <= max_yticks:
                    ytick_positions = list(range(n_actions))
                    ytick_labels = [f'A-{j}' for j in range(n_actions)]
                else:
                    step = max(1, n_actions // (max_yticks - 2))
                    ytick_positions = [0] + list(range(step, n_actions - 1, step)) + [n_actions - 1]
                    ytick_labels = [f'A-{j}' for j in ytick_positions]
                axes[i].set_yticks(ytick_positions)
                axes[i].set_yticklabels(ytick_labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode attention heatmap saved to: {save_path}")

    return fig


def analyze_attention_patterns(attn_weights: torch.Tensor, attn_type: str = "decoder_history") -> dict[str, float]:
    """
    Analyze attention patterns to extract insights.

    Args:
        attn_weights: Attention weights tensor with different shapes depending on type:
            - decoder_history: (B, chunk_size, n_history_tokens) or (chunk_size, n_history_tokens)
            - encoder_history: (B, encoder_seq_len, n_history_tokens) or (encoder_seq_len, n_history_tokens)
        attn_type: Type of attention ("decoder_history" or "encoder_history")

    Returns:
        Dictionary with attention pattern statistics
    """
    # Average over batch dimension if present
    attn_avg = attn_weights.mean(dim=0) if attn_weights.dim() == 3 else attn_weights

    if attn_type == "encoder_history":
        # For encoder-history attention: (encoder_seq_len, n_history_tokens)
        # Find which history token gets most attention for each encoder token
        max_attn_positions = attn_avg.argmax(dim=1)  # (encoder_seq_len,)

        # Calculate entropy of attention distribution for each encoder token
        entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-10), dim=1)

        # Calculate attention sparsity (fraction of weights above threshold)
        sparsity = (attn_avg > 0.1).float().mean()

        # Calculate temporal focus (how spread out attention is across history)
        temporal_focus = max_attn_positions.float().std()

        return {
            'avg_entropy': entropy.mean().item(),
            'max_attention_positions': max_attn_positions.tolist(),
            'attention_sparsity': sparsity.item(),
            'temporal_focus': temporal_focus.item(),
            'mean_max_attention': max_attn_positions.float().mean().item(),
        }
    else:
        # Original decoder history attention: (chunk_size, n_history_tokens)
        # Find which history token gets most attention for each action position
        max_attn_positions = attn_avg.argmax(dim=1)  # (chunk_size,)

        # Calculate entropy of attention distribution for each query
        entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-10), dim=1)

        # Calculate attention sparsity (fraction of weights above threshold)
        sparsity = (attn_avg > 0.1).float().mean()

        # Calculate temporal focus (how spread out attention is across history)
        temporal_focus = max_attn_positions.float().std()

        return {
            'avg_entropy': entropy.mean().item(),
            'max_attention_positions': max_attn_positions.tolist(),
            'attention_sparsity': sparsity.item(),
            'temporal_focus': temporal_focus.item(),
            'mean_max_attention': max_attn_positions.float().mean().item(),
        }


def create_attention_summary_report(episode_attn_history: list[dict[str, torch.Tensor]],
                                  save_path: str | None = None) -> str:
    """
    Create a summary report of attention patterns over an episode.

    Args:
        episode_attn_history: List of attention weight dictionaries from episode
        save_path: Optional path to save the report

    Returns:
        Summary report as string
    """
    if not episode_attn_history:
        return "No attention data available."

    report_lines = ["Attention Pattern Analysis Report", "=" * 40, ""]

    # Analyze each attention type
    for layer_key in episode_attn_history[0]:
        # Skip keys that are not attention weights
        if layer_key not in ['decoder_history_attn', 'decoder_encoder_attn', 'encoder_history_cross_attn']:
            continue

        report_lines.append(f"Attention Type: {layer_key}")
        report_lines.append("-" * 30)

        layer_weights = [attn_dict[layer_key] for attn_dict in episode_attn_history]
        weights_tensor = torch.stack(layer_weights, dim=0)  # (timesteps, batch, ...)

        if layer_key == 'encoder_history_cross_attn':
            # Shape: (timesteps, batch, encoder_seq_len, n_history)
            episode_avg = weights_tensor.mean(dim=[0, 1])  # (encoder_seq_len, n_history)
            attn_type = "encoder_history"
        elif layer_key in ['decoder_history_attn', 'decoder_encoder_attn']:
            # Shape: (timesteps, batch, chunk_size, n_tokens)
            episode_avg = weights_tensor.mean(dim=[0, 1])  # (chunk_size, n_tokens)
            attn_type = "decoder_history"
        else:
            continue

        # Episode-level analysis
        episode_analysis = analyze_attention_patterns(episode_avg, attn_type)

        report_lines.append(f"  Average Entropy: {episode_analysis['avg_entropy']:.3f}")
        report_lines.append(f"  Attention Sparsity: {episode_analysis['attention_sparsity']:.3f}")
        report_lines.append(f"  Temporal Focus (std): {episode_analysis['temporal_focus']:.3f}")
        report_lines.append(f"  Mean Max Attention Position: {episode_analysis['mean_max_attention']:.1f}")
        report_lines.append("")

        # Per-timestep analysis (summary)
        timestep_entropies = []
        timestep_sparsities = []
        for t_weights in layer_weights:
            t_analysis = analyze_attention_patterns(t_weights.mean(dim=0))
            timestep_entropies.append(t_analysis['avg_entropy'])
            timestep_sparsities.append(t_analysis['attention_sparsity'])

        report_lines.append(f"  Entropy over time: {np.mean(timestep_entropies):.3f} ± {np.std(timestep_entropies):.3f}")
        report_lines.append(f"  Sparsity over time: {np.mean(timestep_sparsities):.3f} ± {np.std(timestep_sparsities):.3f}")
        report_lines.append("")

    report = "\n".join(report_lines)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Attention summary report saved to: {save_path}")

    return report


def plot_detailed_encoder_history_attention(episode_attn_history: list[dict[str, torch.Tensor]],
                                          encoder_token_types: list[str] | None = None,
                                          save_path: str | None = None,
                                          max_yticks: int = 25,
                                          average_image_patches: bool = False) -> plt.Figure:
    """
    Create a detailed visualization of encoder-to-history attention with labeled token types.

    Args:
        episode_attn_history: List of attention weight dictionaries from episode
        encoder_token_types: Optional list describing each encoder token type
                           (e.g., ['latent', 'robot_state', 'env_state', 'img_0_0', 'img_0_1', ...])
        save_path: Optional path to save the figure
        max_yticks: Maximum number of y-axis tick labels to show (for readability)
        average_image_patches: If True, average all image patches into single token

    Returns:
        matplotlib Figure object
    """
    if not episode_attn_history or 'encoder_history_cross_attn' not in episode_attn_history[0]:
        return None

    # Get encoder-history attention weights
    encoder_attn_list = [attn_dict['encoder_history_cross_attn'] for attn_dict in episode_attn_history]
    weights_tensor = torch.stack(encoder_attn_list, dim=0)  # (timesteps, batch, encoder_seq_len, n_history)
    avg_weights = weights_tensor.mean(dim=[0, 1]).numpy()  # (encoder_seq_len, n_history)

    # If averaging image patches, modify the weights and labels
    if average_image_patches and len(avg_weights) > 2:  # More than latent + state tokens
        # Assume first token is latent, second is robot state, rest are images
        latent_state_avg = avg_weights[:2]  # (2, n_history)
        image_patches = avg_weights[2:]  # (n_images, n_history)
        image_avg = image_patches.mean(axis=0, keepdims=True)  # (1, n_history)
        avg_weights = np.concatenate([latent_state_avg, image_avg], axis=0)  # (3, n_history)

        # Update encoder_token_types for averaged case
        if encoder_token_types:
            encoder_token_types = ['latent', 'robot_state', 'avg_images']

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 14))

    # Create heatmap using matplotlib
    # Auto-scale to data range
    im = ax.imshow(avg_weights, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)

    ax.set_title('Encoder → History Cross-Attention\n(Episode Average)', fontsize=14, pad=20)
    ax.set_xlabel('History Token Position', fontsize=12)
    ax.set_ylabel('Encoder Token Position', fontsize=12)

    # X-axis: show all history tokens
    n_history = avg_weights.shape[1]
    ax.set_xticks(range(n_history))
    ax.set_xticklabels([f'H-{j}' for j in range(n_history)])

    # Y-axis: handle encoder tokens with intelligent subsampling
    n_encoder = avg_weights.shape[0]

    # If we have token type labels and not too many tokens, show them all
    if encoder_token_types and len(encoder_token_types) == n_encoder and n_encoder <= max_yticks:
        ax.set_yticks(range(n_encoder))
        ax.set_yticklabels(encoder_token_types, fontsize=8)
        ax.tick_params(axis='y', labelrotation=0)
    elif encoder_token_types and len(encoder_token_types) == n_encoder:
        # Too many tokens - subsample but preserve important ones (latent, state, etc.)
        # Always include first few non-image tokens and subsample image tokens
        important_indices = []
        image_indices = []

        for idx, label in enumerate(encoder_token_types):
            if 'img' in label or 'cam' in label:
                image_indices.append(idx)
            else:
                important_indices.append(idx)

        # Keep all important tokens, subsample images
        if len(image_indices) > 0:
            n_image_ticks = max(5, max_yticks - len(important_indices))
            step = max(1, len(image_indices) // n_image_ticks)
            sampled_images = image_indices[::step]
            if image_indices[-1] not in sampled_images:
                sampled_images.append(image_indices[-1])
            ytick_positions = sorted(important_indices + sampled_images)
        else:
            ytick_positions = important_indices

        ytick_labels = [encoder_token_types[i] for i in ytick_positions]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.tick_params(axis='y', labelrotation=0)
    else:
        # No labels provided or length mismatch - use generic subsampling
        if n_encoder <= max_yticks:
            ytick_positions = list(range(n_encoder))
            ytick_labels = [f'E-{j}' for j in range(n_encoder)]
        else:
            step = max(1, n_encoder // (max_yticks - 2))
            ytick_positions = [0] + list(range(step, n_encoder - 1, step)) + [n_encoder - 1]
            ytick_labels = [f'E-{j}' for j in ytick_positions]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed encoder-history attention plot saved to: {save_path}")

    return fig


def visualize_attention_flow(episode_attn_history: list[dict[str, torch.Tensor]],
                           action_idx: int = 0,
                           save_path: str | None = None) -> plt.Figure:
    """
    Visualize how attention to a specific action position evolves over time.

    Args:
        episode_attn_history: List of attention weight dictionaries from episode
        action_idx: Which action position to focus on
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    layer_key = 'decoder_history_attn'
    if layer_key not in episode_attn_history[0]:
        return None

    # Extract attention weights for specific action position over time
    time_attention = []
    for attn_dict in episode_attn_history:
        weights = attn_dict[layer_key]  # (batch, chunk_size, n_history_tokens)
        action_weights = weights[:, action_idx, :]  # (batch, n_history_tokens)
        time_attention.append(action_weights.mean(dim=0).cpu().numpy())

    time_attention = np.array(time_attention)  # (timesteps, n_history_tokens)

    fig, ax = plt.subplots(figsize=(12, 6))
    for hist_idx in range(time_attention.shape[1]):
        ax.plot(range(len(time_attention)), time_attention[:, hist_idx],
               label=f'History Token {hist_idx}', marker='o', markersize=3)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Attention Flow for Action Position {action_idx} (Final Decoder Layer)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
