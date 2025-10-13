

"""
True Streaming Training Script with Persistent Per-Stream State

This script implements TRUE streaming training with stateful parallel streams:
- B = number of parallel streams (true batch size)
- K = window length (number of sequential frames per stream)

Key innovation: Per-stream cursors that persist across training steps
- Each stream maintains (episode_id, frame_idx) cursor
- Streams advance frame-by-frame continuously
- Only reset specific stream's cache when it hits episode boundary
- History cache persists across windows for continuous streaming

Each training step:
1. Get next K timesteps from B persistent streams
2. Reset cache ONLY for streams that hit episode boundaries (not all streams)
3. Stream through K timesteps with DataLoader (workers + pin_memory for speed)
4. Accumulate gradients across all K timesteps
5. Step the optimizer, advance all stream cursors

Performance optimizations:
- Multi-worker DataLoader for parallel data loading
- Pin memory for faster GPU transfer
- Frames/sec metric (B x K frames per step)

Key differences from other training scripts:
- train.py: Random batches, no temporal ordering
- train_sequential.py: Processes entire episodes sequentially  
- train_sequential_small.py (this): TRUE streaming with persistent per-stream state

Configuration:
- cfg.batch_size = B (number of parallel streams)
- cfg.policy.window_size = K (sequential frames per stream)
- cfg.num_workers = number of data loading workers
"""

import logging
import random
import time
from contextlib import nullcontext
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


class StreamingBatchSampler:
    """
    Custom batch sampler that yields indices for B streams at each timestep.
    Maintains per-stream cursors and only resets streams that hit episode boundaries.
    """

    def __init__(self, episode_data_index: dict[str, torch.Tensor], batch_size: int, window_size: int):
        self.from_list = episode_data_index["from"].tolist()
        self.to_list = episode_data_index["to"].tolist()
        self.num_episodes = len(self.from_list)
        self.batch_size = batch_size
        self.window_size = window_size
        
        # Filter episodes that have at least 1 frame
        self.valid_episodes = []
        for ep_idx in range(self.num_episodes):
            ep_length = self.to_list[ep_idx] - self.from_list[ep_idx]
            if ep_length >= 1:
                self.valid_episodes.append(ep_idx)
        
        if not self.valid_episodes:
            raise ValueError("No valid episodes found")
        
        # Per-stream state: (episode_id, frame_idx) for each of B streams
        self.stream_episode_ids = []
        self.stream_frame_indices = []
        
        # Initialize all streams with random episodes and positions
        for _ in range(batch_size):
            ep_idx = random.choice(self.valid_episodes)
            ep_start = self.from_list[ep_idx]
            ep_end = self.to_list[ep_idx]
            # Start at random position in episode
            frame_idx = random.randint(ep_start, ep_end - 1)
            self.stream_episode_ids.append(ep_idx)
            self.stream_frame_indices.append(frame_idx)
        
        logging.info(f"StreamingBatchSampler: {len(self.valid_episodes)}/{self.num_episodes} valid episodes "
                    f"(B={batch_size} streams, K={window_size} frames per window)")
    
    def get_streams_to_reset(self):
        """Returns list of stream indices that need to be reset."""
        streams_to_reset = []
        for b in range(self.batch_size):
            ep_idx = self.stream_episode_ids[b]
            frame_idx = self.stream_frame_indices[b]
            ep_end = self.to_list[ep_idx]
            
            # Reset if at episode boundary
            if frame_idx >= ep_end:
                streams_to_reset.append(b)
        
        return streams_to_reset
    
    def reset_stream(self, stream_idx: int):
        """Reset a specific stream to a new random episode and position."""
        ep_idx = random.choice(self.valid_episodes)
        ep_start = self.from_list[ep_idx]
        ep_end = self.to_list[ep_idx]
        frame_idx = random.randint(ep_start, ep_end - 1)
        
        self.stream_episode_ids[stream_idx] = ep_idx
        self.stream_frame_indices[stream_idx] = frame_idx
    
    def __iter__(self):
        return self
    
    def __next__(self) -> list[int]:
        """
        Returns B frame indices for the current timestep (one per stream).
        Then advances all streams by 1 frame.
        """
        # Get current frame indices for all B streams
        batch_indices = list(self.stream_frame_indices)
        
        # Advance all streams by 1 frame
        for b in range(self.batch_size):
            self.stream_frame_indices[b] += 1
        
        # Check and reset streams that hit episode boundaries
        streams_to_reset = self.get_streams_to_reset()
        for stream_idx in streams_to_reset:
            self.reset_stream(stream_idx)
        
        return batch_indices


class ParallelWindowIterator:
    """
    Iterator that yields windows of K timesteps for true streaming.
    Uses StreamingBatchSampler to maintain per-stream state.
    """

    def __init__(self, episode_data_index: dict[str, torch.Tensor], batch_size: int, window_size: int):
        self.batch_sampler = StreamingBatchSampler(episode_data_index, batch_size, window_size)
        self.window_size = window_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple[list[list[int]], list[int]]:
        """
        Returns:
            timestep_indices: List of K batches, where each batch contains B frame indices
            streams_reset_at_start: List of stream indices that were reset at window start
        """
        # Track which streams need reset at the start of this window
        streams_reset_at_start = self.batch_sampler.get_streams_to_reset()
        
        # Collect K timesteps of B indices each
        timestep_indices = []
        for _ in range(self.window_size):
            batch_indices = next(self.batch_sampler)
            timestep_indices.append(batch_indices)
        
        return timestep_indices, streams_reset_at_start


def _collate_fn(batch_list):
    """Custom collate function for streaming batches."""
    batch = {}
    for k in batch_list[0].keys():
        v_list = [item[k] for item in batch_list]
        if isinstance(v_list[0], torch.Tensor):
            batch[k] = torch.stack(v_list, dim=0)
        else:
            batch[k] = v_list
    return batch


def _reset_stream_cache(policy, stream_indices: list[int], batch_size: int, device: torch.device):
    """Reset history cache for specific streams only."""
    if not hasattr(policy, "history_encoder") or not hasattr(policy, "history_cache"):
        return
    
    if policy.history_cache is None:
        # Initialize full cache if not exists
        policy.history_cache = policy.history_encoder.init_cache(batch_size)
        # Move cache to device
        def _to_device(x):
            if isinstance(x, (tuple, list)):
                return type(x)(_to_device(t) for t in x)
            return x.to(device) if torch.is_tensor(x) else x
        policy.history_cache = _to_device(policy.history_cache)
        return
    
    # Reset only specific streams
    for stream_idx in stream_indices:
        # Get fresh cache for single stream
        fresh_cache = policy.history_encoder.init_cache(1)
        
        # Replace the stream's cache with fresh one
        # Cache structure depends on the encoder, handle common cases
        if isinstance(policy.history_cache, torch.Tensor):
            # Simple tensor cache: shape [batch_size, ...]
            policy.history_cache[stream_idx:stream_idx+1] = fresh_cache.to(device)
        elif isinstance(policy.history_cache, (tuple, list)):
            # Tuple/list of tensors
            for i in range(len(policy.history_cache)):
                if torch.is_tensor(policy.history_cache[i]):
                    policy.history_cache[i][stream_idx:stream_idx+1] = fresh_cache[i].to(device)


def _streaming_forward_step(
    policy: PreTrainedPolicy,
    batch: Any,
    grad_scaler: GradScaler,
    use_amp: bool,
    accumulation_steps: int,
    device: torch.device,
) -> tuple[float, dict]:
    """
    Perform a single streaming forward and backward pass, accumulating gradients.
    
    Args:
        policy: The policy to train
        batch: Batch of B frames at current timestep
        grad_scaler: Gradient scaler for mixed precision
        use_amp: Whether to use automatic mixed precision
        accumulation_steps: Number of steps to accumulate over (for scaling loss)
        
    Returns:
        loss_value: The loss value (scaled)
        loss_dict: Dictionary with loss components
    """    
    # Forward pass
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, loss_dict = policy.forward(batch)
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
    
    # Backward pass (accumulates gradients)
    grad_scaler.scale(loss).backward()
    
    return float(loss.item()), loss_dict


def _optimizer_step(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
) -> float:
    """
    Perform optimizer step after gradient accumulation.
    
    Args:
        policy: The policy being trained
        optimizer: The optimizer
        grad_scaler: Gradient scaler for mixed precision
        lr_scheduler: Learning rate scheduler
        grad_clip_norm: Gradient clipping norm
        
    Returns:
        grad_norm: The gradient norm after clipping
    """
    # Unscale gradients before clipping
    grad_scaler.unscale_(optimizer)
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), 
        grad_clip_norm, 
        error_if_nonfinite=False
    )
    
    # Optimizer step
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    
    # Learning rate scheduler step
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return float(grad_norm)


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(cfg.to_dict())

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Optional eval env
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # B = batch size (number of parallel streams)
    # K = window size (number of sequential frames per stream)
    batch_size = cfg.batch_size
    window_size = getattr(cfg.policy, "window_size", 2)  # Default to 2 if not specified
    
    logging.info(f"Streaming training config: B={batch_size} streams, K={window_size} frames per stream")
    
    # Create iterator that samples B windows with K frames each
    iterator = ParallelWindowIterator(
        dataset.episode_data_index, 
        batch_size=batch_size, 
        window_size=window_size
    )

    policy.train()
    
    # Initialize history cache once at start
    if hasattr(policy, "history_encoder"):
        policy.history_cache = None  # Will be initialized on first reset
    
    # Training metrics - added frames_per_sec
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "frames_per_sec": AverageMeter("frames/s", ":.1f"),
    }
    tracker = MetricsTracker(
        batch_size * window_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info(f"Start streaming training: {batch_size} parallel streams x {window_size} sequential frames")
    logging.info(f"Using DataLoader with {cfg.num_workers} workers and pin_memory={device.type == 'cuda'}")

    # Main training loop
    for _ in range(step, cfg.steps):
        t_step_start = time.perf_counter()
        
        # Step 1: Get next window of K timesteps (B indices per timestep)
        t0 = time.perf_counter()
        timestep_indices, streams_to_reset = next(iterator)
        data_sampling_time = time.perf_counter() - t0

        # Step 2: Reset history cache ONLY for streams that hit boundaries
        _reset_stream_cache(policy, streams_to_reset, batch_size, device)

        # Step 3: Create DataLoader for this window's timesteps
        # Use workers for parallel loading
        t_dataloader_start = time.perf_counter()
        
        # Custom dataset that just returns items by index
        class IndexDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        # Flatten timestep_indices: K lists of B indices -> K*B indices with proper ordering
        # We want to load all B frames for timestep 0, then all B for timestep 1, etc.
        all_indices = []
        for t in range(window_size):
            all_indices.extend(timestep_indices[t])
        
        # Create dataloader for this window
        window_dataset = IndexDataset(dataset, all_indices)
        window_dataloader = torch.utils.data.DataLoader(
            window_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=_collate_fn,
            drop_last=False,
        )
        
        tracker.dataloading_s = time.perf_counter() - t_dataloader_start + data_sampling_time

        # Step 4: Stream through K timesteps, accumulating gradients
        accumulated_loss = 0.0
        accumulated_loss_dict = {}

        optimizer.zero_grad(set_to_none=True)

        for t, batch in enumerate(window_dataloader):
            # Move tensors to device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=device.type == "cuda")
            
            # Forward and backward pass (accumulate gradients)
            loss_value, loss_dict = _streaming_forward_step(
                policy=policy,
                batch=batch,
                grad_scaler=grad_scaler,
                use_amp=cfg.policy.use_amp,
                accumulation_steps=window_size,
                device=device,
            )
            
            accumulated_loss += loss_value
            
            # Accumulate loss dict components
            for k, v in loss_dict.items():
                v = float(v) if torch.is_tensor(v) else v
                if k not in accumulated_loss_dict:
                    accumulated_loss_dict[k] = 0.0
                accumulated_loss_dict[k] += v / window_size

        # Step 5: Optimizer step after accumulating gradients from K timesteps
        grad_norm = _optimizer_step(
            policy=policy,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            grad_clip_norm=cfg.optimizer.grad_clip_norm,
        )
        
        update_time = time.perf_counter() - t_step_start
        
        # Calculate frames per second (B * K frames per step)
        frames_processed = batch_size * window_size
        frames_per_sec = frames_processed / update_time if update_time > 0 else 0

        # Update metrics
        tracker.loss = accumulated_loss
        tracker.grad_norm = grad_norm
        tracker.lr = optimizer.param_groups[0]["lr"]
        tracker.update_s = update_time
        tracker.frames_per_sec = frames_per_sec

        # Increment step and tracker
        step += 1
        tracker.step()

        # Determine logging/saving/eval steps
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Logging
        if is_log_step:
            logging.info(tracker)
            if wandb_logger:
                wandb_log_dict = tracker.to_dict()
                if accumulated_loss_dict:
                    wandb_log_dict.update(accumulated_loss_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            tracker.reset_averages()

        # Save checkpoint
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # Evaluation
        if cfg.env and is_eval_step and eval_env is not None:
            if hasattr(policy, "reset"): policy.reset()
            else: setattr(policy, "history_cache", None)
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of sequential training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
