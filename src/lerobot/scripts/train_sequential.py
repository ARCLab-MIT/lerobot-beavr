import logging
import multiprocessing
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from typing import Any

# Suppress FFmpeg debug messages from torchcodec before any imports
os.environ["AV_LOG_LEVEL"] = "error"
os.environ["FFMPEG_LOGLEVEL"] = "error"

# Set FFmpeg logging to error level before importing av
import av.logging
av.logging.set_level(av.logging.ERROR)

import torch
import av

# Also try to suppress any remaining FFmpeg messages at C level
try:
    import ctypes
    # Try to load avutil and set log level if possible
    avutil = ctypes.CDLL("libavutil.so.58")  # Adjust version as needed
    avutil.av_log_set_level(8)  # AV_LOG_ERROR = 8
except:
    pass  # If this fails, continue anyway
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
from lerobot.policies.utils import get_device_from_parameters
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


def sample_episode_batch(episode_data_index: dict[str, torch.Tensor], batch_size: int, shuffle: bool = True) -> list[int]:
    """
    Sample a batch of episode indices.
    
    Args:
        episode_data_index: Dictionary with 'from' and 'to' tensors indicating episode boundaries
        batch_size: Number of episodes to sample
        shuffle: Whether to shuffle episode order
        
    Returns:
        List of episode indices
    """
    num_episodes = len(episode_data_index["from"])
    if shuffle:
        return random.sample(range(num_episodes), min(batch_size, num_episodes))
    else:
        # Sequential sampling with wraparound
        return [i % num_episodes for i in range(batch_size)]


def get_episode_frames(episode_data_index: dict[str, torch.Tensor], episode_idx: int) -> tuple[int, int, int]:
    """
    Get frame range for an episode.
    
    Returns:
        (start_frame, end_frame, episode_length)
    """
    start = episode_data_index["from"][episode_idx].item()
    end = episode_data_index["to"][episode_idx].item()
    return start, end, end - start


class FrameIndexDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that fetches frames by dynamically set indices.
    Avoids recreating DataLoader and workers for each window.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.frame_indices: list[int] = []
    
    def set_indices(self, indices: list[int]):
        """Set the frame indices to fetch for the next iteration."""
        self.frame_indices = indices
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.frame_indices[idx]]


def _gather_windowed_batch_dataloader(
    frame_dataset: FrameIndexDataset,
    dataloader: torch.utils.data.DataLoader,
    episode_info: list,
    window_size: int,
    window_start: int,
):
    """
    Gather windowed batch using a persistent DataLoader.
    
    Notes:
        - Reuses the same DataLoader and workers across windows
        - Preserves clamping-at-end behavior to keep labels stable
        - Builds masks identical to the original helper
    """
    B = len(episode_info)
    K = window_size

    # Build flat list of frame indices in (b, t) order
    frame_indices: list[int] = []
    valid = torch.zeros(B, K, dtype=torch.bool)
    for b, (start_frame, end_frame, ep_length) in enumerate(episode_info):
        for t in range(K):
            actual_t = window_start + t
            is_valid = actual_t < ep_length
            valid[b, t] = is_valid
            if is_valid:
                frame_idx = start_frame + actual_t
            else:
                frame_idx = end_frame - 1
            frame_indices.append(frame_idx)

    # Set indices in the persistent dataset wrapper
    frame_dataset.set_indices(frame_indices)
    
    # Fetch single batch from persistent DataLoader
    flat_batch = next(iter(dataloader))

    out: dict[str, torch.Tensor] = {}

    # Reshape tensors from (B*K, ...) -> (B, K, ...)
    for k, v in flat_batch.items():
        if not torch.is_tensor(v):
            continue
        out[k] = v.reshape(B, K, *v.shape[1:])

    # Add window metadata
    out["window_start"] = torch.tensor([window_start] * B, dtype=torch.long)
    out["window_size"] = torch.tensor([K] * B, dtype=torch.long)

    # Ensure (B, K, S) action_is_pad semantics for the last step
    S = None
    if "action_is_pad" in out and out["action_is_pad"].ndim == 3:
        pad = out["action_is_pad"].clone()
        S = pad.shape[-1]
    else:
        if "action" in out and out["action"].ndim == 4:
            S = out["action"].shape[2]
        else:
            raise RuntimeError("Need (B,K,S,...) ACTION or (B,K,S) action_is_pad to construct masks.")
        pad = torch.zeros(B, K, S, dtype=torch.bool)

    invalid_last = (~valid[:, -1]).unsqueeze(-1)
    pad[:, -1, :] = torch.where(invalid_last, torch.ones_like(pad[:, -1, :], dtype=pad.dtype), pad[:, -1, :])

    out["action_is_pad"] = pad
    out["valid_mask"] = valid
    out["alive_mask"] = valid.any(dim=1)
    out["ended_mask"] = ~valid[:, -1]

    return out


def _accumulate_windowed_step(policy: PreTrainedPolicy, batch: Any, grad_scaler: GradScaler, use_amp: bool) -> tuple[float, dict]:
    """
    Accumulate gradients for a windowed sequence batch.
    
    Args:
        policy: The policy model
        batch: Windowed batch of shape (B, K, ...)
        grad_scaler: Gradient scaler for mixed precision
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        (loss_value, loss_dict)
    """
    device = get_device_from_parameters(policy)
    policy.train()
    
    # Detailed timing for each step
    step_timings = {}
    
    # Forward pass timing
    forward_start = time.perf_counter()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, loss_dict = policy.forward(batch)
    forward_time = (time.perf_counter() - forward_start) * 1000  # ms
    step_timings['forward_ms'] = forward_time
    
    # Store unscaled loss for logging
    unscaled_loss = float(loss.item())
    
    # Backward pass timing
    backward_start = time.perf_counter()
    grad_scaler.scale(loss).backward()
    backward_time = (time.perf_counter() - backward_start) * 1000  # ms
    step_timings['backward_ms'] = backward_time
    
    # Add timing info to loss_dict for logging
    loss_dict.update(step_timings)
    
    # Return unscaled loss for logging purposes
    return unscaled_loss, loss_dict


def _optimizer_step(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
) -> float:
    """
    Perform optimizer step with gradient clipping.
    
    Note: No gradient scaling is performed. Accumulated gradients represent
    an effective batch size of (batch_size * num_accumulation_steps).
    """
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return float(grad_norm)


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(cfg.to_dict())
    
    # TBPTT Configuration
    logging.info(f"TBPTT Configuration: window_size={cfg.policy.window_size}, windows_per_step={cfg.policy.windows_per_optimizer_step}")
    
    # DataLoader configuration
    num_workers = cfg.num_workers
    if num_workers > 0:
        logging.info(f"DataLoader will use {num_workers} workers with spawn context and persistent_workers=True")
        logging.info("Note: First window may be slow while workers initialize video decoders")

        # Ensure FFmpeg logging is suppressed in worker processes too
        # The environment variables set at module level should be inherited

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

    # Create persistent DataLoader for efficient frame fetching across windows
    # This avoids recreating workers and FFmpeg decoders for each window
    frame_dataset = FrameIndexDataset(dataset)
    
    # Use spawn context to avoid video decoder fork issues
    mp_context = multiprocessing.get_context('spawn') if num_workers > 0 else None
    
    persistent_dataloader = torch.utils.data.DataLoader(
        frame_dataset,
        batch_size=cfg.batch_size * cfg.policy.window_size,  # Max batch size (B * K frames)
        shuffle=False,
        num_workers=max(int(num_workers), 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context=mp_context,
        prefetch_factor=6 if num_workers > 0 else 2,  # Prefetch more batches
    )
    
    logging.info(f"Created persistent DataLoader with {num_workers} workers (reuses decoders across windows)")

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
    total_optimizer_steps = 0  # Track total optimizer steps for logging
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

    # Sequential training batch size (B episodes processed in parallel)
    batch_size = cfg.batch_size
    num_episodes = dataset.num_episodes
    
    # Create episode index pool for sampling
    episode_indices = list(range(num_episodes))
    random.shuffle(episode_indices)
    episode_idx_pool = deque(episode_indices)

    policy.train()
    meters = {
        "loss": AverageMeter("loss", ":.3f"),
        "l1_loss": AverageMeter("l1_loss", ":.3f"),
        "kld_loss": AverageMeter("kld_loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "episode_batch_s": AverageMeter("ep_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    tracker = MetricsTracker(batch_size, dataset.num_frames, dataset.num_episodes, meters, initial_step=step)

    logging.info("Start offline sequential training with windowed TBPTT")

    # Initialize history cache if policy has history encoder
    if hasattr(policy, "history_encoder"):
        policy.history_cache = None

    while step < cfg.steps:
        episode_batch_start_time = time.perf_counter()
        t0 = time.perf_counter()
        
        # Sample B episodes for this batch
        # Refill pool if needed
        if len(episode_idx_pool) < batch_size:
            episode_indices = list(range(num_episodes))
            random.shuffle(episode_indices)
            episode_idx_pool.extend(episode_indices)
        
        # Sample batch of episode indices
        sampled_episode_indices = [episode_idx_pool.popleft() for _ in range(batch_size)]
        
        # Get episode frame ranges
        episode_info = []
        max_episode_length = 0
        for ep_idx in sampled_episode_indices:
            start_frame, end_frame, ep_length = get_episode_frames(dataset.episode_data_index, ep_idx)
            episode_info.append((start_frame, end_frame, ep_length))
            max_episode_length = max(max_episode_length, ep_length)
        
        tracker.dataloading_s = time.perf_counter() - t0
        
        # Reset hidden states for new batch of episodes
        if hasattr(policy, "history_cache"):
            policy.history_cache = None
        
        # Process episodes using windowed TBPTT
        acc_steps = 0
        total_loss = 0.0
        loss_dict_accumulator = {}
        
        # Slide window across episode length
        for window_start in range(0, max_episode_length, cfg.policy.window_size):
            # Calculate actual window size (may be smaller at episode ends)
            actual_window_size = min(cfg.policy.window_size, max_episode_length - window_start)
            if actual_window_size <= 0:
                break
            
            # Gather windowed batch (B, K, ...) using persistent DataLoader
            t_gather0 = time.perf_counter()
            batch = _gather_windowed_batch_dataloader(
                frame_dataset,
                persistent_dataloader,
                episode_info,
                actual_window_size,
                window_start,
            )
            gather_ms = (time.perf_counter() - t_gather0) * 1000

            # Move tensors to device — under optimization, we may move only last-step later
            t_h2d0 = time.perf_counter()
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=device.type == "cuda")
            h2d_ms = (time.perf_counter() - t_h2d0) * 1000
            
            # ---- Drop "dead" streams (no valid steps in this window) to save compute ----
            if "alive_mask" in batch:
                alive = batch["alive_mask"]
                if alive.ndim != 1:
                    alive = alive.view(-1)
                if not bool(alive.all()):
                    # Filter all (B, ..) tensors by alive
                    for k, v in list(batch.items()):
                        if isinstance(v, torch.Tensor) and v.shape[:1] == alive.shape:
                            batch[k] = v[alive]
                    # Keep TBPTT cache in sync (only if it exists and matches B)
                    if hasattr(policy, "history_cache") and policy.history_cache is not None:
                        if torch.is_tensor(policy.history_cache) and policy.history_cache.shape[0] == alive.shape[0]:
                            policy.history_cache = policy.history_cache[alive]
            # ---------------------------------------------------------------------------

            # Accumulate gradients for this window
            loss_val, loss_dict = _accumulate_windowed_step(
                policy=policy,
                batch=batch,
                grad_scaler=grad_scaler,
                use_amp=cfg.policy.use_amp,
            )
            # attach gather/h2d timings for accumulation
            loss_dict.setdefault("gather_ms", 0.0)
            loss_dict.setdefault("h2d_ms", 0.0)
            loss_dict["gather_ms"] += gather_ms
            loss_dict["h2d_ms"] += h2d_ms
            
            acc_steps += 1
            total_loss += loss_val
            
            # Accumulate loss_dict components
            for k, v in loss_dict.items():
                if k not in loss_dict_accumulator:
                    loss_dict_accumulator[k] = 0.0
                loss_dict_accumulator[k] += v
            
            # Step optimizer every N windows
            if acc_steps % cfg.policy.windows_per_optimizer_step == 0:
                duration = time.perf_counter() - episode_batch_start_time
                grad_norm = _optimizer_step(
                    policy=policy,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=lr_scheduler,
                    grad_clip_norm=cfg.optimizer.grad_clip_norm,
                )
                step += 1
                total_optimizer_steps += 1
                
                # Update metrics
                avg_loss = total_loss / acc_steps
                meters["loss"].update(avg_loss)
                if "l1_loss" in loss_dict_accumulator:
                    meters["l1_loss"].update(loss_dict_accumulator["l1_loss"] / acc_steps)
                if "kld_loss" in loss_dict_accumulator:
                    meters["kld_loss"].update(loss_dict_accumulator["kld_loss"] / acc_steps)
                
                # Debug timing logs (only when debug logging is enabled)
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if "forward_ms" in loss_dict_accumulator:
                        fwd_time = loss_dict_accumulator["forward_ms"] / acc_steps
                        logging.debug(f"Avg forward time: {fwd_time:.1f}ms")
                    if "backward_ms" in loss_dict_accumulator:
                        bwd_time = loss_dict_accumulator["backward_ms"] / acc_steps
                        logging.debug(f"Avg backward time: {bwd_time:.1f}ms")
                
                tracker.grad_norm = grad_norm
                tracker.lr = optimizer.param_groups[0]["lr"]
                tracker.episode_batch_s = duration
                tracker.step()
                
                # Reset accumulators
                acc_steps = 0
                total_loss = 0.0
                loss_dict_accumulator = {}
                
                # Logging and checkpointing
                # Use total_optimizer_steps for more consistent logging
                is_log_step = cfg.log_freq > 0 and total_optimizer_steps % cfg.log_freq == 0
                is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
                is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
                
                if is_log_step:
                    logging.info(tracker)
                    if wandb_logger:
                        wandb_log_dict = tracker.to_dict()
                        wandb_logger.log_dict(wandb_log_dict, step)
                    tracker.reset_averages()
                
                if cfg.save_checkpoint and is_saving_step:
                    logging.info(f"Checkpoint policy after step {step}")
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)
                
                if cfg.env and is_eval_step and eval_env is not None:
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
                    eval_tracker = MetricsTracker(batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
                    eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                    eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                    eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                    logging.info(eval_tracker)
                    if wandb_logger:
                        wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                        wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                        wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                
                if step >= cfg.steps:
                    break

        # after the for-window loop (still inside the while step loop)
        if acc_steps > 0:
            duration = time.perf_counter() - episode_batch_start_time
            grad_norm = _optimizer_step(
                policy=policy,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                grad_clip_norm=cfg.optimizer.grad_clip_norm,
            )
            step += 1
            total_optimizer_steps += 1

            # Update meters
            avg_loss = total_loss / acc_steps
            meters["loss"].update(avg_loss)
            if "l1_loss" in loss_dict_accumulator:
                meters["l1_loss"].update(loss_dict_accumulator["l1_loss"] / acc_steps)
            if "kld_loss" in loss_dict_accumulator:
                meters["kld_loss"].update(loss_dict_accumulator["kld_loss"] / acc_steps)

            tracker.grad_norm = grad_norm
            tracker.lr = optimizer.param_groups[0]["lr"]
            tracker.episode_batch_s = duration
            tracker.step()

            # Logging and checkpointing (consolidated logic)
            is_log_step = cfg.log_freq > 0 and total_optimizer_steps % cfg.log_freq == 0
            is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
            
            if is_log_step:
                logging.info(tracker)
                if wandb_logger:
                    wandb_log_dict = tracker.to_dict()
                    wandb_logger.log_dict(wandb_log_dict, step)
                tracker.reset_averages()
            
            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            
            if cfg.env and is_eval_step and eval_env is not None:
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
                eval_tracker = MetricsTracker(batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

            # Reset accumulators
            acc_steps = 0
            total_loss = 0.0
            loss_dict_accumulator = {}


    # Clean up history cache before ending training
    if hasattr(policy, "history_cache") and policy.history_cache is not None:
        policy.history_cache = None
        
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
