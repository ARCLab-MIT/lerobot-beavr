"""
TBPTT Training Script - Clean, Efficient Window-Based Training

This script implements standard Truncated Backpropagation Through Time (TBPTT) with:
- Direct window streaming from LeRobot datasets (no conversion needed)
- Pre-shaped (B, K, ...) batches ready for training
- Persistent DataLoader with video decoder reuse
- Clean state management with detach/carry pattern
- Zero post-processing overhead

Key improvements over train_sequential.py:
1. Windows pre-shaped in dataset → zero reshaping in loop
2. Provides *_last views → zero slicing for ACT
3. Simple collate (just stack) → minimal overhead
4. Clean TBPTT truncation via detach at window boundaries
"""

import logging
import multiprocessing
import os
import time
from contextlib import nullcontext

# Suppress FFmpeg debug messages
os.environ["AV_LOG_LEVEL"] = "error"
os.environ["FFMPEG_LOGLEVEL"] = "error"

import av.logging
av.logging.set_level(av.logging.ERROR)

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.tbptt_windows import TBPTTWindowStreamer, collate_tbptt
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


def _forward_window(
    policy: PreTrainedPolicy,
    batch: dict,
    grad_scaler: GradScaler,
    use_amp: bool,
) -> tuple[float, dict]:
    """
    Forward pass + backward for a single window with gradient accumulation.
    
    Args:
        policy: The policy model
        batch: Window batch with *_seq and *_last keys
        grad_scaler: Gradient scaler for mixed precision
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        (loss_value, loss_dict)
    """
    device = get_device_from_parameters(policy)
    policy.train()
    
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, loss_dict = policy.forward(batch)
    
    grad_scaler.scale(loss).backward()
    
    return float(loss.item()), loss_dict


def _optimizer_step(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
    num_accumulated: int,
) -> float:
    """
    Execute optimizer step after gradient accumulation.
    
    Args:
        policy: The policy model
        optimizer: Optimizer
        grad_scaler: Gradient scaler
        lr_scheduler: Learning rate scheduler
        grad_clip_norm: Max gradient norm
        num_accumulated: Number of accumulated gradients (for averaging)
        
    Returns:
        Gradient norm before clipping
    """
    # Unscale gradients
    grad_scaler.unscale_(optimizer)
    
    # Average gradients across accumulated windows
    denom = max(num_accumulated, 1)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / denom)
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False
    )
    
    # Step optimizer
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # Step scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return float(grad_norm)


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """Main TBPTT training loop."""
    cfg.validate()
    logging.info(cfg.to_dict())
    
    # TBPTT Configuration
    window_size = cfg.policy.window_size
    windows_per_step = cfg.policy.windows_per_optimizer_step
    logging.info(f"TBPTT: window_size={window_size}, windows_per_step={windows_per_step}")
    
    # Logging setup
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Device setup
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create dataset
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    
    # Create TBPTT window streamer
    logging.info(f"Creating TBPTT window streamer (K={window_size})")
    window_dataset = TBPTTWindowStreamer(
        dataset=dataset,
        window_size=window_size,
        stride=window_size,  # Non-overlapping windows
        clamp_pad=True,
        shuffle_episodes=True,
    )
    
    # Create persistent DataLoader
    num_workers = cfg.num_workers
    mp_context = multiprocessing.get_context('spawn') if num_workers > 0 else None
    
    dataloader = torch.utils.data.DataLoader(
        window_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # IterableDataset handles shuffling
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context=mp_context,
        prefetch_factor=6 if num_workers > 0 else 2,
        collate_fn=collate_tbptt,
    )
    
    logging.info(f"Created persistent DataLoader with {num_workers} workers")
    if num_workers > 0:
        logging.info("Note: First batch may be slow while workers initialize video decoders")
    
    # Optional eval env
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    # Create policy
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    
    # Create optimizer and scheduler
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.policy.use_amp)
    
    # Load checkpoint if resuming
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    # Log model info
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
    
    # Metrics tracking
    policy.train()
    meters = {
        "loss": AverageMeter("loss", ":.3f"),
        "l1_loss": AverageMeter("l1_loss", ":.3f"),
        "kld_loss": AverageMeter("kld_loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "window_s": AverageMeter("win_s", ":.3f"),
    }
    tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, meters, initial_step=step)
    
    logging.info("Start TBPTT training")
    
    # Create infinite iterator
    data_iter = iter(dataloader)
    
    # Training loop
    acc_steps = 0
    total_loss = 0.0
    loss_dict_accumulator = {}
    optimizer.zero_grad(set_to_none=True)
    
    step_start_time = time.perf_counter()
    
    while step < cfg.steps:
        # Fetch next window batch
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator for new epoch
            data_iter = iter(dataloader)
            batch = next(data_iter)
            # Reset policy state at epoch boundaries
            if hasattr(policy, 'reset'):
                policy.reset()
        
        # Move batch to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=device.type == "cuda")
            elif isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                # Handle lists of tensors (e.g., images per camera)
                batch[k] = [t.to(device, non_blocking=device.type == "cuda") for t in v]
        
        # Reset policy state when episodes end
        # The ended_mask indicates which streams in the batch ended in this window
        if hasattr(policy, 'reset') and "ended_mask" in batch:
            if batch["ended_mask"].any():
                # For batched training, we can't selectively reset individual streams easily
                # For now, we'll reset when ANY stream ends (conservative approach)
                # A more sophisticated approach would maintain per-stream state
                policy.reset()
        
        # Forward + backward for this window
        loss_val, loss_dict = _forward_window(
            policy=policy,
            batch=batch,
            grad_scaler=grad_scaler,
            use_amp=cfg.policy.use_amp,
        )
        
        acc_steps += 1
        total_loss += loss_val
        
        # Accumulate loss components
        for k, v in loss_dict.items():
            if k not in loss_dict_accumulator:
                loss_dict_accumulator[k] = 0.0
            loss_dict_accumulator[k] += v
        
        # Step optimizer after accumulating enough windows
        if acc_steps >= windows_per_step:
            step_duration = time.perf_counter() - step_start_time
            
            grad_norm = _optimizer_step(
                policy=policy,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                grad_clip_norm=cfg.optimizer.grad_clip_norm,
                num_accumulated=acc_steps,
            )
            
            step += 1
            
            # Update metrics
            avg_loss = total_loss / acc_steps
            meters["loss"].update(avg_loss)
            if "l1_loss" in loss_dict_accumulator:
                meters["l1_loss"].update(loss_dict_accumulator["l1_loss"] / acc_steps)
            if "kld_loss" in loss_dict_accumulator:
                meters["kld_loss"].update(loss_dict_accumulator["kld_loss"] / acc_steps)
            
            tracker.grad_norm = grad_norm
            tracker.lr = optimizer.param_groups[0]["lr"]
            tracker.window_s = step_duration
            tracker.step()
            
            # Reset accumulators
            acc_steps = 0
            total_loss = 0.0
            loss_dict_accumulator = {}
            step_start_time = time.perf_counter()
            
            # Logging
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
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
                
                # Reset policy for evaluation
                if hasattr(policy, 'reset'):
                    policy.reset()
                
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
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                
                # Return to training mode
                policy.train()
    
    # Cleanup
    if eval_env:
        eval_env.close()
    
    logging.info("End of TBPTT training")
    
    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
