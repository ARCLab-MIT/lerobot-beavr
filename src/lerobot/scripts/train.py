#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
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
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.constants import OBS_STATE

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def accumulate_step(
    policy: PreTrainedPolicy,
    batch: Any,
    grad_scaler: GradScaler,
    use_amp: bool,
) -> tuple[float, dict]:
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()
    return float(loss.item()), output_dict

def optimizer_step_only(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
    episode_length: int = 1,
) -> float:
    grad_scaler.unscale_(optimizer)

    # Normalize gradients by episode length to ensure consistent step size
    # regardless of episode length
    if episode_length > 0:
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(1.0 / episode_length)

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
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    print(f"cfg: {cfg}")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

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

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # Strict ordering inside each episode, no shuffling.
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,  # <<< must be False to preserve temporal order
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False  # <<< force off to preserve order even without sampler

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "episode_s": AverageMeter("ep_s", ":.3f"),
        "frame_s": AverageMeter("frm_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "episode_length": AverageMeter("ep_len", ":1.0f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # State for episode-aware stepping
    first_frame_seen = False
    episode_loss_sum = 0.0
    episode_frame_count = 0
    last_output_dict = {}
    current_episode_length = 0
    episode_start_time = None
    prev_episode_index = 0  # Track previous episode_index for boundary detection

    logging.info("Start offline training on a fixed dataset")
    # Interpret cfg.steps as "number of optimizer updates" (i.e., episodes)
    while step < cfg.steps:
        # -------------------- get next frame --------------------
        t0 = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - t0

        # Boundary detection using episode_index field
        # This is much simpler and more reliable than index-based detection
        current_episode_index = batch["episode_index"].item()
        is_boundary = (current_episode_index != prev_episode_index)
        
        # Debug: Log wrap-around detection
        if is_boundary and first_frame_seen and current_episode_index < prev_episode_index:
            logging.info(f"[WRAP-AROUND DETECTED] prev_ep={prev_episode_index} -> current_ep={current_episode_index} at step={step}")


        # If we encounter the start of a *new* episode and this isn't the very first frame
        # we've processed, we finalize the previous episode: clip+step+zero+sched+log.
        if is_boundary and first_frame_seen:
            # Record episode completion time
            episode_end_time = time.perf_counter()
            episode_duration = episode_end_time - episode_start_time if episode_start_time else 0

            grad_norm = optimizer_step_only(
                policy=policy,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                grad_clip_norm=cfg.optimizer.grad_clip_norm,
                episode_length=current_episode_length,
            )
            # bookkeeping for this optimizer update (= one episode)
            step += 1

            # Update episode-level metrics
            train_tracker.episode_s = episode_duration
            train_tracker.episode_length = current_episode_length
            if current_episode_length > 0:
                train_tracker.frame_s = episode_duration / current_episode_length
                # Calculate throughput metrics
                frames_per_sec = current_episode_length / episode_duration if episode_duration > 0 else 0

            train_tracker.grad_norm = grad_norm
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.step()

            # ---------- logging / checkpoint / eval happen on optimizer steps ----------
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                # Log episode-averaged loss (normalized by number of frames)
                if episode_frame_count > 0:
                    episode_avg_loss = episode_loss_sum / episode_frame_count
                    train_tracker.loss = episode_avg_loss
                    # Debug: Check for suspicious loss values
                    if episode_avg_loss < 0.001 or episode_avg_loss == 0.0:
                        logging.warning(f"[SUSPICIOUS LOSS] step={step}, loss={episode_avg_loss:.6f}, "
                                      f"episode_loss_sum={episode_loss_sum:.6f}, episode_frame_count={episode_frame_count}")
                else:
                    logging.warning(f"[NO FRAMES] step={step}, episode_frame_count=0, cannot compute loss")
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log = train_tracker.to_dict()
                    if last_output_dict:
                        wandb_log.update(last_output_dict)
                    wandb_logger.log_dict(wandb_log, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
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
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

            # If we've reached the target number of optimizer steps, stop training.
            if step >= cfg.steps:
                break

            # We just started a new episode → reset recurrent state/history and timing
            policy.reset()
            # Reset episode length counter, loss tracking, and timing for new episode
            current_episode_length = 0
            episode_loss_sum = 0.0
            episode_frame_count = 0
            episode_start_time = time.perf_counter()

        # First frame ever (seed episode): reset before accumulating
        if not first_frame_seen:
            policy.reset()
            first_frame_seen = True
            # Initialize episode timing for the first episode
            episode_start_time = time.perf_counter()

        # -------------------- move to device & accumulate this frame --------------------
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=device.type == "cuda")

        loss_val, output_dict = accumulate_step(
            policy=policy,
            batch=batch,
            grad_scaler=grad_scaler,
            use_amp=cfg.policy.use_amp,
        )
        # Accumulate loss for episode averaging
        episode_loss_sum += loss_val
        episode_frame_count += 1
        last_output_dict = output_dict
        
        # Debug: Log frames with suspicious loss
        if loss_val < 0.001 or loss_val == 0.0:
            num_valid_actions = (~batch["action_is_pad"]).sum().item()
            logging.warning(f"[FRAME WITH ZERO LOSS] step={step}, ep_idx={current_episode_index}, "
                          f"loss={loss_val:.6f}, valid_actions={num_valid_actions}, "
                          f"frame_count={episode_frame_count}")

        # Track episode length for gradient normalization
        current_episode_length += 1
        
        # Update prev_episode_index after processing this frame
        # (Critical: must be after processing so boundary detection works correctly next iteration)
        prev_episode_index = current_episode_index

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
