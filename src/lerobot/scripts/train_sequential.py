#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
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


@dataclass
class SequentialConfig:
    step_tick: int | None = None  # if None, defaults to B (cfg.batch_size)


class LockstepEpisodeIterator:
    """
    Advances B episode streams in lockstep, yielding per-step frame indices and reset_mask.

    Maintains per-stream pointers (from, to, cur). When a stream reaches its end, it is reset by
    pulling the next episode index from the queue.
    """

    def __init__(self, episode_data_index: dict[str, torch.Tensor], b: int):
        self.from_list = episode_data_index["from"].tolist()
        self.to_list = episode_data_index["to"].tolist()
        self.num_episodes = len(self.from_list)
        self.B = b

        # Queue of remaining episode ids
        self.remaining = deque(range(self.num_episodes))

        # Per-stream episode id and pointers
        self.stream_ep = [-1] * b
        self.ptr_from = [0] * b
        self.ptr_to = [0] * b
        self.ptr_cur = [0] * b
        # Per-stream random offset within episode (for decorrelating initial steps)
        self.stream_offset = [0] * b

        # Initialize streams
        for i in range(b):
            if not self.remaining:
                break
            ep = self.remaining.popleft()
            self.stream_ep[i] = ep
            self.ptr_from[i] = self.from_list[ep]
            self.ptr_to[i] = self.to_list[ep]
            # Choose random offset within episode bounds
            episode_length = self.ptr_to[i] - self.ptr_from[i]
            if episode_length > 1:
                self.stream_offset[i] = random.randint(0, episode_length - 1)
            else:
                self.stream_offset[i] = 0
            self.ptr_cur[i] = self.ptr_from[i] + self.stream_offset[i]

    def _reload_stream(self, i: int) -> bool:
        # If exhausted, wrap-around by refilling the queue
        if not self.remaining:
            self.remaining = deque(range(self.num_episodes))
        ep = self.remaining.popleft()
        self.stream_ep[i] = ep
        self.ptr_from[i] = self.from_list[ep]
        self.ptr_to[i] = self.to_list[ep]
        # Choose random offset within episode bounds for new episode
        episode_length = self.ptr_to[i] - self.ptr_from[i]
        if episode_length > 1:
            self.stream_offset[i] = random.randint(0, episode_length - 1)
        else:
            self.stream_offset[i] = 0
        self.ptr_cur[i] = self.ptr_from[i] + self.stream_offset[i]
        return True

    def __iter__(self):
        return self

    def __next__(self) -> tuple[list[int], torch.Tensor]:
        frame_indices: list[int] = []
        reset_mask = torch.zeros(self.B, dtype=torch.bool)

        # Determine next frames; always keep B active via wrap-around
        for i in range(self.B):
            # If at stream's starting position (i.e., just started or just reloaded), signal reset for this stream
            if self.ptr_cur[i] == self.ptr_from[i] + self.stream_offset[i]:
                reset_mask[i] = True
            frame_indices.append(self.ptr_cur[i])

        # Advance pointers and handle boundaries for next call
        for i in range(self.B):
            self.ptr_cur[i] += 1
            if self.ptr_cur[i] >= self.ptr_to[i]:
                # reached end of episode; reload next (wrap if needed)
                self._reload_stream(i)

        return frame_indices, reset_mask


def _gather_batch(dataset, frame_indices: list[int]) -> dict:
    # Note: -1 indices denote inactive streams; skip them to keep tight B across active streams only.
    # We will still build a dense batch of size B by repeating the last valid frame if needed.
    items = []
    last_valid = None
    for idx in frame_indices:
        if idx >= 0:
            item = dataset[idx]
            last_valid = item
            items.append(item)
        else:
            # Fallback to last valid to keep shapes; action_is_pad should mask it out downstream
            if last_valid is None:
                # If nothing valid yet, duplicate a dummy first item (index 0)
                last_valid = dataset[0]
            items.append(last_valid)

    # Collate dict of tensors; dataset returns tensors already
    batch = {}
    for k in items[0].keys():
        v_list = [it[k] for it in items]
        if isinstance(v_list[0], torch.Tensor):
            batch[k] = torch.stack(v_list, dim=0)
        else:
            batch[k] = v_list
    return batch

def _splice_rows(cache, fresh, mask):
    if cache is None or fresh is None:
        return cache
    if isinstance(cache, (tuple, list)):
        return type(cache)(_splice_rows(c,f,mask) for c,f in zip(cache, fresh, strict=False))
    if torch.is_tensor(cache):
        m = mask.view(cache.shape[0], *([1] * (cache.dim()-1)))
        return torch.where(m, fresh.to(cache.device, cache.dtype), cache)
    return cache


def _masked_history_reset(policy, reset_mask, fresh_template=None):
    if not hasattr(policy, "history_encoder"):
        if reset_mask.any():
            policy.reset()
        return
    if (
        getattr(policy, "history_cache", None) is None
        or not reset_mask.any()
    ):
        return
    if fresh_template is None:
        fresh_template = policy.history_encoder.init_cache(reset_mask.shape[0])
    policy.history_cache = _splice_rows(policy.history_cache, fresh_template, reset_mask)


def _accumulate_step(policy: PreTrainedPolicy, batch: Any, grad_scaler: GradScaler, use_amp: bool, loss_scale: int) -> tuple[float, dict]:
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, loss_dict = policy.forward(batch)
        loss = loss / loss_scale
    grad_scaler.scale(loss).backward()
    return float(loss.item()), loss_dict


def _optimizer_step(
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
    denom: int,
) -> float:
    grad_scaler.unscale_(optimizer)
    if denom > 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / denom)
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

    # Sequential training batch size
    batch_size = cfg.batch_size
    step_tick = cfg.policy.step_tick if cfg.policy.step_tick is not None else batch_size

    iterator = LockstepEpisodeIterator(dataset.episode_data_index, b=batch_size)

    policy.train()
    meters = {
        "loss": AverageMeter("loss", ":.3f"),
        "l1_loss": AverageMeter("l1_loss", ":.3f"),
        "kld_loss": AverageMeter("kld_loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "episode_s": AverageMeter("ep_s", ":.3f"),
        "frame_s": AverageMeter("frm_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    tracker = MetricsTracker(batch_size, dataset.num_frames, dataset.num_episodes, meters, initial_step=step)

    frames_since_step = 0
    episode_start_time = time.perf_counter()

    logging.info("Start offline sequential training with lockstep episodes")

    if hasattr(policy, "history_encoder"):
        policy.history_cache = policy.history_encoder.init_cache(batch_size)
        # device/dtype safety
        def _to(x):
            if isinstance(x, (tuple, list)):
                return type(x)(_to(t) for t in x)
            return x.to(device) if torch.is_tensor(x) else x
        policy.history_cache = _to(policy.history_cache)
        _fresh_cache_template = policy.history_encoder.init_cache(batch_size)
    else:
        _fresh_cache_template = None

    while step < cfg.steps:
        t0 = time.perf_counter()
        try:
            frame_indices, reset_mask = next(iterator)
        except StopIteration:
            break
        tracker.dataloading_s = time.perf_counter() - t0

        # Use sequential/streaming training
        batch = _gather_batch(dataset, frame_indices)

        # Move tensors to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=device.type == "cuda")

        # Masked per-stream history reset for sequential training
        _masked_history_reset(policy, reset_mask.to(device), _fresh_cache_template)

        # Accumulate gradients per frame
        loss_val, loss_dict = _accumulate_step(policy, batch, grad_scaler, cfg.policy.use_amp, batch_size)


        meters["loss"].update(loss_val)
        if "l1_loss" in loss_dict:
            meters["l1_loss"].update(loss_dict["l1_loss"])
        if "kld_loss" in loss_dict:
            meters["kld_loss"].update(loss_dict["kld_loss"])
        frames_since_step += batch_size

        # Optimizer step when we hit step_tick frames
        if frames_since_step >= step_tick:
            duration = time.perf_counter() - episode_start_time
            grad_norm = _optimizer_step(
                policy=policy,
                optimizer=optimizer,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                grad_clip_norm=cfg.optimizer.grad_clip_norm,
                denom=1,
            )
            step += 1
            frames_since_step = 0
            episode_start_time = time.perf_counter()

            tracker.grad_norm = grad_norm
            tracker.lr = optimizer.param_groups[0]["lr"]
            if duration > 0:
                tracker.frame_s = duration / max(step_tick, 1)
            tracker.step()

            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps)
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                logging.info(tracker)
                if wandb_logger:
                    wandb_logger.log_dict(tracker.to_dict(), step)
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
