#!/usr/bin/env python3
"""
Script to upload the last checkpoint to the HuggingFace Hub.
Run this from the lerobot-beavr directory.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import lerobot modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class

def upload_checkpoint():
    # Path to the last checkpoint directory
    checkpoint_dir = Path("outputs/train/2025-10-09/22-43-51_aloha_dact_a/checkpoints/last/pretrained_model")

    # Path to the training config JSON
    config_path = Path("outputs/train/2025-10-09/22-43-51_aloha_dact_a/checkpoints/last/pretrained_model/train_config.json")

    # Load the training configuration
    cfg = TrainPipelineConfig.from_pretrained(config_path)

    # Load the pretrained model from the checkpoint directory
    # The model type is determined from the config using the factory pattern
    policy_class = get_policy_class(cfg.policy.type)

    # Load the model from the local checkpoint
    policy = policy_class.from_pretrained(
        pretrained_name_or_path=checkpoint_dir,
        config=cfg.policy
    )

    # Push the model to the hub using the configuration
    policy.push_model_to_hub(cfg)

    print(f"Successfully uploaded checkpoint to {cfg.policy.repo_id}")

if __name__ == "__main__":
    upload_checkpoint()