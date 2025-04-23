python lerobot/scripts/train.py \
  --dataset.repo_id=aposadasn/lander1 \
  --policy.type=act \
  --output_dir=outputs/train/lander_mini \
  --job_name=lander_mini \
  --policy.device=cuda \
  --wandb.enable=true