python lerobot/scripts/train.py \
  --dataset.repo_id=aposadasn/lander19 \
  --policy.type=act \
  --output_dir=outputs/train/lander \
  --job_name=lander \
  --policy.device=cuda \
  --wandb.enable=true