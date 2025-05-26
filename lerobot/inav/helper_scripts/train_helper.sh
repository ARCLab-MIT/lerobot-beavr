python lerobot/scripts/train.py \
  --dataset.repo_id=aposadasn/iss_docking \
  --policy.type=act \
  --output_dir=outputs/train/iss_docking \
  --job_name=iss_docking \
  --policy.device=cuda \
  --wandb.enable=true