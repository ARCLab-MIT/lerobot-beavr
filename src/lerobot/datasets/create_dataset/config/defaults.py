"""Default configuration values for dataset conversion."""

DEFAULT_CONFIG = {
    "repo_id": "user/image2image_point_action",
    "fps": 1,
    "robot_type": "custom",
    "task_name": "point_action",
    "input_dir": "/mnt/data/aposadasn/datafromEnrico/diff_train",
    "output_dir": "/mnt/data/aposadasn/datafromEnrico/diff_train/lerobot_dataset",
    # csv_image parser patterns (unused when parser_type == image_pair)
    "csv_pattern": "trajectory_{episode}.csv",
    "image_pattern": "img_episode_{episode}_frame_{frame}",
    "image_extension": ".png",
    "action_columns": [],
    "state_columns": [],
    "image_keys": ["observation.images.camera"],
    "use_videos": False,
    "debug": True,
    "test_mode": False,
    "push_to_hub": False,
    "private_repo": False,
    "tolerance_s": 1.0,
    # image_pair parser options
    "parser_type": "image_pair",
    "input_subdir": "input_ag2",
    "action_subdir": "action_ag2",
    "action_threshold": 200,
}
