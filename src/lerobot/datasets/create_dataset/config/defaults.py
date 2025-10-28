"""Default configuration values for dataset conversion."""

DEFAULT_CONFIG = {
    "repo_id": "aposadasn/image2image_point_action",
    "fps": 30,
    "robot_type": "custom",
    "task_name": "Sensor tasking",
    "input_dir": "/mnt/data/aposadasn/datafromEnrico/diff_train",
    "output_dir": "/home/aposadasn/lerobot-beavr/outputs/lerobot_dataset",
    # csv_image parser patterns (unused when parser_type == image_pair)
    "csv_pattern": "trajectory_{episode}.csv",
    "image_pattern": "ep_{episode}_agent_{agent}_img_{img}.png",
    "image_extension": ".png",
    "action_columns": [],
    "state_columns": [],
    "image_keys": ["observation.images.camera"],
    "use_videos": True,
    "debug": True,
    "test_mode": False,
    "push_to_hub": True,
    "private_repo": False,
    "tolerance_s": 1e-4,
    # image_pair parser options
    "parser_type": "image_pair",
    "input_subdir": "input_ag3_byepisode",
    "action_subdir": "action_ag3_byepisode",
    "action_threshold": 200,
    "num_agents": 3,
    # "batch_encoding_size": 1,
    # "max_test_episodes": 2,
}
