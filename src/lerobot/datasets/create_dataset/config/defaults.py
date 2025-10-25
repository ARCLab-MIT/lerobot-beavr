"""Default configuration values for dataset conversion."""

DEFAULT_CONFIG = {
    "repo_id": "aposadasn/image2image_point_action",
    "fps": 1,
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
    "tolerance_s": 1.0,
    # image_pair parser options
    "parser_type": "image_pair",
    "input_subdir": "input_ag3_byepisode",
    "action_subdir": "action_ag3_byepisode",
    "action_threshold": 200,
    "num_agents": 3,
    # Chunking and file size options for large datasets (100K episodes)
    # "chunks_size": 1000,  # Max files per chunk directory
    # "data_files_size_in_mb": 500,  # Max size for data parquet files in MB
    # "video_files_size_in_mb": 2000,  # Max size for video files in MB
    # "metadata_buffer_size": 50,  # Buffer size for metadata batching
    # "batch_encoding_size": 4,  # Batch episodes before encoding videos
}
