"""Configuration management for dataset conversion."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lerobot.datasets.create_dataset.config.defaults import DEFAULT_CONFIG


@dataclass
class DatasetConfig:
    """Configuration for dataset conversion."""

    # Dataset identification
    repo_id: str
    fps: int
    robot_type: str = "custom"
    task_name: str = "custom_task"

    # Parser selection and options
    parser_type: str = "csv_image"  # one of: "csv_image", "image_pair"
    input_subdir: str = "input_ag2"  # used by image_pair parser
    action_subdir: str = "action_ag2"  # used by image_pair parser
    action_threshold: int = 200  # used by image_pair parser

    # Input/Output paths
    input_dir: str | Path = ""
    output_dir: str | Path | None = None

    # Data format configuration
    csv_pattern: str = "trajectory_{episode}.csv"
    image_pattern: str = "ep_{episode}_agent_{agent}_img_{img}.png"
    image_extension: str = ".png"

    # Features configuration
    action_columns: list[str] = field(default_factory=list)
    state_columns: list[str] = field(default_factory=list)
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.camera"])

    # Processing options
    use_videos: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    tolerance_s: float = 1e-4

    # Validation and debugging
    debug: bool = False
    validate_data: bool = True
    test_mode: bool = False
    max_test_episodes: int = 1

    # HuggingFace Hub options
    push_to_hub: bool = True
    private_repo: bool = False

    num_agents: int | None = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.input_dir = Path(self.input_dir)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
        else:
            self.output_dir = self.input_dir / "lerobot_dataset"


def load_config(config_path: str | Path) -> DatasetConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return DatasetConfig(**config_dict)


def create_sample_config(output_path: str | Path) -> None:
    """Create a sample configuration file."""
    with open(output_path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
