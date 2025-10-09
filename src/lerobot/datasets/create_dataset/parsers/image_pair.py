"""Parser for paired image datasets (input image + action label image).

This parser expects two flat subdirectories inside `config.input_dir`:
  - `input_ag3_byepisode`: contains observation/input images (chronological order)
  - `action_ag3_byepisode`: contains action label images with a white pixel indicating the action

Frames are paired by lexicographic filename order and grouped into a single multi-frame episode.
Action images are converted to normalized [x, y] coordinates in [0, 1] by computing the centroid
of pixels above a brightness threshold (default 200).
"""

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from lerobot.datasets.create_dataset.parsers.parse_data import DataParser
from lerobot.datasets.create_dataset.parsers.utils import get_image_dimensions, load_image


def _extract_xy_from_label_image(img_uint8: np.ndarray, threshold: int = 200) -> tuple[float, float] | None:
    if img_uint8.ndim == 3:
        gray = img_uint8.mean(axis=2).astype(np.uint8)
    else:
        gray = img_uint8

    mask = gray >= threshold
    if not mask.any():
        return None

    ys, xs = np.nonzero(mask)
    y_center = ys.mean()
    x_center = xs.mean()
    h, w = gray.shape[:2]

    # Normalize with pixel-center convention (top-left origin)
    x_norm = float((x_center + 0.5) / w)
    y_norm = float((y_center + 0.5) / h)
    return x_norm, y_norm



class ImagePairParser(DataParser):
    """Memory-efficient parser for multi-agent paired image datasets with 3 agents.

    Groups files by episode number (ep_{episode}_...) and creates multiple episodes:
    - Uses streaming/generator approach for memory efficiency
    - Processes and yields episodes one at a time instead of loading all into memory
    - Each episode is indicated by the ep_{episode} prefix in filenames
    - Files are processed in sorted order to maintain image number sequence
    - Each episode contains:
      - Input image (observation)
      - Agent indicator: 3D vector showing which agent is acting [1,0,0], [0,1,0], or [0,0,1]
      - Actions: 6D vector with (x,y) coordinates for each agent
      - Timestamps are sequential within each episode (0.0, 1.0, 2.0, etc.)
    """

    def __init__(self, config):
        super().__init__(config)
        # Allow overriding subdir names and threshold via config if present
        self.input_subdir = getattr(config, "input_subdir", "input_ag3_byepisode")
        self.action_subdir = getattr(config, "action_subdir", "action_ag3_byepisode")
        self.threshold = getattr(config, "action_threshold", 200)
        self.num_agents = getattr(config, "num_agents", 3)

    def get_episode_files(self) -> list[Path]:
        """Return a synthetic list with a single placeholder Path.

        The parser is directory-driven; we don't rely on CSV files. We use a single
        episode and return a list containing the input directory path to satisfy the API.
        """
        return [self.config.input_dir]

    def parse_episode(self, episode_file: Path):
        """Parse multiple episodes based on ep_ prefix in filenames.

        Uses a memory-efficient streaming approach: processes episodes one at a time
        and yields them as generators to avoid loading all data into memory.
        """
        input_dir = self.config.input_dir / self.input_subdir
        action_dir = self.config.input_dir / self.action_subdir

        if not input_dir.exists() or not action_dir.exists():
            raise FileNotFoundError(f"Expected subdirectories '{self.input_subdir}' and '{self.action_subdir}' under {self.config.input_dir}")

        # Count files without loading all into memory
        self.logger.info("Counting input files...")
        input_files = [p for p in input_dir.iterdir() if p.is_file()]
        input_count = len(input_files)

        self.logger.info("Counting action files...")
        action_files = [p for p in action_dir.iterdir() if p.is_file()]
        action_count = len(action_files)

        if input_count == 0 or action_count == 0:
            raise ValueError("No files found in input/action subdirectories")

        if input_count != action_count:
            self.logger.warning(
                f"Mismatched counts: {input_count} input images vs {action_count} action images. Using min length."
            )

        num_pairs = min(input_count, action_count)
        self.logger.info(f"Processing {num_pairs} image pairs across multiple episodes...")

        # Group and process episodes one at a time for memory efficiency
        episode_groups = self._group_files_by_episode(input_files[:num_pairs], action_files[:num_pairs])

        for episode_num in sorted(episode_groups.keys()):
            episode_files = episode_groups[episode_num]
            episode_data = self._process_episode_files_streaming(episode_files['input_files'], episode_files['action_files'], episode_num)
            if episode_data is not None:
                yield episode_data

    def _group_files_by_episode(self, input_files: list[Path], action_files: list[Path]) -> dict[int, dict[str, list[Path]]]:
        """Group files by episode number without loading all into memory at once."""
        episode_groups = {}

        for input_file, action_file in zip(input_files, action_files):
            episode_num = self._extract_episode_from_filename(input_file.stem)

            if episode_num is not None:
                if episode_num not in episode_groups:
                    episode_groups[episode_num] = {'input_files': [], 'action_files': []}
                episode_groups[episode_num]['input_files'].append(input_file)
                episode_groups[episode_num]['action_files'].append(action_file)

        return episode_groups

    def _process_episode_files_streaming(self, input_files: list[Path], action_files: list[Path], episode_num: int) -> dict[str, list[Any]] | None:
        """Process files for a single episode."""
        episode_data: dict[str, list[Any]] = {
            "actions": [],
            "states": [],
            "images": {},
            "timestamps": [],
            "tasks": [],
            "task": []
        }

        img_key = self.config.image_keys[0] if self.config.image_keys else "observation.images.camera"
        episode_data["images"][img_key] = []

        # Process each frame in this episode
        for frame_idx in tqdm(range(min(len(input_files), len(action_files))), desc=f"Episode {episode_num}", unit="frame"):
            input_path = input_files[frame_idx]
            action_path = action_files[frame_idx]

            try:
                # Load observation image
                obs_img = load_image(input_path)

                # Load action image and extract coordinates
                action_img = load_image(action_path)
                xy = _extract_xy_from_label_image(action_img, threshold=self.threshold)
                if xy is None:
                    self.logger.warning(f"No white pixel found in action image: {action_path}. Skipping frame {frame_idx} in episode {episode_num}.")
                    continue

                # For sequential processing within episode, assume sequential agent assignment
                agent_id = frame_idx % self.num_agents
                agent_indicator = np.zeros(3, dtype=np.float32)
                agent_indicator[agent_id] = 1.0

                # Create 6D action vector with only one agent having non-zero values
                action_6d = np.zeros(6, dtype=np.float32)
                action_6d[agent_id * 2] = xy[0]
                action_6d[agent_id * 2 + 1] = xy[1]

                episode_data["images"][img_key].append(obs_img)
                episode_data["actions"].append(action_6d)
                episode_data["states"].append(agent_indicator)
                episode_data["timestamps"].append(float(frame_idx))  # Sequential within episode
                episode_data["tasks"].append(self.config.task_name)
                episode_data["task"].append(self.config.task_name)

            except Exception as e:
                self.logger.error(f"Error processing frame {frame_idx} in episode {episode_num}: {e}")
                if self.config.debug:
                    raise

        if len(episode_data["timestamps"]) == 0:
            self.logger.warning(f"Episode {episode_num} had no valid frames, skipping.")
            return None

        return episode_data

    def _extract_episode_from_filename(self, filename: str) -> int | None:
        """Extract episode number from filename. Expected format: ep_{episode}_..."""
        import re
        # Look for ep_{number} pattern
        match = re.search(r'ep_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return None


    def get_features(self) -> dict[str, dict]:
        """Define features for multi-agent actions and observations."""
        features: dict[str, dict] = {}

        # Multi-agent action features: 6D vector (x,y for each of 3 agents)
        features["action"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["agent0_x", "agent0_y", "agent1_x", "agent1_y", "agent2_x", "agent2_y"],
        }

        # Agent indicator state: 3D one-hot vector showing which agent is acting
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["agent0_active", "agent1_active", "agent2_active"],
        }

        # Observation image features
        # Determine dimensions from the first image in input_subdir
        input_dir = self.config.input_dir / self.input_subdir
        sample_files = sorted([p for p in input_dir.iterdir() if p.is_file()])
        if not sample_files:
            raise ValueError(f"No images found in {input_dir} to infer image shape")

        h, w, c = get_image_dimensions(sample_files[0])
        img_key = self.config.image_keys[0] if self.config.image_keys else "observation.images.camera"
        features[img_key] = {
            "dtype": "video" if self.config.use_videos else "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }

        return features
