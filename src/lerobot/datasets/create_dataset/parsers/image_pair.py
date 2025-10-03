"""Parser for paired image datasets (input image + action label image).

This parser expects two flat subdirectories inside `config.input_dir`:
  - `input_ag2`: contains observation/input images (chronological order)
  - `action_ag2`: contains action label images with a white pixel indicating the action

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
    """Extract normalized (x, y) from an action label image.

    Args:
        img_uint8: Image array (H, W) or (H, W, C) with values in [0, 255].
        threshold: Brightness threshold to consider a pixel as white.

    Returns:
        (x, y) normalized to [0, 1] or None if no white pixels are found.
    """
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

    # Normalize with top-left origin convention
    x_norm = float(x_center / max(1, (w - 1)))
    y_norm = float(y_center / max(1, (h - 1)))
    return x_norm, y_norm


class ImagePairParser(DataParser):
    """Parser for paired image datasets (input_ag2/action_ag2)."""

    def __init__(self, config):
        super().__init__(config)
        # Allow overriding subdir names and threshold via config if present
        self.input_subdir = getattr(config, "input_subdir", "input_ag2")
        self.action_subdir = getattr(config, "action_subdir", "action_ag2")
        self.threshold = getattr(config, "action_threshold", 200)

    def get_episode_files(self) -> list[Path]:
        """Return a synthetic list with a single placeholder Path.

        The parser is directory-driven; we don't rely on CSV files. We use a single
        episode and return a list containing the input directory path to satisfy the API.
        """
        return [self.config.input_dir]

    def parse_episode(self, episode_file: Path) -> dict[str, list[Any]]:
        """Parse a single episode comprised of all image pairs in the two subdirs."""
        input_dir = self.config.input_dir / self.input_subdir
        action_dir = self.config.input_dir / self.action_subdir

        if not input_dir.exists() or not action_dir.exists():
            raise FileNotFoundError(f"Expected subdirectories '{self.input_subdir}' and '{self.action_subdir}' under {self.config.input_dir}")

        # Show progress while loading file lists
        self.logger.info("Loading input files...")
        input_files = sorted([p for p in input_dir.iterdir() if p.is_file()])
        self.logger.info(f"Found {len(input_files)} input files")
        
        self.logger.info("Loading action files...")
        action_files = sorted([p for p in action_dir.iterdir() if p.is_file()])
        self.logger.info(f"Found {len(action_files)} action files")

        if len(input_files) == 0 or len(action_files) == 0:
            raise ValueError("No files found in input/action subdirectories")

        if len(input_files) != len(action_files):
            self.logger.warning(
                f"Mismatched counts: {len(input_files)} input images vs {len(action_files)} action images. Zipping to min length."
            )

        num_pairs = min(len(input_files), len(action_files))

        episode_data: dict[str, list[Any]] = {
            "actions": [], 
            "states": [], 
            "images": {}, 
            "timestamps": [], 
            "tasks": [],
            "task": []  # Add task field for frame validation
        }
        # Initialize image list for the primary camera key
        img_key = self.config.image_keys[0] if self.config.image_keys else "observation.images.camera"
        episode_data["images"][img_key] = []

        self.logger.info(f"Processing {num_pairs} image pairs...")
        for idx in tqdm(range(num_pairs), desc="Loading images", unit="pair"):
            input_path = input_files[idx]
            action_path = action_files[idx]

            try:
                # Load observation image
                obs_img = load_image(input_path)

                # Load action image in uint8 without forcing channel repeat (load_image returns 3ch)
                action_img = load_image(action_path)
                xy = _extract_xy_from_label_image(action_img, threshold=self.threshold)
                if xy is None:
                    self.logger.warning(f"No white pixel found in action image: {action_path}. Skipping frame {idx}.")
                    continue

                episode_data["images"][img_key].append(obs_img)
                episode_data["actions"].append(np.array([xy[0], xy[1]], dtype=np.float32))
                episode_data["timestamps"].append(float(idx))
                episode_data["tasks"].append(self.config.task_name)
                episode_data["task"].append(self.config.task_name)  # Add task for each frame
            except Exception as e:
                self.logger.error(f"Error processing pair {idx}: {e}")
                if self.config.debug:
                    raise

        if len(episode_data["timestamps"]) == 0:
            raise ValueError("All frames skipped; no valid action pixels found.")

        return episode_data

    def get_features(self) -> dict[str, dict]:
        """Define features for action (x,y) and observation image."""
        features: dict[str, dict] = {}

        # Action features: (x, y)
        features["action"] = {
            "dtype": "float32",
            "shape": (2,),
            "names": ["x", "y"],
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


