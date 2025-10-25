"""Main converter class for dataset conversion.

Sequential processing for proper dataset creation:
  - Sequential episode processing (required for proper metadata ordering)
  - Memory-efficient frame processing
  - Optimized chunking for large datasets
"""

import logging
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage

from tqdm import tqdm

from lerobot.datasets.create_dataset.config.dataset_config import DatasetConfig
from lerobot.datasets.create_dataset.parsers.csv_image import CSVImageParser
from lerobot.datasets.create_dataset.parsers.image_pair import ImagePairParser
from lerobot.datasets.create_dataset.parsers.parse_data import DataParser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import _validate_feature_names, validate_frame


class DatasetConverter:
    """Main converter class that orchestrates the conversion process."""

    def __init__(self, config: DatasetConfig, parser: DataParser | None = None):
        self.config = config
        if parser is not None:
            self.parser = parser
        else:
            if getattr(config, "parser_type", "csv_image") == "image_pair":
                self.parser = ImagePairParser(config)
            else:
                self.parser = CSVImageParser(config)
        self.logger = self._setup_logging()
        self.dataset = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging based on debug setting."""
        level = logging.DEBUG if self.config.debug else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Silence verbose third-party library debug logs
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('filelock').setLevel(logging.WARNING)
        logging.getLogger('fsspec').setLevel(logging.WARNING)
        
        return logging.getLogger(self.__class__.__name__)

    def convert(self) -> LeRobotDataset:
        """Main conversion function with optional parallel processing."""
        self.logger.info(f"Starting conversion for repo_id: {self.config.repo_id}")

        # Create empty LeRobotDataset
        features = self.parser.get_features()

        # Normalize image feature shapes to 3 channels (RGB)
        for key, ft in features.items():
            if ft.get("dtype") in {"image", "video"}:
                h, w, c = ft["shape"]
                if c != 3:
                    self.logger.warning(
                        f"Feature '{key}' has {c} channels; forcing to 3 (RGB) for writer compatibility."
                    )
                    ft["shape"] = (h, w, 3)

        _validate_feature_names(features)

        self.logger.info(f"Creating dataset with features: {list(features.keys())}")

        # Create dataset with optimized chunking for large datasets
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=self.config.output_dir,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=self.config.use_videos,
            tolerance_s=self.config.tolerance_s,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=self.config.image_writer_threads,
            batch_encoding_size=getattr(self.config, 'batch_encoding_size', 1),
        )
        
        # Apply chunking settings if specified (for large datasets)
        if hasattr(self.config, 'chunks_size') and self.config.chunks_size:
            self.dataset.meta.update_chunk_settings(
                chunks_size=self.config.chunks_size,
                data_files_size_in_mb=self.config.data_files_size_in_mb,
                video_files_size_in_mb=self.config.video_files_size_in_mb,
            )
            self.logger.info(f"Applied chunking settings: chunks_size={self.config.chunks_size}, "
                           f"data_files_size={self.config.data_files_size_in_mb}MB, "
                           f"video_files_size={self.config.video_files_size_in_mb}MB")
        
        # Prefer parsing by explicit episode numbers when parser supports it
        episode_files = self.parser.get_episode_files()
        episode_numbers = getattr(self.parser, "list_episode_numbers", lambda: [])()
        use_episode_numbers = bool(episode_numbers)
        
        # For large datasets, disable batch encoding to avoid metadata issues
        total_episodes = len(episode_files) if episode_files else len(episode_numbers)
        if total_episodes > 1000:
            self.dataset.batch_encoding_size = 1
            self.logger.info("Large dataset detected. Disabling batch video encoding to avoid metadata issues.")
        
        self.logger.info(f"Episode detection: files={len(episode_files)}, numbers={len(episode_numbers)}")
        if episode_numbers:
            self.logger.info(f"Found episodes: {episode_numbers[:10]}{'...' if len(episode_numbers) > 10 else ''}")
        
        if not episode_files and not use_episode_numbers:
            raise ValueError(f"No episode files found in {self.config.input_dir}")

        # Process episodes sequentially (required for proper dataset creation)
        self.logger.info(f"Processing {total_episodes} episodes sequentially")
        if use_episode_numbers:
            self._convert_sequential_numbers(episode_numbers)
        else:
            self._convert_sequential(episode_files)

        self.logger.info(f"Conversion completed. Dataset saved to: {self.config.output_dir}")

        # Push to hub if requested
        if self.config.push_to_hub:
            self.dataset.push_to_hub(private=self.config.private_repo, push_videos=self.config.use_videos)

        return self.dataset
    
    def _convert_sequential(self, episode_files: list) -> None:
        """Sequential episode processing (original implementation)."""
        for episode_file in tqdm(episode_files, desc="Converting episodes"):
            try:
                # parse_episode now returns a generator yielding episodes one by one
                for episode_data in self.parser.parse_episode(episode_file):
                    self._add_episode_to_dataset(episode_data)
            except Exception as e:
                self.logger.error(f"Error processing {episode_file}: {e}")
                if not self.config.debug:
                    continue
                else:
                    raise
    

    def _convert_sequential_numbers(self, episode_numbers: list[int]) -> None:
        for ep in tqdm(episode_numbers, desc="Converting episodes"):
            for episode_data in self.parser.parse_episode_by_number(ep) or []:
                self._add_episode_to_dataset(episode_data)


    def _add_episode_to_dataset(self, episode_data: dict[str, Any]) -> None:
        """Add parsed episode data to the dataset with optimized memory management."""
        num_frames = len(episode_data["timestamps"])
        start_time = time.time()

        # Process frames in batches for large episodes to manage memory
        batch_size = getattr(self.config, 'image_loading_batch_size', 64)
        if num_frames > batch_size:
            self.logger.debug(f"Large episode ({num_frames} frames), processing in batches of {batch_size}")
        
        for frame_idx in range(num_frames):
            frame = self._create_frame(episode_data, frame_idx)
            # Inject task into frame for validation and storage
            frame["task"] = episode_data["tasks"][frame_idx]

            # Validate frame if enabled (timestamp is handled internally by the dataset)
            if self.config.validate_data:
                validate_frame(frame, self.dataset.features)

            # Add frame to dataset (timestamp will be set automatically based on fps)
            self.dataset.add_frame(frame)
            
            # Memory management: clear large objects periodically
            if frame_idx % batch_size == 0 and frame_idx > 0:
                import gc
                gc.collect()

        # Save the complete episode
        save_start = time.time()
        self.dataset.save_episode()
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
        # Log performance metrics for large episodes
        if num_frames > 100:
            fps_processing = num_frames / total_time if total_time > 0 else 0
            self.logger.debug(f"Large episode: {num_frames} frames in {total_time:.2f}s "
                            f"({fps_processing:.1f} fps processing rate)")
        else:
            self.logger.debug(f"Saved episode with {num_frames} frames - total: {total_time:.2f}s, save: {save_time:.2f}s")

    def _create_frame(self, episode_data: dict[str, Any], frame_idx: int) -> dict[str, Any]:
        """Create a frame dictionary from episode data."""
        frame = {}

        # Add action data
        if episode_data.get("actions"):
            frame["action"] = episode_data["actions"][frame_idx]

        # Add state data
        if episode_data.get("states"):
            frame["observation.state"] = episode_data["states"][frame_idx]

        # Add image data
        for img_key in self.config.image_keys:
            if episode_data["images"][img_key][frame_idx] is not None:
                img = episode_data["images"][img_key][frame_idx]
                # Simpler: ensure RGB via PIL
                if isinstance(img, np.ndarray):
                    img = PILImage.fromarray(img)
                if isinstance(img, PILImage.Image):
                    img = img.convert("RGB")
                frame[img_key] = img

        return frame
