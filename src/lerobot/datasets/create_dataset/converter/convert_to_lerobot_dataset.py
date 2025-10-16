"""Main converter class for dataset conversion.

Performance optimizations:
  - Parallel episode processing using multiprocessing
  - Thread-safe dataset writing
  - Configurable parallelization levels
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # Lock for thread-safe dataset operations
        self._dataset_lock = threading.Lock()

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

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=self.config.output_dir,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=self.config.use_videos,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=self.config.image_writer_threads,
            tolerance_s=self.config.tolerance_s,
        )

        # Prefer parsing by explicit episode numbers when parser supports it
        episode_files = self.parser.get_episode_files()
        episode_numbers = getattr(self.parser, "list_episode_numbers", lambda: [])()
        use_episode_numbers = bool(episode_numbers)
        
        self.logger.info(f"Episode detection: files={len(episode_files)}, numbers={len(episode_numbers)}")
        if episode_numbers:
            self.logger.info(f"Found episodes: {episode_numbers[:10]}{'...' if len(episode_numbers) > 10 else ''}")
        
        if not episode_files and not use_episode_numbers:
            raise ValueError(f"No episode files found in {self.config.input_dir}")

        # Determine if parallel processing should be used
        enable_parallel = getattr(self.config, 'enable_parallel_processing', True)
        num_parallel_episodes = getattr(self.config, 'num_parallel_episodes', 4)
        
        # Disable parallel processing in debug mode for easier debugging
        if self.config.debug:
            enable_parallel = False
            self.logger.info("Parallel processing disabled in debug mode")
        
        if enable_parallel and num_parallel_episodes > 0:
            self.logger.info(f"Processing episodes in parallel with {num_parallel_episodes} workers")
            if use_episode_numbers:
                self._convert_parallel_numbers(episode_numbers, num_parallel_episodes)
            else:
                self._convert_parallel(episode_files, num_parallel_episodes)
        else:
            self.logger.info("Processing episodes sequentially")
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
    
    def _convert_parallel(self, episode_files: list, num_workers: int) -> None:
        """Parallel episode processing using ThreadPoolExecutor.
        
        Uses threading instead of multiprocessing because:
        1. Image loading is I/O bound (benefits from threading)
        2. Avoids serialization overhead of the dataset object
        3. Simpler coordination with thread-safe locks
        """
        # Submit all episode parsing tasks
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create a mapping of futures to episode files
            future_to_episode = {
                executor.submit(self._process_episode_file, episode_file): episode_file 
                for episode_file in episode_files
            }
            
            # Process completed episodes as they finish
            with tqdm(total=len(episode_files), desc="Converting episodes") as pbar:
                for future in as_completed(future_to_episode):
                    episode_file = future_to_episode[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error processing {episode_file}: {e}")
                        pbar.update(1)
                        if self.config.debug:
                            raise
    
    def _process_episode_file(self, episode_file) -> None:
        """Process a single episode file and add all its episodes to the dataset.
        
        This method is called in parallel, so it needs to be thread-safe.
        """
        try:
            # parse_episode returns a generator yielding episodes one by one
            for episode_data in self.parser.parse_episode(episode_file):
                # Use lock to ensure thread-safe addition to dataset
                with self._dataset_lock:
                    self._add_episode_to_dataset(episode_data)
        except Exception as e:
            self.logger.error(f"Error in episode file {episode_file}: {e}")
            if self.config.debug:
                raise

    def _convert_sequential_numbers(self, episode_numbers: list[int]) -> None:
        for ep in tqdm(episode_numbers, desc="Converting episodes"):
            for episode_data in self.parser.parse_episode_by_number(ep) or []:
                self._add_episode_to_dataset(episode_data)

    def _convert_parallel_numbers(self, episode_numbers: list[int], num_workers: int) -> None:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_ep = {executor.submit(self._process_episode_number, ep): ep for ep in episode_numbers}
            with tqdm(total=len(episode_numbers), desc="Converting episodes") as pbar:
                for future in as_completed(future_to_ep):
                    ep = future_to_ep[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error processing episode {ep}: {e}")
                        pbar.update(1)
                        if self.config.debug:
                            raise

    def _process_episode_number(self, ep: int) -> None:
        for episode_data in self.parser.parse_episode_by_number(ep) or []:
            with self._dataset_lock:
                self._add_episode_to_dataset(episode_data)

    def _add_episode_to_dataset(self, episode_data: dict[str, Any]) -> None:
        """Add parsed episode data to the dataset."""
        num_frames = len(episode_data["timestamps"])
        start_time = time.time()

        for frame_idx in range(num_frames):
            frame = self._create_frame(episode_data, frame_idx)
            # Inject task into frame for validation and storage
            frame["task"] = episode_data["tasks"][frame_idx]

            # Validate frame if enabled (timestamp is handled internally by the dataset)
            if self.config.validate_data:
                validate_frame(frame, self.dataset.features)

            # Add frame to dataset (timestamp will be set automatically based on fps)
            self.dataset.add_frame(frame)

        # Save the complete episode
        save_start = time.time()
        self.dataset.save_episode()
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
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
