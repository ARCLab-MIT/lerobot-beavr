import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import huggingface_hub
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_CHUNK_SIZE


def _ensure_minimum_ndim(obj):
    """Recursively ensure that every numeric value is at least a 1-D NumPy array."""
    if isinstance(obj, dict):
        return {k: _ensure_minimum_ndim(v) for k, v in obj.items()}
    # If it is already an array, use np.atleast_1d
    elif isinstance(obj, np.ndarray):
        return np.atleast_1d(obj)
    # If it is a numeric type, wrap it.
    elif isinstance(obj, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return np.array([obj])
    # If it is a list, try to convert to a NumPy array (and then ensure at least 1-D)
    elif isinstance(obj, list):
        try:
            return np.atleast_1d(np.array(obj))
        except Exception:
            return obj
    # Fallback: try to force it with np.atleast_1d
    try:
        return np.atleast_1d(obj)
    except Exception:
        return obj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVToLeRobotDatasetConverter:
    """
    Converts CSV trajectory data and associated images to LeRobotDataset v2.1 format.
    
    This class handles the conversion of CSV files containing trajectory data (position, velocity, 
    attitude, etc.) and associated images into the LeRobotDataset v2.1 format, which can be used
    for training and evaluation with the LeRobot framework.
    """
    
    def __init__(
        self,
        csv_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        repo_id: str,
        fps: Optional[int] = None,
        task_name: str = "meta-RL policy rollout",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        image_key: str = "observation.images",
        image_pattern: str = "img_traj_{episode}_step_{frame}",
        image_extension: str = ".png",
        csv_pattern: str = "trajectory_{episode}.csv",
        use_videos: bool = False,
        debug: bool = False,
        image_height: int = 480,
        image_width: int = 640,
        robot_type: str = "inav",
    ):
        """
        Initialize the converter.
        
        Args:
            csv_dir: Directory containing CSV files
            image_dir: Directory containing image files
            output_dir: Directory to save the dataset
            repo_id: Repository ID for the dataset
            fps: Frames per second (if None, calculated from data)
            task_name: Name of the task
            chunk_size: Number of episodes per chunk
            image_key: Key for image feature
            image_pattern: Pattern for image filenames
            image_extension: File extension for images
            csv_pattern: Pattern for CSV filenames
            use_videos: Whether to encode images as videos
            debug: Whether to enable debug logging
            image_height: Height of images in pixels
            image_width: Width of images in pixels
            robot_type: Type of robot
        """
        self.csv_dir = Path(csv_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.task_name = task_name
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.image_pattern = image_pattern
        self.image_extension = image_extension
        self.csv_pattern = csv_pattern
        self.use_videos = use_videos
        self.debug = debug
        self.image_height = image_height
        self.image_width = image_width
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize features and stats
        self.features = {}
        self.stats = {}
        self.episodes_stats = {}
        self.robot_type = robot_type
        
        # Set up logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Verify that CSV and image directories exist
        if not self.csv_dir.exists():
            raise ValueError(f"CSV directory does not exist: {self.csv_dir}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        
        # Check for image files
        image_files = list(self.image_dir.glob(f"*{self.image_extension}"))
        if not image_files:
            logger.warning(f"No image files found in {self.image_dir} with extension {self.image_extension}")
        else:
            logger.info(f"Found {len(image_files)} image files")
            
            # Log a few sample image filenames
            sample_images = image_files[:5]
            logger.info(f"Sample image files: {[img.name for img in sample_images]}")

    def _copy_image(self, episode_index: int, frame_index: int, dest_path: Path) -> None:
        """
        Copy an image file from the source directory to the destination path.
        
        Args:
            episode_index: Episode index
            frame_index: Frame index
            dest_path: Destination path
        """
        # Create the destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the source image filename using the pattern
        source_filename = self.image_pattern.format(episode=episode_index, frame=frame_index) + self.image_extension
        source_path = self.image_dir / source_filename
        
        # Check if the source image exists
        if not source_path.exists():
            logger.warning(f"Image file not found: {source_path}")
            # Create a blank image as a fallback
            blank_image = Image.new('RGB', (self.image_height, self.image_width), color='black')
            blank_image.save(dest_path)
            return
        
        # Copy the image
        try:
            shutil.copy2(source_path, dest_path)
            if self.debug:
                logger.debug(f"Copied image from {source_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy image: {e}")
            # Create a blank image as a fallback
            blank_image = Image.new('RGB', (self.image_height, self.image_width), color='black')
            blank_image.save(dest_path)

    def _create_video(self, episode_index: int, frame_count: int, chunk_dir: Path) -> None:
        """
        Create a video from a sequence of images.
        
        Args:
            episode_index: Episode index
            frame_count: Number of frames
            chunk_dir: Directory to save the video
        """
        # Import cv2 here to avoid dependency issues
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for video encoding. Please install it with 'pip install opencv-python'.")
            return
        
        # Create the video directory
        video_dir = chunk_dir / self.image_key
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the video file path
        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.image_width, self.image_height))
        
        # Add each frame to the video
        for frame_index in range(frame_count):
            # Format the source image filename using the pattern
            source_filename = self.image_pattern.format(episode=episode_index, frame=frame_index) + self.image_extension
            source_path = self.image_dir / source_filename
            
            # Check if the source image exists
            if not source_path.exists():
                logger.warning(f"Image file not found: {source_path}")
                # Create a blank frame as a fallback
                blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                video.write(blank_frame)
                continue
            
            # Read the image and add it to the video
            try:
                frame = cv2.imread(str(source_path))
                if frame is None:
                    logger.warning(f"Failed to read image: {source_path}")
                    blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                    video.write(blank_frame)
                    continue
                
                # Resize the frame if necessary
                if frame.shape[:2] != (self.image_height, self.image_width):
                    frame = cv2.resize(frame, (self.image_width, self.image_height))
                
                video.write(frame)
            except Exception as e:
                logger.error(f"Failed to process image for video: {e}")
                blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                video.write(blank_frame)
        
        # Release the video
        video.release()
        
        if self.debug:
            logger.debug(f"Created video: {video_path}")

    def _find_csv_files(self) -> List[Tuple[int, Path]]:
        """
        Find all CSV files in the CSV directory that match the pattern.
        
        Returns:
            List of tuples containing episode index and path to CSV file
        """
        csv_files = []
        
        # Check if the pattern contains a placeholder for the episode index
        if "{episode}" in self.csv_pattern:
            # Find all CSV files that match the pattern
            for csv_path in self.csv_dir.glob("*.csv"):
                try:
                    # Extract the episode index from the filename
                    filename = csv_path.name
                    pattern_parts = self.csv_pattern.split("{episode}")
                    if len(pattern_parts) != 2:
                        logger.warning(f"Invalid CSV pattern: {self.csv_pattern}")
                        continue
                    
                    prefix, suffix = pattern_parts
                    if not filename.startswith(prefix) or not filename.endswith(suffix):
                        continue
                    
                    episode_str = filename[len(prefix):-len(suffix)]
                    episode_index = int(episode_str)
                    
                    csv_files.append((episode_index, csv_path))
                except Exception as e:
                    logger.warning(f"Failed to parse episode index from {csv_path}: {e}")
        else:
            # If no pattern is specified, use all CSV files and assign sequential indices
            for i, csv_path in enumerate(sorted(self.csv_dir.glob("*.csv"))):
                csv_files.append((i, csv_path))
        
        # Sort by episode index
        csv_files.sort(key=lambda x: x[0])
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_dir} that match the pattern {self.csv_pattern}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        return csv_files

    def _process_csv_file(self, episode_index: int, csv_path: Path) -> pd.DataFrame:
        """
        Process a CSV file to add frame index and episode index.
        
        Args:
            episode_index: Episode index
            csv_path: Path to CSV file
            
        Returns:
            Processed DataFrame
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Add frame index
        df["frame_index"] = np.arange(len(df))
        
        # Add episode index
        df["episode_index"] = episode_index
        
        # Add task index (always 0 for now)
        df["task_index"] = 0
        
        # Add index (unique identifier for each row)
        df["index"] = np.arange(len(df)) + episode_index * 1000000  # Ensure uniqueness across episodes
        
        # Add task (string identifier for the task)
        df["task"] = self.task_name
        
        return df

    def _restructure_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restructure the DataFrame to match the expected schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Restructured DataFrame
        """
        # Create a new DataFrame with the required structure
        restructured_data = {}
        
        # Extract observation.state columns in the correct order
        state_columns = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3", "w1", "w2", "w3"]
        
        # Create observation.state sequences
        observation_states = []
        for _, row in df.iterrows():
            state_values = []
            for col in state_columns:
                # Use the column name from the CSV file
                state_values.append(float(row[col]))
            observation_states.append(state_values)
        
        restructured_data["observation.state"] = observation_states
        
        # Extract action columns in the correct order
        action_columns = ["Tx", "Ty", "Tz", "Lx", "Ly", "Lz"]
        
        # Create action sequences
        actions = []
        for _, row in df.iterrows():
            action_values = []
            for col in action_columns:
                # Use the column name from the CSV file
                action_values.append(float(row[col]))
            actions.append(action_values)
        
        restructured_data["action"] = actions
        
        # Add other required columns
        restructured_data["timestamp"] = df["time"].values
        restructured_data["frame_index"] = df["frame_index"].values
        restructured_data["episode_index"] = df["episode_index"].values
        restructured_data["index"] = df["index"].values
        restructured_data["task_index"] = df["task_index"].values
        
        # Create the restructured DataFrame
        restructured_df = pd.DataFrame(restructured_data)
        
        return restructured_df

    def _calculate_statistics(self, all_dfs: Dict[int, pd.DataFrame]) -> None:
        """
        Calculate statistics for each episode using the compute_stats module.
        Assumes DataFrames in all_dfs have image paths relative to self.output_dir if images are used.
        
        Args:
            all_dfs: Dictionary mapping episode indices to DataFrames
        """
        from lerobot.common.datasets.compute_stats import compute_episode_stats
        
        logger.info("Calculating episode statistics...")
        
        self.episodes_stats = {}
        
        features = {
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"},
            "timestamp": {"dtype": "float32"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "task_index": {"dtype": "int64"}
        }

        if not self.use_videos and self.image_key:
            features[self.image_key] = {"dtype": "image"}
        
        for episode_index, df in tqdm(all_dfs.items(), desc="Calculating Episode Statistics"):
            episode_data = {}
            
            # Extract observation.state data
            if "observation.state" in df.columns:
                episode_data["observation.state"] = df["observation.state"].values
            else:
                # Try to construct from individual state columns
                state_columns = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3", "w1", "w2", "w3"]
                if all(col in df.columns for col in state_columns):
                    episode_data["observation.state"] = df[state_columns].values
                else:
                    logger.warning(f"State columns not found for episode {episode_index}. Using restructured data.")
                    # Use the restructured data if available
                    if "observation.state" in df.columns:
                        episode_data["observation.state"] = _ensure_minimum_ndim(df["observation.state"].values)
            
            # Extract action data
            if "action" in df.columns:
                episode_data["action"] = df["action"].values
            else:
                # Try to construct from individual action columns
                action_columns = ["Tx", "Ty", "Tz", "Lx", "Ly", "Lz"]
                if all(col in df.columns for col in action_columns):
                    episode_data["action"] = df[action_columns].values
                else:
                    logger.warning(f"Action columns not found for episode {episode_index}. Skipping action stats.")
            
            # Add other required columns
            for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
                if col in df.columns:
                    episode_data[col] = df[col].values
            
            if not self.use_videos and self.image_key:
                # Fix: Use the correct path structure for images
                try:
                    # Create paths to the organized images
                    chunk_index = episode_index // self.chunk_size
                    episode_dir = f"episode_{episode_index:06d}"
                    image_paths = []
                    
                    # Get frame count from the DataFrame
                    frame_count = len(df)
                    
                    # Generate paths for all frames
                    for frame_index in range(frame_count):
                        # Fix: Use the correct path structure that matches _organize_images method
                        image_path = self.output_dir / "images" / f"chunk-{chunk_index:03d}" / self.image_key / episode_dir / f"frame_{frame_index:06d}{self.image_extension}"
                        
                        # Check if the file exists before adding it
                        if image_path.exists():
                            image_paths.append(str(image_path))
                        else:
                            # Try with absolute path
                            abs_path = image_path.resolve()
                            if abs_path.exists():
                                image_paths.append(str(abs_path))
                    
                    if image_paths:
                        episode_data[self.image_key] = image_paths
                        logger.debug(f"Found {len(image_paths)} images for episode {episode_index}")
                    else:
                        logger.warning(f"No image paths found for episode {episode_index}. Skipping image stats.")
                        # Create a dummy image path to avoid errors
                        dummy_image = Image.new('RGB', (self.image_width, self.image_height), color='black')
                        dummy_path = self.output_dir / f"dummy_image_{episode_index}.png"
                        dummy_image.save(dummy_path)
                        episode_data[self.image_key] = [str(dummy_path)]
                except Exception as e:
                    logger.warning(f"Error generating image paths for episode {episode_index}: {e}")
            
            try:
                if episode_data:  # Only compute stats if we have data
                    # Ensure all arrays are at least 1D
                    episode_data = _ensure_minimum_ndim(episode_data)
                    self.episodes_stats[episode_index] = compute_episode_stats(episode_data, features)
                else:
                    logger.warning(f"No data available for episode {episode_index}. Skipping stats.")
                    self.episodes_stats[episode_index] = {}
            except Exception as e:
                logger.error(f"Error computing stats for episode {episode_index}: {e}")
                self.episodes_stats[episode_index] = {}

        logger.info(f"Calculated statistics for {len(self.episodes_stats)} episodes")

    def _create_hf_features(self) -> datasets.Features:
        """Create HuggingFace features dictionary."""
        import datasets
        
        feature_dict = {
            "action": datasets.Sequence(datasets.Value("float32")),
            "observation.state": datasets.Sequence(datasets.Value("float32")),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
        }
        
        # Add image feature if using images
        if self.use_videos and self.image_key:
            feature_dict[self.image_key] = datasets.Image()
        
        return datasets.Features(feature_dict)

    def _create_metadata(self, csv_files: List[Tuple[int, Path]], episode_lengths: List[int]) -> None:
        """Create metadata files for the dataset."""
        # Create meta directory
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Create info.json
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": len(csv_files),
            "total_frames": sum(episode_lengths),
            "total_tasks": 1,
            "total_videos": len(csv_files) if self.use_videos else 0,
            "total_chunks": (len(csv_files) + self.chunk_size - 1) // self.chunk_size,
            "chunks_size": self.chunk_size,
            "splits": {"train": f"0:{len(csv_files)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4" if self.use_videos else None,
            "image_path": "images/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png" if not self.use_videos else None,
            "features": {
                "observation.images.cam": {
                    "dtype": "video" if self.use_videos else "image",
                    "shape": [3, self.image_height, self.image_width],
                    "names": ["channels", "height", "width"],
                    "video_info": {
                        "video.fps": self.fps,
                        "video.height": self.image_height,
                        "video.width": self.image_width,
                        "video.channels": 3,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    } if self.use_videos else None
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [13],
                    "names": [
                        "x", "y", "z",
                        "vx", "vy", "vz",
                        "q0", "q1", "q2", "q3",
                        "w1", "w2", "w3"
                    ]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["Tx", "Ty", "Tz", "Lx", "Ly", "Lz"]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        
        # Helper function to convert numpy arrays to Python types
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        # Write info.json
        with open(meta_dir / "info.json", "w") as f:
            json.dump(convert_numpy_to_python(info), f, indent=2)
        
        # Create tasks.jsonl (single line)
        with open(meta_dir / "tasks.jsonl", "w") as f:
            task_data = {
                "task_index": 0,
                "task": self.task_name
            }
            f.write(json.dumps(task_data) + "\n")
        
        # Create episodes.jsonl
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for episode_index, _ in csv_files:
                episode_data = {
                    "episode_index": episode_index,
                    "tasks": [self.task_name],
                    "length": episode_lengths[episode_index]
                }
                f.write(json.dumps(episode_data) + "\n")
        
        # Create episodes_stats.jsonl
        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for episode_index, stats in self.episodes_stats.items():
                episode_stats_dict = {
                    "episode_index": episode_index,
                    "stats": convert_numpy_to_python(stats),
                }
                f.write(json.dumps(episode_stats_dict) + "\n")
        
        # Create image_keys.json to explicitly define image keys
        with open(meta_dir / "image_keys.json", "w") as f:
            json.dump(["observation.images.cam"], f, indent=2)

    def _embed_images(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Embed images in the dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with embedded images
        """
        # This is a placeholder for future implementation
        # Currently, images are stored as separate files and not embedded in the dataset
        return dataset

    def _process_chunk(self, chunk_index: int, episodes: List[Tuple[int, Path]]) -> None:
        """Process a chunk of episodes."""
        logger.info(f"Processing chunk {chunk_index} with {len(episodes)} episodes")
        
        # Create chunk directory
        chunk_dir = self.output_dir / f"data/chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HuggingFace features
        hf_features = self._create_hf_features()
        
        # Process each episode in the chunk
        for episode_index, csv_file in episodes:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Add required columns if they don't exist
            if "frame_index" not in df.columns:
                df["frame_index"] = range(len(df))
            if "episode_index" not in df.columns:
                df["episode_index"] = episode_index
            if "index" not in df.columns:
                df["index"] = range(len(df))
            if "task_index" not in df.columns:
                df["task_index"] = 0  # Default task index
            
            # Restructure DataFrame
            restructured_df = self._restructure_dataframe(df)
            
            # Convert to HuggingFace dataset
            try:
                hf_dataset = datasets.Dataset.from_pandas(restructured_df, features=hf_features)
                
                # Save as parquet
                output_file = chunk_dir / f"episode_{episode_index:06d}.parquet"
                hf_dataset.to_parquet(output_file)
                
                logger.debug(f"Saved episode {episode_index} to {output_file}")
            except Exception as e:
                logger.error(f"Failed to process episode {episode_index}: {str(e)}")
                raise

    def _organize_images(self, csv_files: List[Tuple[int, Path]]) -> None:
        """
        Organize images into the standard LeRobotDataset directory structure.
        
        When use_videos is False, this method copies images from the source directory
        to the target directory with the structure:
        {dataset_root}/images/chunk-{chunk_index:03d}/{image_key_as_directory}/episode_{episode_index:06d}/frame_{frame_index:06d}.png
        
        Args:
            csv_files: List of tuples containing episode index and path to CSV file
        """
        if self.use_videos:
            logger.info("Using videos mode, skipping image organization")
            return
        
        logger.info("Organizing images into standard directory structure...")
        
        # Create image directory structure
        image_base_dir = self.output_dir / "images"
        image_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each episode
        for episode_index, csv_path in tqdm(csv_files, desc="Organizing images"):
            # Calculate chunk index
            chunk_index = episode_index // self.chunk_size
            
            # Create directory structure for this episode
            # Fix: Use consistent naming format for episode directories
            target_dir = image_base_dir / f"chunk-{chunk_index:03d}" / self.image_key / f"episode_{episode_index:06d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Read CSV to get frame count
            df = pd.read_csv(csv_path)
            frame_count = len(df)
            
            # Copy each image
            for frame_index in range(frame_count):
                # Format the source image filename using the pattern
                source_filename = self.image_pattern.format(episode=episode_index, frame=frame_index) + self.image_extension
                source_path = self.image_dir / source_filename
                
                # Format the target image filename
                target_path = target_dir / f"frame_{frame_index:06d}{self.image_extension}"
                
                # Copy the image if it exists
                if source_path.exists():
                    try:
                        shutil.copy2(source_path, target_path)
                        if self.debug:
                            logger.debug(f"Copied image from {source_path} to {target_path}")
                    except Exception as e:
                        logger.error(f"Failed to copy image: {e}")
                        # Create a blank image as a fallback
                        blank_image = Image.new('RGB', (self.image_width, self.image_height), color='black')
                        blank_image.save(target_path)
                else:
                    logger.warning(f"Image file not found: {source_path}")
                    # Create a blank image as a fallback
                    blank_image = Image.new('RGB', (self.image_width, self.image_height), color='black')
                    blank_image.save(target_path)
        
        logger.info(f"Image organization complete. Images stored in {image_base_dir}")

    def convert(self) -> None:
        """
        Convert CSV files and images to LeRobotDataset format.
        """
        # Find all CSV files
        csv_files = self._find_csv_files()
        
        # Process all CSV files first to get the data
        all_dfs = {}
        episode_lengths = {}
        
        for episode_index, csv_path in tqdm(csv_files, desc="Processing CSV files"):
            # Process the CSV file
            df = self._process_csv_file(episode_index, csv_path)
            all_dfs[episode_index] = df
            episode_lengths[episode_index] = len(df)
            
            # Calculate FPS if not provided
            if self.fps is None and "timestamp" in df.columns:
                # Calculate FPS from timestamps
                timestamps = df["timestamp"].values
                if len(timestamps) > 1:
                    time_diff = timestamps[-1] - timestamps[0]
                    if time_diff > 0:
                        self.fps = int(len(timestamps) / time_diff)
                        logger.info(f"Calculated FPS: {self.fps}")
        
        # Set default FPS if not calculated
        if self.fps is None:
            self.fps = 5  # Default FPS
            logger.info(f"Using default FPS: {self.fps}")
        
        # Organize images if not using videos
        self._organize_images(csv_files)
        
        # Create chunks of episodes
        chunks = []
        current_chunk = []
        
        for episode_index, csv_path in csv_files:
            current_chunk.append((episode_index, csv_path))
            
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process each chunk
        for chunk_index, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            self._process_chunk(chunk_index, chunk)
        
        # Calculate statistics
        self._calculate_statistics(all_dfs)
        
        # Create metadata files
        self._create_metadata(csv_files, episode_lengths)
        
        logger.info(f"Dataset conversion complete. Output directory: {self.output_dir}")
    
    def _create_videos(self, episode_index: int, chunk_index: int, df: pd.DataFrame) -> None:
        """
        Create videos from images for an episode.
        
        Args:
            episode_index: Episode index
            chunk_index: Chunk index
            df: DataFrame for the episode
        """
        import cv2
        
        # Create video directory structure
        video_dir = self.output_dir / "videos" / f"chunk-{chunk_index:03d}" / self.image_key
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Video path
        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        
        # Collect all images for this episode
        images = []
        for frame_index in range(len(df)):
            # Format episode and frame indices
            episode_str = f"{episode_index}"
            frame_str = f"{frame_index}"
            
            # Construct image filename based on pattern
            image_filename = self.image_pattern.format(episode=episode_str, frame=frame_str) + self.image_extension
            
            # Image path
            image_path = self.image_dir / image_filename
            
            # Check if image exists
            if image_path.exists():
                # Read the image
                img = cv2.imread(str(image_path))
                if img is not None:
                    images.append(img)
                else:
                    logger.warning(f"Failed to read image: {image_path}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        # Create video if we have images
        if images:
            # Get image dimensions
            height, width = images[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
            
            # Write images to video
            for img in images:
                video_writer.write(img)
            
            # Release video writer
            video_writer.release()
            
            logger.info(f"Created video for episode {episode_index} with {len(images)} frames")
        else:
            logger.warning(f"No images found for episode {episode_index}")
    
    def load_dataset(self) -> LeRobotDataset:
        """Load the converted dataset using LeRobotDataset."""
        return LeRobotDataset(
            repo_id=self.repo_id,
            root=self.output_dir,
        )
    
    def push_to_hub(self, private: bool = False) -> None:
        """
        Push the dataset to the Hugging Face Hub.
        
        Args:
            private: Whether to make the dataset private
        """
        # Create a .gitattributes file to ensure large files are tracked with Git LFS
        gitattributes_path = self.output_dir / ".gitattributes"
        with open(gitattributes_path, "w") as f:
            f.write("*.png filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.jpg filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.jpeg filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
        
        # Create a README.md file with dataset information
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# {self.repo_id.split('/')[-1]}\n\n")
            f.write(f"Dataset created with CSVToLeRobotDatasetConverter for {self.robot_type} robot.\n\n")
            f.write(f"- Task: {self.task_name}\n")
            f.write(f"- FPS: {self.fps}\n")
            f.write(f"- Image dimensions: {self.image_width}x{self.image_height}\n")
        
        # Use huggingface_hub directly to upload the dataset
        logger.info(f"Pushing dataset to {self.repo_id}")
        
        try:
            # First create the repo if it doesn't exist
            huggingface_hub.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            
            # Then upload all files
            huggingface_hub.upload_folder(
                folder_path=str(self.output_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
                ignore_patterns=["*.pyc", "__pycache__", ".git*"],
            )
            logger.info(f"Successfully pushed dataset to {self.repo_id}")
        except Exception as e:
            logger.error(f"Failed to push dataset to hub: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSV trajectory data to LeRobotDataset format")
    parser.add_argument("--csv-dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second (if None, calculated from data)")
    parser.add_argument("--task-name", type=str, default="meta-RL policy rollout", help="Name of the task")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of episodes per chunk")
    parser.add_argument("--image-key", type=str, default="observation.images", help="Key for image feature")
    parser.add_argument("--image-pattern", type=str, default="img_traj_{episode}_step_{frame}", help="Pattern for image filenames")
    parser.add_argument("--image-extension", type=str, default=".png", help="File extension for images")
    parser.add_argument("--csv-pattern", type=str, default="trajectory_{episode}.csv", help="Pattern for CSV filenames")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--push", action="store_false", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--robot-type", type=str, default="inav", help="Type of robot")
    # Add image dimension arguments
    parser.add_argument("--image-height", type=int, default=480, help="Height of images in pixels")
    parser.add_argument("--image-width", type=int, default=640, help="Width of images in pixels")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter
    converter = CSVToLeRobotDatasetConverter(
        csv_dir=args.csv_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task_name=args.task_name,
        chunk_size=args.chunk_size,
        image_key=args.image_key,
        image_pattern=args.image_pattern,
        image_extension=args.image_extension,
        csv_pattern=args.csv_pattern,
        debug=args.debug,
        image_height=args.image_height,
        image_width=args.image_width,
        robot_type=args.robot_type,
    )
    
    # Convert dataset
    converter.convert()
    
    # Push to HuggingFace Hub if requested
    if args.push:
        converter.push_to_hub(private=args.private)


if __name__ == "__main__":
    main()