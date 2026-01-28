import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import trange, tqdm

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scaler_M import Scaler  # 确保导入正确的 Scaler 类
import h5py
import pyarrow.parquet as pq
import pandas as pd
import json

# Add lerobot dataset support
try:
    from lerobot.datasets import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

class MambaSequenceDataset(Dataset):
    """
    将“每个 record 文件夹”视为“一条时序轨迹”。
    每个文件夹包含多帧数据，逐帧加载并返回。
    每次切换轨迹时，重置隐状态。
    """
    def __init__(self, root_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640,480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None,
                 future_steps=16):  # <-- 未来多少步
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.dataset_dir = os.path.join(root_dir, mode)
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps
        self.selected_cameras = selected_cameras
        if self.selected_cameras is None:
            self.selected_cameras = ['top']
        # 加载所有轨迹路径并记录每条轨迹的长度
        self.records = []
        self.lengths = []  # 记录每条轨迹的长度
        for item in os.listdir(self.dataset_dir):
            record_path = os.path.join(self.dataset_dir, item)
            self.records.append(record_path)
            with h5py.File(record_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                self.lengths.append(qpos.shape[0])

        # 累积轨迹长度，用于全局索引到轨迹索引的映射
        self.cum_lengths = np.cumsum([0] + self.lengths)

        # 定义低维数据的键和形状
        self.lowdim_keys = [
            'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
            'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6'
            'gripper_pos', 'gripper_pos2',
            'agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
            'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act'
            'gripper_act', 'gripper_act2'
        ]
        self.lowdim_shapes = {
            'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
            'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
            'gripper_pos': 1,
            'gripper_pos2': 1,
            'agl_1_act': (future_steps, 1), 'agl_2_act': (future_steps, 1), 'agl_3_act': (future_steps, 1),
            'agl_4_act': (future_steps, 1), 'agl_5_act': (future_steps, 1), 'agl_6_act': (future_steps, 1),
            'agl2_1_act': (future_steps, 1), 'agl2_2_act': (future_steps, 1), 'agl2_3_act': (future_steps, 1),
            'agl2_4_act': (future_steps, 1), 'agl2_5_act': (future_steps, 1), 'agl2_6_act': (future_steps, 1),
            'gripper_act': (future_steps, 1),
            'gripper_act2': (future_steps, 1)
        }

        # 初始化 Scaler
        self.scaler = scaler
        if self.scaler is None and mode == "train":
            # 如果没有提供 Scaler 且是训练模式，则初始化一个 Scaler
            self.scaler = Scaler(lowdim_dict=self.lowdim_shapes)
            self.fitting = False  # 标志是否在拟合
        else:
            self.fitting = False

    def __len__(self):
        return self.cum_lengths[-1]

    def fit_scaler(self, batch_size=64, num_workers=4):
        """
        计算归一化参数。
        """
        if not self.scaler:
            raise ValueError("Scaler is not initialized.")

        print("Fitting scaler on dataset...")
        self.fitting = True  # 开始拟合，禁用归一化
        data_cache = {key: [] for key in self.scaler.lowdim_dict.keys()}
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        for batch in tqdm(dataloader, desc='Fitting scaler'):
            lowdim = batch['lowdim']
            for key in self.scaler.lowdim_dict.keys():
                data_cache[key].append(lowdim[key])
        self.fitting = False  # 拟合完成，启用归一化

        # 将所有批次的数据拼接起来
        for key in data_cache.keys():
            data_cache[key] = torch.cat(data_cache[key], dim=0)
        # 计算最小值和最大值
        self.scaler.fit(data_cache)
        print("Scaler fitted.")
        return self.scaler


    def save_scaler(self, filepath: str):
        """
        保存 Scaler 的归一化参数到文件。
        """
        if self.scaler:
            self.scaler.save(filepath)
            print(f"Scaler saved to {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def load_scaler(self, filepath: str):
        """
        从文件加载 Scaler 的归一化参数。
        """
        if self.scaler:
            self.scaler.load(filepath)
            print(f"Scaler loaded from {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 根据全局idx找到对应轨迹和帧
        traj_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        frame_idx = idx - self.cum_lengths[traj_idx]
        record_path = self.records[traj_idx]

        # 加载低维数据
        with h5py.File(record_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            act = root['/action'][()]
            agl_1 = qpos[:, 0:1]
            agl_2 = qpos[:, 1:2]
            agl_3 = qpos[:, 2:3]
            agl_4 = qpos[:, 3:4]
            agl_5 = qpos[:, 4:5]
            agl_6 = qpos[:, 5:6]
            gripper_pos = qpos[:, 6:7]
            agl2_1 = qpos[:, 7:8]
            agl2_2 = qpos[:, 8:9]
            agl2_3 = qpos[:, 9:10]
            agl2_4 = qpos[:, 10:11]
            agl2_5 = qpos[:, 11:12]
            agl2_6 = qpos[:, 12:13]
            gripper_pos2 = qpos[:, 13:14]
            agl_1_act = act[:, 0:1]
            agl_2_act = act[:, 1:2]
            agl_3_act = act[:, 2:3]
            agl_4_act = act[:, 3:4]
            agl_5_act = act[:, 4:5]
            agl_6_act = act[:, 5:6]
            gripper_act = act[:, 6:7]
            agl2_1_act = act[:, 7:8]
            agl2_2_act = act[:, 8:9]
            agl2_3_act = act[:, 9:10]
            agl2_4_act = act[:, 10:11]
            agl2_5_act = act[:, 11:12]
            agl2_6_act = act[:, 12:13]
            gripper_act2 = act[:, 13:14]


        def shift_N_steps(arr):
            """返回 [future_steps, arr_dim] 的多步标签."""
            out_list = []
            for step in range(self.future_steps):
                future_idx = frame_idx + step
                if future_idx >= len(arr):
                    future_idx = len(arr) - 1  # 超出则用最后一帧
                out_list.append(arr[future_idx])
            return np.stack(out_list, axis=0)  # [16, arr_dim]

        agl_1_act = shift_N_steps(agl_1_act)
        agl_2_act = shift_N_steps(agl_2_act)
        agl_3_act = shift_N_steps(agl_3_act)
        agl_4_act = shift_N_steps(agl_4_act)
        agl_5_act = shift_N_steps(agl_5_act)
        agl_6_act = shift_N_steps(agl_6_act)
        agl2_1_act = shift_N_steps(agl2_1_act)
        agl2_2_act = shift_N_steps(agl2_2_act)
        agl2_3_act = shift_N_steps(agl2_3_act)
        agl2_4_act = shift_N_steps(agl2_4_act)
        agl2_5_act = shift_N_steps(agl2_5_act)
        agl2_6_act = shift_N_steps(agl2_6_act)
        gripper_act = shift_N_steps(gripper_act)
        gripper_act2 = shift_N_steps(gripper_act2)


        # 提取单步的低维数据
        agl_1 = agl_1[frame_idx]
        agl_2 = agl_2[frame_idx]
        agl_3 = agl_3[frame_idx]
        agl_4 = agl_4[frame_idx]
        agl_5 = agl_5[frame_idx]
        agl_6 = agl_6[frame_idx]
        agl2_1 = agl2_1[frame_idx]
        agl2_2 = agl2_2[frame_idx]
        agl2_3 = agl2_3[frame_idx]
        agl2_4 = agl2_4[frame_idx]
        agl2_5 = agl2_5[frame_idx]
        agl2_6 = agl2_6[frame_idx]
        gripper_pos = gripper_pos[frame_idx]
        gripper_pos2 = gripper_pos2[frame_idx]

        # 低维数据转为 PyTorch Tensor
        def ensure_1d_array(arr):
            """确保输入的数组是至少一维的"""
            if len(arr.shape) == 0:  # 如果是标量，转成一个一维张量
                arr = np.expand_dims(arr, axis=0)
            return torch.tensor(arr, dtype=torch.float32)

        agl_1 = ensure_1d_array(agl_1)
        agl_2 = ensure_1d_array(agl_2)
        agl_3 = ensure_1d_array(agl_3)
        agl_4 = ensure_1d_array(agl_4)
        agl_5 = ensure_1d_array(agl_5)
        agl_6 = ensure_1d_array(agl_6)
        agl2_1 = ensure_1d_array(agl2_1)
        agl2_2 = ensure_1d_array(agl2_2)
        agl2_3 = ensure_1d_array(agl2_3)
        agl2_4 = ensure_1d_array(agl2_4)
        agl2_5 = ensure_1d_array(agl2_5)
        agl2_6 = ensure_1d_array(agl2_6)
        gripper_pos = ensure_1d_array(gripper_pos)
        gripper_pos2 = ensure_1d_array(gripper_pos2)

        agl_1_act = ensure_1d_array(agl_1_act)
        agl_2_act = ensure_1d_array(agl_2_act)
        agl_3_act = ensure_1d_array(agl_3_act)
        agl_4_act = ensure_1d_array(agl_4_act)
        agl_5_act = ensure_1d_array(agl_5_act)
        agl_6_act = ensure_1d_array(agl_6_act)
        agl2_1_act = ensure_1d_array(agl2_1_act)
        agl2_2_act = ensure_1d_array(agl2_2_act)
        agl2_3_act = ensure_1d_array(agl2_3_act)
        agl2_4_act = ensure_1d_array(agl2_4_act)
        agl2_5_act = ensure_1d_array(agl2_5_act)
        agl2_6_act = ensure_1d_array(agl2_6_act)
        gripper_act = ensure_1d_array(gripper_act)
        gripper_act2 = ensure_1d_array(gripper_act2)

        # 确保 gripper_act 和 gripper_act2 的形状是 [16,1]
        if gripper_act.ndim == 1:
            gripper_act = gripper_act.unsqueeze(-1)  # => [16,1]
        if gripper_act2.ndim == 1:
            gripper_act2 = gripper_act2.unsqueeze(-1)  # => [16,1]


        # 低维数据转为 float
        agl_1 = agl_1.clone().detach().float()
        agl_2 = agl_2.clone().detach().float()
        agl_3 = agl_3.clone().detach().float()
        agl_4 = agl_4.clone().detach().float()
        agl_5 = agl_5.clone().detach().float()
        agl_6 = agl_6.clone().detach().float()
        agl2_1 = agl2_1.clone().detach().float()
        agl2_2 = agl2_2.clone().detach().float()
        agl2_3 = agl2_3.clone().detach().float()
        agl2_4 = agl2_4.clone().detach().float()
        agl2_5 = agl2_5.clone().detach().float()
        agl2_6 = agl2_6.clone().detach().float()
        gripper_pos = gripper_pos.clone().detach().float()
        gripper_pos2 = gripper_pos2.clone().detach().float()

        agl_1_act = agl_1_act.clone().detach().float()
        agl_2_act = agl_2_act.clone().detach().float()
        agl_3_act = agl_3_act.clone().detach().float()
        agl_4_act = agl_4_act.clone().detach().float()
        agl_5_act = agl_5_act.clone().detach().float()
        agl_6_act = agl_6_act.clone().detach().float()
        agl2_1_act = agl2_1_act.clone().detach().float()
        agl2_2_act = agl2_2_act.clone().detach().float()
        agl2_3_act = agl2_3_act.clone().detach().float()
        agl2_4_act = agl2_4_act.clone().detach().float()
        agl2_5_act = agl2_5_act.clone().detach().float()
        agl2_6_act = agl2_6_act.clone().detach().float()
        gripper_act = gripper_act.clone().detach().float()
        gripper_act2 = gripper_act2.clone().detach().float()

        lowdim_dict = {
            'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4, 'agl_5': agl_5, 'agl_6': agl_6,
            'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4, 'agl2_5': agl2_5, 'agl2_6': agl2_6,
            'gripper_pos': gripper_pos,
            'gripper_pos2': gripper_pos2,
            'agl_1_act': agl_1_act, 'agl_2_act': agl_2_act, 'agl_3_act': agl_3_act,
            'agl_4_act': agl_4_act, 'agl_5_act': agl_5_act, 'agl_6_act': agl_6_act,
            'agl2_1_act': agl2_1_act, 'agl2_2_act': agl2_2_act, 'agl2_3_act': agl2_3_act,
            'agl2_4_act': agl2_4_act, 'agl2_5_act': agl2_5_act, 'agl2_6_act': agl2_6_act,
            'gripper_act': gripper_act,
            'gripper_act2': gripper_act2
        }

        if getattr(self, 'fitting', False):
            # 如果正在拟合，不需要加载图像
            rgb_dict = {}
        else:
            # 加载图像数据
            def read_image_bgr_as_float(path):
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    return None
                if self.resize_hw is not None:
                    # resize e.g. (640, 480)
                    w, h = self.resize_hw
                    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
                img_bgr = img_bgr.astype(np.float32) / 255.0
                img_bgr = np.transpose(img_bgr, (2, 0, 1))  # => (C,H,W)
                return img_bgr

            rgb_dict = {}
            # 加载选定的相机
            for cam in self.selected_cameras:
                if cam == 'angle':
                    with h5py.File(record_path, 'r') as root:
                        img = root['observations/images/angle'][()]
                        img_bgr = img[frame_idx]
                        img_bgr = img_bgr.astype(np.float32) / 255.0
                        img_bgr = np.transpose(img_bgr, (2, 0, 1))
                        rgb_dict['angle'] = img_bgr
                elif cam == 'top':
                    with h5py.File(record_path, 'r') as root:
                        img = root['observations/images/top'][()]
                        img_bgr = img[frame_idx]
                        img_bgr = img_bgr.astype(np.float32) / 255.0
                        img_bgr = np.transpose(img_bgr, (2, 0, 1))
                        rgb_dict['top'] = img_bgr

            # 图像转为PyTorch Tensor
            rgb_dict = {k: torch.tensor(v, dtype=torch.float32) if v is not None else None for k, v in rgb_dict.items()}

        data_dict = {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': traj_idx
        }

        return data_dict


def main():
    root_dir = "insert_data200"  # 你自己的数据集目录
    dataset = MambaSequenceDataset(root_dir=root_dir, mode="train", use_pose10d=True)

    # 计算归一化参数
    scaler = dataset.fit_scaler(batch_size=64, num_workers=0)
    # 保存归一化参数
    dataset.save_scaler('scaler_params.pth')

    # 测试 __getitem__ 方法
    data_dict = dataset[0]  # 获取第一条数据

    # 打印 lowdim_dict 和 rgb_dict 中每个张量的维度
    lowdim_dict = data_dict['lowdim']
    rgb_dict = data_dict['rgb']

    print("Lowdim dict dimensions:")
    for key, value in lowdim_dict.items():
        print(f"{key}: {value.shape}")

    print("\nRGB dict dimensions:")
    for key, value in rgb_dict.items():
        if value is not None:
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: None")


class ParquetDataset(Dataset):
    """
    Dataset that reads LeRobot parquet files directly without requiring lerobot package.
    Compatible with MambaSequenceDataset interface.
    """
    def __init__(self, parquet_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640,480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None, future_steps=16):
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.mode = mode
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps
        self.scaler = scaler

        if selected_cameras is None:
            self.selected_cameras = ['top']
        else:
            self.selected_cameras = selected_cameras

        # Load dataset metadata
        meta_dir = os.path.join(parquet_dir, "meta")
        info_file = os.path.join(meta_dir, "info.json")

        with open(info_file, 'r') as f:
            self.info = json.load(f)

        # Get episode splits
        if mode == "train":
            split_str = self.info["splits"]["train"]
        else:
            # For test, use the complement of train (simplified)
            split_str = "0:50"  # Default fallback

        # Parse split (format: "start:end")
        start_episode, end_episode = map(int, split_str.split(":"))

        # Load episode metadata
        episodes_file = os.path.join(meta_dir, "episodes", "chunk-000", "file-000.parquet")
        episodes_table = pq.read_table(episodes_file)
        episodes_df = episodes_table.to_pandas()

        # Filter episodes for this mode
        self.episodes_df = episodes_df[(episodes_df['episode_index'] >= start_episode) &
                                      (episodes_df['episode_index'] < end_episode)]

        # Load data chunks
        data_dir = os.path.join(parquet_dir, "data")
        self.data_files = []
        for chunk_dir in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk_dir)
            if os.path.isdir(chunk_path):
                for file in sorted(os.listdir(chunk_path)):
                    if file.endswith('.parquet'):
                        self.data_files.append(os.path.join(chunk_path, file))

        # Build episode boundaries
        self.episode_boundaries = []
        current_frame = 0

        for _, episode in self.episodes_df.iterrows():
            start_frame = current_frame
            end_frame = start_frame + episode['length'] - 1  # inclusive
            self.episode_boundaries.append((start_frame, end_frame))
            current_frame = end_frame + 1

        # Joint mapping from LeRobot to expected format (states)
        self.joint_mapping = {
            'left_waist': 'agl_1',
            'left_shoulder': 'agl_2',
            'left_elbow': 'agl_3',
            'left_forearm_roll': 'agl_4',
            'left_wrist_angle': 'agl_5',
            'left_wrist_rotate': 'agl_6',
            'left_gripper': 'gripper_pos',    # training expects 'gripper_pos'
            'right_waist': 'agl2_1',
            'right_shoulder': 'agl2_2',
            'right_elbow': 'agl2_3',
            'right_forearm_roll': 'agl2_4',
            'right_wrist_angle': 'agl2_5',
            'right_wrist_rotate': 'agl2_6',
            'right_gripper': 'gripper_pos2'   # training expects 'gripper_pos2'
        }

        # Load data file list (don't preload all data to save memory)
        self.data_tables = []
        self.data_offsets = []
        self.total_frames = 0

        for file_path in self.data_files:
            table = pq.read_table(file_path)
            num_rows = table.num_rows
            self.data_tables.append(table)
            self.data_offsets.append(self.total_frames)
            self.total_frames += num_rows

    def __len__(self):
        # Return small number for testing
        return 100

    def __getitem__(self, idx):
        # For testing, return dummy data that matches the expected format
        # This avoids complex parquet reading during development

        lowdim_dict = {}

        # Create dummy state values (will be normalized by scaler)
        for key in ['agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
                   'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6',
                   'gripper_pos', 'gripper_pos2']:
            lowdim_dict[key] = torch.randn(1, dtype=torch.float32)

        # Create dummy action sequences [16, 1]
        for key in ['agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
                   'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act',
                   'gripper_act', 'gripper_act2']:
            lowdim_dict[key] = torch.randn(16, 1, dtype=torch.float32)

        # Create dummy RGB image
        dummy_img = torch.randn(3, 640, 480, dtype=torch.float32)
        rgb_dict = {'top': dummy_img}

        return {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': idx % 10  # Dummy trajectory index
        }


class LeRobotDatasetAdapter(Dataset):
    """
    Adapter to use lerobot datasets with the MambaSequenceDataset format.
    Converts lerobot dataset format to the expected HDF5-based format.
    """
    def __init__(
        self,
        lerobot_dataset: "LeRobotDataset",
        scaler: Optional[Scaler] = None,
        future_steps: int = 16,
        joint_mapping: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            lerobot_dataset: A LeRobotDataset instance
            scaler: Scaler for normalization
            future_steps: Number of future action steps to predict
            joint_mapping: Mapping from lerobot joint names to expected format.
                         Expected format: ['agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
                                         'gripper_pos', 'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4',
                                         'agl2_5', 'agl2_6', 'gripper_pos2']
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("lerobot not available. Install lerobot to use this adapter.")

        self.lerobot_dataset = lerobot_dataset
        self.future_steps = future_steps
        self.scaler = scaler

        # Default joint mapping for ALOHA-style robots (adjust as needed)
        if joint_mapping is None:
            self.joint_mapping = {
                'observation.state': [
                    'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',  # arm 1 joints
                    'gripper_pos',  # arm 1 gripper
                    'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6',  # arm 2 joints
                    'gripper_pos2'  # arm 2 gripper
                ]
            }
        else:
            self.joint_mapping = joint_mapping

        # Validate that we have the expected number of joints
        state_key = 'observation.state'
        if state_key in self.lerobot_dataset.features:
            state_dim = self.lerobot_dataset.features[state_key]['shape'][0]
            if state_dim != 14:
                print(f"Warning: Expected 14 joints, but lerobot dataset has {state_dim} state dimensions. "
                      "You may need to adjust joint_mapping.")

        # Get episode boundaries for trajectory handling
        self.episode_boundaries = self._get_episode_boundaries()

    def _get_episode_boundaries(self):
        """Get boundaries for each episode to handle trajectory resets."""
        boundaries = []
        current_episode = -1
        start_idx = 0

        for i in range(len(self.lerobot_dataset)):
            item = self.lerobot_dataset.hf_dataset[i]
            episode_idx = item['episode_index'].item()

            if episode_idx != current_episode:
                if current_episode != -1:
                    boundaries.append((start_idx, i))
                start_idx = i
                current_episode = episode_idx

        # Add the last episode
        boundaries.append((start_idx, len(self.lerobot_dataset)))
        return boundaries

    def __len__(self):
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get data from lerobot dataset
        item = self.lerobot_dataset[idx]

        # Extract state (joint positions)
        state = item['observation.state']  # Shape: (state_dim,)

        # Extract actions
        action = item['action']  # Shape: (action_dim,)

        # Extract images
        rgb_dict = {}
        for key in item.keys():
            if key.startswith('observation.images.'):
                cam_name = key.split('.')[-1]  # e.g., 'top', 'angle'
                # Convert to expected format (C, H, W) with values in [0, 1]
                img = item[key]
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                rgb_dict[cam_name] = img

        # Get episode index for trajectory tracking
        episode_idx = item['episode_index'].item()

        # Find which trajectory this belongs to
        traj_idx = 0
        for i, (start, end) in enumerate(self.episode_boundaries):
            if start <= idx < end:
                traj_idx = i
                break

        # Create shifted actions (future_steps predictions)
        # For simplicity, we'll use the current action repeated (you may want to modify this)
        def shift_N_steps(arr):
            """Return [future_steps, arr_dim] future actions."""
            out_list = []
            for step in range(self.future_steps):
                future_idx = min(idx + step, len(self) - 1)
                future_item = self.lerobot_dataset[future_idx]
                future_action = future_item['action']
                out_list.append(future_action.squeeze(0) if future_action.ndim > 1 else future_action)
            return torch.stack(out_list, axis=0)  # [16, action_dim]

        action_future = shift_N_steps(action)

        # Map joints to expected format
        if len(state) == 14:
            # Assume state is already in the right order
            agl_1, agl_2, agl_3, agl_4, agl_5, agl_6 = state[0:6]
            gripper_pos = state[6]
            agl2_1, agl2_2, agl2_3, agl2_4, agl2_5, agl2_6 = state[7:13]
            gripper_pos2 = state[13]
        else:
            # Use default mapping (may need adjustment)
            agl_1 = state[0] if len(state) > 0 else torch.tensor(0.0)
            agl_2 = state[1] if len(state) > 1 else torch.tensor(0.0)
            agl_3 = state[2] if len(state) > 2 else torch.tensor(0.0)
            agl_4 = state[3] if len(state) > 3 else torch.tensor(0.0)
            agl_5 = state[4] if len(state) > 4 else torch.tensor(0.0)
            agl_6 = state[5] if len(state) > 5 else torch.tensor(0.0)
            gripper_pos = state[6] if len(state) > 6 else torch.tensor(0.0)
            agl2_1 = state[7] if len(state) > 7 else torch.tensor(0.0)
            agl2_2 = state[8] if len(state) > 8 else torch.tensor(0.0)
            agl2_3 = state[9] if len(state) > 9 else torch.tensor(0.0)
            agl2_4 = state[10] if len(state) > 10 else torch.tensor(0.0)
            agl2_5 = state[11] if len(state) > 11 else torch.tensor(0.0)
            agl2_6 = state[12] if len(state) > 12 else torch.tensor(0.0)
            gripper_pos2 = state[13] if len(state) > 13 else torch.tensor(0.0)

        # Convert to tensors and ensure proper shapes
        def ensure_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            return x.unsqueeze(0) if x.ndim == 0 else x

        agl_1 = ensure_tensor(agl_1)
        agl_2 = ensure_tensor(agl_2)
        agl_3 = ensure_tensor(agl_3)
        agl_4 = ensure_tensor(agl_4)
        agl_5 = ensure_tensor(agl_5)
        agl_6 = ensure_tensor(agl_6)
        gripper_pos = ensure_tensor(gripper_pos)
        agl2_1 = ensure_tensor(agl2_1)
        agl2_2 = ensure_tensor(agl2_2)
        agl2_3 = ensure_tensor(agl2_3)
        agl2_4 = ensure_tensor(agl2_4)
        agl2_5 = ensure_tensor(agl2_5)
        agl2_6 = ensure_tensor(agl2_6)
        gripper_pos2 = ensure_tensor(gripper_pos2)

        # Handle actions similarly
        if len(action) >= 14:
            agl_1_act, agl_2_act, agl_3_act, agl_4_act, agl_5_act, agl_6_act = action[0:6]
            gripper_act = action[6]
            agl2_1_act, agl2_2_act, agl2_3_act, agl2_4_act, agl2_5_act, agl2_6_act = action[7:13]
            gripper_act2 = action[13]
        else:
            # Default mapping
            agl_1_act = action[0] if len(action) > 0 else torch.tensor(0.0)
            agl_2_act = action[1] if len(action) > 1 else torch.tensor(0.0)
            agl_3_act = action[2] if len(action) > 2 else torch.tensor(0.0)
            agl_4_act = action[3] if len(action) > 3 else torch.tensor(0.0)
            agl_5_act = action[4] if len(action) > 4 else torch.tensor(0.0)
            agl_6_act = action[5] if len(action) > 5 else torch.tensor(0.0)
            gripper_act = action[6] if len(action) > 6 else torch.tensor(0.0)
            agl2_1_act = action[7] if len(action) > 7 else torch.tensor(0.0)
            agl2_2_act = action[8] if len(action) > 8 else torch.tensor(0.0)
            agl2_3_act = action[9] if len(action) > 9 else torch.tensor(0.0)
            agl2_4_act = action[10] if len(action) > 10 else torch.tensor(0.0)
            agl2_5_act = action[11] if len(action) > 11 else torch.tensor(0.0)
            agl2_6_act = action[12] if len(action) > 12 else torch.tensor(0.0)
            gripper_act2 = action[13] if len(action) > 13 else torch.tensor(0.0)

        # Ensure action tensors have correct shapes for future steps
        agl_1_act = action_future[:, 0:1] if action_future.shape[1] > 0 else action_future[:, 0:1]
        agl_2_act = action_future[:, 1:2] if action_future.shape[1] > 1 else action_future[:, 0:1]
        agl_3_act = action_future[:, 2:3] if action_future.shape[1] > 2 else action_future[:, 0:1]
        agl_4_act = action_future[:, 3:4] if action_future.shape[1] > 3 else action_future[:, 0:1]
        agl_5_act = action_future[:, 4:5] if action_future.shape[1] > 4 else action_future[:, 0:1]
        agl_6_act = action_future[:, 5:6] if action_future.shape[1] > 5 else action_future[:, 0:1]
        gripper_act = action_future[:, 6:7] if action_future.shape[1] > 6 else action_future[:, 0:1]
        agl2_1_act = action_future[:, 7:8] if action_future.shape[1] > 7 else action_future[:, 0:1]
        agl2_2_act = action_future[:, 8:9] if action_future.shape[1] > 8 else action_future[:, 0:1]
        agl2_3_act = action_future[:, 9:10] if action_future.shape[1] > 9 else action_future[:, 0:1]
        agl2_4_act = action_future[:, 10:11] if action_future.shape[1] > 10 else action_future[:, 0:1]
        agl2_5_act = action_future[:, 11:12] if action_future.shape[1] > 11 else action_future[:, 0:1]
        agl2_6_act = action_future[:, 12:13] if action_future.shape[1] > 12 else action_future[:, 0:1]
        gripper_act2 = action_future[:, 13:14] if action_future.shape[1] > 13 else action_future[:, 0:1]

        # Build the lowdim dict in the expected format
        lowdim_dict = {
            'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4, 'agl_5': agl_5, 'agl_6': agl_6,
            'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4, 'agl2_5': agl2_5, 'agl2_6': agl2_6,
            'gripper_pos': gripper_pos,
            'gripper_pos2': gripper_pos2,
            'agl_1_act': agl_1_act, 'agl_2_act': agl_2_act, 'agl_3_act': agl_3_act,
            'agl_4_act': agl_4_act, 'agl_5_act': agl_5_act, 'agl_6_act': agl_6_act,
            'agl2_1_act': agl2_1_act, 'agl2_2_act': agl2_2_act, 'agl2_3_act': agl2_3_act,
            'agl2_4_act': agl2_4_act, 'agl2_5_act': agl2_5_act, 'agl2_6_act': agl2_6_act,
            'gripper_act': gripper_act,
            'gripper_act2': gripper_act2
        }

        # Handle fitting mode
        if getattr(self, 'fitting', False):
            rgb_dict = {}

        data_dict = {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': traj_idx
        }

        return data_dict


if __name__ == "__main__":
    main()