# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""
import io
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler


class Policy(object):
    """
    Custom ACTPolicy class that inherits from the ACTPolicy provided by the lerobot library.
    This class is used to load a pretrained policy and run inference on it.
    """
    def __init__(self):

        model_id = "arclabmit/iss_docking_act_model"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model directly from Hugging Face
        self.policy = ACTPolicy.from_pretrained(model_id)
        cfg = self.policy.config
        self.policy.config.temporal_ensemble_coeff = 0.01  # or your preferred value
        self.policy.temporal_ensembler = ACTTemporalEnsembler(0.01, self.policy.config.chunk_size)
        # print(f"Available attributes: {dir(cfg)}")
        print("=== ACT CONFIG ===")
        print("chunk_size:", cfg.chunk_size)
        print("n_action_steps:", cfg.n_action_steps)
        print("temporal_ensemble_coeff:", getattr(cfg, "temporal_ensemble_coeff", None))
        print("input_features:", cfg.input_features)
        print("output_features:", cfg.output_features)
        if self.policy.config.temporal_ensemble_coeff is not None:
            print(f"Temporal ensembling is ENABLED with coeff={self.policy.config.temporal_ensemble_coeff}")
        else:
            print("Temporal ensembling is DISABLED")


        self.policy.to(self.device)  # Move to appropriate device
        self.policy.eval()  # Set to evaluation mode
        self.policy.reset()



    def preprocess_image(self, img: Any) -> torch.Tensor:
        """
        Convert *img* to a PyTorch tensor with shape (C, H, W), ``dtype=torch.float32`` and
        values in \[0, 1].

        Supported input types
        ---------------------
        1. ``torch.Tensor`` (HWC **or** CHW, uint8/float32/float64)
        2. ``np.ndarray``   (HWC **or** CHW, uint8/float32/float64)
        3. ``PIL.Image``
        4. raw PNG/JPEG bytes

        The function makes **no** assumption about channel order (HWC/CHW) - it
        automatically detects and rearranges the axes. Images with an alpha channel
        are converted to RGB by dropping the alpha component. Grayscale images are
        expanded to a single channel.
        """
        # ------------------------------------------------------------------
        # Helper: convert a *numeric* array-like object to (C, H, W) float32
        # ------------------------------------------------------------------
        def _to_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
            """Handle ndarray/tensor separately but keep logic shared."""
            is_numpy = isinstance(arr, np.ndarray)
            tensor = torch.from_numpy(arr) if is_numpy else arr

            # If uint8 -> float32 in [0,1]
            tensor = tensor.float().div_(255.0) if tensor.dtype == torch.uint8 else tensor.to(torch.float32)

            # Detect layout / channels
            if tensor.ndim == 2:  # (H, W) grayscale â†’ add channel dimension
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                c_first = tensor.shape[0] in (1, 3, 4)
                c_last  = tensor.shape[-1] in (1, 3, 4)
                if c_first and not c_last:
                    # Already CHW
                    pass
                elif c_last and not c_first:
                    # HWC â†’ CHW
                    tensor = tensor.permute(2, 0, 1)
                else:
                    raise ValueError(
                        "Ambiguous image shape - cannot determine channel position: "
                        f"{tuple(tensor.shape)}"
                    )
            else:
                raise ValueError(f"Unsupported number of dimensions: {tensor.ndim} (expected 2 or 3)")

            # Remove alpha if present (convert RGBA â†’ RGB)
            if tensor.shape[0] == 4:
                tensor = tensor[:3]

            return tensor.contiguous()

        # ------------------------------------------------------------------
        # Dispatch on *img* type
        # ------------------------------------------------------------------
        if isinstance(img, torch.Tensor):
            return _to_tensor(img)

        if isinstance(img, np.ndarray):
            return _to_tensor(img)

        if isinstance(img, Image.Image):
            return _to_tensor(np.array(img.convert("RGB")))

        if isinstance(img, (bytes, bytearray)):
            with Image.open(io.BytesIO(img)) as pil_img:
                return _to_tensor(np.array(pil_img.convert("RGB")))

        raise TypeError(f"Unsupported image type: {type(img)}")



    def run_inference(self, obs):

        # prepare observation for the policy running in Pytorch
        state = torch.from_numpy(obs["state"])
        image = self.preprocess_image(obs["image"])

        state = state.to(torch.float32)

        # Send data tensors from CPU to GPU
        state = state.to(self.device, non_blocking=True)
        image = image.to(self.device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image.cam": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = self.policy.select_action(observation)

        action = action.squeeze(0).to("cpu").numpy()

        return action


# if __name__ == "__main__":
#     from pathlib import Path

#     import numpy as np
#     import pandas as pd
#     from PIL import Image

#     # Path setup
#     dataset_root = Path("datasets/iss_docking_images")
#     # Load metadata (optional, for future use)
#     # with open(dataset_root / "features.json", "r") as f:
#     #     metadata = json.load(f)

#     # Load the first episode parquet file (adjust if needed)
#     chunk_idx = 0
#     episode_idx = 0
#     parquet_path = dataset_root / f"data/chunk-{chunk_idx:03d}/episode_{episode_idx:06d}.parquet"
#     df = pd.read_parquet(parquet_path)

#     # Create policy
#     policy = Policy()
#     policy.policy.reset()

#     # Test with 3 observations
#     for i in range(47, 50):
#         policy.policy.reset()
#         row = df.iloc[i]
#         # State: ensure it's np.float32
#         state = np.array(row["observation.state"]).astype(np.float32)
#         # Frame index
#         frame_index = row["frame_index"] if isinstance(row["frame_index"], (int, np.integer)) else row["frame_index"][0]
#         # Image path
#         image_path = dataset_root / f"images/chunk-{chunk_idx:03d}/observation.image.cam/episode_{episode_idx:06d}/frame_{frame_index:06d}.png"
#         image = Image.open(image_path)
#         # Prepare observation
#         obs = {"state": state, "image": image}
#         # Run inference
#         pred_action = policy.run_inference(obs)
#         # Print results
#         print(f"\nObservation {i+1}:")
#         print(f"Frame index: {frame_index}")
#         print(f"Predicted action: {pred_action}")
#         print(f"Ground truth action: {row['action']}")