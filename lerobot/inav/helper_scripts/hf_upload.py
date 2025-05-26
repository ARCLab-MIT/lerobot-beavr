from huggingface_hub import HfApi as api
from pathlib import Path

api.upload_folder(
    folder_path=Path("/home/demo/lerobot/datasets/iss_docking_images"),
    repo_id="aposadasn/iss_docking_images2",
    repo_type="dataset",
    path_in_repo="",  # Upload to the root of the repository
    commit_message=f"Upload dataset: {Path.cwd().name} with correct structure",
)