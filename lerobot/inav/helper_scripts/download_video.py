from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aposadasn/lander31",
    repo_type="dataset",
    allow_patterns=["videos/chunk-000/observation.image.cam/episode_000005.mp4"],
    local_dir="/home/demo/.cache/huggingface/lerobot/aposadasn/lander31",
    ignore_patterns=[],  # make sure nothing is ignored
    force_download=True  # force the fetch
)
