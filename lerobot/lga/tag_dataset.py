from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("arclabmit/koch_cubebin_dataset", tag="v2.1", repo_type="dataset")
