from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("aposadasn/iss_docking_images_parquet", tag="v2.1", repo_type="dataset")
