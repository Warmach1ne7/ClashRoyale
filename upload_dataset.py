from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="training.parquet",  # Local file path
    path_in_repo="training.parquet",     # Destination path in repo
    repo_id="chrisrca/clash-royale-tv-replays",
    repo_type="dataset"
)