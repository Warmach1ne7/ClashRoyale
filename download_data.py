from huggingface_hub import hf_hub_download, snapshot_download
import joblib

REPO_ID = "chrisrca/clash-royale-tv-replays"

snapshot_download(
    repo_id = REPO_ID,
    repo_type = "dataset",
    allow_patterns = "arena_02/2ceed9bf-ff35-4787-9304-e72ca0b1a9b3/*",
    local_dir = "E:\Projects\CS541\ClashRoyale\data"
)