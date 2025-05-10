# download.py
from huggingface_hub import snapshot_download
import os

# 다운로드 경로 설정
local_dir = os.path.expanduser("~/models/hyperclovax-vision-3b")
os.makedirs(local_dir, exist_ok=True)

# 모델 다운로드
snapshot_download(
    repo_id="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print("다운로드 완료!")
