# model-test.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 경로를 절대 경로로 변경
model_path = os.path.expanduser("~/models/hyperclovax-vision-3b")
# 또는 직접 절대 경로 지정
# model_path = "/mnt/ssd/hyperclovax/models/hyperclovax-vision-3b"

# 경로 확인
print(f"모델 경로: {model_path}")
print(f"경로 존재 여부: {os.path.exists(model_path)}")

# 경로의 파일들 확인
if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"디렉토리 내 파일들: {files}")

# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
