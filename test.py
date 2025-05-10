# test.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "~/models/hyperclovax-vision-3b"

# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("설치 성공!")
