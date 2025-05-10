# simple_test.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 가상환경 내에서 실행
model_path = "~/models/hyperclovax-vision-3b"

# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 간단한 텍스트 생성 테스트
chat = [{"role": "user", "content": "안녕하세요! 오늘 날씨가 어때요?"}]

input_ids = tokenizer.apply_chat_template(
    chat, return_tensors="pt", add_generation_prompt=True
).to("cuda")

output_ids = model.generate(
    input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7
)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"AI 응답: {response}")
