from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# carrega o modelo pre treinado
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = "def soma(a, b):"  # exemplo de inicio de código
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_length=50,       # max saída | 1024tokens no gpt2
    do_sample=True,      # True = respostas despadronizadas
    temperature=1.1,     # criatividade do modelo
    top_k=50,            # top_k para limitar diversidade
    top_p=0.95,          # top_p para limitar diversidade
    pad_token_id=tokenizer.eos_token_id  # evita erro de padding
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("codigo gerado")
print(generated_code)