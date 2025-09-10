from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# carrega o modelo pre treinado
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

while True:
    prompt = input("digite o prompt para auto-complete: ")

    if prompt.lower() == exit:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n -- codigo gerado --")
    print(generated_code)