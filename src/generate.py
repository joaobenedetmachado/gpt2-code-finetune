from transformers import AutoTokenizer, AutoModelForCausalLM

# carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# carrega o modelo pre treinado
model = AutoModelForCausalLM.from_pretrained("gpt2")
