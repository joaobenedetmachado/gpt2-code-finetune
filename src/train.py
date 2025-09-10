import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

OUTPUT_DIR = "fine_tuned_code"   # pasta onde salvar o modelo
DATA_PATH = "data/dataset.jsonl" # dataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def load_dataset(tokenizer, file_path, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

train_dataset = load_dataset(tokenizer, DATA_PATH)