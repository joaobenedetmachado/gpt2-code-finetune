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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # importante: False porque GPT-2 n é masked language model
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,       # número de epocas
    per_device_train_batch_size=2,  # batch pequeno (GPU fraca)
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),  # mixed precision se tiver GPU
)