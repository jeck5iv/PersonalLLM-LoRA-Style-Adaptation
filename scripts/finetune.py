import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, default_data_collator

from utils.io import load_model
from utils.text import training_labels


def run(model_cfg, lora_cfg, train_cfg, paths):
    model, tokenizer = load_model(paths.original_model, model_cfg)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        result = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
        result["labels"] = training_labels(result["input_ids"], result["attention_mask"])
        return result

    if model_cfg.get("quantized", False):
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**{
        key: value for key, value in lora_cfg.items() if key != "name"
    }))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    def dataset(path):
        texts = pd.read_csv(path)["cleaned_text"].fillna("").tolist()
        return Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(paths.resolve(train_cfg["output_dir"])),
        per_device_train_batch_size=train_cfg["batch_size"],
        num_train_epochs=train_cfg["num_train_epochs"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        fp16=train_cfg["fp16"] and torch.cuda.is_available(),
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        logging_dir=str(paths.resolve(train_cfg["logging_dir"])),
        save_total_limit=train_cfg["save_total_limit"],
        seed=42, data_seed=42, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset(paths.train_data),
        eval_dataset=dataset(paths.val_data), data_collator=default_data_collator,
    )
    trainer.train()
    return model, tokenizer
