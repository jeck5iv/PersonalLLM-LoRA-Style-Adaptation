import csv
import math
from collections import Counter
from datetime import datetime

import pandas as pd

from utils.text import continuation_ids


def token_repetition_ratio(text):
    counts = Counter(text.split())
    return sum(count > 1 for count in counts.values()) / len(counts) if counts else 0.0


def build_prompt_from_train_data(train_path, num_messages=5):
    texts = pd.read_csv(train_path)["cleaned_text"].dropna()
    samples = texts.sample(n=min(num_messages, len(texts)), random_state=42)
    return "\n\n".join(samples) + "\n\n"


def calculate_perplexity(model, tokenizer, texts):
    import torch
    loss_sum, token_count = 0.0, 0
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        count = encoded["input_ids"].shape[1] - 1
        if count <= 0:
            continue
        with torch.no_grad():
            loss = model(**encoded, labels=encoded["input_ids"]).loss.item()
        loss_sum += loss * count
        token_count += count
    return math.exp(loss_sum / token_count) if token_count else float("nan")


def evaluate_metrics(model_cfg, paths, model_name, num_samples=100, prompt=None, use_gpt_api=False):
    if use_gpt_api:
        raise NotImplementedError("The original API wrapper is not included. Use local evaluation.")
    import evaluate
    import torch
    from scripts.save_load_model import load_model_adapted_with_lora

    model, tokenizer = load_model_adapted_with_lora(model_cfg, paths)
    model.eval()
    tokenizer.truncation_side = "left"
    texts = pd.read_csv(paths.test_data)["cleaned_text"].dropna().tolist()[:num_samples]
    generations, targets = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
        if len(ids) < 5:
            continue
        cutoff = int(len(ids) * 0.8)
        prefix = tokenizer.decode(ids[:cutoff], skip_special_tokens=True)
        targets.append(tokenizer.decode(ids[cutoff:], skip_special_tokens=True))
        inputs = tokenizer((prompt or "") + prefix, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            sequence = model.generate(
                **inputs, max_new_tokens=len(ids) - cutoff, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        generated = continuation_ids(sequence, inputs["input_ids"].shape[1])
        generations.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    if not generations:
        raise ValueError("No evaluable test posts")
    bleu = evaluate.load("bleu").compute(predictions=generations, references=[[x] for x in targets])["bleu"]
    rouge = evaluate.load("rouge").compute(predictions=generations, references=targets)["rougeL"]
    bert = evaluate.load("bertscore").compute(
        predictions=generations, references=targets, lang=model_cfg.get("language", "ru"),
    )["f1"]
    return {
        "bleu": bleu, "rougeL": rouge, "bertscore": sum(bert) / len(bert),
        "perplexity": calculate_perplexity(model, tokenizer, texts),
        "repetition": sum(map(token_repetition_ratio, generations)) / len(generations),
        "n_samples": len(generations), "source": "local",
    }


def save_metrics_csv(metrics, model_cfg, lora_cfg, paths, with_prompt):
    path = paths.resolve("metrics_log.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_cfg.get("full_name", model_cfg["name"]),
        "quant_cfg_name": model_cfg.get("quant_name", "none"),
        "lora_cfg_name": lora_cfg.get("name", "none"),
        "with_prompt": with_prompt, **metrics,
    }
    exists = path.is_file()
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), []) != list(row):
                raise ValueError("Existing metrics_log.csv has a different schema. Archive it before this run.")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
