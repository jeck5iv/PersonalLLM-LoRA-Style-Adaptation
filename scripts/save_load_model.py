import datetime
from peft import PeftModel
from utils.io import load_model, load_tokenizer


def generate_model_name(model_cfg, lora_cfg):
    base = model_cfg["name"].replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{model_cfg.get('quant_name', 'fp16')}_lora{lora_cfg.get('r', 'none')}_{timestamp}"


def save_model(model, tokenizer, model_cfg, lora_cfg, paths, train_cfg):
    name = generate_model_name(model_cfg, lora_cfg)
    model_cfg["full_name"] = name
    target = paths.model_weights_dir / name
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return name


def load_model_adapted_with_lora(model_cfg, paths):
    adapter_path = paths.model_weights_dir / model_cfg["full_name"]
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter is missing: {adapter_path}")
    base_model, _ = load_model(paths.original_model, model_cfg)
    tokenizer = load_tokenizer(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    return PeftModel.from_pretrained(base_model, adapter_path), tokenizer
