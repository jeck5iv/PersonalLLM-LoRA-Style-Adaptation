import argparse
from pathlib import Path

from utils.io import load_config
from utils.paths import Paths


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a LoRA adapter")
    parser.add_argument("--model", default="configs/models_configs/Qwen2.5-1.5B-Instruct.json")
    parser.add_argument("--lora", default="configs/lora_configs/lora_optimal.json")
    parser.add_argument("--train", default="configs/train_configs/train_shared.json")
    parser.add_argument("--quant", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = lambda name: load_config(root / name)
    model_cfg = {**config(args.model), **config("configs/base.json")}
    if args.quant:
        quant_cfg = config(args.quant)
        # Keep the selected model when applying a quantization preset.
        for key in ("name", "base_model_id", "original_model", "quantized_model"):
            quant_cfg.pop(key, None)
        model_cfg.update(quant_cfg)
    lora_cfg, train_cfg = config(args.lora), config(args.train)
    paths = Paths(model_cfg, config("configs/telegram_config.json"))
    for data in (paths.train_data, paths.val_data, paths.test_data):
        if not data.is_file():
            parser.error(f"Missing {data}. Run python -m scripts.prepare_data first.")

    from scripts.download_model import run as download_model
    from scripts.finetune import run as finetune
    from scripts.metrics import evaluate_metrics, save_metrics_csv
    from scripts.save_load_model import save_model

    model_cfg["original_model"] = str(paths.original_model)
    paths.model_weights_dir.mkdir(parents=True, exist_ok=True)
    download_model(model_cfg)
    model, tokenizer = finetune(model_cfg, lora_cfg, train_cfg, paths)
    name = save_model(model, tokenizer, model_cfg, lora_cfg, paths, train_cfg)
    del model, tokenizer
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics = evaluate_metrics(model_cfg, paths, name, num_samples=args.num_samples)
    save_metrics_csv(metrics, model_cfg, lora_cfg, paths, with_prompt=False)
    print(metrics)


if __name__ == "__main__":
    main()
