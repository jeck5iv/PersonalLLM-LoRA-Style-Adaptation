import json


def load_config(config_path):
    with open(config_path, encoding="utf-8") as stream:
        return json.load(stream)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def load_model(model_dir, model_config=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = model_config or {}
    tokenizer = load_tokenizer(model_dir)
    kwargs = {"device_map": "auto"}
    if cfg.get("quantized", False):
        if not torch.cuda.is_available():
            raise RuntimeError("The 4-bit training path requires a supported CUDA environment")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16")),
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    return model, tokenizer
