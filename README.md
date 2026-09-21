# Personal writing style with LoRA

I built this project to study whether a small adapter could make an open language model write more like a particular author. The input is a collection of posts used with the author's permission. The task is held-out text continuation, not question answering.

The pipeline separates the base model, quantization, adapter and training settings. It supports the Mistral and Qwen configurations in `configs/`, with optional 4-bit loading and LoRA on the attention projections.

## What is available

The repository contains data preparation, adapter training, generation and evaluation code. It does not contain personal messages, trained weights or a verified results table. I therefore do not claim a measured improvement in writing style from this public snapshot.

The evaluation uses the first 80% of a held-out post as a prompt and compares the generated continuation with the rest. BLEU, ROUGE-L and BERTScore describe text overlap or semantic similarity. They do not establish that a model has learned someone's style. Perplexity and repetition are useful diagnostics, with the same limitation.

[Research notes](docs/research.md) explain the experiment, evaluation problems and changes made while preparing this release.

## Try the data pipeline

Python 3.11 or 3.12. This part runs on a CPU and needs no credentials.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-checks.txt
python -m scripts.prepare_data --input examples/posts.csv --output /tmp/style-data --min-chars 20
python -m unittest discover -s tests
```

The example posts are synthetic. For your own data, supply a CSV with `date` and `text` columns. Preparation removes links and handles, keeps punctuation, removes exact cleaned duplicates and splits in time order. It writes only cleaned text and timestamps, not original messages or account metadata. This is basic redaction, not guaranteed anonymization.

## Train an adapter

GPU training has not been rerun as part of this cleanup. The lightweight tests do not certify the CUDA or model-download path.

```bash
pip install -r requirements.txt
python -m scripts.prepare_data --input /path/to/consented-posts.csv --output data
python main.py --model configs/models_configs/Qwen2.5-1.5B-Instruct.json
```

Use `--lora` and `--train` to select other JSON settings. Add `--quant configs/quantisation_configs/quant_nf4.json` for a 4-bit run. The base checkpoint is downloaded from Hugging Face on the first run; access and license terms depend on the selected model. Model weights can be large.

Training uses causal language-model loss and ignores padding tokens. Adapter weights go to `checkpoints/`. Evaluation uses the saved adapter and writes aggregate metrics to the ignored `metrics_log.csv`. Generation is deterministic. The language setting defaults to Russian, matching the original experiment.

`requirements-legacy.txt` preserves the original environment snapshot. It includes CUDA and notebook packages and is not the recommended installation path.

## Privacy and scope

Use only data you are allowed to process. Removing a username does not remove names, addresses or other identifying details inside a post. Review text before training, and do not publish adapters that may memorize private material.

Training and local evaluation do not send messages to an external inference API. The older `main_gpt.py` entry point is retained for provenance, but its API wrapper is absent from the public snapshot and it is not part of the supported path.

Telegram credentials and session files belong in local, ignored files. The current example configuration contains no account credentials. Removing credentials from the current tree does not remove them from Git history.

## Code

`scripts/prepare_data.py` prepares chronological splits. `scripts/finetune.py` fits the adapter. `scripts/metrics.py` evaluates held-out continuations. `tests/` checks data boundaries, token handling and configuration paths without downloading a model.
