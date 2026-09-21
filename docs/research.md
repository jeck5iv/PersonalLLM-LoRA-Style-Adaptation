# Research notes

## Question

Can parameter-efficient adaptation capture an author's writing style from a limited collection of posts?

I separated the base model from the adaptation settings so I could compare Mistral and Qwen checkpoints, LoRA ranks and training schedules without maintaining a separate pipeline for each run. FP4 and NF4 configurations provide a lower-memory option. Their presence is not evidence that every combination was evaluated successfully.

## Data and validation

The original script stripped punctuation, dropped short messages and randomly assigned posts to train, validation and test. Punctuation carries style, repeated posts can cross the split, and a random split does not test transfer to later writing.

The cleanup preserves punctuation, removes exact cleaned duplicates before splitting and orders posts by timestamp. Posts sharing a timestamp stay together. The first 80% of unique timestamps are used for training, the next 10% for validation and the rest for testing. Post counts can differ from these proportions.

Near-duplicate content and shared discussion threads can remain. A stronger study should group related posts and inspect overlap before fixing a holdout. Removing URLs and handles is not full anonymization. No original posts are included.

## Model

The base language model stays frozen while LoRA adapters learn a low-rank update to attention projections. Configuration controls rank, scale and dropout. Quantization reduces the memory needed for base weights, with hardware and numerical tradeoffs.

The target is the next token in each sequence. Padding positions must not contribute to the loss. The original implementation did not mask these positions; this version does. Original measurements, if recovered, must be kept separate from reruns of this version.

## Evaluation

The tokenizer determines an 80/20 prefix-continuation split. The model sees the prefix and generates at most the continuation length. Generated token IDs are sliced after the input length before decoding. Slicing decoded text by character count is not equivalent.

BLEU and ROUGE-L measure overlap with one recorded continuation. BERTScore measures semantic similarity in the configured language. Token-weighted perplexity measures probability assigned to held-out text. Repetition is the fraction of distinct whitespace-separated tokens that appear more than once.

A plausible continuation may differ from the recorded one. A controlled style study should compare the untouched model and adapter on frozen prompts, use multiple seeds and add blinded author-style judgments. Memorization checks should accompany qualitative examples.

## Evidence

The repository establishes the implementation and experiment structure. There are no committed run logs or checkpoints sufficient to reconstruct a numerical comparison. No numerical improvement is claimed.

CPU tests exercise cleaning, chronological separation, padding masks and continuation slicing. GPU training, generation quality and memory use have not been verified in this cleanup.

## Release changes

Original source remains in Git history. This release resolves the case-sensitive configuration collision, removes credentials from the current configuration, masks padding labels, corrects continuation slicing and documents the unsupported API path. A smaller dependency list replaces the environment dump, which is retained for reference.
