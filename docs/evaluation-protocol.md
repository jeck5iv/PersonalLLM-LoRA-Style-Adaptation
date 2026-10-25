# Evaluation protocol

This is the specification for a future controlled comparison, not a record of completed runs. The current public snapshot establishes the pipeline and CPU checks; it does not establish an improvement in author style.

## Research question and comparison

Compare a frozen base model with its LoRA-adapted version on the same held-out continuations. Hold the tokenizer, checkpoint revision, prompt construction, context budget and decoding policy fixed. Use the existing greedy decoding path for the primary comparison so that generation randomness is not an extra factor.

Choose adapter rank, learning rate and stopping rules on the validation partition. After these choices are fixed, run at least three training seeds and report per-seed results as well as the aggregate. This is a proposed evaluation design; those runs have not yet been completed.

## Data boundaries

1. Use text with permission to process it and record the dataset version and cleaning configuration.
2. Remove exact cleaned duplicates before splitting. Inspect near-duplicates and related discussion threads before fixing the final holdout.
3. Split chronologically and keep equal timestamps together. Record post counts and date ranges for each partition.
4. Tokenise the held-out post once, use its first 80% of tokens as the prompt, and retain the remaining tokens as the target continuation.
5. Decode generated tokens only after the prompt boundary. A decoded character count is not a substitute for a token boundary.

## Measures and interpretation

| Measure | Intended interpretation | Limitation |
| --- | --- | --- |
| Token-weighted perplexity | Probability assigned to held-out text | Sensitive to tokenizer and token accounting; compare matched checkpoints |
| BERTScore | Semantic similarity to the recorded continuation | A valid alternative continuation may score lower |
| BLEU / ROUGE-L | Lexical overlap | Overlap is not author-style fidelity |
| Repetition diagnostic | Reuse of distinct whitespace-separated tokens | This implementation is not a repeated-n-gram rate |
| Blinded paired style judgments | Preference for the author's style on matched prompts | Requires a specified rubric, raters and uncertainty assessment |

For quantitative comparisons, retain paired per-post scores and resample whole posts or related-post groups, rather than treating tokens as independent observations. Report between-training-seed variation separately from evaluation-sample uncertainty. Do not select the best test seed.

## Artifacts required for a results claim

Store checkpoint revisions, package versions, seeds, configuration hashes, partition summaries, per-post metric records and aggregate comparisons. Keep private text and adapters outside the public repository unless publication is authorised. Add a reproducible summary generated from these records before inserting a numerical result into the README.

Qualitative examples should include both improvements and failures, selected by a rule fixed before examining model output. Check memorisation and near-verbatim continuation of training material alongside stylistic preference.
