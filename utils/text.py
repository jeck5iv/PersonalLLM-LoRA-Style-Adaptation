import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def training_labels(input_ids, attention_mask):
    return [
        [token if visible else -100 for token, visible in zip(ids, mask)]
        for ids, mask in zip(input_ids, attention_mask)
    ]


def continuation_ids(sequence, input_length):
    return sequence[input_length:]
