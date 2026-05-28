"""Demo: translate English -> Tok Pisin with NLLB-200-3.3B.

Run on a GPU node:
    uv run python scripts/translate_tok_pisin.py
"""

from tokpisin_demo import load_env

load_env()  # must run before importing torch/transformers (HF_HOME read at import)

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "facebook/nllb-200-3.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "tpi_Latn"
TEXT = "We have become one of the most sought-after teams."


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    inputs = tokenizer(TEXT, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
        max_new_tokens=128,
        num_beams=5,
    )
    translation = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    print(f"[{SRC_LANG}] {TEXT}")
    print(f"[{TGT_LANG}] {translation}")


if __name__ == "__main__":
    main()
