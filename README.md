# tokpisin-models

Demo inference scripts for **Tok Pisin** (`tpi_Latn`):

- **Translation** (text to/from English) with [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) via HF Transformers.
- **Transcription** (speech to text) with [`facebook/omniASR-CTC-3B`](https://huggingface.co/facebook/omniASR-CTC-3B) via Meta's [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr).

developed/tested on an A100 80GB GPU

## Demo scripts

```bash
# Translate English -> Tok Pisin with NLLB-200-3.3B
uv run python scripts/translate_tok_pisin.py

# Transcribe Tok Pisin speech (tok_pisin.mp3) with OmniASR-CTC-3B
uv run python scripts/transcribe_tok_pisin.py
```

## Requirements

- A GPU node with an NVIDIA driver supporting CUDA 12.x (this repo targets `cu126`; verified on driver 12.4 via CUDA minor-version compatibility).
- Python **3.12** (max supported by `omnilingual-asr` 0.2.0) and [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv python install 3.12
uv sync
```

### GPU stack

The torch / fairseq2 / CUDA versions are pinned together in `pyproject.toml` and
**must stay in sync**:

| Component   | Version        | Source                                                            |
|-------------|----------------|-------------------------------------------------------------------|
| Python      | 3.12           | `requires-python = "==3.12.*"`                                    |
| torch       | 2.8.0 (+cu126) | `https://download.pytorch.org/whl/cu126`                          |
| fairseq2    | 0.6.0          | PyPI                                                              |
| fairseq2n   | 0.6 (+cu126)   | `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126`          |


