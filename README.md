# tokpisin-models

Demo inference scripts for **Tok Pisin** (`tpi_Latn`):

- **Translation** (text to/from English) with [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) via HF Transformers.
- **Transcription** (speech to text) with [`facebook/omniASR-CTC-3B`](https://huggingface.co/facebook/omniASR-CTC-3B) via Meta's [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr).

Built for a CUDA GPU node on the HPC (developed/tested on an A100 80GB).

## Requirements

- A GPU node with an NVIDIA driver supporting CUDA 12.x (this repo targets `cu126`; verified on driver 12.4 via CUDA minor-version compatibility).
- Python **3.12** (max supported by `omnilingual-asr` 0.2.0) and [uv](https://docs.astral.sh/uv/).

## Setup

```bash
# 1. Python 3.12 (uv will fetch a standalone build if needed)
uv python install 3.12

# 2. Credentials & cache paths
cp .env.example .env   # then edit HF_TOKEN / HF_HOME / VLLM_CACHE_ROOT

# 3. Install the locked GPU stack (run on a GPU node)
uv sync
```

`uv sync` installs the project (including the `tokpisin_demo` helper package) plus
the pinned GPU stack described below.

### GPU stack

The torch / fairseq2 / CUDA versions are pinned together in `pyproject.toml` and
**must stay in sync**:

| Component   | Version        | Source                                                            |
|-------------|----------------|-------------------------------------------------------------------|
| Python      | 3.12           | `requires-python = "==3.12.*"`                                    |
| torch       | 2.8.0 (+cu126) | `https://download.pytorch.org/whl/cu126`                          |
| fairseq2    | 0.6.0          | PyPI                                                              |
| fairseq2n   | 0.6 (+cu126)   | `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126`          |

`fairseq2n` is a native (C++/ABI) library that is built against an exact torch
version and is served only from Meta's variant index, so it cannot float. To
move to a different torch/CUDA build, change `torch`'s version and both index
URLs in `pyproject.toml` together (see the fairseq2 install matrix).

## The `load_env` convention

**Every Python script must call `load_env()` first**, before importing `torch`,
`transformers`, or `omnilingual_asr`. Those libraries read `HF_HOME`, `HF_TOKEN`,
and `VLLM_CACHE_ROOT` at import time, so the values from `.env` have to be in the
environment before the import happens. Skipping this makes models download to the
home directory instead of the shared HPC cache.

```python
from tokpisin_demo import load_env

load_env()  # loads .env into os.environ

import torch                       # safe: HF_HOME etc. are now set
from transformers import AutoModelForSeq2SeqLM
```

## Language codes

Both models use FLORES-style `{lang}_{script}` codes:

- Tok Pisin: `tpi_Latn`
- English: `eng_Latn`

## Demo scripts

```bash
# Translate English -> Tok Pisin with NLLB-200-3.3B
uv run python scripts/translate_tok_pisin.py
```

Always launch with `uv run` so the project venv (and `tokpisin_demo`) is on the
path. On a Slurm cluster, run these inside a GPU allocation (e.g. `srun`/`sbatch`),
not on the login node.

## Layout

```
pyproject.toml            # uv project + pinned GPU stack
.python-version           # 3.12
.env / .env.example       # HF + API settings (load_env reads .env)
src/tokpisin_demo/        # shared helper package
  __init__.py             # load_env()
scripts/
  translate_tok_pisin.py  # English -> Tok Pisin demo (NLLB-200-3.3B)
```
