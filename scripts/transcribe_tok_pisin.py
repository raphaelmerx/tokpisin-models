"""Demo: transcribe Tok Pisin speech with OmniASR-CTC-3B.

Run on a GPU node:
    uv run python scripts/transcribe_tok_pisin.py
"""

import os

from tokpisin_demo import PROJECT_ROOT, load_env

load_env()  # must run before importing omnilingual_asr (reads HF_HOME / FAIRSEQ2_CACHE_DIR at import)

# When CONDA_PREFIX is set (a conda base env is active in the shell), fairseq2n
# skips the system-linker lookup for libsndfile and only searches the conda/venv
# dir, where it isn't installed. Drop it so the system libsndfile loads. We decode
# the mp3 with soundfile below, so the old system libsndfile only needs to import.
os.environ.pop("CONDA_PREFIX", None)

import soundfile as sf
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

MODEL_CARD = "omniASR_CTC_3B"  # facebook/omniASR-CTC-3B
LANG = "tpi_Latn"
AUDIO_PATH = PROJECT_ROOT / "tok_pisin.mp3"


def main() -> None:
    # The system libsndfile (1.0.31) cannot decode mp3, so decode with soundfile
    # (ships libsndfile 1.2.x) and hand the pipeline a pre-decoded waveform; it
    # resamples to 16 kHz internally.
    waveform, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
    if waveform.ndim > 1:  # downmix stereo to mono
        waveform = waveform.mean(axis=1)

    pipeline = ASRInferencePipeline(model_card=MODEL_CARD)  # cuda + bfloat16 by default
    transcriptions = pipeline.transcribe(
        [{"waveform": waveform, "sample_rate": sample_rate}],
        lang=[LANG],  # ignored by the CTC model; kept to document intent
        batch_size=1,
    )

    print(f"[audio]    {AUDIO_PATH.name}")
    print(f"[{LANG}] {transcriptions[0]}")


if __name__ == "__main__":
    main()
