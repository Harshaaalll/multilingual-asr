# Automatic Speech Recognition in Multilingual Languages

> Zero-shot Urdu ASR system using OpenAI Whisper with a two-stage post-processing pipeline including noise reduction and IndicBERT MLM-based error correction. Reduces Word Error Rate by 14%.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-412991?style=flat-square)](https://openai.com/research/whisper)
[![IndicBERT](https://img.shields.io/badge/IndicBERT-AI4Bharat-FF6B35?style=flat-square)](https://huggingface.co/ai4bharat/indic-bert)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

ASR (Automatic Speech Recognition) for low-resource languages like Urdu is especially challenging — limited training data, complex phonology, and code-switching between Urdu and Hindi. This project builds a two-stage pipeline: Whisper for acoustic modelling, IndicBERT for linguistic post-correction.

**Key insight: rather than trying to build one perfect ASR model, separate the acoustic problem (Whisper) from the linguistic correction problem (IndicBERT). Each model does what it does best.**

---

## Pipeline

```
Raw Audio File
      │
      ▼ ── Convert to mono (stereo → single channel)
      │
      ▼ ── Normalise loudness to -20 dBFS
      │      (standardises across different microphones)
      │
      ▼ ── Trim leading/trailing silence
      │
      ▼ ── Resample to 16kHz
      │      (Whisper training rate)
      │
      ▼ ── Spectral Subtraction (librosa + noisereduce)
      │      Estimate noise floor from silent segments
      │      Subtract from speech signal → -10dB reduction
      │
      ▼ ── OpenAI Whisper (small)
      │      language="ur" (language biasing)
      │      Zero-shot — no fine-tuning required
      │
      ▼ ── Transcribed text (raw, may have phonetic errors)
      │
      ▼ ── IndicBERT MLM Corrector
      │      For each word: mask → predict → compare
      │      If probability(original word) < 0.4:
      │        replace with highest-probability alternative
      │
      ▼
Final Transcription (WER measured)
```

---

## Audio Preprocessing

### Why each step matters

**Mono conversion:**
Whisper expects single-channel audio. Stereo audio contains two channels that may have slight phase offsets from different microphone positions. These phase differences can create frequency cancellation artifacts that confuse the acoustic model.

**Loudness normalisation (-20 dBFS):**
Audio collected from different sources — phone recordings, laptop microphones, external mics — have wildly different volume levels. Normalisation to -20 dBFS creates a consistent input that the model processes reliably. Without this, a very quiet recording might produce empty transcriptions.

**16kHz resampling:**
Whisper was trained exclusively on 16kHz audio. Feeding it 44.1kHz audio (CD quality) or 8kHz audio (telephone quality) causes the model to process frequency information it wasn't trained to interpret, degrading accuracy.

**Spectral subtraction:**
Real-world recordings contain background noise — fans, traffic, other speakers. The algorithm:
1. Identifies silent segments in the audio (where speech is absent)
2. Estimates the noise profile from those segments
3. Subtracts the noise profile from the full signal
Result: -10dB noise reduction, making speech components more prominent.

---

## IndicBERT MLM Correction

### The problem it solves
Whisper makes phonetically plausible errors — it transcribes what it "hears" acoustically but sometimes picks the wrong word. In Urdu, many words have similar phonemes and different meanings. The acoustic model alone cannot resolve ambiguity.

### How it works

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def correct_transcription(text: str, threshold: float = 0.4) -> str:
    words = text.split()
    corrected = []
    
    for i, word in enumerate(words):
        # Build context with current word masked
        masked_words = words.copy()
        masked_words[i] = tokenizer.mask_token
        masked_text = " ".join(masked_words)
        
        # Tokenise and get predictions
        inputs = tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Find the mask token position
        mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()[0][1]
        logits = outputs.logits[0, mask_idx]
        probs = torch.softmax(logits, dim=-1)
        
        # Check confidence for the original word
        orig_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(word)[0]
        )
        orig_prob = probs[orig_token_id].item()
        
        if orig_prob < threshold:
            # Replace with highest-probability token
            best_token_id = probs.argmax().item()
            corrected_word = tokenizer.decode([best_token_id])
            corrected.append(corrected_word)
        else:
            corrected.append(word)
    
    return " ".join(corrected)
```

### Why threshold = 0.4?
Below 0.4, the model has less than 40% confidence that the original transcribed word is correct in context. At this confidence level, the MLM's alternative is likely better. Values too low (0.1) over-correct and introduce errors; values too high (0.7) miss real errors. 0.4 was tuned on validation set WER.

---

## Results

| Stage | WER |
|-------|-----|
| Raw Whisper (no preprocessing) | Baseline |
| + Audio preprocessing | -6% relative |
| + IndicBERT correction (threshold=0.4) | -14% relative |

---

## Project Structure

```
multilingual-asr/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── src/
│   ├── audio/
│   │   ├── preprocessor.py     # pydub pipeline
│   │   └── noise_reducer.py    # librosa + noisereduce
│   ├── asr/
│   │   └── whisper_transcriber.py
│   ├── correction/
│   │   └── indicbert_corrector.py
│   └── pipeline.py
├── tests/
│   ├── test_preprocessor.py
│   └── test_corrector.py
├── notebooks/
│   ├── 01_audio_preprocessing_analysis.ipynb
│   └── 02_wer_evaluation.ipynb
├── run_asr.py                  # Batch CLI tool
└── Dockerfile
```

---

## Requirements

```
openai-whisper==20230314
transformers==4.28.1
torch==1.13.1
pydub==0.25.1
librosa==0.10.0
noisereduce==2.0.1
soundfile==0.12.1
numpy==1.24.3
tqdm==4.65.0
```

---

## Usage

```bash
# Transcribe a single file
python run_asr.py --input audio/speech.wav --language ur

# Batch transcription
python run_asr.py --input_dir audio/ --language ur --output transcriptions/

# With WER evaluation (if reference transcripts available)
python run_asr.py --input audio/speech.wav \
                  --reference transcripts/reference.txt \
                  --language ur
```

---

*BITS Hyderabad · Aug–Nov 2024*
*Harshal Bhambhani · 2026*
