# src/asr_pipeline.py
"""
ASR pipeline helpers:
- audio preprocessing with pydub (silence trim + normalization)
- optional noise reduction with librosa (or noisereduce)
- transcription using Whisper
- optional contextual correction using a masked LM (IndicBERT-like)
- evaluation: WER (jiwer) and CER (simple Levenshtein implementation)

NOTE: whisper here refers to the "openai/whisper" pip package.
"""

import os
from typing import List, Dict, Tuple, Optional
from io import BytesIO

import numpy as np
import pandas as pd

# audio libraries
from pydub import AudioSegment, effects, silence
import librosa
import soundfile as sf

# ASR
import whisper

# evaluation
from jiwer import wer

# transformers (for MLM correction) - optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# ----------------- Audio preprocess helpers -----------------

def normalize_and_trim(in_path: str, out_path: str, target_dBFS: float = -20.0,
                       silence_thresh: int = -40, min_silence_len: int = 500) -> None:
    """
    Load audio with pydub, normalize to target_dBFS, trim leading/trailing silence, export 16k mono WAV.
    - in_path: input audio file path
    - out_path: output file path (.wav)
    - target_dBFS: desired loudness
    - silence_thresh: dBFS threshold to consider silence (lower = quieter)
    - min_silence_len: ms of silence to consider trimming
    """
    audio = AudioSegment.from_file(in_path)
    # convert to mono
    audio = audio.set_channels(1)
    # normalize
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)
    # detect nonsilent regions and crop
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if nonsilent_ranges:
        start = nonsilent_ranges[0][0]
        end = nonsilent_ranges[-1][1]
        audio = audio[start:end]
    # export wav at 16k
    audio = audio.set_frame_rate(16000)
    audio.export(out_path, format="wav")
    return out_path

def reduce_noise_librosa(in_wav: str, out_wav: str, sr: int = 16000, use_noisereduce: bool = True) -> str:
    """
    Load audio with librosa, apply noise reduction, save out_wav.
    - If noisereduce is available, uses it; otherwise performs a simple spectral gating-like approach.
    """
    y, _ = librosa.load(in_wav, sr=sr)
    # simple fallback trimming (librosa)
    y_trim, _ = librosa.effects.trim(y)
    # optional noisereduce
    if use_noisereduce:
        try:
            import noisereduce as nr
            # estimate noise from first 0.5 seconds
            noise_sample = y_trim[: int(0.5 * sr)]
            y_reduced = nr.reduce_noise(audio_clip=y_trim, noise_clip=noise_sample, verbose=False)
        except Exception:
            y_reduced = y_trim
    else:
        y_reduced = y_trim
    # write to wav
    sf.write(out_wav, y_reduced, sr)
    return out_wav

# ----------------- ASR (Whisper) -----------------

def load_whisper_model(size: str = "small"):
    """
    Load whisper model. Choose sizes: tiny, base, small, medium, large.
    medium/large are heavier but more accurate.
    """
    model = whisper.load_model(size)
    return model

def transcribe_whisper(model, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> Dict:
    """
    Transcribe audio using whisper model.
    Returns transcription dict produced by model.transcribe(), containing 'text' and 'segments'.
    - language: ISO language code (e.g., 'ur' for Urdu) to bias the model (optional)
    - task: "transcribe" or "translate"
    """
    options = {"task": task}
    if language:
        options["language"] = language
    result = model.transcribe(audio_path, **options)
    # result has keys: text, segments (list), language, etc.
    return result

# ----------------- IndicBERT-based correction (optional) -----------------

class MLMCorrector:
    """
    Simple masked-language-model (MLM) based corrector.
    Approach (simple / naive):
      - For each sentence, for each token we create a masked version and ask the MLM for top predictions.
      - If top predicted token differs from the original and probability is reasonably high, replace.
    This is not a production-grade spell-corrector but can fix common token substitutions.
    """

    def __init__(self, model_name: str = "ai4bharat/indic-bert", device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        # prefer cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # mask token (some tokenizers use '<mask>' or '[MASK]')
        self.mask_token = self.tokenizer.mask_token

    def correct_sentence(self, sentence: str, top_k: int = 1, prob_threshold: float = 0.4) -> str:
        """
        Naively attempt to correct tokens in a sentence.
        - top_k: number of candidates to consider (we check top-1)
        - prob_threshold: minimum softmax prob to accept replacement
        """
        # simple whitespace tokenization for iteration (not subword): we will place mask for single token
        words = sentence.split()
        corrected = words.copy()
        for i, w in enumerate(words):
            # create masked sentence
            masked_words = words.copy()
            masked_words[i] = self.mask_token
            masked_text = " ".join(masked_words)
            # prepare inputs
            inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (1, seq_len, vocab)
            mask_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_index[0]) == 0:
                continue
            seq_idx = mask_index[1].item()
            # get softmax on that position
            scores = logits[0, seq_idx, :]
            probs = torch.softmax(scores, dim=0)
            topk = torch.topk(probs, top_k)
            top_ids = topk.indices.cpu().tolist()
            top_probs = topk.values.cpu().tolist()
            # decode top candidate
            candidate = self.tokenizer.decode([top_ids[0]]).strip()
            if candidate and top_probs[0] >= prob_threshold and candidate.lower() != w.lower():
                corrected[i] = candidate
        return " ".join(corrected)

# ----------------- Evaluation helpers -----------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute WER using jiwer (word error rate)."""
    return wer(reference, hypothesis)

def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (CER) using simple Levenshtein distance implementation."""
    # simple DP Levenshtein
    r = reference
    h = hypothesis
    n = len(r)
    m = len(h)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    # init matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,    # deletion
                          d[i][j - 1] + 1,    # insertion
                          d[i - 1][j - 1] + cost)  # substitution
    cer = d[n][m] / float(n)
    return cer

# ----------------- Batch-run helper -----------------

def process_single_audio(in_path: str,
                         tmp_dir: str = "/tmp",
                         do_denoise: bool = True,
                         use_noisereduce: bool = True,
                         whisper_model_size: str = "small",
                         language: Optional[str] = None,
                         mlm_corrector: Optional[MLMCorrector] = None) -> Dict:
    """
    Full pipeline on one file:
     - normalize/trim -> tmp_clean.wav
     - optional noise reduction -> tmp_denoised.wav
     - transcribe with Whisper
     - optional MLM correction
     - return dict with fields: original_file, cleaned_audio, transcript_raw, transcript_corrected
    """
    # prepare temp file names
    base = os.path.basename(in_path)
    name = os.path.splitext(base)[0]
    tmp_clean = os.path.join(tmp_dir, f"{name}_clean.wav")
    tmp_denoised = os.path.join(tmp_dir, f"{name}_den.wav")

    # 1) normalize & trim
    normalize_and_trim(in_path, tmp_clean)

    # 2) optional denoise
    if do_denoise:
        reduce_noise_librosa(tmp_clean, tmp_denoised, use_noisereduce=use_noisereduce)
        final_wav = tmp_denoised
    else:
        final_wav = tmp_clean

    # 3) load whisper model (lazy)
    model = load_whisper_model(whisper_model_size)

    # 4) transcribe
    result = transcribe_whisper(model, final_wav, language=language)
    transcript_raw = result.get("text", "").strip()

    # 5) optional MLM correction
    transcript_corrected = transcript_raw
    if mlm_corrector is not None:
        try:
            transcript_corrected = mlm_corrector.correct_sentence(transcript_raw)
        except Exception:
            # if MLM fails, keep raw
            transcript_corrected = transcript_raw

    return {
        "file": in_path,
        "clean_wav": final_wav,
        "transcript_raw": transcript_raw,
        "transcript_corrected": transcript_corrected,
        "meta": result.get("segments", [])
    }
