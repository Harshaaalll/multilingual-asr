# src/streamlit_asr.py
import streamlit as st
import tempfile
import os
import pandas as pd
from asr_pipeline import load_whisper_model, transcribe_whisper, MLMCorrector, compute_wer, compute_cer, normalize_and_trim, reduce_noise_librosa

st.set_page_config(page_title="ASR Demo (Urdu)", layout="wide")
st.title("ASR Demo — Preprocess → Whisper → IndicBERT correction")

# ---------------- Session state init ----------------
if "text_raw" not in st.session_state:
    st.session_state.text_raw = None
if "corrected" not in st.session_state:
    st.session_state.corrected = None

uploaded = st.file_uploader("Upload audio file (wav/mp3/m4a)", type=["wav","mp3","m4a","flac"])
lang = st.selectbox("Language (for Whisper bias)", options=[None,"ur","hi","en"], index=1)
model_size = st.selectbox("Whisper model size", options=["tiny","base","small","medium"], index=2)
do_denoise = st.checkbox("Apply noise reduction (librosa/noisereduce)", value=True)
enable_correct = st.checkbox("Try MLM correction (IndicBERT)", value=False)

if uploaded:
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp_in.write(uploaded.read())
    tmp_in.flush()
    tmp_in.close()

    st.audio(tmp_in.name)

    # ----------------- Run ASR -----------------
if st.button("Run ASR"):
    st.info("Running — this may take time (model downloads on first run)...")
    tmp_clean = tmp_in.name + "_clean.wav"
    normalize_and_trim(tmp_in.name, tmp_clean)
    tmp_denoised = tmp_in.name + "_den.wav"
    if do_denoise:
        reduce_noise_librosa(tmp_clean, tmp_denoised, use_noisereduce=True)
        final_wav = tmp_denoised
    else:
        final_wav = tmp_clean

    model = load_whisper_model(model_size)
    res = transcribe_whisper(model, final_wav, language=lang)
    st.session_state["text_raw"] = res.get("text", "").strip()

    corrected = st.session_state["text_raw"]
    if enable_correct:
        with st.spinner("Loading MLM corrector..."):
            corr = MLMCorrector()
            corrected = corr.correct_sentence(st.session_state["text_raw"])
    st.session_state["corrected"] = corrected

# ----------------- Show Results -----------------
if "text_raw" in st.session_state:
    st.subheader("Raw transcript")
    st.write(st.session_state["text_raw"])

if "corrected" in st.session_state:
    if enable_correct:
        st.subheader("Corrected (MLM)")
    st.write(st.session_state["corrected"])

# ----------------- Ground truth evaluation -----------------
st.subheader("Evaluate with Ground-Truth")
gt = st.text_area("Paste ground-truth text here", height=100, key="gt_input")

if st.button("Compute WER/CER"):
    if gt and gt.strip() and "corrected" in st.session_state:
        w = compute_wer(gt.strip(), st.session_state["corrected"])
        c = compute_cer(gt.strip(), st.session_state["corrected"])
        st.success(f"WER: {w:.3f} — CER: {c:.3f}")
    else:
        st.warning("Please enter ground-truth text and run ASR first.")


        # ----------------- Download transcript (Excel) -----------------
        df = pd.DataFrame([{
            "file": uploaded.name,
            "transcript_raw": text_raw,
            "transcript_corrected": corrected
        }])

        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Transcript")

        st.download_button(
            "Download Excel with transcript",
            output.getvalue(),
            file_name="transcript.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



