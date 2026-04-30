# src/run_asr.py
import os
import argparse
import glob
import pandas as pd
from .asr_pipeline import process_single_audio, MLMCorrector

def main(input_folder, output_csv, do_denoise=True, use_noisereduce=True, model_size="small", language=None, enable_correction=False, mlm_model_name="ai4bharat/indic-bert"):
    files = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        files.extend(glob.glob(os.path.join(input_folder, ext)))
    files = sorted(files)
    if not files:
        print("No audio files found in", input_folder)
        return

    mlm = None
    if enable_correction:
        print("Loading MLM corrector (this may take time)...")
        mlm = MLMCorrector(model_name=mlm_model_name)
    results = []
    for f in files:
        print("Processing:", f)
        res = process_single_audio(f, do_denoise=do_denoise, use_noisereduce=use_noisereduce, whisper_model_size=model_size, language=language, mlm_corrector=mlm)
        results.append(res)
    # assemble dataframe
    rows = []
    for r in results:
        rows.append({
            "file": os.path.basename(r["file"]),
            "transcript_raw": r["transcript_raw"],
            "transcript_corrected": r["transcript_corrected"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("Wrote", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--noisereduce", action="store_true")
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--lang", default=None)  # e.g., "ur"
    parser.add_argument("--correct", action="store_true")
    args = parser.parse_args()
    main(args.input_folder, args.output_csv, do_denoise=args.denoise, use_noisereduce=args.noisereduce, model_size=args.model_size, language=args.lang, enable_correction=args.correct)
