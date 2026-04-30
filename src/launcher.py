# src/launcher.py
import os, subprocess, webbrowser

def launch_application():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(this_dir, "streamlit_asr.py")
    subprocess.Popen(f"streamlit run \"{script}\"", shell=True)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    launch_application()
