import os
import subprocess
import sys
import threading
import time
import webbrowser

here = os.path.dirname(os.path.abspath(__file__))
print("Python being used:", sys.executable)
print("Folder:", here)
print("Files here:", os.listdir(here))

try:
    import streamlit
    print("Streamlit version:", streamlit.__version__)
except ImportError:
    print("\nStreamlit is NOT installed for this Python.")
    print("Fix: open Command Prompt and run:")
    print(f'"{sys.executable}" -m pip install streamlit plotly pandas numpy')
    raise SystemExit

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "tbcc_app.py",
     "--server.headless", "true", "--server.port", "8501"],
    cwd=here,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

def open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")

threading.Thread(target=open_browser, daemon=True).start()

for line in proc.stdout:
    print(line, end="")
