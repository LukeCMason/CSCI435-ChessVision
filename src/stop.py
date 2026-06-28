"""
stop.py — Stop any running ChessVision (Streamlit) server.

Usage:
    python stop.py
"""
import subprocess
import sys

if sys.platform == "win32":
    subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq streamlit*"],
                   capture_output=True)
    subprocess.run(["taskkill", "/F", "/FI", "IMAGENAME eq streamlit.exe"],
                   capture_output=True)
else:
    subprocess.run(["pkill", "-f", "streamlit run main.py"], capture_output=True)

print("ChessVision server stopped.")
