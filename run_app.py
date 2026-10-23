import os
import sys
import time
import webbrowser
import subprocess
import uvicorn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Add backend directory to sys.path
sys.path.insert(0, BACKEND_DIR)

def main():
    print("=" * 70)
    print(" VoiceAI: AI-Powered Speech Emotion Recognition Platform")
    print("=" * 70)
    print(" Machine Learning (SVM, MLP) + Deep Learning (CNN, CNN-LSTM)")
    print(" Real-Time Audio + Whisper STT + Explainable AI + MySQL/SQLite")
    print("=" * 70)

    # Check if user wants dev mode (runs both Vite dev server and FastAPI)
    is_dev = "--dev" in sys.argv

    if is_dev:
        print("\nStarting Vite Dev Server (port 5173)...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        vite_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=FRONTEND_DIR,
            shell=True
        )
        print("Frontend running at: http://localhost:5173")
        time.sleep(2)
        webbrowser.open("http://localhost:5173")

    print("\nStarting FastAPI Server on http://localhost:8000 ...")
    print("API Documentation available at: http://localhost:8000/docs")
    if not is_dev:
        print("Single-port full-stack UI available at: http://localhost:8000")
        # Open browser in production mode
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://localhost:8000")
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    os.chdir(BACKEND_DIR)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=is_dev)

if __name__ == "__main__":
    main()
