"""Download Google's official Face Landmarker model, without running the app."""
from pathlib import Path
import urllib.request
import zipfile

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main():
    if MODEL_PATH.is_file() and zipfile.is_zipfile(MODEL_PATH):
        print("Model already available.")
        return
    MODEL_PATH.parent.mkdir(exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Downloaded model is not a valid task bundle.")
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    print("Face Landmarker model downloaded.")


if __name__ == "__main__":
    main()
