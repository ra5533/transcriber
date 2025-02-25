import os
import time
import logging
import whisper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
SUPPORTED_FORMATS = {".mp3", ".wav", ".mp4", ".mkv", ".mov", ".flv", ".aac", ".m4a"}
LOG_FILE = "processed_files.log"
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

class TranscriptionHandler(FileSystemEventHandler):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.processed_files = self.load_processed_files()

    def load_processed_files(self):
        if not os.path.exists(LOG_FILE):
            return set()
        with open(LOG_FILE, "r") as f:
            return {line.strip() for line in f}

    def log_processed_file(self, file_path):
        with open(LOG_FILE, "a") as f:
            f.write(f"{file_path}\n")
        self.processed_files.add(file_path)

    def is_supported(self, file_path):
        return os.path.splitext(file_path)[1].lower() in SUPPORTED_FORMATS

    def transcribe_file(self, file_path):
        try:
            result = self.model.transcribe(file_path)
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write(result["text"])
            self.log_processed_file(file_path)
            logging.info(f"Transcribed: {file_path}")
        except Exception as e:
            logging.error(f"Error transcribing {file_path}: {str(e)}")

    def on_created(self, event):
        if not event.is_directory and self.is_supported(event.src_path):
            if event.src_path not in self.processed_files:
                self.transcribe_file(event.src_path)

def scan_existing_files(directory, handler):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if handler.is_supported(file_path) and file_path not in handler.processed_files:
                handler.transcribe_file(file_path)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    target_directory = input("Enter directory to monitor: ").strip()

    if not os.path.isdir(target_directory):
        logging.error("Invalid directory path!")
        return

    # Load Whisper model
    logging.info("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE)
    logging.info("Model loaded.")

    # Initialize handler and observer
    event_handler = TranscriptionHandler(model)
    observer = Observer()
    observer.schedule(event_handler, target_directory, recursive=True)

    # Initial scan for existing files
    scan_existing_files(target_directory, event_handler)

    # Start monitoring
    observer.start()
    logging.info(f"Monitoring started for: {target_directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()