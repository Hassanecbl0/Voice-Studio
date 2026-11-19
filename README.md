# Voice Studio

Repository containing the **xtts‑voice‑studio** project.

## Project structure

- `app.py` – entry point for the web app
- `core/` – core Python modules (audio utils, TTS engine, etc.)
- `requirements.txt` – Python dependencies
- `outputs/` – generated audio files (ignore in Git via `.gitignore`)
- `voices/` – sample voice wav files
- `run.bat` / `run.ps1` – convenient launch scripts

## Setup

```bash
# Clone the repo (if you haven’t already)
git clone https://github.com/Hassanecbl0/Voice-Studio.git
cd Voice-Studio

# Create a virtual environment and install deps
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run the app
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit your changes (`git commit -m "Add …"`)
4. Push and open a Pull Request

> **Note** – This README was added automatically to keep the repository organised.
