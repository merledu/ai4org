# Reflect Clone Desktop (Python + HTML/CSS)

A minimal Reflect-style hero page with a looping background video, wrapped as a desktop app using **pywebview** (Python).

## Run

1. Install Python 3.10+.
2. Open terminal in this folder and run:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

> Note: Put your looping video as `assets/bg.mp4`. A placeholder path is already referenced in `index.html`.

## Build EXE (Windows)

```bash
pip install pyinstaller
pyinstaller -F --add-data "index.html;." --add-data "styles.css;." --add-data "script.js;." --add-data "assets;assets" main.py
```
The built exe will be in `dist/main.exe`.

## Customize
- Edit text, links, and buttons in `index.html`.
- Tweak colors and glow in `styles.css`.
